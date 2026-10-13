---
title: "LangGraph ソースコード解析 Part 5：Checkpointer 永続化・interrupt() 例外制御と Resume 機構（pregel/_checkpoint.py・types.py）"
date: 2026-09-18 05:58:39
categories: [AI]
tags: [Python, LLM, AI Agent, LangGraph, ソースコード解析, Checkpointer, Human-in-the-Loop]
lang: ja
description: "LangGraph の耐障害性・タイムトラベル・Human-in-the-Loop を支える Checkpoint 生成・復元ロジックと、interrupt() 関数による例外制御・Resume マッチング・スキーマ検証の内部実装を詳解します。"
---

LangGraph が単なる DAG ランナーと一線を画す最大の理由は、**任意のスーパーステップでの完全な状態永続化（Checkpointing）** と、ノードの途中で人間の承認や追加情報を待って再開できる **Human-in-the-Loop（`interrupt()` / `Command(resume=...)`）** です。

Part 5 では、これらを支える以下の 2 ファイルを中心にソースコードを解析します。

- [`pregel/_checkpoint.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_checkpoint.py)：`empty_checkpoint()`、`create_checkpoint()`、`channels_from_checkpoint()`、`_mark_bumps_seen()`、`DeltaChannel` のスナップショット＆リプレイ制御
- [`types.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py)：`Interrupt`、`_interrupt_id()`、`interrupt()`、`Command`、`_validate_resume()`、`Overwrite`

---

## 1. `Checkpoint` のデータ構造と `create_checkpoint()`

[`pregel/_checkpoint.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_checkpoint.py) を見ると、現在のチェックポイントスキーマは `LATEST_VERSION = 4` で、ID には時刻順ソート可能な **UUIDv6**（`uuid6(clock_seq=step)`）が使われています。

```python
def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,                         # スキーマバージョン (4)
        id=str(uuid6(clock_seq=-2)),              # 単調増加する UUIDv6（ステップ番号を clock_seq に埋込）
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},                        # 各チャネルの永続化値（または _DeltaSnapshot）
        channel_versions={},                      # 各チャネルの現在バージョン（単調増加）
        versions_seen={},                         # {node_name: {channel: version}} 各ノードの既読バージョン
    )
```

`create_checkpoint()` は、現在の `channels` から新しい `Checkpoint` を組み立てる中心関数です。特に **`DeltaChannel` のスナップショット最適化とバージョンバンプの整合性** が緻密に実装されています。

```python
def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    channels_to_snapshot: set[str] | None = None,
    stored_versions: ChannelVersions | None = None,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
) -> Checkpoint:
    ts = datetime.now(timezone.utc).isoformat()
    channels_to_snapshot = channels_to_snapshot or set()
    bumped: dict[str, tuple[Any, Any]] = {}

    if channels is None:
        values = checkpoint["channel_values"]
        channel_versions = checkpoint["channel_versions"]
    else:
        values = {}
        channel_versions = dict(checkpoint["channel_versions"])
        for k in channels:
            ch = channels[k]
            if k not in channel_versions:
                # 一度も書かれていないチャネルへの強制スナップショット（フォーク時など）
                if k in channels_to_snapshot and get_next_version is not None:
                    channel_versions[k] = get_next_version(None, None)
                    bumped[k] = (None, channel_versions[k])
                    values[k] = _DeltaSnapshot(ch.get())
                continue
            if k in channels_to_snapshot:
                # Checkpointer.put() はバージョンが進んだチャネルの blob だけを保存するため、
                # 今回更新がなくてもスナップショットを強制保存したい場合はバージョンを 1 つ進める
                if (
                    get_next_version is not None
                    and stored_versions is not None
                    and channel_versions[k] == stored_versions.get(k)
                ):
                    old = channel_versions[k]
                    channel_versions[k] = get_next_version(old, None)
                    bumped[k] = (old, channel_versions[k])
                values[k] = _DeltaSnapshot(ch.get())
            else:
                # 通常チャネルは ch.checkpoint() を格納（DeltaChannel は MISSING を返し省略）
                v = ch.checkpoint()
                if v is not MISSING:
                    values[k] = v

    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=channel_versions,
        # スナップショット目的の擬似バージョンバンプで購読ノードが誤発火しないよう
        # versions_seen を同期的に進める（_mark_bumps_seen）
        versions_seen=_mark_bumps_seen(
            checkpoint["versions_seen"], bumped, trigger_to_nodes or {}
        ),
        updated_channels=None if updated_channels is None else sorted(updated_channels),
    )
```

### `_mark_bumps_seen()` の巧妙な役割
Checkpointer はネットワーク・ストレージ帯域を節約するため、「前回保存時から `channel_versions[k]` が変化したチャネルの blob だけ」を書き込みます。そのため、`DeltaChannel` が `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` 到達などで「書き込みの無いステップにスナップショットを保存したい」場合、意図的にバージョンをバンプする必要があります。

しかし、単純にバージョンだけ上げると「新しい書き込みがあった」と誤認され、購読ノードや `interrupt_before` が再発火してしまいます。そこで `_mark_bumps_seen()` が全ノードの `versions_seen` を新バージョンへ進めつつ、`SNAPSHOT_BUMPS` に「本当はどのバージョンを読んでいたか」を退避記録することで、ストレージ最適化と実行セマンティクスを完全に分離しています。

---

## 2. チェックポイント復元：`channels_from_checkpoint()`

スレッド再開や `get_state()` 時にチェックポイントからライブなチャネル辞書を再構築するのが `channels_from_checkpoint()`（および非同期版 `achannels_from_checkpoint()`）です。

```python
def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    channel_specs = {k: v for k, v in specs.items() if isinstance(v, BaseChannel)}
    managed_specs = {k: v for k, v in specs.items() if not isinstance(v, BaseChannel)}

    # 1. channel_versions には存在するが channel_values に blob が無い
    #    DeltaChannel（または旧 DeltaChannel からの移行 BinaryOperatorAggregate）を抽出
    delta_channels = _delta_channels_to_replay(channel_specs, checkpoint)
    _require_saver_for_history(delta_channels, saver, config)

    # 2. リプレイが必要な全 DeltaChannel の祖先履歴を 1 回のバッチ呼び出しで一括取得
    histories: Mapping[str, DeltaChannelHistory] = {}
    if delta_channels and saver is not None and config is not None:
        histories = saver.get_delta_channel_history(
            config=config, channels=delta_channels
        )

    # 3. 各チャネルをハイドレート（リプレイ対象は seed + writes で復元、他は from_checkpoint）
    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        if k in histories:
            ch = _replay(spec, histories[k])
        else:
            ch = spec.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
        channels[k] = ch
    return channels, managed_specs
```

{% mermaid %}
flowchart LR
    CP["ロード済み Checkpoint<br/>(channel_values / channel_versions)"] --> Check{"channel_values[k] は<br/>存在する?"}
    Check -- "Yes (通常チャネル / _DeltaSnapshot)" --> Direct["spec.from_checkpoint(val)"]
    Check -- "No (DeltaChannel かつ version あり)" --> Batch["saver.get_delta_channel_history()<br/>全対象チャネルを 1 回でバッチ取得"]
    Batch --> Seed["seed (_DeltaSnapshot) で初期化"]
    Seed --> Replay["DeltaChannel.replay_writes(writes)<br/>最後の Overwrite 以降を一括 fold"]
    Direct --> Live["ライブな channels 辞書"]
    Replay --> Live

    style CP fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Batch fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style Live fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

---

## 3. `interrupt()` の正体：例外ベースの協調的中断と Resume マッチング

LangGraph の Human-in-the-Loop API である `interrupt()`（[`types.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py)）は、実はわずか 35 行ほどの極めて洗練された実装です。Python のコルーチンや継続（continuation）を使わず、**「タスク単位の Scratchpad＋例外（`GraphInterrupt`）＋ノード先頭からの決定的リプレイ」** で中断・再開を実現しています。

```python
def _interrupt_id(ns: str, answered: int) -> str:
    # タスクの namespace と「既に回答済みの interrupt 数 (answered)」から
    # 決定的な 128-bit xxHash3 ID を生成する。
    # 0 件目のときは純粋な ns ハッシュとなり、旧バージョンとの後方互換性を維持。
    return xxh3_128_hexdigest((f"{ns}{NS_SEP}{answered}" if answered else ns).encode())


def interrupt(
    value: Any, *, response_schema: dict[str, Any] | type | None = None
) -> Any:
    conf = get_config()["configurable"]
    # Pydantic モデル / TypedDict / dataclass が渡されたら TypeAdapter を構築
    adapter = (
        None
        if response_schema is None or isinstance(response_schema, dict)
        else TypeAdapter(response_schema)
    )

    # 1. このタスク実行内で何回目の interrupt() 呼び出しかを単調カウンタで取得 (0, 1, 2, ...)
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()

    # 2. 既に過去の再開で回答済みの interrupt であれば、保存済みリストから対応する値を返す
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            v = scratchpad.resume[idx]
            validated = _validate_resume(adapter, v) if adapter else v
            # 再開済み履歴をタスク writes の RESUME チャネルへ再記録しておく
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume[: idx + 1])])
            return validated

    # 3. 今回の Command(resume=...) で渡された新しい回答値を取得（1 回消費）
    v = scratchpad.get_null_resume(True)
    if v is not None:
        assert len(scratchpad.resume) == idx, (scratchpad.resume, idx)
        # スキーマ検証（失敗時は _mark_invalid_resume を付けて ValidationError 送出、何も保存しない）
        validated = _validate_resume(adapter, v) if adapter else v
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return validated

    # 4. 回答がまだ無ければ GraphInterrupt 例外を送出してノード実行を中断
    raise GraphInterrupt(
        (
            Interrupt(
                value=value,
                id=_interrupt_id(
                    conf[CONFIG_KEY_CHECKPOINT_NS], len(scratchpad.resume)
                ),
                response_schema=adapter.json_schema() if adapter else response_schema,
            ),
        )
    )
```

### 1 つのノードに複数の `interrupt()` がある場合の動作フロー

ノード内に `a = interrupt("Q1")` と `b = interrupt("Q2")` の 2 回の中断がある場合、再開時の挙動は以下のシーケンスになります。

{% mermaid %}
sequenceDiagram
    participant Client as "クライアント"
    participant Loop as "PregelLoop / Runner"
    participant Node as "ノード関数 (先頭から再実行)"
    participant SP as "PregelScratchpad"

    Note over Client,SP: 1回目: 初回実行
    Client->>Loop: stream(input, config)
    Loop->>Node: ノード開始 (scratchpad.resume = [])
    Node->>SP: interrupt("Q1") -> idx=0, 回答なし
    Node--xLoop: raise GraphInterrupt(Interrupt("Q1", id=id0))
    Loop->>Loop: commit() が pending_writes に (INTERRUPT, ...) 保存
    Loop-->>Client: {"__interrupt__": (Interrupt("Q1", id=id0),)}

    Note over Client,SP: 2回目: Q1 へ回答
    Client->>Loop: stream(Command(resume="A1"), config)
    Loop->>Node: ノードを先頭から再実行 (null_resume="A1")
    Node->>SP: interrupt("Q1") -> idx=0, get_null_resume()="A1"
    SP-->>Node: "A1" を返却 & task.writes に (RESUME, ["A1"]) 記録
    Node->>SP: interrupt("Q2") -> idx=1, 回答なし
    Node--xLoop: raise GraphInterrupt(Interrupt("Q2", id=id1))
    Loop->>Loop: commit() が (INTERRUPT, Q2) と (RESUME, ["A1"]) を両方保存
    Loop-->>Client: {"__interrupt__": (Interrupt("Q2", id=id1),)}

    Note over Client,SP: 3回目: Q2 へ回答
    Client->>Loop: stream(Command(resume="A2"), config)
    Loop->>Node: ノードを先頭から再実行 (resume=["A1"], null_resume="A2")
    Node->>SP: interrupt("Q1") -> idx=0 < len(resume) -> "A1" を即返却
    Node->>SP: interrupt("Q2") -> idx=1, get_null_resume()="A2" -> "A2" を返却
    Node-->>Loop: 最終 State 更新を返却して完了
{% endmermaid %}

### 実装から導かれる重要な実務上の注意点
1. **副作用は `interrupt()` より前に書かない（または `@task` に包む）**：再開時にノードは必ず先頭から再実行されるため、`interrupt()` 前の API 呼び出しや DB 書き込みは再実行されます。これを避けたい処理は Part 6 で解説する `@task` として切り出すことで、Checkpointer の `pending_writes` から結果を再利用できます。
2. **`interrupt()` の呼び出し順序は決定的でなければならない**：`idx = scratchpad.interrupt_counter()` と `scratchpad.resume[idx]` のインデックス照合に依存しているため、条件分岐で `interrupt()` の順序が実行ごとに変わると回答値がずれます。
3. **`response_schema` のバリデーション失敗は非破壊**：`_validate_resume()` が `ValidationError` に `_mark_invalid_resume` を付与して送出すると、`scratchpad.resume.append(v)` も `CONFIG_KEY_SEND` も実行されません。さらに Part 4 で見た `_should_stop_others()` が `is_invalid_resume(exc)` を除外するため、同じステップの並列タスクを破壊せずにクライアントへ再入力を促せます。

---

## 4. `Command` の設計と親子グラフ間通信

[`types.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) の `Command` は、`update`（状態更新）・`goto`（動的遷移）・`resume`（中断再開）・`graph`（`None` または `Command.PARENT`）の 4 機能を 1 つのイミュータブルな dataclass に統合しています。

- **`_update_as_tuples()`**：`dict`、`list[tuple[str, Any]]`、Pydantic / dataclass インスタンス、あるいは単一値（`ROOT`）のいずれが `update` に渡されても統一的な `(channel, value)` タプル列へ正規化します。
- **`Command.PARENT`**：Part 1 で見たように、サブグラフのノードが `Command(graph=Command.PARENT, ...)` を返すと `_control_branch()` が `ParentCommand(command)` 例外を送出し、`GraphBubbleUp` としてサブグラフの実行を抜け出し、親グラフの実行コンテキストで状態更新とルーティングが適用されます。

---

## 5. まとめ

Part 5 では、LangGraph のチェックポイント永続化と Human-in-the-Loop の中核実装を読み解きました。

- **`create_checkpoint()` と `_mark_bumps_seen()`** は、バージョン差分保存と `DeltaChannel` のスナップショット最適化を、購読ノードの誤発火なしに両立させている
- **`channels_from_checkpoint()`** は、リプレイが必要な `DeltaChannel` をバッチ取得して `_DeltaSnapshot` 基点から高速復元する
- **`interrupt()`** は `PregelScratchpad` のインデックス照合と `GraphInterrupt` 例外・`commit()` の `RESUME` 永続化の組み合わせにより、複数回の中断や並列中断、型付き `response_schema` 検証までを驚くほど簡潔に実現している

最終回となる Part 6 では、**Functional API（`func/__init__.py`）** と、v1/v2/v3 を支える **Streaming 多重化パイプライン（`stream/run_stream.py`, `pregel/_messages.py`）** を解析します。
