---
title: "LangGraph ソースコード解析 Part 4：アルゴリズム中核（pregel/_algo.py・_runner.py・_write.py）— タスク準備・並列実行・Send / Command 解決"
date: 2026-09-14 18:42:49
categories: [AI]
tags: [Python, LLM, AI Agent, LangGraph, ソースコード解析, Pregel, 並列実行]
lang: ja
description: "LangGraph の BSP ステップを支えるアルゴリズム層（prepare_next_tasks / apply_writes）、並列タスク実行器（PregelRunner / FuturesDict）、および ChannelWrite による Send・Command 解決の内部実装を詳解します。"
---

Part 3 で Pregel の外側のループ構造（`tick()` → `runner.tick()` → `after_tick()`）を把握しました。Part 4 では、その各フェーズの計算実体を担うアルゴリズム中核の 3 ファイルを読み解きます。

- [`pregel/_algo.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_algo.py)：`prepare_next_tasks()`、`prepare_single_task()`、`apply_writes()`、`local_read()`
- [`pregel/_runner.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_runner.py)：`PregelRunner`、`FuturesDict`、`commit()`、`_panic_or_proceed()`、`_call()` / `_acall()`
- [`pregel/_write.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_write.py)：`ChannelWrite`、`_assemble_writes()` による `Send` / `Command` / State 更新の組み立て

---

## 1. タスク準備：`prepare_next_tasks()` の PUSH / PULL 統合モデル

[`pregel/_algo.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_algo.py) の `prepare_next_tasks()` は、次のスーパーステップで実行する `PregelExecutableTask` 辞書を生成します。LangGraph のタスクには **PUSH**（動的 `Send` パケットや Functional API の `@task` 呼び出し）と **PULL**（エッジ・チャネル更新によってトリガーされる通常ノード）の 2 系統があり、1 つの関数で統一的に計画されます。

```python
def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    *,
    for_execution: bool,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
    updated_channels: set[str] | None = None,
    ...
) -> dict[str, PregelTask] | dict[str, PregelExecutableTask]:
    input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] = {}
    checkpoint_id_bytes = binascii.unhexlify(checkpoint["id"].replace("-", ""))
    null_version = checkpoint_null_version(checkpoint)
    tasks: list[PregelTask | PregelExecutableTask] = []

    # 1. PUSH タスクの収集: TASKS チャネル（Topic[Send]）に積まれた Send パケットを展開
    tasks_channel = cast(Topic[Send] | None, channels.get(TASKS))
    if tasks_channel and tasks_channel.is_available():
        for idx, _ in enumerate(tasks_channel.get()):
            if task := prepare_single_task(
                (PUSH, idx), None, ...
            ):
                tasks.append(task)

    # 2. PULL 候補ノードの絞り込み最適化:
    #    前ステップで更新された updated_channels と逆引きマップ trigger_to_nodes があれば
    #    全ノード走査を避け、該当ノードだけを決定的なソート順で検査する
    if updated_channels and trigger_to_nodes:
        triggered_nodes: set[str] = set()
        for channel in updated_channels:
            if node_ids := trigger_to_nodes.get(channel):
                triggered_nodes.update(node_ids)
        candidate_nodes: Iterable[str] = sorted(triggered_nodes)
    elif not checkpoint["channel_versions"]:
        candidate_nodes = ()
    else:
        candidate_nodes = processes.keys()

    # 3. 各候補ノードについて _triggers() でバージョン比較し、実行タスク化
    for name in candidate_nodes:
        if task := prepare_single_task((PULL, name), None, ...):
            tasks.append(task)

    return {t.id: t for t in tasks}
```

### 決定的なタスク ID と `configurable` 注入（`prepare_single_task`）
`prepare_single_task()` が PULL タスクを生成する際、以下の 3 点が実行時の正確性を支えています。

1. **決定的タスク ID（`_xxhash_str`）**：`checkpoint_id_bytes`・`checkpoint_ns`・`step`・`name`・`PULL`・`*triggers` を連結して xxHash3-128 でハッシュ化します。これにより、クラッシュ後に同一チェックポイントから再起動しても**全く同じ `task_id`** が再生成され、`pending_writes` の照合やキャッシュヒットが成立します。
2. **スレッドセーフな書き込み口（`CONFIG_KEY_SEND`）**：各タスク専用の `writes: deque[tuple[str, Any]] = deque()` を作成し、その `writes.extend` を `CONFIG_KEY_SEND` として注入します（Python の `deque.extend` はスレッドセーフ）。
3. **ローカル先読み関数（`CONFIG_KEY_READ`）**：Part 1 で見た条件分岐の `fresh=True` 読み取りを支える `local_read(scratchpad, channels, managed, task)` を注入します。`local_read` は `channels[k].copy()` にそのタスク自身の `task.writes` だけを仮適用するため、他ノードと共有する本体 `channels` を一切汚さずに最新 State を計算できます。

---

## 2. BSP バリア適用：`apply_writes()` の 5 段階アルゴリズム

全タスクの実行終了後、`after_tick()` から呼ばれる [`pregel/_algo.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_algo.py) の `apply_writes()` が、チャネルへの一括更新を行います。

```python
def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
    # 1. task.path で決定的にソートし、並列タスク間の更新適用順を固定化
    tasks = sorted(tasks, key=lambda t: writes_sort_key(task_path_str(t.path)))
    bump_step = any(t.triggers for t in tasks)

    # 2. 各ノードが「どのトリガーチャネルのどのバージョンを見たか」を versions_seen に記録
    for task in tasks:
        checkpoint["versions_seen"].setdefault(task.name, {}).update(
            {
                chan: checkpoint["channel_versions"][chan]
                for chan in task.triggers
                if chan in checkpoint["channel_versions"]
            }
        )

    # 3. 今回購読されたトリガーチャネルに対し consume() を通知
    for chan in {c for t in tasks for c in t.triggers if c not in RESERVED and c in channels}:
        if channels[chan].consume() and next_version is not None:
            checkpoint["channel_versions"][chan] = next_version

    # 4. 制御用特殊チャネル（INTERRUPT / ERROR / PUSH 等）を除外し、チャネル別にグルーピングして update()
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        for chan, val in task.writes:
            if chan in (NO_WRITES, PUSH, RESUME, INTERRUPT, RETURN, ERROR, ERROR_SOURCE_NODE):
                pass
            elif chan in channels:
                pending_writes_by_channel[chan].append(val)

    updated_channels: set[str] = set()
    for chan, vals in pending_writes_by_channel.items():
        if channels[chan].update(vals) and next_version is not None:
            checkpoint["channel_versions"][chan] = next_version
            if channels[chan].is_available():
                updated_channels.add(chan)

    # 5. 未更新チャネルへ EMPTY_SEQ を通知（EphemeralValue / Topic の自動クリア）
    if bump_step:
        for chan in channels:
            if channels[chan].is_available() and chan not in updated_channels:
                if channels[chan].update(EMPTY_SEQ) and next_version is not None:
                    checkpoint["channel_versions"][chan] = next_version
                    if channels[chan].is_available():
                        updated_channels.add(chan)

    # 6. 次にトリガーされるノードが無い（暫定終了）場合、全チャネルに finish() を通知
    #    -> LastValueAfterFinish (defer=True) 等がここで利用可能になり、追加ステップが起動する
    if bump_step and updated_channels.isdisjoint(trigger_to_nodes):
        for chan in channels:
            if channels[chan].finish() and next_version is not None:
                checkpoint["channel_versions"][chan] = next_version
                if channels[chan].is_available():
                    updated_channels.add(chan)

    return updated_channels
```

Part 2 で見た `BaseChannel` の `consume()` / `update(EMPTY_SEQ)` / `finish()` の 3 フックが、この `apply_writes()` の中で完璧に噛み合って動いていることが確認できます。

---

## 3. 並列実行と障害ハンドリング：`PregelRunner` と `FuturesDict`

[`pregel/_runner.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_runner.py) の `PregelRunner` は、1 ステップ内の複数タスクをスレッドプール（`tick`）または `asyncio`（`atick`）で並列実行します。

{% mermaid %}
flowchart TD
    Start["runner.tick(tasks)"] --> Fast{"タスク1件 &<br/>timeout=None &<br/>waiter=None?"}
    Fast -- "Yes (高速パス)" --> Direct["スレッド生成せず直接 run_with_retry()"]
    Fast -- "No (並列パス)" --> Submit["各タスクを submit(run_with_retry) -> FuturesDict へ登録"]
    Submit --> Wait["FIRST_COMPLETED で待機"]
    Wait --> DoneCB["Future 完了 -> FuturesDict.on_done() -> runner.commit()"]
    DoneCB --> Yield["呼び出し元へ yield (ストリーム即時出力)"]
    Yield --> ErrCheck{"通常例外 &<br/>error_handler あり?"}
    ErrCheck -- "Yes" --> EH["schedule_error_handler() でハンドラタスクを動的投入"]
    EH --> Wait
    ErrCheck -- "No" --> StopCheck{"_should_stop_others()?"}
    StopCheck -- "致命的例外" --> Break["残タスクをキャンセル & _panic_or_proceed() で再送出"]
    StopCheck -- "継続 / GraphInterrupt" --> More{"未完了タスク残あり?"}
    More -- "Yes" --> Wait
    More -- "No" --> Panic["_panic_or_proceed()<br/>全 GraphInterrupt を統合して送出"]

    style Start fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Direct fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style DoneCB fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style Panic fill:#fce4ec,stroke:#c62828,color:#b71c1c
{% endmermaid %}

### 3.1 単一タスク高速パスと `FuturesDict` のコールバック設計
- **高速パス**：`len(tasks) == 1 and timeout is None and get_waiter is None` のときは、スレッドプールや Future を経由せず直接 `run_with_retry(t, ...)` を呼び出してオーバーヘッドをゼロにしています。
- **`FuturesDict.on_done`**：各 Future 完了時に即座に `self.commit(task, exc)` を呼び出し、完了カウンタをデクリメントします。また、`_should_stop_others` は**「`GraphBubbleUp`（`GraphInterrupt` / `ParentCommand`）やエラーハンドラで処理済みの例外では他タスクを中断しない」**ルールを実装しています。これにより、並列ノードの 1 つが `interrupt()` を呼んでも、兄弟ノードは途中キャンセルされず完了まで走り切ってから、最後に `_panic_or_proceed()` がすべての `GraphInterrupt` を 1 つのタプルに統合して送出します。

### 3.2 `commit()` によるタスク結果の永続化
```python
def commit(self, task: PregelExecutableTask, exception: BaseException | None) -> None:
    if isinstance(exception, asyncio.CancelledError):
        task.writes.append((ERROR, exception))
        self.put_writes()(task.id, task.writes)
    elif exception:
        if isinstance(exception, GraphInterrupt):
            # 中断ペイロードと、そのタスク内で既に消費した RESUME 値を保存
            if exception.args[0]:
                writes = [(INTERRUPT, exception.args[0])]
                if resumes := [w for w in task.writes if w[0] == RESUME]:
                    writes.extend(resumes)
                self.put_writes()(task.id, writes)
        elif isinstance(exception, GraphBubbleUp):
            pass  # ParentCommand 等は _panic_or_proceed で親へ伝播
        else:
            # 通常例外: ERROR を保存し、ノードに error_handler があればルーティング印を記録
            task.writes.append((ERROR, exception))
            if self._should_route_to_error_handler(task):
                task.writes.append((ERROR_SOURCE_NODE, task.name))
                self._handled_exception_ids.add(id(exception))
            self.put_writes()(task.id, task.writes)
    else:
        # 出力ゼロで正常終了したタスクには NO_WRITES を記録（再開時に再実行しないため）
        if all(chan in CONTROL_WRITES for chan, _ in task.writes):
            task.writes.append((NO_WRITES, None))
        self.put_writes()(task.id, task.writes)
```

また、Functional API の `@task` 呼び出しを実現する `_call()` / `_acall()` も `PregelRunner` 内に定義されており、実行中タスクから `schedule_task(parent, call_counter(), Call(...))` を介して動的に PUSH タスクを同一 `FuturesDict` に追加し、`chain_future` で `commit()` 完了後に親へ結果を返す仕組みになっています。

---

## 4. `ChannelWrite` による `Send` / `Command` / State 更新の解決

ノード関数の戻り値が最終的にどのように `(channel, value)` タプル列へ変換されるかを担うのが [`pregel/_write.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_write.py) の `ChannelWrite` です。

```python
class ChannelWrite(RunnableCallable):
    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        allow_passthrough: bool = True,
    ) -> None:
        for w in writes:
            if isinstance(w, ChannelWriteEntry):
                if w.channel == TASKS:
                    # ユーザーが直接 TASKS チャネルへ書くことは禁止（Send オブジェクト経由のみ許可）
                    raise InvalidUpdateError("Cannot write to the reserved channel TASKS")
        write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        write(_assemble_writes(writes))


def _assemble_writes(
    writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
) -> list[tuple[str, Any]]:
    tuples: list[tuple[str, Any]] = []
    for w in writes:
        if isinstance(w, Send):
            # Send パケットは自動的に予約チャネル TASKS ("__pregel_tasks") へ変換
            tuples.append((TASKS, w))
        elif isinstance(w, ChannelWriteTupleEntry):
            # StateGraph の _get_updates / _control_branch がここで実行される
            if ww := w.mapper(w.value):
                tuples.extend(ww)
        elif isinstance(w, ChannelWriteEntry):
            value = w.mapper(w.value) if w.mapper is not None else w.value
            if value is SKIP_WRITE:
                continue
            if w.skip_none and value is None:
                continue
            tuples.append((w.channel, value))
    return tuples
```

{% mermaid %}
flowchart LR
    Ret["ノード戻り値<br/>dict / Command / Send"] --> CW["ChannelWrite._write()"]
    CW --> AW["_assemble_writes()"]
    AW --> E1["ChannelWriteTupleEntry(_get_updates)<br/>-> (state_key, value)"]
    AW --> E2["ChannelWriteTupleEntry(_control_branch)<br/>-> (branch:to:node, None)<br/>or (TASKS, Send)<br/>or raise ParentCommand"]
    AW --> E3["Send インスタンス<br/>-> (TASKS, Send)"]
    E1 --> SEND["config[CONF][CONFIG_KEY_SEND]<br/>(task.writes.extend)"]
    E2 --> SEND
    E3 --> SEND

    style Ret fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style AW fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style SEND fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

---

## 5. まとめ

Part 4 では、Pregel エンジン内部のアルゴリズムと並列実行制御を解析しました。

- **`prepare_next_tasks()`** は `TASKS` チャネルからの PUSH タスクと、`trigger_to_nodes` で絞り込んだ PULL タスクを統一的に計画し、xxHash3 による決定的 ID を付与する
- **`apply_writes()`** は `task.path` 順の決定的適用と `consume` / `update` / `finish` の 3 段階フックで BSP バリアを厳密に実行する
- **`PregelRunner`** は `FuturesDict` と `commit()` により、高速パス・並列実行・エラーハンドラ動的投入・並列 `GraphInterrupt` 統合をクリーンに実現している
- **`ChannelWrite`** は `Send`・`Command`・State 更新をすべてスレッドセーフな `task.writes.extend` へのタプル列へ正規化する

次回の Part 5 では、Human-in-the-Loop とタイムトラベルを支える **Checkpointer 永続化・`interrupt()` 例外制御と Resume 機構（`pregel/_checkpoint.py`, `types.py`）** を深掘りします。
