---
title: "LangGraph ソースコード解析 Part 3：Pregel BSP 実行ループ（pregel/main.py・_loop.py）の Plan → Execution → Update サイクル"
date: 2026-09-13 00:44:13
categories: [AI]
tags: [Python, LLM, AI Agent, LangGraph, ソースコード解析, Pregel, BSP]
lang: ja
description: "LangGraph の実行エンジン中核である Pregel.stream() と PregelLoop の BSP（Bulk Synchronous Parallel）サイクルを、tick()・runner.tick()・after_tick()・Durability モードの実装から徹底解析します。"
---

Part 1・Part 2 で `StateGraph` のコンパイルと `BaseChannel` の仕組みを追い終えました。Part 3 では、いよいよコンパイル済みグラフを実際に駆動する **Pregel 実行エンジン** の心臓部に迫ります。

LangGraph の実行モデルは Google の分散グラフ処理モデル **Pregel（BSP: Bulk Synchronous Parallel）** に基づいています。各「スーパーステップ（superstep）」は厳密に **Plan（タスク計画）→ Execution（並列実行）→ Update（チャネル一括適用・永続化）** の 3 フェーズに分かれ、ステップ実行中はチャネルが不変に保たれるため、並列ノード間の競合や部分的な状態汚染を防げます。

本記事では以下の 2 ファイルを中心に解析します。

- [`pregel/main.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)：`Pregel.stream()` / `astream()` のオーケストレーション、コールバック・Runtime 構築、Durability 制御
- [`pregel/_loop.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_loop.py)：`PregelLoop`・`SyncPregelLoop`・`AsyncPregelLoop` の状態機械、`_first()`、`tick()`、`after_tick()`、`_put_checkpoint()`、`_suppress_interrupt()`

---

## 1. `Pregel.stream()` から見る BSP メインループ

`invoke()` や `ainvoke()` も内部的には `stream()` / `astream()` をラップして最終結果を取り出しているに過ぎません。つまり [`pregel/main.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py) の `Pregel.stream()` こそが LangGraph 実行の単一エントリーポイントです。

```python
with SyncPregelLoop(
    input,
    stream=StreamProtocol(stream.put, stream_modes),
    config=config,
    store=store,
    cache=cache,
    checkpointer=checkpointer,
    nodes=self.nodes,
    specs=self.channels,
    output_keys=output_keys,
    input_keys=self.input_channels,
    stream_keys=self.stream_channels_asis,
    interrupt_before=interrupt_before_,
    interrupt_after=interrupt_after_,
    manager=run_manager,
    durability=durability_,
    trigger_to_nodes=self.trigger_to_nodes,
    migrate_checkpoint=self._migrate_checkpoint,
    retry_policy=self.retry_policy,
    cache_policy=self.cache_policy,
    has_graph_lifecycle_callbacks=bool(graph_callback_manager.handlers),
) as loop:
    # 1. タスク並列実行器 PregelRunner を生成（loop.submit / loop.put_writes を弱参照で接続）
    runner = PregelRunner(
        submit=config[CONF].get(
            CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
        ),
        put_writes=weakref.WeakMethod(loop.put_writes),
        node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
        node_error_handler_map=self.node_error_handler_map,
        schedule_error_handler=loop.schedule_error_handler,
    )

    # 2. BSP スーパーステップループ: Plan -> Execution -> Update
    while loop.tick():
        # [Cache] キャッシュヒットしたタスクの writes を事前復元（実行をスキップ）
        for task in loop.match_cached_writes():
            loop.put_writes(task.id, task.writes, cached=True)

        # [Execution] 未完了タスク (not t.writes) を並列実行し、完了ごとにストリーム出力を yield
        for _ in runner.tick(
            [t for t in loop.tasks.values() if not t.writes],
            timeout=self.step_timeout,
            get_waiter=get_waiter,
            schedule_task=loop.accept_push,
        ):
            yield from _output(
                stream_mode, print_mode, subgraphs,
                stream.get, queue.Empty, version,
                _output_mapper, _state_mapper,
            )

        # [Update] 全タスクの writes をチャネルへ一括適用し、チェックポイントを保存
        loop.after_tick()
        emit_graph_lifecycle_events(loop)

        # durability="sync" の場合はここでチェックポイント保存の完了を同期的に待機
        if durability_ == "sync":
            loop._put_checkpoint_fut.result()
```

### 注目すべきアーキテクチャの工夫
1. **`weakref.WeakMethod` による循環参照回避**：`PregelRunner` に渡す `loop.submit` と `loop.put_writes` は弱参照で渡されており、ジェネレータの早期クローズや例外発生時にも `PregelLoop` がリークしない設計になっています。
2. **リアルタイムストリーミング用 `get_waiter`**：`messages` / `custom` / `tools` / `subgraphs` モードが有効な場合、`stream.wait` をバックグラウンドスレッドで待機する単一の `waiter` Future を `runner.tick()` へ差し込みます。これにより、ノードが実行中でもトークンが `SyncQueue` に届いた瞬間に `concurrent.futures.wait(..., FIRST_COMPLETED)` が起床し、即座に `yield from _output(...)` へ制御を返せます。
3. **`Runtime` のマージ**：親グラフからサブグラフが呼ばれた場合、`parent_runtime.merge(runtime)` によって `context`・`store`・`stream_writer`・`control`（協調的ドレイン用 `RunControl`）が安全に継承されます。

{% mermaid %}
flowchart TD
    Enter["SyncPregelLoop.__enter__()<br/>チェックポイント復元 & _first() 入力適用"] --> Tick{"loop.tick()<br/>Plan フェーズ"}
    Tick -- "step > stop" --> Out["status = out_of_steps<br/>GraphRecursionError"]
    Tick -- "tasks が空" --> Done["status = done<br/>正常終了"]
    Tick -- "interrupt_before 該当" --> IntB["GraphInterrupt 送出<br/>(__exit__ で抑制 & 出力)"]
    Tick -- "タスクあり (True)" --> Cache["match_cached_writes()<br/>キャッシュ済みタスクをスキップ"]
    Cache --> Exec["runner.tick()<br/>Execution フェーズ (並列実行 + リアルタイム yield)"]
    Exec --> After["loop.after_tick()<br/>Update フェーズ (apply_writes + _put_checkpoint)"]
    After -- "interrupt_after 該当" --> IntA["GraphInterrupt 送出"]
    After -- "継続" --> Sync{"durability == sync?"}
    Sync -- "Yes" --> Wait["put_checkpoint_fut.result()"]
    Sync -- "No" --> Tick
    Wait --> Tick

    style Enter fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Tick fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style Exec fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style After fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

---

## 2. コンテキスト開始時の初期化：`_first()` の分岐ロジック

`SyncPregelLoop.__enter__()`（非同期は `AsyncPregelLoop.__aenter__()`）は、チェックポインタから最新（または指定 `checkpoint_id`）の状態をロードして `self.channels` を復元した後、[`pregel/_loop.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_loop.py) の `_first()` を呼び出します。

`_first()` は「新規実行」「中断からの再開（Resume）」「過去チェックポイントへのタイムトラベル（Fork）」の 3 パターンを厳密に判別します。

```python
# 1. 再開 (is_resuming) の判定:
#    既存チェックポイントが存在し、かつ入力が None / Command / 同一 run_id / サブグラフからの明示フラグ
is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
    configurable.get(
        CONFIG_KEY_RESUMING,
        self.input is None
        or input_is_command
        or (
            not self.is_nested
            and self.config.get("metadata", {}).get("run_id")
            == self.checkpoint_metadata.get("run_id", MISSING)
        ),
    )
)

# 2. タイムトラベル (is_time_traveling) の判定:
#    checkpoint_id が明示指定されており、かつ通常の Command(resume=...) ではない場合
is_time_traveling = self._addressed and (
    (
        self.is_nested
        and configurable.get(CONFIG_KEY_CHECKPOINT_NS, "")
        in configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {})
    )
    or not (
        (input_is_command and cast(Command, self.input).resume is not None)
        or configurable.get(CONFIG_KEY_RESUMING, False)
    )
)

# タイムトラベル時は古い RESUME 書き込みを破棄し、interrupt() が再発火できるようにする
self.is_replaying = is_time_traveling or self._addressed_checkpoint_superseded
if is_time_traveling:
    self.checkpoint_pending_writes = [
        w for w in self.checkpoint_pending_writes if w[1] != RESUME
    ]
```

この判別結果に応じて、`_first()` は以下の 3 つのいずれかを実行します。
- **`Command` 入力時**：`map_command()` でタスク別 writes に展開し、複数 interrupt がある場合の ID 指定チェック（`CONFIG_KEY_STRICT_RESUME` / `UnknownInterruptError`）を行った上で `put_writes()`（フォーク時は `carried` に退避）。
- **再開 / フォーク時（`is_resuming=True`）**：`INTERRUPT` の `versions_seen` を最新バージョンへ進めて同じ `interrupt_before` で再停止しないようにし、タイムトラベル時は `source="fork"` の分岐チェックポイントを先に切って新ブランチを確立します。
- **新規入力時（`elif input_writes := deque(map_input(...))`）**：前回未完了だったタスクを破棄しつつ `apply_writes()` で入力チャネル（`START`）を更新し、`source="input"` の初期チェックポイントを保存します。

---

## 3. Plan フェーズ：`PregelLoop.tick()` の 6 ステップ

各スーパーステップの冒頭で呼ばれる `tick()`（[`pregel/_loop.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_loop.py)）は、実行すべきタスク集合を確定します。

```python
def tick(self) -> bool:
    # 1. recursion_limit（self.stop）到達チェック
    if self.step > self.stop:
        self.status = "out_of_steps"
        self._prepare_tasks_and_emit_checkpoint()
        return False

    # 2. prepare_next_tasks() で次ステップのタスク辞書 self.tasks を構築し、
    #    "checkpoints" ストリームイベントを発行
    self._prepare_tasks_and_emit_checkpoint()

    # 3. 実行すべきタスクが 0 件ならグラフ終了
    if not self.tasks:
        self.status = "done"
        return False

    # 4. 前回途中で中断・障害が起きたステップの再開時、既に成功していたノードの
    #    pending_writes を復元し（再実行をスキップ）、失敗タスクのエラーハンドラを復旧
    if self._reapplies_pending_writes and self.checkpoint_pending_writes:
        self._reapply_writes_to_succeeded_nodes(self.tasks)
        self._resume_error_handlers_if_applicable()

    # 5. interrupt_before ブレークポイント判定
    if self.interrupt_before and should_interrupt(
        self.checkpoint, self.interrupt_before, self.tasks.values()
    ):
        self.status = "interrupt_before"
        raise GraphInterrupt()

    # 6. 協調的ドレイン（グレースフルシャットダウン）要求チェック
    if self.control is not None and self.control.drain_requested:
        self.status = "draining"
        return False

    # 7. "tasks" デバッグイベント出力 & 復元済みタスクの出力を再送出
    self._emit("tasks", map_debug_tasks, self.tasks.values())
    for task in self.tasks.values():
        if task.writes:
            self.output_writes(task.id, task.writes, cached=True)

    return True
```

ここで極めて重要なのが **ステップ 4 の `_reapply_writes_to_succeeded_nodes()`** です。例えば並列実行されたノード A と B のうち、A が成功して B が失敗（または `interrupt()`）した場合、チェックポインタの `pending_writes` には A の出力だけが保存されています。再実行時、`tick()` は A の `task.writes` に保存済みの出力を戻して空でなくすることで、直後の `runner.tick([t for t in loop.tasks.values() if not t.writes])` が **A をスキップして B だけを再実行する** という部分再実行（fault tolerance）を実現しています。

---

## 4. Update フェーズ：`after_tick()` とチェックポイント永続化順序保証

`runner.tick()` でステップ内の全タスクが完了すると、`loop.after_tick()` が呼ばれます。

```python
def after_tick(self) -> None:
    writes = [w for t in self.tasks.values() for w in t.writes]

    # 1. 全タスクの writes を決定的な順序でチャネルへ一括適用（BSP バリア）
    self.updated_channels = apply_writes(
        self.checkpoint,
        self.channels,
        self.tasks.values(),
        self.checkpointer_get_next_version,
        self.trigger_to_nodes,
    )

    # 2. output_keys に変化があれば "values" ストリームイベントを発行
    if not self.updated_channels.isdisjoint(
        (self.output_keys,) if isinstance(self.output_keys, str) else self.output_keys
    ):
        self._emit("values", map_output_values, self.output_keys, writes, self.channels)

    # 3. 適用済みの pending_writes をクリアし、タイムトラベル再実行フラグを解除
    self.checkpoint_pending_writes.clear()
    self.is_replaying = False

    # 4. 新しいチェックポイントを作成し、バックグラウンド保存をスケジュール
    self._put_checkpoint({"source": "loop"})

    # 5. interrupt_after ブレークポイント判定
    if self.interrupt_after and should_interrupt(
        self.checkpoint, self.interrupt_after, self.tasks.values()
    ):
        self.status = "interrupt_after"
        self._prepare_tasks_and_emit_checkpoint()
        raise GraphInterrupt()

    self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
```

### 3 つの Durability モードと順序保証（`_put_checkpoint_after_previous`）
LangGraph では `durability` 引数に `"sync"`・`"async"`（デフォルト）・`"exit"` の 3 つを選べます。

| モード | `put_writes` | `_put_checkpoint` | 特徴 |
|---|---|---|---|
| `"sync"` | タスク完了時に即時送信 | 毎ステップ送信＋**完了を同期待機** | 最高の一貫性。DB 書き込み完了後に次ステップ開始 |
| `"async"` | タスク完了時にバックグラウンド送信 | 毎ステップ、**前回の保存 Future 完了後に直列実行** | 性能と耐久性のバランス。順序逆転を防止 |
| `"exit"` | 実行中はメモリ保持のみ | 実行中は保存せず、`__exit__`（`_suppress_interrupt`）で最終状態を一括保存 | 最高スループット。途中クラッシュ時は復元不可 |

特に `"async"` モードで順序逆転を防ぐ仕組みが秀逸です。`_put_checkpoint()` は直前の保存タスク `self._put_checkpoint_fut` と未完了の Delta 書き込み `delta_write_futs` を `_put_checkpoint_after_previous()` へチェーンして `self.submit(...)` します。

```python
def _put_checkpoint_after_previous(
    self,
    prev: concurrent.futures.Future | None,
    delta_write_futs: Sequence[concurrent.futures.Future],
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    # 先行する DeltaChannel の put_writes が永続化されるのを待機
    for fut in delta_write_futs:
        fut.result()
    # 1 つ前のステップの checkpoint.put() が完了するのを待機（順序保証）
    if prev is not None:
        prev.result()
    cast(BaseCheckpointSaver, self.checkpointer).put(
        config, checkpoint, metadata, new_versions
    )
```

{% mermaid %}
sequenceDiagram
    participant Main as "メインスレッド (Pregel.stream)"
    participant BG as "BackgroundExecutor"
    participant DB as "Checkpointer"

    Main->>Main: Step 1: runner.tick() & apply_writes()
    Main->>BG: submit(_put_checkpoint_after_previous, prev=None, cp1)
    Main->>Main: Step 2: 即座に次ステップ開始 (durability="async")
    BG->>DB: put(cp1)
    Main->>BG: submit(_put_checkpoint_after_previous, prev=fut1, cp2)
    Note over BG: fut1.result() 完了まで cp2 の put を待機
    DB-->>BG: cp1 保存完了
    BG->>DB: put(cp2)
{% endmermaid %}

---

## 5. 終了処理と `GraphInterrupt` の抑制：`_suppress_interrupt()`

`SyncPregelLoop.__exit__()` から呼ばれる `_suppress_interrupt()` は、コンテキストマネージャの終了フックを巧みに利用しています。

1. **`durability == "exit"` の最終フラッシュ**：トップレベルグラフ、または例外・中断が発生したサブグラフの場合、ここで初めて `_put_exit_delta_writes()` → `_put_checkpoint()` → `_put_pending_writes()` を実行して永続化します。
2. **トップレベルでの `GraphInterrupt` 抑制**：`isinstance(exc_value, GraphInterrupt) and not self.is_nested` のとき、中断時点の `pending_writes` を仮適用して最後の `"values"` / `"updates"` イベントを発行し、`self.output` を保存した上で **`return True` で例外を握りつぶします**。だからこそ、`interrupt()` が発生してもトップレベルの `graph.invoke()` はクラッシュせず、`{"__interrupt__": [...]}` を含んだ正常なレスポンスを返せるのです（サブグラフの場合は `is_nested=True` なので親グラフまで例外がバブルアップします）。

---

## 6. まとめ

Part 3 では、`Pregel.stream()` と `PregelLoop` が実現する BSP 実行ループの全体像を追いました。

- **厳格な BSP 境界**：`tick()`（Plan）→ `runner.tick()`（Execution）→ `after_tick()`（Update）の 3 段階が明確に分離されている
- **耐障害性と部分再実行**：成功済みタスクの `pending_writes` を `tick()` で再適用することで、失敗・中断ノードだけを再実行できる
- **非同期永続化と順序保証**：`_put_checkpoint_after_previous` により、ステップ実行を止めずにチェックポイント保存の直列順序と `DeltaChannel` の可視性不変条件を担保している

次回の Part 4 では、このループの内部で実際に動いているアルゴリズム中核——[`pregel/_algo.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_algo.py)・[`pregel/_runner.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_runner.py)・[`pregel/_write.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_write.py)——を読み解き、タスク準備・並列スケジューリング・`Send` / `Command` 解決の仕組みを詳解します。
