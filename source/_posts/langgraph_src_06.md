---
title: "LangGraph ソースコード解析 Part 6：Functional API（func/__init__.py）と Streaming 多重化パイプライン（stream/run_stream.py・pregel/_messages.py）"
date: 2026-09-19 01:20:29
categories: [AI]
tags: [Python, LLM, AI Agent, LangGraph, ソースコード解析, Functional API, Streaming]
lang: ja
description: "LangGraph の Functional API（@entrypoint / @task）が単一ノードの Pregel グラフへ変換される仕組みと、LLM トークンをリアルタイム配信する StreamMessagesHandler および v3 多重化ストリーム（GraphRunStream / StreamMux）をソースコードから解析します。"
---

シリーズの最終回となる Part 6 では、LangGraph の 2 つのモダンな拡張領域——グラフ宣言を書かずに通常の Python 制御構文で耐久実行（durable execution）を実現する **Functional API** と、LLM トークンやサブグラフイベントを多重化して配信する **Streaming パイプライン**——を解析します。

対象となるコアファイルは以下の 4 つです。

- [`func/__init__.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/func/__init__.py)：`@entrypoint`、`@task`、`entrypoint.final`
- [`pregel/_messages.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_messages.py)：`StreamMessagesHandler`、`StreamMessagesHandlerV2`、`graph_namespace()`
- [`stream/run_stream.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/stream/run_stream.py)：v3 ストリーミングの `GraphRunStream` / `AsyncGraphRunStream` と `interleave()`
- [`stream/_mux.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/stream/_mux.py)：プロジェクションとサブグラフ用ミニ Mux を管理する `StreamMux`

---

## 1. Functional API：`@entrypoint` と `@task` の正体

`StateGraph` はノードとエッジを明示的に定義しますが、既存の Python コードベースにチェックポイント永続化や `interrupt()` を組み込みたい場合、グラフ定義への書き換えは負担になります。そこで用意されたのが [`func/__init__.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/func/__init__.py) の **Functional API** です。

実は `@entrypoint` デコレータは独自エンジンを持つわけではなく、**「たった 1 つの `PregelNode` と 3 つのチャネル（`START`, `END`, `PREVIOUS`）だけを持つ `Pregel` インスタンス」** へと関数を変換しています。

```python
class entrypoint(Generic[ContextT]):
    @dataclass(**_DC_KWARGS)
    class final(Generic[R, S]):
        value: R  # 呼び出し元へ返す戻り値
        save: S   # 次回実行時に previous 引数として渡すために保存する値

    def __call__(self, func: Callable[..., Any]) -> Pregel:
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise NotImplementedError("Generators are not supported in the Functional API.")

        bound = get_runnable_for_entrypoint(func)

        def _pluck_return_value(value: Any) -> Any:
            return value.value if isinstance(value, entrypoint.final) else value

        def _pluck_save_value(value: Any) -> Any:
            return value.save if isinstance(value, entrypoint.final) else value

        # 関数 1 つをそのまま 1 ノードの Pregel グラフとして構築する
        graph: Pregel[Any, ContextT, Any, Any] = Pregel(
            nodes={
                func.__name__: PregelNode(
                    bound=bound,
                    triggers=[START],
                    channels=START,
                    timeout=self.timeout,
                    writers=[
                        ChannelWrite(
                            [
                                # 戻り値と次回用保存値を 2 つのチャネルへ分離書き込み
                                ChannelWriteEntry(END, mapper=_pluck_return_value),
                                ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                            ]
                        )
                    ],
                )
            },
            channels={
                START: EphemeralValue(input_type),      # 入力チャネル
                END: LastValue(output_type, END),       # 最終出力チャネル
                PREVIOUS: LastValue(save_type, PREVIOUS),  # 次回呼び出しの previous 注入元
            },
            input_channels=START,
            output_channels=END,
            stream_channels=END,
            stream_mode="updates",
            stream_eager=True,   # @task 完了ごとに即座にストリームへ流す
            checkpointer=self.checkpointer,
            store=self.store,
            cache=self.cache,
            cache_policy=self.cache_policy,
            retry_policy=self.retry_policy or (),
            context_schema=self.context_schema,
        )
        return graph
```

### `@task` が Checkpointer でメモ化・並列化される仕組み
一方の `@task` デコレータは、対象関数を `_TaskFunction` でラップします。呼び出されると `_call_with_options(...)` を経由して `config[CONF][CONFIG_KEY_CALL]`（Part 4 で見た `PregelRunner` の `_call` / `_acall`）を叩き、`SyncAsyncFuture[T]` を即座に返却します。

1. `PregelLoop.accept_push()` が `(PUSH, parent_path, call_counter, parent_id, Call(...))` から決定的な `task_id` を生成
2. 再開時（`interrupt()` 後のリプレイなど）に、その `task_id` の出力が既に `pending_writes` に保存されていれば、`_call()` が **タスクを再実行せず保存済みの `RETURN` 値で解決済みの Future を即返却**
3. 未実行であれば `FuturesDict` へ非同期投入され、複数の `@task` を呼んで `asyncio.gather(*futures)` や `[f.result() for f in futures]` で並列実行可能

{% mermaid %}
flowchart LR
    subgraph Entry["@entrypoint (1ノードの Pregel)"]
        ST["START (EphemeralValue)"] --> FN["PregelNode (ユーザー関数)"]
        FN --> END_CH["END (LastValue)<br/>entrypoint.final.value"]
        FN --> PREV_CH["PREVIOUS (LastValue)<br/>entrypoint.final.save"]
    end

    subgraph TaskCall["@task 呼び出し (実行中)"]
        FN -- "CONFIG_KEY_CALL" --> CALL["PregelRunner._call / _acall"]
        CALL --> AP["loop.accept_push() -> prepare_push_task_functional()"]
        AP --> HIT{"pending_writes に<br/>結果あり?"}
        HIT -- "Yes (再開時)" --> RET["保存済み RETURN を即返却"]
        HIT -- "No" --> RUN["FuturesDict で並列実行 & commit() 保存"]
    end

    style Entry fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style TaskCall fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

このように、`StateGraph` と Functional API は**完全に同一の Pregel ランタイム・Checkpointer・`interrupt()`・Retry/Cache 機構を共有**しています。

---

## 2. LLM トークンストリーミング：`StreamMessagesHandler`（`pregel/_messages.py`）

LangGraph で `stream_mode="messages"` を指定すると、ノード内部の奥深くで呼ばれた ChatModel のトークンがリアルタイムに `(AIMessageChunk, metadata)` として流れてきます。これを実現するのが [`pregel/_messages.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_messages.py) の `StreamMessagesHandler` です。

```python
class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    # 重要: メインスレッド（イベントループスレッド）上で同期実行させ、順序逆転やロック競合を防ぐ
    run_inline = True

    def __init__(self, stream: Callable[[StreamChunk], None], subgraphs: bool, *, parent_ns=None):
        self.stream = stream
        self.subgraphs = subgraphs
        self.metadata: dict[UUID, Meta] = {}  # run_id -> (graph_ns, metadata)
        self.seen: set[int | str] = set()     # 重複送出防止用のメッセージ ID 集合
        self.parent_ns = parent_ns

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs):
        # TAG_NOSTREAM ("nostream") が付いた LLM 呼び出しはストリーミング対象外
        if metadata and (not tags or (TAG_NOSTREAM not in tags)):
            ns = graph_namespace(metadata)
            # subgraphs=False の場合、別名前空間（サブグラフ内）の LLM は無視
            if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
                return
            if (filtered_tags := filter_to_user_tags(tags)) is not None:
                metadata["tags"] = filtered_tags
            self.metadata[run_id] = (ns, metadata)

    def on_llm_new_token(self, token, *, chunk=None, run_id, **kwargs):
        if not isinstance(chunk, ChatGenerationChunk):
            return
        # 登録済みの run_id であれば即座に SyncQueue / AsyncQueue へプッシュ
        if meta := self.metadata.get(run_id):
            self._emit(meta, chunk.message)

    def on_chain_end(self, response, *, run_id, **kwargs):
        # LLM を経由せずノードが直接返した BaseMessage（ToolMessage 等）も
        # self.seen で重複排除しつつ messages ストリームへ流す
        if meta := self.metadata.pop(run_id, None):
            if isinstance(response, Command):
                self._find_and_emit_messages(meta, response.update)
            else:
                self._find_and_emit_messages(meta, response)
```

### 実装上の 3 つの巧妙なポイント
1. **`graph_namespace()` による `@task` ネスト補正**：Functional API の `@task` は親グラフ内で動くにも関わらず `langgraph_checkpoint_ns` に自身のセグメントを積みます。`graph_namespace()` は `langgraph_path` が `(PUSH, caller_path, idx, True)` である間だけ末尾セグメントを剥がし、正しい所属グラフの名前空間を復元します。
2. **`self.seen` による重複排除**：`on_chain_start()` で入力に含まれる既存メッセージ ID を `self.seen` に登録し、`on_llm_new_token()` で既に流したメッセージ ID も記録しておくことで、ノード終了時の `on_chain_end()` で同じ `AIMessage` が二重に流れるのを防いでいます。
3. **`StreamMessagesHandlerV2`**：v3 プロトコル（`CONFIG_KEY_STREAM_MESSAGES_V2`）では `_V2StreamingCallbackHandler` を継承することで、`BaseChatModel.invoke` をコンテンツブロック単位のイベントストリームへ切り替えつつ、`on_llm_new_token` を意図的に no-op 化し、さらに `ToolMessage` は専用 `tools` チャネルに任せて `messages` からは除外しています。

---

## 3. v3 ストリーミング多重化パイプライン：`GraphRunStream` と `StreamMux`

従来の `stream(stream_mode=["values", "messages"])` では、呼び出し元が単一の `(mode, data)` タプル列を `if mode == ...` で分岐処理する必要がありました。実験的な v3 ストリーミング（`stream_events(version="v3")` / `astream_events(version="v3")`）では、[`stream/run_stream.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/stream/run_stream.py) の **`GraphRunStream` / `AsyncGraphRunStream`** と [`stream/_mux.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/stream/_mux.py) の **`StreamMux`** により、**呼び出し元駆動（caller-driven pump）の型付きプロジェクション**を提供しています。

```python
class GraphRunStream:
    # ネイティブプロジェクション（__init__ で mux.extensions から動的に setattr される）
    values: StreamChannel[dict[str, Any]]
    messages: StreamChannel[ChatModelStream]
    lifecycle: StreamChannel[LifecyclePayload]
    subgraphs: StreamChannel[SubgraphRunStream]

    def _pump_next(self) -> bool:
        """
        バックグラウンドスレッドを持たず、利用者が run.messages や run.values を
        for ループで読むたびに 1 イベントずつグラフを前進させる pull 型ポンプ。
        """
        if self._exhausted or self._graph_iter is None:
            return False
        try:
            part = next(self._graph_iter)
            event = convert_to_protocol_event(part)  # v2 StreamPart -> ProtocolEvent 変換
            self._observe_event(event)               # output / interrupted / interrupts を追跡
            self._mux.push(event)                    # 全 StreamTransformer へ配信
            return True
        except StopIteration:
            self._mux.close()
            self._exhausted = True
            return False
        except Exception as e:
            self._mux.fail(e)
            self._exhausted = True
            return False
```

{% mermaid %}
flowchart LR
    Graph["Pregel.stream() イテレータ"] -- "next()" --> Pump["GraphRunStream._pump_next()"]
    Pump --> Conv["convert_to_protocol_event()"]
    Conv --> Mux["StreamMux.push(event)"]

    subgraph Pipeline["StreamTransformer パイプライン"]
        Pre["before_builtins=True<br/>(PII 編集・カスタムフィルタ)"]
        Built["Built-in Transformers<br/>(Values / Messages / Lifecycle / Subgraph)"]
        Pre --> Built
    end

    Mux --> Pipeline
    Built --> P1["run.values (StreamChannel)"]
    Built --> P2["run.messages (ChatModelStream)"]
    Built --> P3["run.subgraphs (子 StreamMux 継承)"]
    P1 --> IL["run.interleave('messages', 'values')<br/>単調 push_seq 順で厳密マージ"]
    P2 --> IL

    style Pipeline fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style Mux fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style IL fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

### `StreamMux` と `interleave()` の設計ハイライト
1. **バックグラウンドスレッド不要の Pull ポンプ**：`GraphRunStream` は `mux.bind_pump(self._pump_next)` で全プロジェクションの `_request_more` にポンプ関数を配線します。非同期版の `AsyncGraphRunStream` では単一の `asyncio.Lock` で並列コンシューマのポンプ競合を防ぎつつ、バッファ上限（`maxlen`）によるバックプレッシャを効かせています。
2. **サブグラフ用ミニ Mux（`_make_child`）**：サブグラフを検出すると、`StreamMux._make_child(scope)` が `factories` からそのスコープ専用の Transformer 群を新規生成し、**親と同じ `_pump_fn` を継承**します。そのため、ネストされた `SubgraphRunStream` 上のプロジェクションをイテレートしても、ルートのグラフポンプが透過的に駆動されます。
3. **`before_builtins` レーン**：`StreamMux.__init__` は `before_builtins = True` を持つカスタム Transformer を組み込み Transformer（`MessagesTransformer` 等）より前に登録します。これにより、PII マスキングのようなコンテンツ変換をビルトインのプロジェクション確定前に安全に挟み込めます。
4. **厳密な到着順マージ（`interleave()`）**：複数プロジェクションを同時に読みたい場合、単純なラウンドロビンではなく、各 `StreamChannel` への `push()` 時に採番される単調増加のスタンプ（`_push_seq`）を比較して最小スタンプのアイテムから順に `yield (name, item)` します。

---

## 4. 全 6 回シリーズ総括

全 6 回にわたり、LangGraph のコアソースコードを上から下まで横断解析してきました。最後に、LangGraph 全体のレイヤー構造を振り返ります。

| Part | レイヤー | コアファイル | 設計の核心 |
|---|---|---|---|
| **Part 1** | 宣言的グラフ構築 | `graph/state.py`, `_branch.py`, `_node.py` | すべての制御フローを `branch:to:*` / `join:*` チャネル書き込みへ還元 |
| **Part 2** | 状態抽象 | `channels/*.py`, `graph/message.py` | `update` / `consume` / `finish` フックと `DeltaChannel` による差分リプレイ |
| **Part 3** | BSP 実行ループ | `pregel/main.py`, `pregel/_loop.py` | Plan → Execution → Update の厳格な分離と `durability` 順序保証 |
| **Part 4** | アルゴリズム・並列実行 | `pregel/_algo.py`, `_runner.py`, `_write.py` | PUSH/PULL 統合タスク計画、`FuturesDict` 並列制御、`ChannelWrite` 正規化 |
| **Part 5** | 永続化・HITL | `pregel/_checkpoint.py`, `types.py` | UUIDv6 スナップショットと `Scratchpad`＋`GraphInterrupt` 例外による中断再開 |
| **Part 6** | 関数型 API・配信 | `func/__init__.py`, `stream/run_stream.py`, `pregel/_messages.py` | 単一ノード Pregel 変換、`run_inline` トークン配信、Pull 型 `StreamMux` |

一見複雑に見えるエージェント実行基盤も、ソースコードの根底を貫くのは **「Channel 抽象 × BSP スーパーステップ × 決定的リプレイ」** という極めて一貫した少数プリミティブの組み合わせです。本シリーズが、LangGraph を実戦投入する際のデバッグやパフォーマンス最適化、カスタム Checkpointer / Channel 設計の一助となれば幸いです。
