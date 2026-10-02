---
title: "LangGraph ソースコード解析 Part 1：StateGraph と CompiledStateGraph のコンパイル機構"
date: 2026-09-11 12:29:19
categories: [AI]
tags: [Python, LLM, AI Agent, LangGraph, ソースコード解析, StateGraph]
lang: ja
description: "LangGraph の高レベル API である StateGraph が、どのようにスキーマ解析・ノード登録・条件分岐（BranchSpec）を経て Pregel ベースの CompiledStateGraph へコンパイルされるのかをソースコードから徹底解析します。"
---

LangGraph は、LLM エージェントの状態遷移を有向グラフ（ループを含む）として記述し、Google の Pregel（BSP: Bulk Synchronous Parallel）モデル上で実行するためのフレームワークです。ユーザーは通常 `StateGraph` にノードやエッジを登録し、`.compile()` を呼び出して `CompiledStateGraph` を得ますが、この「コンパイル」の裏側では高レベルな宣言的グラフ定義が低レベルな **Channel / PregelNode / ChannelWrite** の配線へと変換されています。

本シリーズでは全 6 回にわたり、LangGraph のコアソースコードを直接読み解きながら内部アーキテクチャを解説します。Part 1 では、グラフ構築とコンパイルを担う以下の 3 ファイルを中心に掘り下げます。

- [`graph/state.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)：`StateGraph`、`CompiledStateGraph`、スキーマ→Channel 変換ロジック
- [`graph/_node.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/_node.py)：`StateNode` プロトコル群と `StateNodeSpec`
- [`graph/_branch.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/_branch.py)：条件分岐の仕様を表す `BranchSpec` と実行時ルーティング

---

## 1. 全体アーキテクチャ：宣言的 Builder から Pregel グラフへの変換

`StateGraph` はあくまで**ミュータブルな Builder クラス**であり、それ自体は実行機能を持ちません。`compile()` を呼び出すことで、`Pregel` を継承した `CompiledStateGraph` が生成され、以下の変換が一気に行われます。

1. **スキーマ解析（`_get_channels`）**：`TypedDict` や Pydantic `BaseModel` の型ヒントから各フィールドに対応する `BaseChannel` / `ManagedValueSpec` を生成
2. **ノード配線（`attach_node`）**：各 `StateNodeSpec` を `PregelNode` に変換し、専用トリガーチャネル `branch:to:<node>` と出力書き込み器 `ChannelWrite` を接続
3. **エッジ配線（`attach_edge`）**：通常の有向エッジを `branch:to:<end>` への書き込みに、複数ノードの合流（`waiting_edges`）を `NamedBarrierValue` チャネルによるバリア同期に変換
4. **条件分岐配線（`attach_branch`）**：`BranchSpec` を `ChannelRead.do_read(..., fresh=True)` 付きの `RunnableCallable` に変換し、遷移元ノードの `writers` 末尾へ連結

{% mermaid %}
flowchart LR
    subgraph Builder["StateGraph (宣言的ビルダー)"]
        S["state_schema / input_schema / output_schema"]
        N["nodes: dict[str, StateNodeSpec]"]
        E["edges / waiting_edges"]
        B["branches: dict[str, dict[str, BranchSpec]]"]
    end

    subgraph Compile["StateGraph.compile()"]
        V["validate() & デフォルトポリシー適用"]
        AN["attach_node()"]
        AE["attach_edge()"]
        AB["attach_branch()"]
    end

    subgraph Runtime["CompiledStateGraph (Pregel サブクラス)"]
        CH["channels: Stateチャネル + branch:to:* + join:*"]
        PN["nodes: dict[str, PregelNode]"]
        CW["writers: ChannelWrite + Branch ルーター"]
    end

    S --> V
    N --> AN
    E --> AE
    B --> AB
    V --> AN --> AE --> AB
    AN --> CH
    AN --> PN
    AE --> CH
    AE --> CW
    AB --> CW

    style Builder fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Compile fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style Runtime fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

---

## 2. スキーマ解析：`TypedDict` / `Annotated` から Channel への変換

`StateGraph.__init__` では、`state_schema`・`input_schema`・`output_schema` のそれぞれに対して `self._add_schema()` を呼び出します。その実体である `_get_channels` と `_get_channel`（[`graph/state.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)）を見てみましょう。

```python
def _get_channels(
    schema: type[dict],
) -> tuple[dict[str, BaseChannel], dict[str, ManagedValueSpec], dict[str, Any]]:
    # __annotations__ を持たない単一型（例: list[Any] や str）の場合は
    # 特殊なルートキー ROOT ("__root__") を 1 つだけ持つチャネル辞書として扱う
    if not hasattr(schema, "__annotations__"):
        return (
            {ROOT: _get_channel(ROOT, schema, allow_managed=False)},
            {},
            {},
        )

    # include_extras=True により Annotated[T, metadata...] のメタデータを保持したまま型ヒントを取得
    type_hints = get_type_hints(schema, include_extras=True)
    all_keys = {
        name: _get_channel(name, typ)
        for name, typ in type_hints.items()
        if name != "__slots__"
    }
    # 通常チャネルと ManagedValue（IsLastStep など実行時管理値）を分離して返す
    return (
        {k: v for k, v in all_keys.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
        type_hints,
    )


def _get_channel(
    name: str, annotation: Any, *, allow_managed: bool = True
) -> BaseChannel | ManagedValueSpec:
    # TypedDict の Required[T] / NotRequired[T] ラッパーを剥がして中身を取り出す
    if hasattr(annotation, "__origin__") and annotation.__origin__ in (
        Required,
        NotRequired,
    ):
        annotation = annotation.__args__[0]

    # 優先順位 1: ManagedValue（例: IsLastStep）
    if manager := _is_field_managed_value(name, annotation):
        if allow_managed:
            return manager
        else:
            # input_schema / output_schema では ManagedValue を禁止
            raise ValueError(f"This {annotation} not allowed in this position")
    # 優先順位 2: 明示的な BaseChannel インスタンスまたはクラス（DeltaChannel / EphemeralValue 等）
    elif channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    # 優先順位 3: Annotated[T, reducer] 形式の 2 引数関数 -> BinaryOperatorAggregate
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel

    # デフォルトフォールバック: 単一値上書きの LastValue チャネル
    fallback: LastValue = LastValue(annotation)
    fallback.key = name
    return fallback
```

### 設計上の注目ポイント
- **`Annotated` の解決優先順位**：`ManagedValue` → 明示的 `BaseChannel` → 2 引数 Reducer（`BinaryOperatorAggregate`）→ `LastValue` の順に判定されます。
- **`_is_field_binop` のシグネチャ検証**：`Annotated[list, add_messages]` のような指定では、`meta[-1]` が呼び出し可能であり、位置引数（`POSITIONAL_ONLY` または `POSITIONAL_OR_KEYWORD`）が厳密に **2 つ** であることを `inspect.signature` で検証します。これにより、実行時ではなくグラフ構築時に誤った Reducer を検出できます。
- **複数スキーマ間のチャネル共存**：`_add_schema` では、既に同名キーが登録されていても新しい側が `LastValue` であれば既存チャネル（例：`BinaryOperatorAggregate`）を優先します。これにより、`input_schema` 側で Reducer を省略しても `state_schema` の Reducer が維持されます。

---

## 3. ノード仕様（`StateNodeSpec`）と型推論付き `add_node`

[`graph/_node.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/_node.py) では、ノード関数が受け取れる引数パターン（`state` のみ、`config` 付き、`writer` 付き、`store` 付き、`runtime: Runtime[ContextT]` 付き、`Runnable`）を `Protocol` の Union である `StateNode` として定義し、メタデータ一式を `slots=True` の `StateNodeSpec` データクラスに格納します。

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    is_error_handler: bool = False          # エラーハンドラ専用ノードかどうか
    error_handler_node: str | None = None   # 失敗時に遷移するハンドラノード名
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ  # Command 遷移先の静的ヒント
    defer: bool = False                     # 全タスク完了後まで遅延実行するフラグ
    timeout: TimeoutPolicy | None = None    # 非同期ノードのタイムアウト制御
    trace_policy: TracePolicy | None = None # トレース入力加工ポリシー
```

`StateGraph.add_node()`（[`graph/state.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)）の興味深い点は、**関数の型アノテーションから 2 種類の情報を自動推論する**ことです。

1. **ノード固有の `input_schema` 推論**：ノード関数の第 1 引数に別の `TypedDict` / `BaseModel` が指定されていれば、それを `inferred_input_schema` として自動登録（`_add_schema`）します。これにより、グラフ全体の State とは異なるプライベート入力スキーマをノード単位で持てます。
2. **`Command[Literal["node_a", "node_b"]]` の戻り値型解析**：戻り値型ヒントが `Command[Literal[...]]`（またはそれを含む `Union`）であれば、`Literal` の中身を `ends` に抽出し、エッジを明示的に書かない Command ベースグラフでも可視化とバリデーションを可能にしています。
3. **ノード単位エラーハンドラの自動ノード化**：`error_handler` が渡されると、`__error_handler__<node>` という隠しノードを `is_error_handler=True` で自動登録し、通常のノードと同様に Pregel 上で管理します。

---

## 4. `compile()` と `CompiledStateGraph` の内部配線

`StateGraph.compile()` は、大きく 4 つのフェーズで `CompiledStateGraph` を組み立てます。

```python
# 1. ビルダーデフォルトポリシー（set_node_defaults）の適用
#    ※ retry / timeout は全ノードに適用されるが、
#      cache_policy と error_handler は通常ノードのみに限定（無限再帰や不安全なキャッシュを防止）
for spec in self.nodes.values():
    if not spec.is_error_handler and default_handler_name is not None and spec.error_handler_node is None:
        spec.error_handler_node = default_handler_name
    if defaults.retry_policy is not None and spec.retry_policy is None:
        spec.retry_policy = defaults.retry_policy
    if not spec.is_error_handler and defaults.cache_policy is not None and spec.cache_policy is None:
        spec.cache_policy = defaults.cache_policy
    if defaults.timeout is not None and spec.timeout is None:
        spec.timeout = defaults.timeout

# 2. CompiledStateGraph（Pregel のサブクラス）を初期化
compiled = CompiledStateGraph[StateT, ContextT, InputT, OutputT](
    builder=self,
    schema_to_mapper={},
    context_schema=self.context_schema,
    nodes={},
    channels={
        **self.channels,
        **self.managed,
        START: EphemeralValue(self.input_schema),  # 入力エントリ用の揮発チャネル
    },
    input_channels=START,
    stream_mode="updates",
    output_channels=output_channels,
    stream_channels=stream_channels,
    checkpointer=checkpointer,
    ...
)

# 3. ノード・通常エッジ・合流エッジ・条件分岐を Pregel プリミティブへ配線
compiled.attach_node(START, None)
for key, node in self.nodes.items():
    compiled.attach_node(key, node)

for start, end in self.edges:
    compiled.attach_edge(start, end)
for starts, end in self.waiting_edges:
    compiled.attach_edge(starts, end)
for start, branches in self.branches.items():
    for name, branch in branches.items():
        compiled.attach_branch(start, name, branch)

return compiled.validate()
```

### 4.1 `attach_node`：トリガーチャネルと 2 段構えの `ChannelWrite`

`attach_node` は、すべての通常ノードに対して専用の制御チャネル `branch:to:<key>` を生成します。

```python
# ノード実行をトリガーする制御チャネルを定義
branch_channel = _CHANNEL_BRANCH_TO.format(key)  # "branch:to:<key>"
self.channels[branch_channel] = (
    # defer=True の場合は他タスクが全て終わるまで発火しない遅延チャネルを使用
    LastValueAfterFinish(Any)
    if node.defer
    else EphemeralValue(Any, guard=False)
)

self.nodes[key] = PregelNode(
    triggers=[branch_channel],                    # branch:to:<key> への書き込みで起動
    channels=(ROOT if is_single_input else input_channels),  # 読み取る State キー群
    mapper=mapper,                                # Pydantic / dataclass への型変換関数
    writers=[ChannelWrite(write_entries)],        # 戻り値を State 更新 & 制御コマンドへ分解
    bound=node.runnable,
    ...
)
```

ここで極めて重要なのが `write_entries` の 2 つの `ChannelWriteTupleEntry` です。

1. **State 更新マッパー（`_get_updates` / `_get_root`）**：ノードの戻り値が `dict`、`Command`、`Command` のリスト、あるいは dataclass / Pydantic モデルであるかを判別し、`output_key_set`（有効なチャネル名集合）に含まれる `(channel, value)` タプル列だけを抽出します。なお `Command.PARENT` は親グラフ宛てのためローカル State 更新からは除外されます。
2. **制御フローマッパー（`_control_branch`）**：戻り値に `Send` や `Command(goto=...)` が含まれていた場合、以下のように**制御用チャネルへの書き込み**に変換します。
   - `Send` → `(TASKS, send_obj)`（`__pregel_tasks` への PUSH タスク発行）
   - `Command(goto="next_node")` → `("branch:to:next_node", None)`（次ノードのトリガー発火）
   - `Command(graph=Command.PARENT)` → `ParentCommand(command)` 例外を送出し、親グラフの実行ループへ伝播

### 4.2 `attach_edge`：通常エッジと Fan-In バリア同期

`attach_edge` は、`starts` が単一文字列かシーケンス（`waiting_edges`）かで完全に異なるチャネル配線を行います。

```python
def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        # 単一エッジ: starts ノードの末尾に "branch:to:<end>" への書き込みを追加するだけ
        if end != END:
            self.nodes[starts].writers.append(
                ChannelWrite(
                    (ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),)
                )
            )
    elif end != END:
        # 複数エッジの合流（Fan-In）: NamedBarrierValue チャネルを動的に生成
        channel_name = f"join:{'+'.join(starts)}:{end}"
        if self.builder.nodes[end].defer:
            self.channels[channel_name] = NamedBarrierValueAfterFinish(str, set(starts))
        else:
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
        # 合流先ノードのトリガーにバリアチャネルを追加
        self.nodes[end].triggers.append(channel_name)
        # 各上流ノードの完了時に、自身のノード名をバリアチャネルへ書き込む
        for start in starts:
            self.nodes[start].writers.append(
                ChannelWrite((ChannelWriteEntry(channel_name, start),))
            )
```

通常エッジが単なる `EphemeralValue` への `None` 書き込みで表現される一方、複数ノード待ち合わせ（`add_edge(["A", "B"], "C")`）では `join:A+B:C` という `NamedBarrierValue` チャネルが生成されます。上流の `"A"` と `"B"` がそれぞれ自身の名前を書き込み、集合 `{"A", "B"}` が揃ったスーパーステップで初めて `"C"` がトリガーされるという、Pregel のチャネル抽象を最大限に活かした設計です。

---

## 5. 条件分岐の仕組み：`BranchSpec` と `fresh=True` による最新状態読み取り

条件分岐（`add_conditional_edges`）は [`graph/_branch.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/_branch.py) の `BranchSpec` として表現され、`CompiledStateGraph.attach_branch` で遷移元ノードの `writers` リストに追加されます。

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None

    def _route(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Callable[[RunnableConfig], Any] | None,
        writer: _Writer,
    ) -> Runnable:
        if reader:
            # 重要: fresh=True で直前のノード書き込みを反映した最新 State を読み直す
            value = reader(config)
            # dict State の場合、ノードが返した追加キーもパススルーでマージ
            if (
                isinstance(value, dict)
                and isinstance(input, dict)
                and self.input_schema is None
            ):
                value = {**input, **value}
        else:
            value = input
        # ユーザー定義の分岐関数を実行
        result = self.path.invoke(value, config)
        return self._finish(writer, input, result, config)
```

ここで注目すべきは、`attach_branch` が `BranchSpec.run` に渡す `reader` です。

```python
reader = partial(
    ChannelRead.do_read,
    select=channels[0] if channels == [ROOT] else channels,
    fresh=True,      # 同一タスク内で行われた直前の ChannelWrite を即時適用して読む
    mapper=mapper,
)
```

1 つの `PregelNode` の実行パイプラインでは、**`bound`（ノード本体）→ `writers[0]`（State への `ChannelWrite`）→ `writers[1]`（`BranchSpec` ルーター）** の順で実行されます。`writers[0]` の時点ではまだ次のスーパーステップへコミットされていないため、`fresh=True` を指定することで「現在のチェックポイント値＋このタスクが今書き込んだ pending writes」をメモリ上で仮マージした最新 State を分岐関数へ渡せるようになっています。

{% mermaid %}
sequenceDiagram
    participant Node as "PregelNode.bound (ユーザー関数)"
    participant W0 as "writers[0]: ChannelWrite"
    participant Buf as "タスクローカル writes バッファ"
    participant Br as "writers[1]: BranchSpec._route"
    participant Rd as "ChannelRead.do_read(fresh=True)"

    Node->>W0: Partial State / Command を返却
    W0->>Buf: _get_updates() で (state_key, val) を書き込み
    W0->>Buf: _control_branch() で Send / goto を書き込み
    W0->>Br: ノード戻り値 (input) をそのまま渡す
    Br->>Rd: 最新 State を要求 (fresh=True)
    Rd->>Buf: pending writes をチャネルへ仮適用
    Rd-->>Br: マージ済み最新 State を返却
    Br->>Br: self.path.invoke(value, config) で遷移先決定
    Br->>Buf: ChannelWrite.do_write() で branch:to:<dest> or Send を書き込み
{% endmermaid %}

また、`_finish()` 内で `self.ends` によるマッピング解決後、`Send` であればそのままタスクキュー（`TASKS`）へ、ノード名であれば `branch:to:<dest>` への `ChannelWriteEntry` へと変換し、`END` の場合は書き込みを行わずに静かに終了します。

---

## 6. まとめ：Part 1 で見えた LangGraph の設計哲学

`StateGraph` のコンパイル機構を読み解くと、LangGraph のコア設計が極めてクリーンな **「すべての制御フローをチャネルへの書き込みに還元する」** アプローチに基づいていることが分かります。

- **データフロー**：`state_schema` の各フィールド → `LastValue` / `BinaryOperatorAggregate` / `DeltaChannel` 等
- **制御フロー（通常エッジ / `Command.goto`）**：`branch:to:<node>` という `EphemeralValue`（`defer=True` 時は `LastValueAfterFinish`）への書き込み
- **制御フロー（合流エッジ）**：`join:<starts>:<end>` という `NamedBarrierValue` へのノード名書き込み
- **動的ファンアウト（`Send`）**：`TASKS` チャネルへの直接パブリッシュ

次回の Part 2 では、この状態管理と制御フローの土台となっている **Channels 抽象（`BaseChannel`, `LastValue`, `BinaryOperatorAggregate`, `Topic`, `DeltaChannel`）と `add_messages` リデューサ** の実装を詳しく解析します。
