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
