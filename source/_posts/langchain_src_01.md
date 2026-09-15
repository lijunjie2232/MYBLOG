---
title: "LangChain v1 ソースコード解析 Part 1：create_agent（agents/factory.py）のグラフ構築パイプラインとノード配線"
date: 2026-08-11 09:03:09
categories: [AI]
tags: [Python, LLM, AI Agent, LangChain, LangGraph, ソースコード解析]
lang: ja
description: "LangChain v1 の中核エントリポイント create_agent（agents/factory.py）のソースコードを精読し、スキーママージ、ミドルウェアノード配線、条件付きエッジ、ツール並列ディスパッチの内部設計を解説する。"
---

LangChain v1 では、従来の `AgentExecutor` や LangGraph の `create_react_agent` を統合・再設計した新しい中核 API [`create_agent`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py) が導入されました。本シリーズでは、LangChain v1（`libs/langchain_v1/langchain`）の実際のソースコードを追いながら、その内部アーキテクチャを 6 回にわたって解剖します。

Part 1 では、エージェント構築の司令塔である [`agents/factory.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py) に焦点を当て、`create_agent()` がどのように LangGraph の `StateGraph` を組み立て、ノードと条件付きエッジを配線しているのかをコードレベルで追跡します。

---

## 1. `create_agent()` の全体構築パイプライン

`create_agent()`（[`factory.py#L895-L1925`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L895-L1925)）は、単なる「モデルとツールをつなぐラッパー」ではありません。呼び出し時に以下の 6 段階のパイプラインを経て `CompiledStateGraph` を生成します。

{% mermaid %}
flowchart TD
    A["1. モデル・プロンプト正規化<br/>init_chat_model / SystemMessage"] --> B["2. ResponseFormat & Tools 準備<br/>AutoStrategy -> ToolStrategy / ToolNode"]
    B --> C["3. Middleware 収集・合成<br/>wrap_tool_call / wrap_model_call チェイン化"]
    C --> D["4. State / Input / Output スキーマ解決<br/>_resolve_schemas (OmitFromSchema反映)"]
    D --> E["5. ノード登録 & 境界ノード決定<br/>entry / loop_entry / loop_exit / exit"]
    E --> F["6. エッジ配線 & コンパイル<br/>条件付きエッジ + Transformers 登録"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style B fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style C fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style D fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    style E fill:#e0f7fa,stroke:#00838f,color:#006064
    style F fill:#fce4ec,stroke:#c2185b,color:#880e4f
{% endmermaid %}

### 1.1 ツール分類と `ToolNode` 生成

注目すべきは、`tools` 引数に含まれる **Provider Built-in Tools（`dict` 形式）** と **クライアント側ツール（`BaseTool` / `Callable`）** の分離です（[`factory.py#L1132-L1161`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1132-L1161)）。

```python
# Built-in provider tools（Web検索などプロバイダ側で実行される dict 定義）と通常ツールを分離
built_in_tools = [t for t in tools if isinstance(t, dict)]
regular_tools = [t for t in tools if not isinstance(t, dict)]

# クライアント側で実行する必要があるツール（ミドルウェア由来ツール + ユーザー定義ツール）
available_tools = middleware_tools + regular_tools

# クライアント側ツールが存在するか、wrap_tool_call ミドルウェアがある場合にのみ ToolNode を生成
# ※ 動的ツールを wrap_tool_call 内で処理するユースケースのため、ツール空でもラッパーがあれば生成する
tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
    else None
)

# ModelRequest のデフォルトツール一覧を構築
# ToolNode が Callable を BaseTool に変換済みなので tools_by_name.values() を利用する
# Structured Output 用ツールは実行時の戦略判定に応じて動的に追加するためここには含めない
if tool_node:
    default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
else:
    default_tools = list(built_in_tools)
```

ここで **「`available_tools` が空でも `wrap_tool_call` が定義されていれば `ToolNode` を作る」** という設計は実務上重要です。実行時にミドルウェアが `wrap_model_call` でツールを動的追加し、`wrap_tool_call` でその実行を横取りするパターンを破綻なくサポートするためです。

---

## 2. `_resolve_schemas` による 3 層スキーマの動的マージ

LangChain v1 の各ミドルウェアは独自の `state_schema` を宣言できます。`create_agent()` は、すべてのミドルウェアの `state_schema` とベース状態（`state_schema` 引数または `AgentState`）を統合し、LangGraph の `StateGraph` に渡す **State / Input / Output の 3 種類の `TypedDict`** を動的生成します（[`factory.py#L473-L542`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L473-L542), [`#L1229-L1237`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1229-L1237)）。

```python
base_state = state_schema if state_schema is not None else AgentState
# ミドルウェアのスキーマを登録順に並べ、最後に base_state を配置する
# 後勝ちマージにより、ユーザーが明示指定した state_schema がミドルウェアの定義を上書きできる
state_schemas: list[type] = [*(m.state_schema for m in middleware), base_state]

resolved_state_schema, input_schema, output_schema = _resolve_schemas(state_schemas)
```

`_resolve_schema()` の核心は、フィールドの型アノテーションに含まれる `OmitFromSchema` メタデータを検査し、入力スキーマ・出力スキーマから内部専用フィールドを除外する点にあります。

```python
def _resolve_schema(
    schema_hints: dict[type, dict[str, Any]],
    schema_name: str,
    omit_flag: str | None = None,
) -> type:
    all_annotations = {}

    for hints in schema_hints.values():
        for field_name, field_type in hints.items():
            should_omit = False

            if omit_flag:
                # Required[Annotated[T, ...]] / NotRequired[Annotated[T, ...]] / Annotated[T, ...] を剥がす
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break

            if not should_omit:
                all_annotations[field_name] = field_type

    # 動的に TypedDict クラスを生成して StateGraph へ渡す
    return cast("type", TypedDict(schema_name, all_annotations))
```

たとえば `AgentState` に定義されている `jump_to` には `PrivateStateAttr`（`OmitFromSchema(input=True, output=True)`）が、`structured_response` には `OmitFromInput`（`OmitFromSchema(input=True, output=False)`）が付与されています。これにより、外部の呼び出し元に対して不要な内部状態（`jump_to` など）が API スキーマに露出しない設計になっています。

---

## 3. ノード配線と 4 つの境界ノード設計

`create_agent()` は、ミドルウェアがどのフックをオーバーライドしているかを **クラスメソッドの同一性比較**（`m.__class__.before_model is not AgentMiddleware.before_model`）で判定し、必要なフックだけを LangGraph のノードとして登録します（[`factory.py#L1621-L1716`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1621-L1716)）。

そしてグラフ全体の配線を極めて見通しよく保つために、以下の **4 つの境界ノード** を算出します（[`factory.py#L1718-L1745`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1718-L1745)）。

| 境界変数 | 役割 | 決定ロジック |
| :--- | :--- | :--- |
| `entry_node` | エージェント開始時に 1 回だけ入る先頭ノード | `before_agent[0]` → `before_model[0]` → `"model"` |
| `loop_entry_node` | ツール実行後のループ戻り先（`before_agent` は除外） | `before_model[0]` → `"model"` |
| `loop_exit_node` | 各反復のモデル処理完了点（`after_agent` は除外） | `after_model[0]` → `"model"` |
| `exit_node` | エージェント終了時に 1 回だけ通る出口ノード | `after_agent[-1]` → `END` |

### 3.1 `before_*` は正順、`after_*` は逆順（オニオン構造）

注目すべきはエッジの接続順序です（[`factory.py#L1817-L1899`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1817-L1899)）。ミドルウェア `[M1, M2]` が登録されている場合、グラフ構造は次のようになります。

{% mermaid %}
flowchart LR
    START(["START"]) --> BA1["M1.before_agent"]
    BA1 --> BA2["M2.before_agent"]
    BA2 --> BM1["M1.before_model<br/>(loop_entry_node)"]
    BM1 --> BM2["M2.before_model"]
    BM2 --> MODEL["model<br/>(wrap_model_call stack)"]
    MODEL --> AM2["M2.after_model"]
    AM2 --> AM1["M1.after_model<br/>(loop_exit_node)"]
    AM1 -- "pending_tool_calls" --> TOOLS["tools<br/>(Send 並列実行)"]
    TOOLS -- "ループ継続" --> BM1
    AM1 -- "終了 / structured_response" --> AA2["M2.after_agent<br/>(exit_node)"]
    AA2 --> AA1["M1.after_agent"]
    AA1 --> END_NODE(["END"])

    style START fill:#eceff1,stroke:#455a64,color:#263238
    style END_NODE fill:#eceff1,stroke:#455a64,color:#263238
    style MODEL fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style TOOLS fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style BM1 fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style AM1 fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
{% endmermaid %}

`before_agent` と `before_model` は `M1 -> M2` の **正順** で接続されるのに対し、`after_model` と `after_agent` は `M2 -> M1` の **逆順** で接続されます。つまり、`M1` が最も外側、`M2` が内側となる完全な対称構造（スタック／オニオン構造）になっており、`after_model` の出口である `loop_exit_node` は `M1.after_model`（配列の index `0`）、`after_agent` の入口である `exit_node` は `M2.after_agent`（配列の index `-1`）を指します。

---

## 4. `model_node` の実行と無効ツール呼び出しの自動修復

`"model"` ノードに登録される `model_node` / `amodel_node`（[`factory.py#L1523-L1611`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1523-L1611)）では、モデルを呼び出す前に `_patch_invalid_tool_calls()` が走ります。

```python
def _patch_invalid_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    # すでに ToolMessage で応答済みの tool_call_id を収集
    answered_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    patched_messages: list[AnyMessage] = []
    for message in messages:
        patched_messages.append(message)
        if not isinstance(message, AIMessage):
            continue
        # JSON パース失敗などで invalid_tool_calls に入った呼び出しを検出
        for tool_call in message.invalid_tool_calls:
            if tool_call.get("id") in answered_ids:
                continue
            # エラーを示す合成 ToolMessage を差し込み、履歴の整合性を回復する
            if tool_message := _invalid_tool_call_message(tool_call):
                patched_messages.append(tool_message)
                answered_ids.add(tool_message.tool_call_id)
    return patched_messages
```

OpenAI や Anthropic の API は、`AIMessage` 内に未応答の `tool_call_id` が残ったまま次のリクエストを送ると 400 エラーを返します。`_patch_invalid_tool_calls()` は、引数パースに失敗した `invalid_tool_calls` に対して `status="error"` の `ToolMessage` を自動補完します。さらに修復が発生した場合、`_build_commands()`（[`factory.py#L225-L281`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L225-L281)）が `RemoveMessage(id=REMOVE_ALL_MESSAGES)` を先頭に付けてメッセージ履歴全体を原子的に置き換えます。

また、`_build_commands()` は **`has_structured_output=True` かつ今回のモデル応答に `structured_response` が無い場合、明示的に `state["structured_response"] = None` を書き込む** 点も重要です。これは、チェックポイント経由で複数ターン会話を行う際に「前ターンの `structured_response` が状態に残り続け、誤って即座にループ終了してしまうバグ」を防ぐための防衛的処理です。

---

## 5. 条件付きエッジによるルーティングと `Send` 並列ディスパッチ

エージェントループの遷移制御は、以下の 3 つのエッジファクトリ関数と `_add_middleware_edge` に集約されています。

### 5.1 `_make_model_to_tools_edge`（[`factory.py#L1991-L2042`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1991-L2042)）

モデル（および `after_model`）完了後の遷移先は、次の優先順位で決定されます。

```python
def model_to_tools(state: dict[str, Any]) -> str | list[Send] | None:
    # 1. ミドルウェアが設定した明示的 jump_to を最優先
    if jump_to := state.get("jump_to"):
        return _resolve_jump(
            jump_to,
            model_destination=model_destination,
            end_destination=end_destination,
        )

    last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

    # 2. AIMessage が存在しない場合は終了
    if last_ai_message is None:
        return end_destination

    tool_message_ids = [m.tool_call_id for m in tool_messages]

    # 3. ツール呼び出しが無ければ通常のエージェント終了条件
    if len(last_ai_message.tool_calls) == 0:
        return end_destination

    # 未実行かつ Structured Output 用ツールではない通常ツール呼び出しを抽出
    pending_tool_calls = [
        c
        for c in last_ai_message.tool_calls
        if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
    ]

    # 4. 未実行ツールがあれば LangGraph の Send API でツール呼び出し単位に並列ディスパッチ
    # ※ 以前は state 全体を各 Send に埋め込んでいたが O(N^2) の書き込み負荷になるため、
    #    ToolNode が実行時にチャネルから state をハイドレートする軽量方式に最適化されている
    if pending_tool_calls:
        return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

    # 5. Structured Output が生成済みであれば終了
    if state.get("structured_response") is not None:
        return end_destination

    # 6. tool_calls はあるが pending が空（例: バリデーション失敗で合成 ToolMessage が返された）
    #    → モデルへ戻して再試行させる
    return model_destination
```

特に注目すべきはステップ 4 の `[Send("tools", [tool_call]) for tool_call in pending_tool_calls]` です。モデルが 1 ターンで複数のツールを同時呼び出しした場合、LangGraph の `Send` プリミティブを使って並列に `tools` ノードへディスパッチしつつ、ペイロードには `[tool_call]` のみを載せることでチェックポインタへの書き込みサイズ爆発（$O(N^2)$ 問題）を回避しています。

### 5.2 `_make_tools_to_model_edge`（[`factory.py#L2072-L2111`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L2072-L2111)）

ツール実行後は、以下の条件で終了かモデル再帰かを決定します。
1. 実行されたすべてのクライアント側ツールが `return_direct=True` の場合 → `exit_node` へ終了（HITL によるツール名リダイレクトを考慮し、`ToolMessage.name` を優先して判定）。
2. `status != "error"` の Structured Output ツール結果が含まれる場合 → `exit_node` へ終了。
3. それ以外 → `loop_entry_node`（`before_model[0]` または `"model"`）へ戻り、次の推論サイクルへ。

---

## 6. コンパイル設定と Stream Transformers の統合

`create_agent()` の最終段（[`factory.py#L1900-L1925`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1900-L1925)）では、デフォルトの `recursion_limit` を `9_999` に引き上げ、ストリーム変換パイプライン（`transformers`）を `_dedupe_transformers()` でアイデンティティ重複排除（`id(factory)` ベース）しながら合成します。

```python
return graph.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=interrupt_before,
    interrupt_after=interrupt_after,
    debug=debug,
    name=name,
    cache=cache,
    transformers=_dedupe_transformers(
        [
            ToolCallTransformer,      # ツール呼び出しストリームの標準変換
            SubagentTransformer,      # サブエージェント用ストリーム変換
            *middleware_transformers, # 各ミドルウェアが宣言した transformers
            *(transformers or ()),    # ユーザー指定の transformers
        ]
    ),
).with_config(config)
```

---

## まとめ

`agents/factory.py` の `create_agent()` は、LangGraph の `StateGraph` 上に以下の設計をクリーンに落とし込んでいます。

- **境界ノード抽象化**（`entry_node`, `loop_entry_node`, `loop_exit_node`, `exit_node`）により、ミドルウェアの有無や数に依存せずグラフ配線を統一。
- **`OmitFromSchema` ベースの 3 層スキーマ生成** により、ミドルウェア内部状態を安全にカプセル化。
- **`Send` による軽量並列ツール実行** と **`_patch_invalid_tool_calls` による自動履歴修復** で、本番運用時のエッジケースを吸収。

次回 Part 2 では、Middleware 実行エンジンの核となる [`agents/middleware/types.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py)、ハンドラ合成のオニオン構造、および [`agents/middleware/_execution.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/_execution.py) のシェル実行ポリシー機構を詳しく解析します。
