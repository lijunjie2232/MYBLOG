---
title: "LangChain v1 ソースコード解析 Part 3：Structured Output 実装（agents/structured_output.py）の ToolStrategy vs ProviderStrategy 内部機構"
date: 2026-08-18 17:58:47
categories: [AI]
tags: [Python, LLM, AI Agent, LangChain, Structured Output, ソースコード解析]
lang: ja
description: "LangChain v1 の構造化出力モジュール agents/structured_output.py と factory.py の連携コードを精読し、ToolStrategy・ProviderStrategy・AutoStrategy の自動選択ロジック、Union/oneOf 展開、自動リトライ機構を解説する。"
---

LLM エージェントを実務システムに組み込む際、最終出力を型安全な JSON / Pydantic モデルとして保証する **Structured Output** は不可欠です。LangChain v1 では、プロバイダネイティブの JSON Schema 強制機能と、人工的なツール呼び出しによるエミュレーションを統一的に扱う [`agents/structured_output.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py) が導入されました。

Part 3 では、`structured_output.py` のスキーマ正規化・パース機構と、[`agents/factory.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py) 側でのバインディング・エラー自動リトライの連携をソースコードから紐解きます。

---

## 1. 全体アーキテクチャ：3 つの戦略と `_SchemaSpec`

`structured_output.py` が提供する戦略クラスは以下の 3 つで、これらの Union 型が `ResponseFormat`（[`structured_output.py#L473`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L473)）として定義されています。

- **`ProviderStrategy`**：OpenAI や Anthropic などのプロバイダネイティブ `response_format={"type": "json_schema", ...}` を使用。
- **`ToolStrategy`**：スキーマから人工的な `StructuredTool` を生成し、モデルの Tool Calling 機能で構造化出力を強制。Union 型（複数スキーマ候補）やバリデーション失敗時の自動リトライに対応。
- **`AutoStrategy`**：生の Pydantic モデル等が渡されたときのラッパー。モデル実行直前に `_supports_provider_strategy()` でモデル能力を検査し、`ProviderStrategy` か `ToolStrategy` へ動的に解決する。

{% mermaid %}
flowchart TD
    IN["create_agent(response_format=Schema)"] --> CHK{"戦略オブジェクトか?"}
    CHK -- "Raw Schema (Pydantic等)" --> AUTO["AutoStrategy(schema)<br/>※ ToolStrategy も事前生成してツール名確保"]
    CHK -- "ToolStrategy" --> TS["ToolStrategy"]
    CHK -- "ProviderStrategy" --> PS["ProviderStrategy"]

    AUTO --> DETECT{"_supports_provider_strategy()<br/>(model.profile / regex)"}
    DETECT -- "True" --> PS_EFF["ProviderStrategy 採用"]
    DETECT -- "False" --> TS_EFF["ToolStrategy 採用"]

    PS --> PS_EFF
    TS --> TS_EFF

    PS_EFF --> BIND_P["bind_tools(tools, response_format=json_schema)<br/>OpenAI互換は strict=True"]
    TS_EFF --> BIND_T["bind_tools(tools + structured_tools, tool_choice='any')"]

    BIND_P --> PARSE_P["ProviderStrategyBinding.parse(AIMessage)<br/>JSON抽出 -> TypeAdapter検証"]
    BIND_T --> PARSE_T["OutputToolBinding.parse(tool_call['args'])<br/>TypeAdapter検証 + エラー時自動リトライ"]

    style AUTO fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style PS_EFF fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style TS_EFF fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style PARSE_P fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    style PARSE_T fill:#fce4ec,stroke:#c2185b,color:#880e4f
{% endmermaid %}

---

## 2. `_SchemaSpec` と `TypeAdapter` による 4 種スキーマの統一処理

LangChain v1 の構造化出力は、**Pydantic `BaseModel`・`dataclass`・`TypedDict`・生 JSON Schema `dict`** の 4 種類をシームレスにサポートします。その変換を担うのが `_SchemaSpec`（[`structured_output.py#L109-L195`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L109-L195)）と `_parse_with_schema`（[`#L79-L106`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L79-L106)）です。

```python
# _SchemaSpec.__init__ のスキーマ種別判定と JSON Schema 生成
if isinstance(schema, dict):
    self.schema_kind = "json_schema"
    self.json_schema = schema
elif isinstance(schema, type) and issubclass(schema, BaseModel):
    self.schema_kind = "pydantic"
    self.json_schema = schema.model_json_schema()
elif is_dataclass(schema):
    self.schema_kind = "dataclass"
    # Pydantic v2 の TypeAdapter を使い、dataclass / TypedDict も統一的に JSON Schema 化
    self.json_schema = TypeAdapter(schema).json_schema()
elif is_typeddict(schema):
    self.schema_kind = "typeddict"
    self.json_schema = TypeAdapter(schema).json_schema()
else:
    raise ValueError(f"Unsupported schema type: {type(schema)}...")
```

パース時も `TypeAdapter(schema).validate_python(data)` に一元化されています。ただし重要な注意点として、**`schema_kind == "json_schema"`（生 dict）の場合はバリデーションを行わず `data` をそのまま返す** 仕様になっています（[`structured_output.py#L96-L98`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L96-L98)）。そのため、自動バリデーション＆リトライの恩恵を受けたい場合は Pydantic / dataclass / TypedDict を使う必要があります。

---

## 3. `ToolStrategy` vs `ProviderStrategy` の詳細比較

### 3.1 `ToolStrategy`：Union / `oneOf` 展開と人工ツールバインディング

`ToolStrategy`（[`structured_output.py#L198-L270`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L198-L270)）の強力な特徴は、Python の `Union[A, B]`（`A | B`）や JSON Schema の `oneOf` を **再帰的に展開して複数の人工ツールとしてモデルに提示できる** 点です。

```python
def _iter_variants(schema: Any) -> Iterable[Any]:
    """Union 型や JSON Schema の oneOf からリーフとなる個別スキーマを再帰展開する。"""
    if get_origin(schema) in {UnionType, Union}:
        for arg in get_args(schema):
            yield from _iter_variants(arg)
        return

    if isinstance(schema, dict) and "oneOf" in schema:
        for sub in schema.get("oneOf", []):
            yield from _iter_variants(sub)
        return

    yield schema

self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]
```

各 `_SchemaSpec` は `OutputToolBinding.from_schema_spec()`（[`structured_output.py#L340-L357`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L340-L357)）によって、`args_schema=schema_spec.json_schema` を持つ `StructuredTool` へと変換されます。そしてモデル呼び出し時（[`factory.py#L1461-L1484`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1461-L1484)）には、通常ツールにこれら構造化出力ツールを合流させたうえで **`tool_choice="any"`** を設定し、モデルに「通常ツールか構造化出力ツールのいずれかを必ず呼ぶ」よう強制します。

さらに注目すべきは、`wrap_model_call` ミドルウェアによる **Union 絞り込み（narrowing）** のサポートです（[`factory.py#L1431-L1443`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1431-L1443)）。エージェント作成時に `ToolStrategy(SchemaA | SchemaB)` を宣言しておき、特定の状況下だけミドルウェアが `request.override(response_format=ToolStrategy(SchemaA))` と絞り込むと、そのターンでは `SchemaA` のツールだけがモデルにバインドされます（逆に、事前宣言されていない新規スキーマの追加は `ValueError` で弾かれます）。

### 3.2 `ProviderStrategy`：ネイティブ JSON Schema とテキスト抽出

一方、`ProviderStrategy`（[`structured_output.py#L273-L317`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py#L273-L317)）は `to_model_kwargs()` でプロバイダ向けの `response_format` 辞書を生成します。

```python
def to_model_kwargs(self) -> dict[str, Any]:
    json_schema: dict[str, Any] = {
        "name": self.schema_spec.name,
        "schema": self.schema_spec.json_schema,
    }
    if self.schema_spec.strict:
        json_schema["strict"] = True

    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        }
    }
```

`factory.py` の `_get_bound_model()`（[`factory.py#L1446-L1459`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1446-L1459)）では、`BaseChatOpenAI` 継承モデルかつ Responses API 未使用の場合に後方互換のため `strict=True` を自動付与します。応答時は、`output.tool_calls` が空（＝通常ツールを呼ばず最終回答を返した）場合にのみ `ProviderStrategyBinding.parse(output)` が走り、マルチモーダルな `content` ブロック配列からもテキストを連結抽出して `json.loads()` → `_parse_with_schema()` で検証します。

| 比較軸 | `ProviderStrategy` | `ToolStrategy` |
| :--- | :--- | :--- |
| 仕組み | プロバイダ側デコーダによる JSON Schema 強制 | 人工 `StructuredTool` + `tool_choice="any"` |
| Union / `oneOf` 対応 | 単一スキーマ前提 | 再帰展開して複数ツールとして提示可能 |
| バリデーション失敗時 | `StructuredOutputValidationError` を即送出 | `handle_errors` に従い合成 `ToolMessage` で自動リトライ |
| 履歴への影響 | `AIMessage` のみ追加 | `AIMessage` + 合成 `ToolMessage` を追加 |

---

## 4. `AutoStrategy` のモデル能力判定ロジック

ユーザーが `create_agent(model, response_format=MyPydanticModel)` のように生スキーマを渡すと、`AutoStrategy` が生成され、モデル実行直前の `_get_bound_model()` で `_supports_provider_strategy()`（[`factory.py#L581-L626`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L581-L626)）が評価されます。

```python
def _supports_provider_strategy(
    model: str | BaseChatModel, tools: list[BaseTool | dict[str, Any]] | None = None
) -> bool:
    # 1. BaseChatModel の model.profile["structured_output"] を優先参照
    if isinstance(model, BaseChatModel):
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or getattr(model, "model_id", "")
        model_profile = model.profile
        if (
            model_profile is not None
            and model_profile.get("structured_output")
            # ★ 例外ルール: Gemini 3 未満は「ツール呼び出し + ネイティブ構造化出力」の同時利用に非対応
            and not (
                tools
                and isinstance(model_name, str)
                and "gemini" in model_name.lower()
                and "gemini-3" not in model_name.lower()
            )
        ):
            return True

    # 2. profile が無い場合は FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT（正規表現リスト）で判定
    return (
        any(re.search(pattern, model_name.lower()) for pattern in FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT)
        if model_name
        else False
    )
```

「Gemini 3 未満かつツールありの場合は `ProviderStrategy` を避けて `ToolStrategy` にフォールバックする」といった現実的なプロバイダ制約までコードレベルで丁寧に吸収しているのが分かります。しかも判定を `create_agent()` 時ではなく `_get_bound_model()` 時に遅延実行しているため、`ModelFallbackMiddleware` や `ModelRouterMiddleware` が実行時に別モデルへ差し替えた場合でも、差し替え後のモデル能力に応じて最適な戦略が自動選択されます。

---

## 5. `ToolStrategy` のエラー検知と自動リトライ機構

`ToolStrategy` の真価は、[`_handle_model_output()`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L1249-L1351) と [`_handle_structured_output_error()`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L649-L676) による自己修復ループにあります。

{% mermaid %}
sequenceDiagram
    participant M as "model_node"
    participant H as "_handle_model_output"
    participant E as "_make_model_to_tools_edge"

    M->>H: "AIMessage (structured_tool_calls)"
    alt "複数スキーマを同時呼び出し"
        H->>H: "MultipleStructuredOutputsError 生成"
        H-->>M: "ToolMessage(content=error_msg) を全呼出分返却<br/>structured_response は未設定(None)"
    else "単一呼び出し・バリデーション失敗"
        H->>H: "StructuredOutputValidationError 生成"
        H-->>M: "ToolMessage(content=error_msg) を返却<br/>structured_response は未設定(None)"
    else "単一呼び出し・パース成功"
        H-->>M: "ToolMessage(確定応答) + structured_response=parsed"
    end

    M->>E: "state 更新後にエッジ評価"
    alt "structured_response != None"
        E-->>E: "exit_node へ遷移（完了）"
    else "エラーリトライ（pending_tool_calls なし & structured_response == None）"
        E-->>M: "loop_entry_node へ戻り再推論"
    end
{% endmermaid %}

`ToolStrategy.handle_errors` には 6 パターンの設定（`True` / `False` / カスタムエラー文字列 `str` / 例外型 `type[Exception]` / 例外タプル / カスタム関数 `Callable[[Exception], str]`）を指定でき、リトライ可能な場合はエラー内容を `ToolMessage` として履歴に積みます。すると Part 1 で見た `_make_model_to_tools_edge()` のステップ 6（「`tool_calls` はあるが未実行ツールも `structured_response` もない」）に該当し、グラフが自動的に `model` ノードへ戻って LLM に修正再生成を促します。

---

## まとめ

`agents/structured_output.py` と `factory.py` の協調設計により、LangChain v1 の Structured Output は以下を実現しています。

- **`TypeAdapter` による 4 形式の統一処理** と、Union / `oneOf` の自動展開。
- **モデル実行時の動的戦略解決（`AutoStrategy`）** により、フォールバックやモデルルーティングと組み合わせても破綻しない。
- **`ToolStrategy` の自己修復ループ** により、`MultipleStructuredOutputsError` や `StructuredOutputValidationError` を `ToolMessage` フィードバックとしてモデルに返し、グラフ遷移だけで自動リトライを完結。

次回 Part 4 では、Built-in Middleware の第 1 弾として、長大コンテキストを制御する `SummarizationMiddleware` と `ContextEditingMiddleware` のソースコードを解析します。
