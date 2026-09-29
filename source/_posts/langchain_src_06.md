---
title: "LangChain v1 ソースコード解析 Part 6：init_chat_model（chat_models/base.py）と MCP Adapter（mcp/adapter.py, mcp/tools.py）の動的ロード機構"
date: 2026-09-10 08:14:18
categories: [AI]
tags: [Python, LLM, AI Agent, LangChain, MCP, FastMCP, ソースコード解析]
lang: ja
description: "LangChain v1 のモデル動的ファクトリ init_chat_model（chat_models/base.py）と MCP 統合アダプタ（mcp/adapter.py, mcp/tools.py）のソースコードを精読し、遅延インポート、_ConfigurableModel の宣言的操作キュー、URL 検証セキュリティ境界、LangGraph interrupt 連携 Elicitation を解説する。"
---

シリーズ最終回となる Part 6 では、LangChain v1 の **外部世界との接続レイヤー** を担う 2 つの重要モジュールを解析します。

1. [`chat_models/base.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py)：30 以上の LLM プロバイダを統一文字列から遅延ロードし、実行時切替まで可能にする `init_chat_model()` と `_ConfigurableModel`。
2. [`mcp/adapter.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/adapter.py) & [`mcp/tools.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/tools.py)：FastMCP 経由で MCP サーバーのツールを動的発見・変換し、LangGraph の `interrupt()` による Elicitation まで自動配線する `MCPAdapter` と `as_langchain_tool()`。

---

## 1. `init_chat_model` の動的プロバイダロード機構

`create_agent("openai:gpt-5.5")` のように文字列 1 つで任意のチャットモデルを起動できる背景には、[`chat_models/base.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py#L56-L654) の 3 段階パイプラインがあります。

{% mermaid %}
flowchart TD
    IN["init_chat_model(model, model_provider, configurable_fields, **kwargs)"] --> CFG{"configurable_fields あり<br/>または model is None?"}
    CFG -- "No (固定モデル)" --> PARSE["_parse_model()<br/>1. 'provider:model' 分割<br/>2. _attempt_infer_model_provider() プレフィックス推論"]
    PARSE --> CACHE["_get_chat_model_creator(provider)<br/>@lru_cache + importlib.import_module()"]
    CACHE --> INST["creator_func(model=model, **kwargs)<br/>→ BaseChatModel インスタンス"]

    CFG -- "Yes (動的切替モデル)" --> CONF["_ConfigurableModel(Runnable) 生成<br/>default_config + queued_declarative_operations"]
    CONF --> DECL["bind_tools() / with_structured_output()<br/>→ 操作をキューに積んで新インスタンス返却"]
    DECL --> RUN["invoke(input, config) 実行時<br/>_model(config) で実体化 & キュー再生"]

    style PARSE fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style CACHE fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style CONF fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style RUN fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
{% endmermaid %}

### 1.1 `_BUILTIN_PROVIDERS` と LRU キャッシュ付き遅延インポート

`langchain` 本体には各プロバイダ SDK を直接依存させず、[`_BUILTIN_PROVIDERS`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py#L56-L99) テーブルに `(module_path, class_name, creator_func)` の 3 つ組を宣言しています。

```python
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    "openai": ("langchain_openai", "ChatOpenAI", _call),
    "huggingface": (
        "langchain_huggingface",
        "ChatHuggingFace",
        # コンストラクタではなくファクトリメソッドが必要なプロバイダも統一抽象化
        lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs),
    ),
    "ibm": (
        "langchain_ibm",
        "ChatWatsonx",
        lambda cls, model, **kwargs: cls(model_id=model, **kwargs),
    ),
    "langsmith": ("langchain_openai", "ChatOpenAI", _init_langsmith),
    # ... 全 30 プロバイダ
}

@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(provider: str) -> Callable[..., BaseChatModel]:
    pkg, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    # importlib.import_module() で初回のみ動的ロードし、未導入時は `pip install <pkg>` を案内
    module = _import_module(pkg, class_name)
    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)
```

`_parse_model()`（[`base.py#L614-L653`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py#L614-L653)）では、まず `model.split(":", maxsplit=1)[0]` が `_BUILTIN_PROVIDERS` に含まれるかを確認します（これにより `bedrock:anthropic.claude-...` や `ollama:llama3:8b` のようにモデル名自体にコロンを含むケースでも正しく分割されます）。プレフィックスが無い場合は `_attempt_infer_model_provider()` で `gpt-`/`claude`/`gemini` 等から推論します。

### 1.2 `_ConfigurableModel` の「宣言的操作キュー」設計

`init_chat_model(configurable_fields=("model", "temperature"))` のように実行時切替を有効にすると、`BaseChatModel` ではなく `_ConfigurableModel`（[`base.py#L663-L1081`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py#L663-L1081)）が返されます。

ここでの難題は、**「実行時までどのプロバイダのモデルになるか分からないのに、定義時に `.bind_tools()` や `.with_structured_output()` を呼び出せるようにする」** ことです。`_ConfigurableModel` はこれを `__getattr__` と操作キュー `_queued_declarative_operations` で鮮やかに解決しています。

```python
_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")

class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __getattr__(self, name: str) -> Any:
        if name in _DECLARATIVE_METHODS:
            # 実体化前なので実行せず、(メソッド名, args, kwargs) を不変キューに積んだ新インスタンスを返す
            def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
                queued = [*self._queued_declarative_operations, (name, args, kwargs)]
                return _ConfigurableModel(
                    default_config=dict(self._default_config),
                    configurable_fields=self._configurable_fields,
                    config_prefix=self._config_prefix,
                    queued_declarative_operations=queued,
                )
            return queue
        # デフォルトモデルが存在すれば属性を委譲
        if self._default_config and (model := self._model()) and hasattr(model, name):
            return getattr(model, name)
        raise AttributeError(f"{name} is not a BaseChatModel attribute.")

    def _model(self, config: RunnableConfig | None = None) -> Runnable[Any, Any]:
        # invoke/ainvoke/stream 時に初めて config["configurable"] をマージして実体化
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        # 蓄積した bind_tools / with_structured_output / with_config を順番に再生
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model
```

なお、`batch()` / `abatch()` では `config` が単一なら下流モデルのネイティブ `batch` に委譲し、複数異なる `config` が渡された場合のみ `super().batch()` で並列実行する最適化も入っています（[`#L814-L868`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py#L814-L868)）。

---

## 2. `MCPAdapter`（`mcp/adapter.py`）：セキュリティ境界と Elicitation 武装

LangChain v1 は、Model Context Protocol（MCP）サーバーのツールをエージェントへ統合する公式アダプタ [`MCPAdapter`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/adapter.py#L124-L242) を標準搭載しています。内部では FastMCP クライアントを利用しつつ、**2 つの重要な独自設計** を加えています。

### 2.1 文字列ターゲットによる意図しないサブプロセス起動の防止

FastMCP の `Client(target)` は、`str` が渡されると **URL 判定より先にローカルファイルパス判定を行う** ため、設定ファイルやユーザー入力から渡された文字列が既存の `.py` / `.js` ファイル名だった場合、暗黙的にローカルサブプロセス（stdio）を起動してしまう危険があります。

`mcp/adapter.py` の `_validate_url_target()`（[`#L83-L122`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/adapter.py#L83-L122)）は、この脆弱性を構造的に塞いでいます。

```python
_URL_SCHEMES: Final = frozenset({"http", "https"})
_ANY_URL: Final = TypeAdapter(AnyUrl)

def _validate_url_target(target: str) -> None:
    try:
        url = _ANY_URL.validate_python(target)
    except ValidationError as validation_error:
        raise ValueError(f"MCP target {target!r} is not a valid URL. ...") from validation_error

    # ★ 重要: Pydantic の AnyUrl 単体では Windows パス "C:\server.py" を
    #    スキーム "c" の URL として受理してしまうため、必ず http/https に厳密制限する
    if url.scheme not in _URL_SCHEMES:
        raise ValueError(
            f"MCP target {target!r} has scheme {url.scheme!r}, "
            "but a string target must be an http or https URL."
        )
```

ローカルの stdio サーバーを起動したい場合は、`Path("server.py")` や明示的な `ClientTransport`、`MCPConfig` を使わせる設計になっており、「型によって意図を明示する」優れたセキュリティ境界です。

### 2.2 クライアントの非破壊クローンと LangGraph `interrupt()` 武装

`MCPAdapter.__init__()`（[`adapter.py#L173-L200`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/adapter.py#L173-L200)）では、渡された `FastMCPClient` や `ClientGroup` に対して、呼び出し元オブジェクトを破壊しないよう `client.new()` でクローンしたうえで `_arm_for_interrupts(clone)` を適用します（すでにカスタムの `_elicitation_callback` が設定済みの場合はそれを尊重）。これにより、MCP サーバーがツール実行中にユーザーへ追加質問（Elicitation）を返した際、自動的に LangGraph の `interrupt()` へ橋渡しされます。

---

## 3. `as_langchain_tool`（`mcp/tools.py`）：ツール動的変換とエラー・スキーマ正規化

`adapter.list_tools(cache_mode="use")` は、SEP-2549 のクライアントキャッシュを活用して MCP サーバーからツール一覧を取得し、各ツールを [`as_langchain_tool()`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/tools.py#L234-L305) で `StructuredTool` へ変換します。

{% mermaid %}
sequenceDiagram
    participant AG as "create_agent / ToolNode"
    participant ST as "StructuredTool (call_tool)"
    participant CL as "FastMCP Client / ClientGroup"
    participant CV as "_convert_call_tool_result"

    AG->>ST: "ainvoke(**arguments)"
    ST->>CL: "async with client: (リエントラント接続)"
    alt "_drives_interrupts(client) == True"
        CL->>CL: "_call_tool_with_interrupts() (Elicitation 対応)"
    else "通常呼び出し"
        CL->>CL: "call_tool(name, args, raise_on_error=False)"
    end
    CL-->>ST: "_ToolCallResult(content, structured_content, is_error)"
    ST->>CV: "変換処理"
    alt "is_error == True"
        CV--xST: "_MCPToolExecutionError (ToolException)"
        ST-->>AG: "_handle_mcp_tool_error → ToolMessage(status='error')"
    else "正常終了"
        CV-->>ST: "(list[ToolMessageContentBlock], MCPToolArtifact | None)"
        ST-->>AG: "content_and_artifact 形式で返却"
    end
{% endmermaid %}

### 3.1 `content_and_artifact` 分離と狭いエラーハンドリング

`as_langchain_tool()` の設計で光るのは次の 3 点です。

```python
return StructuredTool(
    name=tool.name,
    description=tool.description or "",
    # 1. 空 object プロパティに additionalProperties=True を補完し、プロバイダ変換時の厳格化を防ぐ
    args_schema=_normalize_mcp_schema(tool.input_schema),
    coroutine=call_tool,
    # 2. モデル向け content ブロックと構造化 artifact（structuredContent）を分離返却
    response_format="content_and_artifact",
    # 3. {"mcp": {"tool": {annotations, _meta}, "server": server_info}} に名前空間化して保持
    metadata=_tool_metadata(tool, requesting_client),
    # 4. MCP の isError=True のみ ToolMessage(status="error") に変換し、トランスポート障害はそのまま伝播
    handle_tool_error=_handle_mcp_tool_error,
)
```

特に **4 番目のエラー境界** は秀逸です。`client.call_tool(..., raise_on_error=False)` で呼び出し、`result.is_error` が `True` のときだけ `_MCPToolExecutionError(ToolException)` を送出します。これを `_handle_mcp_tool_error` がキャッチして `ToolMessage(status="error")` としてモデルに返すため、モデルが引数を修正して自己修復できます。一方、ネットワーク切断やプロトコル不整合などのトランスポート障害は `ToolException` ではないため握りつぶされず、呼び出し側へ即座に伝播します。

また `_convert_content_block()`（[`tools.py#L120-L166`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/tools.py#L120-L166)）は、`TextContent`・`ImageContent`（base64）・`ResourceLink`（URL）・`EmbeddedResource`（text/blob）をすべて LangChain 標準の `TextContentBlock` / `ImageContentBlock` / `FileContentBlock` へと型安全にマッピングしています。

---

## シリーズ全体のまとめ

全 6 回にわたり、LangChain v1（`libs/langchain_v1/langchain`）のコアソースコードを追跡してきました。

- **Part 1（`agents/factory.py`）**：`create_agent()` による境界ノード抽象化、`OmitFromSchema` 3 層スキーマ、`Send` 並列ツールディスパッチ。
- **Part 2（`middleware/types.py`, `_execution.py`）**：`ModelRequest.override()`、オニオン型ラッパー合成のリトライ安全性、隔離シェル実行ポリシー。
- **Part 3（`agents/structured_output.py`）**：`TypeAdapter` 統一処理、`AutoStrategy` の遅延能力判定、`ToolStrategy` の自己修復ループ。
- **Part 4（`summarization.py`, `context_editing.py`）**：破壊的要約（二分探索 + ペア保護 + 中抜き回復）と非破壊ツール結果クリアの二段構成。
- **Part 5（`human_in_the_loop.py`, `pii.py`, `model_fallback.py`, `model_retry.py`）**：動的スキーマ検証付き HITL、128 文字 lookback ストリーミング PII 検閲、クロスプロバイダ・キャッシュサニタイズ。
- **Part 6（`chat_models/base.py`, `mcp/adapter.py`, `mcp/tools.py`）**：`_ConfigurableModel` の操作キューと、セキュリティ境界・Elicitation 武装を備えた FastMCP 統合。

LangChain v1 は、LangGraph のグラフ実行基盤（`StateGraph`, `Command`, `Send`, `interrupt`, `StreamTransformer`）を最大限に活かしながら、本番エージェント開発で必ず直面するエッジケースをミドルウェアとファクトリの抽象化で綺麗に封じ込めた設計になっていることが、ソースコードの隅々から確認できました。
