---
title: "LangChain v1 Part 3 理論編：Structured Output（ToolStrategy vs ProviderStrategy）と Streaming / Event Streaming の設計"
date: 2026-05-07 13:36:43
categories: [AI]
tags: [LangChain, LangChain v1, Structured Output, ToolStrategy, Streaming, Event Streaming]
lang: ja
description: "LangChain v1 の create_agent における構造化出力（ProviderStrategy と ToolStrategy の使い分け・エラー自動リトライ）と、マルチモードストリーミングおよび Event Streaming v3 の設計理論を解説します。"
---

# LangChain v1 Part 3 理論編：Structured Output（`ToolStrategy` vs `ProviderStrategy`）と Streaming / Event Streaming の設計

エージェントを業務システムや Web アプリケーションへ組み込む際、避けて通れない 2 つの要件があります。

1. **Structured Output（構造化出力）**: 自然言語の回答を正規表現でパースするのではなく、Pydantic モデルや JSON として型安全・決定論的にデータを受け取ること。
2. **Streaming（ストリーミング）**: LLM の推論遅延（数秒〜数十秒）をユーザーに感じさせず、思考経過・ツール実行状況・生成トークンをリアルタイムに可視化すること。

本記事（Part 3 理論編）では、LangChain v1 の `create_agent` が備える **2 つの構造化出力戦略（`ProviderStrategy` と `ToolStrategy`）** と、**LangGraph ベースのマルチモード Streaming / Event Streaming（v3）** のアーキテクチャを徹底解説します。

---

## 1. Structured Output の仕組み：`response_format` の 2 大戦略

`create_agent` に `response_format` パラメータを指定すると、エージェントは最終回答を指定スキーマに従って検証し、グラフ状態辞書の **`result["structured_response"]`** キーへ格納して返します。

LangChain v1 では、構造化出力の実現方式として 2 つの戦略（Strategy）を提供しています（ソースコード: [langchain/agents/structured_output.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py) 相当）。

| 比較項目 | `ProviderStrategy` | `ToolStrategy` |
| :--- | :--- | :--- |
| **実現メカニズム** | プロバイダネイティブの構造化出力 API（Constrained Decoding 等）を利用。 | スキーマを仮想的な Tool 定義としてモデルに渡し、Tool Calling の引数から抽出・検証する。 |
| **対応モデル** | ネイティブ対応プロバイダ（OpenAI、Anthropic、Gemini、xAI 等）。 | **Tool Calling に対応したすべてのモデル**（OpenAI 互換エンドポイント・OSS MoE モデル含む）。 |
| **自動選択条件** | `response_format=MySchema` と直接クラスを渡した場合、モデルプロファイルが `structured_output: True` なら自動選択。 | 上記以外のモデルでは自動的に `ToolStrategy` へフォールバック。明示指定も可能。 |
| **Union 型（複数スキーマ択一）** | プロバイダ制約により制限あり。 | `ToolStrategy(Union[SchemaA, SchemaB])` で文脈に応じたスキーマ自動選択が可能。 |
| **バリデーション失敗時** | プロバイダ側でスキーマ強制（`strict=True` 対応プロバイダあり）。 | `handle_errors` により、Pydantic の検証エラーを `ToolMessage` でモデルへ返し**自動リトライ**可能。 |

{% mermaid %}
flowchart TD
    IN["create_agent(..., response_format=...)"] --> CHECK{"strategy 指定は？"}
    CHECK -->|"スキーマ型を直接指定"| PROF{"モデルプロファイルが<br/>ネイティブ構造化対応？"}
    CHECK -->|"ProviderStrategy 明示"| PS["ProviderStrategy<br/>(プロバイダネイティブ制約生成)"]
    CHECK -->|"ToolStrategy 明示"| TS["ToolStrategy<br/>(人工 Tool Calling による抽出)"]

    PROF -->|"Yes (OpenAI / Anthropic 等)"| PS
    PROF -->|"No (OpenAI互換 / OSSモデル等)"| TS

    PS --> VAL_P["スキーマ検証"]
    TS --> VAL_T["Pydantic / Schema 検証"]
    VAL_T -->|"検証エラー / 複数呼出"| ERR["ToolMessage でエラーを<br/>モデルへフィードバック (handle_errors)"]
    ERR -->|"自動リトライ"| TS
    VAL_P --> OUT["state['structured_response']"]
    VAL_T -->|"検証OK"| OUT

    style PS fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style TS fill:#fefcbf,stroke:#b7791f,stroke-width:2px,color:#744210
    style ERR fill:#fff5f5,stroke:#c53030,stroke-width:1.5px,color:#742a2a
    style OUT fill:#f0fff4,stroke:#2f855a,stroke-width:2px,color:#22543d
{% endmermaid %}

### OpenAI 互換エンドポイントで `ToolStrategy` を明示指定すべき理由

vLLM や SGLang などの OpenAI 互換エンドポイント（独自ホストの MoE モデル等）を `ChatOpenAI(base_url=...)` で呼び出す場合、プロバイダ側のネイティブ `json_schema` 厳密モード（`response_format` API）が未実装、あるいは一部制約（`anyOf` やネスト配列など）で 400 エラーになるケースが多々あります。

そのため、**OpenAI 互換エンドポイントでは `response_format=ToolStrategy(IncidentReport)` と明示指定するのが実務上の鉄則**です。これにより、モデルが普段通り得意とする Function Calling（Tool Calling）経路を使って構造化データを安全に抽出できます。

---

## 2. `ToolStrategy` のエラー自動修復と `handle_errors`

`ToolStrategy` の真価は、**モデルがスキーマ違反を起こした際の自動リカバリループ**にあります。

1. **Schema Validation Error（スキーマ検証エラー）**: たとえば `rating: int = Field(ge=1, le=5)` に対してモデルが `rating=10` を返した場合、LangChain は Pydantic のエラー詳細を `ToolMessage` としてモデルへ返却し、次のターンで `rating=5` へ自己修正させます。
2. **Multiple Structured Outputs Error（複数出力エラー）**: `Union[SchemaA, SchemaB]` を渡した際にモデルが両方のツールを同時に呼んでしまった場合、「1 つだけ選択してください」というエラー `ToolMessage` を返して再試行させます。

`handle_errors` パラメータは 5 通りの制御をサポートします。

- `True`（デフォルト）: すべての検証エラーを捕捉し、デフォルトのエラーテンプレートでリトライ。
- `str`: 固定のカスタムメッセージでリトライを促す。
- `type[Exception]` / `tuple[type[Exception], ...]`: 指定した例外型のみリトライし、それ以外は即座に送出。
- `Callable[[Exception], str]`: 例外種別（`StructuredOutputValidationError` / `MultipleStructuredOutputsError`）に応じて動的にフィードバック文を生成。
- `False`: リトライせず即座に例外を送出（厳格なバッチ処理向け）。

また、`tool_message_content="インシデントを登録しました"` を指定すれば、構造化抽出完了時に会話履歴へ残る `ToolMessage` の文言もカスタマイズできます。さらに、モデルが安全ポリシー等で応答を拒否（Refusal）した場合を明示的に検知したい場合は、`ModelRefusalMiddleware()` を併用することで `ModelRefusalError` を送出させられます。

```python
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRefusalMiddleware
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from pydantic import BaseModel, Field


class IncidentReport(BaseModel):
    """SRE インシデント報告書の構造化スキーマ。"""
    service: str = Field(description="障害が発生したサービス名")
    severity: Literal["P0", "P1", "P2"] = Field(description="重大度: P0, P1, P2")
    root_cause: str = Field(description="推定原因の要約")
    action_items: list[str] = Field(description="推奨アクションのリスト")


class MaintenanceNotice(BaseModel):
    """計画メンテナンス通知の構造化スキーマ（Union 択一の例）。"""
    service: str = Field(description="対象サービス名")
    window: str = Field(description="メンテナンス時間帯")


def custom_error_handler(error: Exception) -> str:
    """構造化出力の失敗理由に応じてモデルへ返す修正指示を切り替える。"""
    if isinstance(error, StructuredOutputValidationError):
        return "スキーマ検証に失敗しました。severity は P0/P1/P2 のいずれかを指定してください。"
    if isinstance(error, MultipleStructuredOutputsError):
        return "複数のスキーマが同時に選択されました。最も適切な 1 つだけを選択してください。"
    return f"構造化エラー: {error}"


# ToolStrategy を明示指定することで、OpenAI 互換モデルでも確実に構造化抽出できる
# Union スキーマ・カスタムエラーハンドラ・履歴メッセージの上書きを組み合わせた本番構成
strategy = ToolStrategy(
    schema=Union[IncidentReport, MaintenanceNotice],
    tool_message_content="構造化レポートの抽出が完了しました。",
    handle_errors=custom_error_handler,
)

# モデルの安全フィルタ等による拒否を明示的に検知したい場合は ModelRefusalMiddleware を併用する
# agent = create_agent(model=llm, tools=[], response_format=strategy, middleware=[ModelRefusalMiddleware()])
```

---

## 3. Streaming アーキテクチャ：3 つのストリームモードとマルチモード配信

`create_agent` は LangGraph の `CompiledStateGraph` を返すため、`agent.stream(...)` / `agent.astream(...)` によって以下の 3 種類のストリームモードを単体または複数同時に購読できます。

| `stream_mode` | 配信タイミング | 配信ペイロードの内容 | 主な用途 |
| :--- | :--- | :--- | :--- |
| **`updates`** | 各グラフノード（`model`, `tools`）の完了ごと | ノード名をキーとした状態差分辞書（例: `{"model": {"messages": [AIMessage(...)]}}`）。 | ステップ進捗表示、完了済みツール呼び出し・ツール結果の取得。 |
| **`messages`** | LLM がトークンを生成するたび（リアルタイム） | `(AIMessageChunk, metadata)` のタプル。`metadata["langgraph_node"]` で発生元ノードを判別可能。 | チャット UI でのタイプライター表示、思考トークン（`reasoning`）の逐次表示。 |
| **`custom`** | ツールや Middleware 内で `get_stream_writer()` を呼んだ瞬間 | 任意のユーザー定義データ（文字列や進捗 dict）。 | 長時間ツール（DB スキャン、外部 API ポーリング）の進捗バー更新。 |

### `custom` モードによるツール内部からの進捗配信

時間のかかる検索や外部 API 呼び出しを含むツールでは、`langgraph.config.get_stream_writer()` を使って途中経過をリアルタイムに送信できます。

```python
from langchain_core.tools import tool
from langgraph.config import get_stream_writer


@tool
def scan_cluster_logs(cluster: str) -> str:
    """クラスタ内のエラーログをスキャンする（進捗を custom ストリームへ通知）。"""
    # LangGraph 実行コンテキストからストリームライターを取得
    writer = get_stream_writer()
    writer({"stage": "connect", "cluster": cluster, "progress": 20})
    writer({"stage": "aggregate", "cluster": cluster, "progress": 80})
    return f"{cluster}: 503エラーが14件検出されました。"
```

> **注意**: ツール内で `get_stream_writer()` を呼び出す場合、そのツールは LangGraph の実行コンテキスト（`create_agent` や `StateGraph`）配下で実行する必要があります。単体テスト等で直接呼び出す場合はモックまたはガードを用意してください。

### マルチモードストリーミング：`stream_mode=["updates", "messages"]`

実務のチャット UI では、「LLM の文字は 1 トークンずつ滑らかに表示したい（`messages`）」と同時に、「どのノードが完了し、どのツールが呼ばれたかという状態遷移も把握したい（`updates`）」という要件がほぼ必須になります。

`stream_mode=["updates", "messages"]` のようにリストを渡すと、1 つのストリームループで両方のイベントを統合受信できます。

{% mermaid %}
flowchart LR
    AGENT["create_agent<br/>.stream(input, stream_mode=['updates', 'messages'])"]

    AGENT -->|"トークン生成ごと"| M_CH["mode == 'messages'<br/>(AIMessageChunk, metadata)"]
    AGENT -->|"ノード完了ごと"| U_CH["mode == 'updates'<br/>{'model': ..., 'tools': ...}"]

    M_CH --> UI1["UI テキスト領域へ<br/>リアルタイム追記"]
    U_CH --> UI2["ステップ進捗・ツール実行ログ・<br/>完了メッセージの確定表示"]

    style AGENT fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style M_CH fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
    style U_CH fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
{% endmermaid %}

> **API バージョンによる受信形式の違い**:
> - デフォルト（v1 形式）で複数モードを指定すると、ループは `for mode, chunk in agent.stream(..., stream_mode=["updates", "messages"]):` のように **`(mode, payload)` タプル** を返します。
> - `version="v2"` を指定すると、各要素は `{"type": "messages", "ns": ..., "data": ...}` という統一された `StreamPart` 辞書として返されます。

---

## 4. 次世代の型付き配信：Event Streaming（`stream_events(..., version="v3")`）

マルチモードの `agent.stream(...)` は強力ですが、複雑なフロントエンドやマルチエージェント構成では、タプルや辞書の分岐コードが肥大化しがちです。そこで LangChain v1 の最新世代では、**Event Streaming (`stream_events(..., version="v3")`)** が導入されています。

`stream_events(..., version="v3")` は、生タプルを解析する代わりに**型付きの投影（Projection）**を提供するランオブジェクトを返します。

- `stream.messages`: LLM 呼び出しごとの `ChatModelStream`。`message.text`（本文デルタ）、`message.reasoning`（思考デルタ）、`message.tool_calls`（ツール引数チャンク）、`message.output`（確定 `AIMessage` と `usage_metadata`）を個別に購読可能。
- `stream.tool_calls`: ツール実行のライフサイクル（`tool_name`, `input`, `output_deltas`, `output`, `error`）。
- `stream.subagents`: 名前付きサブエージェント（`create_agent(..., name="...")`）のストリームだけを自動分離。
- `stream.interleave("messages", "tool_calls")`: 同期コードで複数投影を時系列順にまとめて処理。非同期コードでは `astream_events` + `asyncio.gather` で並行消費。
- `stream.output`: ストリーム終了後の最終グラフ状態。

```python
# Event Streaming (v3) による型付きストリーム消費のイメージ
# タプルの手動アンパックが不要になり、テキストとツール実行をクリーンに分離できる
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "障害ログを確認してください"}]},
    version="v3",
)

for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
        # テキストトークンを逐次出力
        for token in item.text:
            print(token, end="", flush=True)
    elif kind == "tool_calls":
        # ツール呼び出し開始と結果をログ出力
        print(f"\n[TOOL] {item.tool_name}({item.input}) -> {item.output}")

# ループ完了後に最終状態へそのままアクセス可能
final_state = stream.output
```

---

## 5. まとめと実践編への橋渡し

Part 3 理論編の要点を整理します。

1. **`ProviderStrategy` vs `ToolStrategy`**: ネイティブ対応モデルでは `ProviderStrategy` が高信頼だが、OpenAI 互換エンドポイントや Union スキーマ、自動リトライ（`handle_errors`）を活かす場面では **`ToolStrategy` の明示指定**が最も堅牢。
2. **マルチモード Streaming**: `stream_mode=["updates", "messages"]` により、ノード単位の状態遷移とトークン単位のリアルタイム描画を 1 ループで両立できる。
3. **Event Streaming (v3)**: より複雑な UI やサブエージェント可視化には、`stream_events(..., version="v3")` の型付き投影が有効。

次の **Part 3 実践編** では、SRE の障害ログから `IncidentReport` を **`ToolStrategy`** で構造化抽出し、さらに **`stream_mode=["updates", "messages"]`** によるマルチモードストリーミングで実際に 27 チャンクの逐次生成が行われる様子を実測データで検証します。
