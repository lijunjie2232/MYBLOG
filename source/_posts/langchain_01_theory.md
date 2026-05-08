---
title: "LangChain v1 Part 1 理論編：create_agent アーキテクチャ（Agent = Model + Harness）と Core Primitives"
date: 2026-05-01 10:35:11
categories: [AI]
tags: [LangChain, LangChain v1, LLM, AI Agent, Content Blocks, Architecture]
lang: ja
description: "LangChain v1 の中核思想である Agent = Model + Harness と create_agent の内部構造、そして Models・Messages・Standard Content Blocks の設計理論を徹底解説します。"
---

# LangChain v1 Part 1 理論編：`create_agent` アーキテクチャ（Agent = Model + Harness）と Core Primitives

LangChain は v1 系への進化において、従来の「多数のチェーンやレガシー抽象（`AgentExecutor` 等）を積み上げるフレームワーク」から、**「最小限かつ高度に構成可能なエージェント・ハーネス（Harness）」**へと設計思想を大きく刷新しました。

本シリーズでは、全 7 Part（各 Part 理論編＋実践編）にわたって LangChain v1 のアーキテクチャと実戦投入の勘所を体系的に掘り下げます。第 1 回となる本記事（理論編）では、すべての土台となる **Agent = Model + Harness** の原則と、それを支える Core Primitives（`create_agent` / Models / Messages / Standard Content Blocks）を解説します。

---

## 1. Agent = Model + Harness という設計哲学

LangChain v1 の公式ドキュメントが最初に掲げる定義は極めて明快です。

> **Agent = Model + Harness.**
> エージェントとは「タスクが完了するまでループ内でツールを呼び出し続けるモデル」であり、ハーネス（Harness）とは「そのループを取り巻くすべて（プロンプト、ツール、そしてモデルの振る舞いを制御するミドルウェア）」である。

従来の LLM アプリケーション開発では、プロンプト結合やツール呼び出しのディスパッチ、リトライ、状態管理がアプリケーションコードと密結合しがちでした。LangChain v1 ではこれらを明確に 2 層に分離します。

1. **Model（推論エンジン）**: コンテキストを読み取り、「どのツールをどの引数で呼ぶか」「ツール結果をどう解釈するか」「いつ最終回答を返すか」を判断する確率的エンジン。
2. **Harness（制御・文脈供給基盤）**: 「適切なタイミングで適切なコンテキストをモデルに届ける（get the model the right context at the right time）」責務を持つ決定論的な実行ループ。

{% mermaid %}
flowchart LR
    U["ユーザー入力<br/>(HumanMessage)"] --> H
    subgraph H["Harness (create_agent / LangGraph ランタイム)"]
        direction TB
        SP["System Prompt /<br/>Middleware 前処理"] --> M["Model Node<br/>(推論エンジン)"]
        M -->|"tool_calls あり"| T["Tools Node<br/>(ツール実行 / 並列ディスパッチ)"]
        T -->|"ToolMessage 追記"| M
        M -->|"tool_calls なし"| POST["Middleware 後処理 /<br/>構造化検証"]
    end
    POST --> OUT["最終出力<br/>(AIMessage / content_blocks)"]

    style H fill:#f0f7ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style M fill:#ebf8ff,stroke:#3182ce,stroke-width:2px,color:#1a365d
    style T fill:#fefcbf,stroke:#b7791f,stroke-width:2px,color:#744210
    style SP fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style POST fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
{% endmermaid %}

### LangChain / LangGraph / Deep Agents のレイヤー関係

LangChain エコシステムには 3 つのレイヤーが存在し、用途に応じて使い分けます。

| レイヤー | エントリポイント | 位置づけ・使いどころ |
| :--- | :--- | :--- |
| **Deep Agents** | Deep Agents SDK / Managed Deep Agents | 計画立案・仮想ファイルシステム・コンテキスト圧縮・サブエージェント等を標準搭載した「Batteries-included」なハーネス。 |
| **LangChain (`create_agent`)** | `langchain.agents.create_agent` | 最小限のハーネスから始め、ツールや Middleware を必要な分だけ合成（Compose）できる標準エージェント基盤。 |
| **LangGraph** | `StateGraph` / `CompiledStateGraph` | 決定論的ワークフローとエージェント的ループを自由に組み合わせる低レベルオーケストレーション基盤。 |

重要なのは、**`create_agent` は内部的に LangGraph の `CompiledStateGraph` として構築されている**という点です。これにより、`create_agent` を使うだけで LangGraph の強みである永続化（Checkpointer）、Human-in-the-Loop（中断と再開）、ストリーミング実行、Durable Execution をそのまま享受できます。

---

## 2. `create_agent` の構成要素とループの終了条件

`create_agent`（ソースコード: [langchain/agents/factory.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py) 相当）は、以下の主要パラメータでエージェントを構成します。

- `model`: モデル識別子文字列（例: `"openai:gpt-4.1"`）または初期化済みの `BaseChatModel` インスタンス（例: `ChatOpenAI`）。
- `tools`: Python の呼び出し可能オブジェクト、`@tool` デコレータで定義したツール、またはツール辞書のリスト。
- `system_prompt`: モデルの役割・行動規範を定める文字列または `SystemMessage`。実行時に動的に差し替えたい場合は Middleware を利用します。
- `response_format`: 最終応答を Pydantic モデル等の構造化データとして強制する設定（Part 3 で詳述）。
- `middleware`: モデル呼び出し前後やツール実行前後をフックする拡張レイヤー（Part 5・6 で詳述）。
- `checkpointer` / `store`: 短期記憶（会話スレッド）と長期記憶（スレッド横断ストレージ）（Part 4 で詳述）。

### エージェントループの 1 サイクルと並列ツール呼び出し

`agent.invoke({"messages": [...]})` を呼び出すと、ハーネスは以下の手順で状態（`AgentState`）を更新しながらループします。

1. 入力の `messages`（辞書形式 `{"role": "user", "content": ...}` は自動的に `HumanMessage` へ正規化）をグラフ状態に格納する。
2. **Model Node** が `system_prompt` と履歴 `messages`、および `tools` の JSON Schema を束ねてモデルへ送信する。
3. モデルが返した `AIMessage` に `tool_calls` が含まれていれば、**Tools Node** へ遷移する。
   - モデルが 1 ターンで複数の `tool_calls` を発行した場合（例: 2 つのリージョンのステータスを同時に照会）、Tools Node はそれらをすべて実行し、各 `tool_call["id"]` に 1 対 1 で対応する `ToolMessage` を生成して履歴へ追加する。
4. `ToolMessage` が追加された最新履歴を再び **Model Node** へ渡す。
5. `AIMessage.tool_calls` が空になった時点でループを終了し、最終的なグラフ状態辞書（`{"messages": [...]}` 等）を返す。

この「`tool_calls` の有無による条件分岐」こそが ReAct ループの最小単位であり、`create_agent` ではこれをわずか数行で宣言的に構築できます。

```python
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. モデルの初期化（OpenAI互換エンドポイントも同一インターフェースで接続可能）
# 環境変数からエンドポイントとAPIキーを読み込み、コードへのハードコードを避ける
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,   # ツール選択と事実要約の再現性を高めるため 0.0 に固定
    max_tokens=512,
    timeout=30,        # ハング防止のため接続・応答タイムアウトを明示
    max_retries=1,     # 一時的なネットワーク揺らぎに対する最小リトライ
)

# 2. ツールの定義（型ヒントと docstring がそのまま LLM 向けの JSON Schema になる）
@tool
def check_health(service: str) -> str:
    """指定されたサービスの稼働ステータスを返す。"""
    return f"{service}: OK"

# 3. create_agent による最小ハーネスの構築
agent = create_agent(
    model=llm,
    tools=[check_health],
    system_prompt="あなたはSREアシスタントです。ツールを用いて簡潔に回答してください。",
)
```

---

## 3. Core Primitive 1：Models（標準モデルインターフェース）

LangChain v1 において、Chat Model は単なるテキスト生成器ではなく、**Tool Calling・Structured Output・Multimodality・Reasoning** を統一インターフェース（`BaseChatModel`）で扱うための抽象層です。

### モデルの初期化方式：`init_chat_model` vs プロバイダクラス

1. **`init_chat_model("provider:model_name")`**:
   - `"openai:..."` や `"anthropic:..."`、`"google_genai:..."` のように文字列指定でプロバイダを切り替えられるファクトリ関数。モデル選定を設定ファイルや環境変数へ外出ししたい場合に強力です。
2. **プロバイダ固有クラス（例: `ChatOpenAI`）**:
   - 社内ゲートウェイや vLLM / SGLang などの **OpenAI 互換 API サーバ**（カスタム `base_url` や独自認証ヘッダ）を利用する場合、`ChatOpenAI(model=..., base_url=..., api_key=...)` を直接インスタンス化して `create_agent(model=llm, ...)` に渡すのが最も確実です。

### 運用を見据えた必須パラメータと接続耐性

本番エージェントでは、モデル初期化時に以下のパラメータを明示することが重要です。

- `temperature`: ツール引数生成や障害調査など決定論性が求められるタスクでは `0.0` を推奨。
- `max_tokens`: 無限生成やコンテキスト浪費を防ぐ上限値。
- `timeout` / `max_retries`: 外部推論エンドポイントの遅延や 429/5xx エラーに対する最低限のガード。さらに上位のフォールバックや回数制限は Middleware（`ModelFallbackMiddleware`、`ModelCallLimitMiddleware`）で分離管理します。

---

## 4. Core Primitive 2：Messages（文脈の基本単位）

Messages は、モデルへの入力と出力、そして会話状態そのものを表現する LangChain の基本データ構造です。すべてのメッセージは **Role（役割）・Content（内容）・Metadata（メタデータ）** の 3 要素を持ちます。

### 4 つの基本メッセージ型

| クラス / `type` | 役割 | 主な属性・特徴 |
| :--- | :--- | :--- |
| `SystemMessage` (`system`) | モデルのペルソナ・制約・行動指針 | `create_agent` の `system_prompt` で指定。 |
| `HumanMessage` (`human`) | ユーザーからの入力（テキスト・マルチモーダル） | 辞書 `{"role": "user", "content": ...}` からの自動変換に対応。 |
| `AIMessage` (`ai`) | モデルからの応答・ツール呼び出し要求 | `content`, `content_blocks`, `tool_calls`, `usage_metadata`, `response_metadata` を保持。 |
| `ToolMessage` (`tool`) | ツール実行結果の返却 | `tool_call_id` により `AIMessage.tool_calls` の各呼び出しと厳密に紐付く。`artifact` でモデル非公開データも保持可能。 |

### `tool_call_id` の厳密な対応関係と `usage_metadata`

モデルが `tool_calls` を含む `AIMessage` を返した場合、履歴上その直後には**すべての `id` に対応する `ToolMessage` が必ず存在しなければなりません**。1 つでも欠落すると、次ターンの API 呼び出しでプロバイダ側から 400 Bad Request エラーが返されます。`create_agent` 内の Tools Node はこの整合性を自動保証します。

また、`AIMessage` には以下のメタデータが格納され、コスト監視や異常検知に利用できます。
- `usage_metadata`: `input_tokens`, `output_tokens`, `total_tokens` および推論トークン数（`output_token_details.reasoning`）やキャッシュヒット数。
- `response_metadata`: プロバイダ固有の `finish_reason` に加え、プロバイダ間で統一された `normalized_finish_reason`（`stop`, `length`, `tool_call`, `content_filter`, `refusal` 等）。

---

## 5. Core Primitive 3：Standard Content Blocks（`content_blocks`）

LangChain v1 における最も重要なデータモデル刷新の一つが **Standard Content Blocks** です。

### なぜ `content` だけでは不十分なのか？

従来の `message.content` は、`str` だったり `list[dict]` だったりと緩やかに型付けされており、しかもリスト内の構造がプロバイダごとに異なっていました。たとえば「思考過程（Reasoning）」ひとつ取っても、プロバイダ間でフォーマットがバラバラです。

- **Anthropic**: `{"type": "thinking", "thinking": "...", "signature": "..."}`
- **OpenAI**: `{"type": "reasoning", "id": "rs_...", "summary": [{"type": "summary_text", "text": "..."}]}`
- **OpenAI 互換エンドポイント（通常応答）**: 単純な文字列 `"..."`

アプリケーションコード側で `isinstance(msg.content, str)` やプロバイダ別の条件分岐を書き続けると、モデル差し替えのたびに UI やパーサが壊れてしまいます。

### `content_blocks` プロパティによる遅延正規化（Lazy Parsing）

LangChain v1 のすべてのメッセージオブジェクトは、**`content_blocks` プロパティ**を実装しています。これにアクセスすると、元の `content`（文字列でもプロバイダ固有の辞書リストでも）が遅延評価によって**プロバイダ非依存の標準 TypedDict リスト**へと正規化されます。

{% mermaid %}
flowchart TD
    subgraph RAW["プロバイダ固有の message.content"]
        R1["文字列<br/>'正常です'"]
        R2["Anthropic 形式<br/>[{type: 'thinking', ...}, {type: 'text', ...}]"]
        R3["OpenAI 形式<br/>[{type: 'reasoning', summary: [...]}, {type: 'text', ...}]"]
    end

    PARSE["message.content_blocks<br/>(遅延パーサ / 標準化レイヤー)"]

    R1 --> PARSE
    R2 --> PARSE
    R3 --> PARSE

    subgraph STD["Standard Content Blocks (統一出力)"]
        B1["TextContentBlock<br/>{'type': 'text', 'text': '...'}"]
        B2["ReasoningContentBlock<br/>{'type': 'reasoning', 'reasoning': '...'}"]
        B3["ToolCall / Image / Audio / Refusal Block"]
    end

    PARSE --> B1
    PARSE --> B2
    PARSE --> B3

    style RAW fill:#fff5f5,stroke:#c53030,stroke-width:1.5px,color:#742a2a
    style PARSE fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style STD fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

主な標準ブロック型は以下の通りです。

- `TextContentBlock`: `{"type": "text", "text": "..."}` — 標準テキスト出力。
- `ReasoningContentBlock`: `{"type": "reasoning", "reasoning": "...", "extras": {...}}` — 思考・推論トレース。
- `RefusalContentBlock`: `{"type": "refusal", "refusal": "..."}` — 安全性ポリシー等による明示的な拒否。
- マルチモーダル系: `ImageContentBlock` (`type: "image"`), `AudioContentBlock` (`type: "audio"`), `VideoContentBlock` (`type: "video"`), `FileContentBlock` (`type: "file"`)。

```python
from langchain_core.messages import AIMessage

# 例: 通常の文字列レスポンスでも content_blocks は常にリスト形式で正規化される
msg = AIMessage(content="ap-northeast-1 は HEALTHY です。")

# アプリケーション側ではプロバイダ差を意識せずブロック種別でディスパッチできる
for block in msg.content_blocks:
    if block["type"] == "text":
        print("[TEXT]", block["text"])
    elif block["type"] == "reasoning":
        print("[REASONING]", block["reasoning"])
```

さらに、LangChain 外部のシステムへ標準ブロック形式のままシリアライズしたい場合は、環境変数 `LC_OUTPUT_VERSION=v1` またはモデル初期化時に `output_version="v1"` を指定することで、`content` 自体に標準ブロックを格納することも可能です。

---

## 6. まとめと実践編への橋渡し

本記事（Part 1 理論編）の要点を整理します。

1. **Agent = Model + Harness**: モデルを推論エンジン、`create_agent` を文脈供給とループ制御のハーネスとして責務分離する。
2. **`create_agent` は LangGraph 上に構築されている**: `model`・`tools`・`system_prompt` の最小構成から始め、必要に応じて並列ツール呼び出しや永続化・Middleware を段階的に追加できる。
3. **Messages と `content_blocks`**: `HumanMessage` / `AIMessage` / `ToolMessage` の厳密な ID 連鎖によってツールループが成立し、`content_blocks` によってプロバイダ間の出力フォーマット差を吸収できる。

次の **Part 1 実践編** では、OpenAI 互換エンドポイント上の MoE モデル（`MiniCPM-V-4.7-35B-A3B`）を用いて最小の SRE エージェントを構築し、**並列ツール呼び出しによる 5 メッセージの遷移**と **`content_blocks` の実際の正規化結果**を、実測ログとともに検証します。
