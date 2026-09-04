---
title: DeepAgents Part 7 理論編：本番運用設計（Streaming・Checkpointer・Rubric 評価・ACP / AG-UI / A2A プロトコル連携）
date: 2026-08-01 07:58:39
categories:
  - AI
tags:
  - DeepAgents
  - LangGraph
  - Streaming
  - RubricMiddleware
  - ACP
  - AG-UI
  - A2A
  - Production
lang: ja
description: DeepAgents を本番システムへ統合するための運用アーキテクチャを網羅解説。サブグラフ対応 Streaming、Checkpointer とマルチテナント設計、RubricMiddleware による実行時 LLM-as-a-Judge 自己改善ループ、そして ACP / AG-UI / A2A / MCP の 4 大プロトコル棲み分けまで整理します。
---

DeepAgents シリーズの最終章となる Part 7 では、ローカルで動くエージェントを**「観測可能・耐障害・品質保証済み・外部エコシステムと相互運用可能な本番システム」**へ昇華させるための設計論を解説します。

扱うテーマは以下の4本柱です。

1. **Streaming アーキテクチャ**：メインエージェントと並列サブエージェントの進捗・トークン・カスタムイベントをリアルタイム可視化する仕組み
2. **本番永続化とマルチテナント（Going to Production）**：`thread_id` と `context` の分離、Checkpointer による Durable Execution、Store Namespace スコーピング、PII ガードレール
3. **Grading Rubrics（`RubricMiddleware`）**：実行時 LLM-as-a-Judge によって「完了の定義（Definition of Done）」を満たすまで自律反復させる品質保証ループ
4. **エコシステム連携プロトコル（ACP / AG-UI / A2A / MCP）**：IDE、Web フロントエンド、他社エージェント、外部ツールとの標準接続仕様

---

## 1. 本番 DeepAgents システムの全体アーキテクチャ

本番環境の DeepAgents は、LangGraph の Durable Execution 基盤の上で動き、外縁部で各種プロトコルを通じてクライアントや他エージェントと接続されます。

{% mermaid %}
flowchart TB
    subgraph Clients["外部クライアント / エコシステム"]
        IDE["コードエディタ / IDE<br/>(Zed, JetBrains, VS Code)"]
        WEB["Web フロントエンド<br/>(React / CopilotKit / useStream)"]
        AGENT2["外部 AI エージェント<br/>(Google ADK 等)"]
    end

    subgraph Protocols["標準プロトコル層"]
        ACP["ACP (stdio)<br/>deepagents-acp"]
        AGUI["AG-UI (SSE)<br/>@ag-ui/langgraph"]
        A2A["A2A (JSON-RPC)<br/>/a2a/{assistant_id}"]
    end

    subgraph Runtime["DeepAgents 本番ランタイム (LangSmith Agent Server / Self-Hosted)"]
        MW["Middleware スタック<br/>PatchToolCalls / PII / HITL / Rubric"]
        GRAPH["LangGraph 実行エンジン<br/>stream_mode: updates / messages / custom"]
        SUB["Subagents (並列サブグラフ)<br/>ns=('tools:call_id',)"]
        MCP["MCP クライアント<br/>外部ツール呼び出し"]
    end

    subgraph Persistence["永続化・観測層"]
        CP[("Checkpointer<br/>thread_id 単位・短期状態/再開")]
        ST[("LangGraph Store<br/>user_id / org_id 単位・長期記憶")]
        LS["LangSmith Tracing<br/>分散トレース / 評価"]
    end

    IDE <--> ACP <--> Runtime
    WEB <--> AGUI <--> Runtime
    AGENT2 <--> A2A <--> Runtime
    MW --> GRAPH --> SUB
    GRAPH <--> MCP
    GRAPH <--> CP
    GRAPH <--> ST
    Runtime --> LS

    style Protocols fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Runtime fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style Persistence fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

---

## 2. Streaming：実行グラフとサブエージェントのリアルタイム可視化

DeepAgents のような自律エージェントは、1リクエストの中で複数回のツール呼び出しやサブエージェント委譲を行うため、`.invoke()` で最終結果だけを待つとユーザー体験（UX）が大きく損なわれます。DeepAgents は LangGraph のストリーミング基盤を拡張し、**メインエージェントと複数サブエージェントのイベントを統一ストリームで購読**できます。

### 2.1 3つの `stream_mode` と使い分け

| `stream_mode` | 流れるデータ | 主な用途 |
| --- | --- | --- |
| `"updates"` | 各グラフノード実行完了ごとの状態差分辞書 `{node_name: state_delta}` | どの内部ノード・Middleware が通過したかの追跡、ツール完了検知、**`__interrupt__` 検知** |
| `"messages"` | `(AIMessageChunk / ToolMessage, metadata)` タプル | LLM トークンの逐次描画、ツール引数のストリーミング表示 |
| `"custom"` | ツール内で `get_stream_writer()` により送信した任意辞書 | 「分析進捗 50%」などのドメイン固有プログレスバー表示 |

> `stream_mode=["updates"]` でストリームを購読すると、表面上の `model` や `tools` だけでなく、DeepAgents ハーネスが内部に組み込んだ **`PatchToolCallsMiddleware.before_agent`** や **`_ParentNotifierMiddleware.after_agent`** といったライフサイクル Middleware ノードの遷移までそのまま観測できる。

### 2.2 サブグラフストリーミングと Namespace 設計

`subgraphs=True` を指定すると、各チャンクに発生源を示すタプル **Namespace（`ns`）** が付与されます。

- `()`（空タプル）：メインエージェントのイベント
- `("tools:abc123",)`：メインエージェントの `task` ツール呼び出しから起動されたサブエージェントのイベント
- `("tools:abc123", "model_request:def456")`：サブエージェント内部の特定ノード

また、`subgraphs` の有無と `stream_mode` の単一／複数指定によって、返却されるチャンクのタプル形状が変わる点に注意が必要です。

| 設定 | チャンク形状 |
| --- | --- |
| `subgraphs=False`, 単一モード（例：`"updates"`） | `data` |
| `subgraphs=False`, 複数モード（例：`["updates", "messages"]`） | `(mode, data)` |
| `subgraphs=True`, 単一モード | `(namespace, data)` |
| `subgraphs=True`, 複数モード | `(namespace, mode, data)` |

LangGraph 1.1 以降では `version="v2"` を指定することで `{"type": ..., "ns": ..., "data": ...}` という統一辞書形式（`StreamPart`）で受け取ることもできます。

---

## 3. 本番永続化・マルチテナント・耐障害設計

[Going to Production](https://docs.langchain.com/oss/python/deepagents/going-to-production) ガイドでは、本番運用で必須となる3つのプリミティブ（**Thread**, **User**, **Assistant**）と運用パラメータの設計を定義しています。

### 3.1 `thread_id` と `context` の厳密な分離

本番環境でのすべての呼び出しは、2つの独立した実行時パラメータを明示的に分離して渡すべきです。

- **`thread_id`**（`config={"configurable": {"thread_id": ...}}`）：会話単位の識別子。Checkpointer がメッセージ履歴やスクラッチファイル（`StateBackend`）、割り込み状態を永続化・復元するためのキー。
- **`context`**（`context_schema` で型定義し、`runtime.context` で参照）：1回の実行ごとに渡すリクエスト固有データ（`user_id`, `org_id`, 機能フラグ、セッションメタデータ等）。

> Checkpointer はステップごとにグラフ状態をスナップショット保存するため、同一の `thread_id` を渡すだけでマルチターンの会話記憶が自動的に復元されるだけでなく、ワーカークラッシュ時の途中再開（Durable Execution）やタイムトラベルによる巻き戻しデバッグも可能になる。

```python
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

# 実行時コンテキストのスキーマ定義
@dataclass
class RunContext:
    user_id: str
    org_id: str

# 環境変数から LLM を初期化（API キーのハードコード禁止）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

# 本番では PostgresSaver 等の永続 Checkpointer を使用する
agent = create_deep_agent(
    model=llm,
    context_schema=RunContext,
    checkpointer=InMemorySaver(),
)

# thread_id で会話の連続性を維持し、context でテナント情報を渡す
config = {"configurable": {"thread_id": "prod-thread-001"}}
agent.invoke(
    {"messages": [{"role": "user", "content": "今月のレポート構成案を作成してください"}]},
    config=config,
    context=RunContext(user_id="user-123", org_id="org-acme"),
)
```

### 3.2 長期記憶（Store）のスコーピングと PII ガードレール

- **Store Namespace 設計**：`CompositeBackend` で `/memories/` を `StoreBackend` にルーティングする際、誰と記憶を共有するかに応じて Namespace を切ります。
  - **User スコープ（推奨デフォルト）**：`(assistant_id, user_id)` または `(user_id,)` — ユーザー個人の好み・履歴
  - **Assistant スコープ**：`(assistant_id,)` — 特定アシスタント全ユーザー共通の運用知識
  - **Global / Org スコープ**：`(org_id,)` — 組織ポリシー。**プロンプトインジェクションによる汚染を防ぐため、必ず `FilesystemPermission(operations=["write"], paths=["/policies/**"], mode="deny")` で読み取り専用にロック**します。
- **Async 徹底**：LLM 呼び出し・DB・Sandbox 作成・MCP 接続はすべて I/O バウンドであるため、本番では `ainvoke`, `astream`, `abefore_agent`, async ツール、および async Graph Factory を採用します。
- **PII 保護（`PIIMiddleware`）**：メールアドレスやクレジットカード番号がモデルやログへ流出するのを防ぐため、`PIIMiddleware("email", strategy="redact", apply_to_input=True)` や `strategy="mask" / "hash" / "block"` を Middleware スタックへ差し込みます。

---

## 4. Grading Rubrics（`RubricMiddleware`）：実行時 LLM-as-a-Judge 自己改善ループ

`deepagents>=0.6.5` で追加された `RubricMiddleware` は、オフライン評価で使われる **LLM-as-a-Judge** を**エージェントの実行時ループ**に組み込む機能です。「全テストが通るコード」「必須セクションを網羅した報告書」のように、1回の推論では満たしにくい明確な完了条件（Rubric）を宣言し、採点者モデル（Grader）が合格を出すまで自律的に修正を繰り返させます。

{% mermaid %}
flowchart TD
    IN["ユーザー入力 + rubric（評価基準）"] --> WORKER["メイン DeepAgent<br/>推論・ツール実行・回答生成"]
    WORKER -->|"回答完了"| GRADER["RubricMiddleware<br/>LLM-as-a-Judge Grader<br/>(任意でテスト実行ツール等を使用)"]
    GRADER --> VERDICT{"Verdict 判定"}
    VERDICT -->|"satisfied"| OK["正常終了<br/>(全基準クリア)"]
    VERDICT -->|"needs_revision<br/>かつ iteration < max_iterations"| FB["不合格基準の gap フィードバックを<br/>HumanMessage (source='rubric_grader') で注入"]
    FB --> WORKER
    VERDICT -->|"max_iterations 到達"| CAP["終了: max_iterations_reached"]
    VERDICT -->|"failed / grader_error"| ERR["終了: 評価不能 / Grader 例外"]

    style GRADER fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style OK fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style CAP fill:#fce4ec,stroke:#c2185b,color:#880e4f
    style ERR fill:#ffebee,stroke:#c62828,color:#b71c1c
{% endmermaid %}

### 4.1 5つの Verdict ステータスと設計ポイント

| ステータス | 意味 | ループ継続 |
| --- | --- | --- |
| `satisfied` | ルーブリックの全基準を通過 | 終了 |
| `needs_revision` | 1つ以上の基準が不合格。各基準の `gap` フィードバックを注入して再実行 | **継続** |
| `max_iterations_reached` | 修正が必要だが `max_iterations` 上限に到達 | 終了 |
| `failed` | ルーブリック自体が不備、または履歴から評価不能 | 終了 |
| `grader_error` | Grader モデルのタイムアウトや例外発生 | 終了 |

実運用では以下の3点が強力な武器になります。
1. **軽量モデルによるコスト最適化**：作業用エージェントには高性能モデルを割り当て、`RubricMiddleware(model=...)` には軽量・低コストな採点モデルを割り当てられる。
2. **Evidence-Based Grading（ツール付き採点）**：`RubricMiddleware(tools=[run_test_suite])` のように検証用ツールを渡すと、Grader が文章の推測だけでなく実際にテストスイートを走らせた証拠に基づいて合否判定できる。
3. **進捗の観測**：`on_evaluation` コールバック、または `stream_events(..., version="v3", transformers=[CustomTransformer])` の `rubric_evaluation_start` / `rubric_evaluation_end` イベントで各イテレーションの採点結果をリアルタイム監視できる。

---

## 5. エコシステム連携プロトコル：ACP・AG-UI・A2A・MCP の棲み分け

DeepAgents は単体で閉じるのではなく、4つのオープンプロトコルによって開発環境・フロントエンド・外部ツール・他エージェントと接続できます。

| プロトコル | 接続先 | 通信方式 | DeepAgents での利用手段 |
| --- | --- | --- | --- |
| **MCP** (Model Context Protocol) | 外部ツール・データソース | stdio / Streamable HTTP | エージェントが外部サーバー上のツールを呼び出す |
| **ACP** (Agent Client Protocol) | コードエディタ・IDE（Zed, JetBrains, VS Code, Neovim） | stdio（JSON-RPC） | `deepagents-acp` の `AgentServerACP(agent)` で公開 |
| **AG-UI** (Agent User Interaction) | Web / モバイルフロントエンド（CopilotKit 等） | HTTP / SSE 型付きイベントストリーム | `langgraph dev` / Agent Server + `@ag-ui/langgraph` |
| **A2A** (Agent2Agent) | 他の自律エージェント（Google ADK や別 LangGraph） | JSON-RPC v1.0 / v0.3 + Agent Card | Agent Server の `/a2a/{assistant_id}` エンドポイント |

### 5.1 ACP（エディタ連携）
`pip install deepagents-acp` を導入し、`AgentServerACP(agent)` を `acp.run_agent(server)` で起動するだけで、自作の DeepAgent を Zed や JetBrains IDE のネイティブエージェントパネルから直接利用できます。エディタがプロジェクトのコンテキストを渡し、DeepAgent の進捗をリッチに描画します。

### 5.2 AG-UI（フロントエンド連携）
AG-UI は LangGraph の内部構造をフロントエンドから隠蔽し、`RUN_STARTED`, `TEXT_MESSAGE_CONTENT`, `TOOL_CALL_START`, `STATE_SNAPSHOT`, `STATE_DELTA`（RFC 6902 JSON Patch）といった型付きイベントへ変換します。`emitInterruptOutcome: true` を有効にすれば、Part 6 で解説した HITL 割り込みも標準の `resume` フィールドで UI から再開可能です。

### 5.3 A2A（エージェント間連携と分散トレース）
LangSmith Agent Server は各アシスタントに対して自動的に `GET /.well-known/agent-card.json?assistant_id={assistant_id}`（Agent Card）と `/a2a/{assistant_id}` エンドポイントを公開します。
- **会話継続と分散トレースの鍵**：A2A の `contextId`（UUID）は Agent Server 側で自動的に LangGraph の `thread_id` にマッピングされます。複数のフレームワーク（例：Google ADK と DeepAgents）が同じ `contextId` を引き回すだけで、LangSmith UI 上の1つの Thread に全エージェントのトレースが統合されます。
- **A2A 経由の割り込み**：グラフが `interrupt()` を呼ぶと、A2A タスクは `TASK_STATE_INPUT_REQUIRED` 状態となり、`Interrupt` アーティファクトを返却します。呼び出し側は同じ `taskId` と `contextId` に `{"data": {"resume": ...}}` を送って再開できます。

---

## 6. まとめ

- **Streaming** は `updates` / `messages` / `custom` の3モードと `subgraphs=True` による Namespace 識別で、複雑なマルチエージェント実行を透明化する。
- **本番永続化**は `thread_id`（会話状態）と `context`（リクエスト属性）を分離し、Checkpointer による Durable Execution と Store Namespace によるテナント隔離を徹底する。
- **`RubricMiddleware`** は実行時 LLM-as-a-Judge によって、完了条件を満たすまで自律修正する品質保証ループを実現する。
- **MCP / ACP / AG-UI / A2A** の4プロトコルにより、ツール・IDE・Web UI・外部エージェントのすべてと疎結合に接続できる。

次回の最終記事（Part 7 実践編）では、`stream_mode=["updates"]` を用いて DeepAgents の内部 Middleware ノード遷移（`PatchToolCallsMiddleware.before_agent` → `model` → `_ParentNotifierMiddleware.after_agent`）を実測し、`InMemorySaver` と `thread_id` によるマルチターン記憶復元を検証します。
