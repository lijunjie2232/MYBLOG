---
title: "LangChain v1 Part 4 理論編：Short-term Memory（Checkpointer / Thread）と Long-term Memory（Store / Namespace）の二層記憶設計"
date: 2026-05-13 05:45:28
categories: [AI]
tags: [LangChain, LangGraph, AI Agent, Memory, Checkpointer, Store, Python]
lang: ja
description: "LangChain v1 におけるエージェント記憶の二層アーキテクチャを徹底解説。Thread スコープの Short-term Memory（Checkpointer / AgentState）と、セッション横断の Long-term Memory（Store / Namespace / Semantic Search）の設計指針とコンテキスト制御パターンを学びます。"
---

# LangChain v1 Part 4 理論編：Short-term Memory と Long-term Memory の二層記憶設計

AI エージェントが単発の Q&A を超えて、継続的なタスク遂行やユーザーごとのパーソナライズを行うためには、**「記憶（Memory）」のアーキテクチャ設計**が不可欠です。しかし、すべての履歴を無制限にプロンプトへ詰め込むと、コンテキストウィンドウの枯渇、推論レイテンシの増大、コスト悪化、さらには古い無関係な情報による精度劣化（Context Distraction）を引き起こします。

LangChain v1（およびその実行基盤である LangGraph）では、この課題を解決するために記憶を以下の **二層（Two-Tier）モデル** に明確に分離しています。

1. **Short-term Memory（短期記憶）**: 単一のスレッド（会話セッション）内での状態を管理する **Checkpointer × Thread** 層
2. **Long-term Memory（長期記憶）**: スレッドやセッションを横断してユーザー知識・嗜好・手順を永続化する **Store × Namespace** 層

本稿（理論編）では、これら二層の記憶が LangChain v1 の `create_agent` 内部でどのように協調し、コンテキスト管理や本番永続化（PostgreSQL 等）へどう接続されるのかを体系的に解説します。

---

## 1. なぜ「二層記憶」が必要なのか：スコープとライフサイクルの違い

人間の認知モデルにおいて「作業記憶（ワーキングメモリ）」と「長期記憶」が分離されているのと同様に、本番品質の AI エージェントでも記憶のスコープを分ける必要があります。

{% mermaid %}
flowchart TB
    subgraph Client["クライアント / アプリケーション層"]
        U1["Thread A (今日のチャット)"]
        U2["Thread B (翌日の別セッション)"]
    end

    subgraph STM["Short-term Memory 層 (Thread スコープ)"]
        CP["Checkpointer<br/>(InMemorySaver / PostgresSaver)"]
        S1["State Snapshot: thread-A<br/>messages: [Human, AI, Tool, ...]"]
        S2["State Snapshot: thread-B<br/>messages: [Human, AI, ...]"]
        CP --> S1
        CP --> S2
    end

    subgraph LTM["Long-term Memory 層 (Cross-Thread スコープ)"]
        ST["BaseStore<br/>(InMemoryStore / PostgresStore)"]
        NS["Namespace: ('users', 'alice')"]
        DOC["Key: 'profile'<br/>JSON Document + Vector Index"]
        ST --> NS --> DOC
    end

    U1 -->|"thread_id='A'"| CP
    U2 -->|"thread_id='B'"| CP
    U1 -.->|"ToolRuntime.store で参照/更新"| ST
    U2 -.->|"ToolRuntime.store で参照/更新"| ST

    style Client fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style STM fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style LTM fill:#fef3c7,stroke:#d97706,color:#78350f
    style CP fill:#bae6fd,stroke:#0369a1,color:#0c4a6e
    style ST fill:#fde68a,stroke:#b45309,color:#78350f
{% endmermaid %}

両者の責務と特性を比較すると、以下の通り明確な境界が浮かび上がります。

| 比較軸 | Short-term Memory（短期記憶） | Long-term Memory（長期記憶） |
| :--- | :--- | :--- |
| **主要コンポーネント** | `Checkpointer`（`InMemorySaver`, `PostgresSaver` 等） | `Store`（`InMemoryStore`, `PostgresStore` 等） |
| **識別子（スコープ）** | `configurable.thread_id`（単一会話スレッド） | `namespace: tuple[str, ...]` + `key: str` |
| **保存されるデータ** | グラフ状態全体（`AgentState`：`messages` やカスタム状態） | 任意の JSON ドキュメント（プロフィール、嗜好、エピソード等） |
| **読み書きのタイミング** | エージェントの**各ステップ（Superstep）開始・完了時に自動** | ツール内（`ToolRuntime.store`）や Middleware から**明示的に操作** |
| **検索能力** | `thread_id` / `checkpoint_id` による時系列復元 | キー完全一致（`get`）＋メタデータフィルタ＆ベクトル類似検索（`search`） |
| **主な用途** | 会話の文脈維持、ツール実行の中断・再開（HITL）、Time-travel | ユーザー横断のパーソナライズ、長期的な事実・手順の蓄積 |

> **設計原則**：短期記憶（Checkpointer）は「いま進行中の会話と実行状態の忠実なスナップショット」を担い、長期記憶（Store）は「セッションが終わっても残すべき構造化・検索可能な知識」を担います。この二つを混同しないことが、スケーラブルなエージェント設計の第一歩です。

---

## 2. Short-term Memory の仕組み：Checkpointer・Thread・AgentState

### 2.1 Checkpointer と Thread による自動永続化

LangChain v1 の `create_agent` は内部的に LangGraph のステートマシンとしてコンパイルされます。`checkpointer` を渡してエージェントを構築すると、**各ステップ（モデル呼び出し、ツール実行など）のたびに `AgentState` がチェックポイントとして保存**されます。

呼び出し側が `config={"configurable": {"thread_id": "..."}}` を指定することで、同じ `thread_id` の最新スナップショットが自動的にロードされ、新しい入力メッセージが `add_messages` リデューサーによって既存履歴へ安全に追記されます。

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# 環境変数からエンドポイントと API キーを読み込む（ハードコード禁止）
load_dotenv()

llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

# 開発・検証用のインメモリ Checkpointer を生成
checkpointer = InMemorySaver()

# create_agent に checkpointer を渡すだけで Thread 単位の短期記憶が有効化される
agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=checkpointer,
    system_prompt="簡潔に日本語で回答してください。",
)

# 同じ thread_id を使うことで 2 ターン目が 1 ターン目の記憶を引き継ぐ
thread_config = {"configurable": {"thread_id": "session-001"}}
agent.invoke({"messages": [{"role": "user", "content": "私の名前は Alice です。"}]}, config=thread_config)
res = agent.invoke({"messages": [{"role": "user", "content": "私の名前を覚えていますか？"}]}, config=thread_config)
```

### 2.2 本番環境での永続化：`PostgresSaver`

`InMemorySaver` はプロセス内のメモリ辞書に状態を保持するため、サーバー再起動やマルチワーカー環境では状態が失われます。本番環境では `langgraph-checkpoint-postgres` の `PostgresSaver`（あるいは SQLite / Cosmos DB 等の永続 Checkpointer）へ差し替えます。

```python
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver

# 本番環境では DB 接続文字列も環境変数から注入する
DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable")

llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

# コンテキストマネージャ経由で接続プールを管理し、初回のみ setup() でテーブルを作成する
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=checkpointer,
    )
```

### 2.3 `AgentState` の拡張とツールからの読み書き

デフォルトの `AgentState` は `messages` キーのみを持ちますが、サブクラス化することでセッション固有のカスタム状態（例：`user_id`、`preferences`、進行中タスクのフラグ等）を同じ Checkpointer に乗せて永続化できます。

さらに、ツール関数は `ToolRuntime` 引数を受け取ることで、**LLM のツールスキーマには一切露出させずに** 短期記憶（`runtime.state`）や設定（`runtime.config`）、型付きコンテキスト（`runtime.context`）へアクセスできます。ツールから短期記憶を更新したい場合は `langgraph.types.Command(update={...})` を返します。

```python
from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

# AgentState を継承して Thread 内で共有したいカスタムフィールドを追加
class CustomAgentState(AgentState):
    user_name: str

@tool
def cache_user_name(name: str, runtime: ToolRuntime) -> Command:
    """ユーザー名を短期記憶（AgentState）に保存する。"""
    # Command(update=...) を使うと、ToolMessage の追加と同時に任意の状態キーを更新できる
    return Command(
        update={
            "user_name": name,
            "messages": [
                ToolMessage(
                    content=f"短期記憶に user_name={name} を保存しました。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

---

## 3. 短期記憶の肥大化対策：Trim / Delete / Summarize

会話が長くなると `messages` リストが膨張し、トークン上限超過や推論精度低下を招きます。LangChain v1 では主に以下の 3 つのコンテキスト制御パターンを組み合わせて対処します。

1. **Trim messages（切り詰め）**: `@before_model` フック等で先頭や直近 N 件のみを残す。実装が軽量だが、切り捨てた文脈は失われる。
2. **Delete messages（明示的削除）**: `RemoveMessage(id=m.id)` や `RemoveMessage(id=REMOVE_ALL_MESSAGES)` を返し、LangGraph の `add_messages` リデューサー経由で状態から永続削除する。
   - **重要制約**: 削除後のメッセージ列は LLM プロバイダの仕様を満たす必要があります（例：多くのプロバイダは `tool_calls` を含む `AIMessage` の直後に対応する `ToolMessage` が必須）。
3. **Summarize messages（自動要約）**: 組み込みの `SummarizationMiddleware` を使い、トークン数やメッセージ数が閾値（`trigger`）を超えた時点で古い履歴を軽量モデルで要約し、直近の `keep` 件だけを原文で保持する。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# トークン数が 4000 を超えたら古い履歴を自動要約し、直近 20 メッセージは原文のまま維持する
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=summary_llm,          # 要約専用の軽量・低コストモデルを指定可能
            trigger=("tokens", 4000),   # 発火条件（tokens / messages / fraction）
            keep=("messages", 20),      # 原文で残す直近コンテキスト量
        )
    ],
    checkpointer=InMemorySaver(),
)
```

---

## 4. Long-term Memory の仕組み：Store・Namespace・Semantic Search

### 4.1 Namespace による階層的な知識整理

Short-term Memory が `thread_id` に閉じているのに対し、Long-term Memory は **LangGraph の `BaseStore` インターフェース**（開発用 `InMemoryStore`、本番用 `PostgresStore` 等）上に構築されます。

データは「フォルダに相当する **Namespace（文字列のタプル）**」と「ファイル名に相当する **Key（文字列）**」の組で、JSON ドキュメント（`dict`）として格納されます。

```text
Namespace: ("users", "alice")          ── Key: "profile"      ──> {"name": "Alice", "role": "ML Engineer", ...}
Namespace: ("users", "alice", "prefs") ── Key: "code_style"   ──> {"language": "Python", "concise": True}
Namespace: ("orgs", "acme", "runbook") ── Key: "db_failover"  ──> {"steps": [...]}
```

このように Namespace をタプルで階層化することで、マルチテナント環境（組織 ID × ユーザー ID × 用途）における厳密なデータ分離と、上位プレフィックス指定による横断検索を両立できます。

### 4.2 セマンティック検索（`IndexConfig`）とフィルタリング

Store は単なる Key-Value ストアではなく、初期化時に `IndexConfig(embed=..., dims=...)` を渡すことで **ベクトル類似検索（Semantic Search）** と **JSON フィールドの等価フィルタ** を併用できます。

```python
from collections.abc import Sequence
from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore

def dummy_embed(texts: Sequence[str]) -> list[list[float]]:
    """本番では OpenAIEmbeddings 等の実 Embedding モデルに差し替える。"""
    return [[0.1, 0.2] for _ in texts]

# ベクトルインデックス付きで Store を初期化
store = InMemoryStore(index=IndexConfig(embed=dummy_embed, dims=2))

namespace = ("users", "alice")

# put() で JSON ドキュメントを保存（自動的にベクトル化されてインデックス登録される）
store.put(
    namespace,
    "preferences",
    {
        "rules": ["簡潔な日本語で答える", "Python と LangGraph を好む"],
        "category": "style",
    },
)

# 1. キー完全一致での高速取得
item = store.get(namespace, "preferences")

# 2. メタデータフィルタ + ベクトル類似度での意味検索
matched_items = store.search(
    namespace,
    filter={"category": "style"},
    query="language and framework preferences",
)
```

---

## 5. 二層記憶の連携フロー：エージェント実行時のデータパス

実際のエージェント実行時に、短期記憶（Checkpointer）と長期記憶（Store）がどのように協調するかを図示します。

{% mermaid %}
sequenceDiagram
    autonumber
    participant User as ユーザー
    participant Agent as create_agent (LangGraph)
    participant CP as Checkpointer (短期記憶)
    participant LLM as ChatModel
    participant Tool as lookup_profile (ToolRuntime)
    participant Store as BaseStore (長期記憶)

    User->>Agent: invoke(Turn 1, config={thread_id, user_id})
    Agent->>CP: thread_id の最新 State をロード
    CP-->>Agent: messages=[] (初回)
    Agent->>LLM: SystemPrompt + HumanMessage
    LLM-->>Agent: AIMessage (tool_calls: lookup_profile)
    Agent->>Tool: 実行 (runtime.store & runtime.config を自動注入)
    Tool->>Store: get(("users", user_id), "profile")
    Store-->>Tool: Item(value={name, role, favorite_stack})
    Tool-->>Agent: ToolMessage (JSON文字列)
    Agent->>LLM: 全メッセージ履歴を送信
    LLM-->>Agent: AIMessage (Turn 1 最終回答)
    Agent->>CP: 更新済み State (4メッセージ) を保存
    Agent-->>User: Turn 1 応答

    User->>Agent: invoke(Turn 2, 同じ thread_id)
    Agent->>CP: thread_id の最新 State をロード (4メッセージ復元)
    Agent->>LLM: 既存4メッセージ + Turn 2 HumanMessage
    LLM-->>Agent: AIMessage (ツール再呼出なしで即答)
    Agent->>CP: 更新済み State (6メッセージ) を保存
    Agent-->>User: Turn 2 応答
{% endmermaid %}

> **キャッシュ効果としての二層連携**：一度長期記憶（Store）からツール経由で読み出された情報は、そのスレッドの短期記憶（`messages` 内の `ToolMessage` / `AIMessage`）へ取り込まれます。その結果、同一スレッドの後続ターンでは、わざわざ Store へ再問い合わせしなくても LLM が短期記憶から文脈を即座に参照できます。一方で、別スレッド（新しい `thread_id`）が開始されたときは短期記憶が空になるため、再び Store から長期記憶をロードすることで「セッションを越えた継続性」が実現されます。

---

## 6. 記憶設計のベストプラクティスまとめ

- **役割分担の徹底**: 会話ログ・途中経過は `Checkpointer`、ユーザー属性・学習済み知識は `Store` に分離する。
- **識別子の分離**: `thread_id`（会話セッション ID）と `user_id`（ユーザー ID）を明確に分け、`config["configurable"]` または型付き `context_schema` 経由で注入する。LLM に `user_id` を引数として推測させてはならない（権限越境リスク）。
- **本番移行の容易さ**: LangChain v1 / LangGraph の設計上、`InMemorySaver` → `PostgresSaver`、`InMemoryStore` → `PostgresStore` の差し替えは `create_agent` の引数変更だけで完結する。
- **長期会話の防御**: `Checkpointer` を有効にしたら、必ず `SummarizationMiddleware` や `@before_model` でのトリミング戦略をセットで検討する。

次回の **Part 4 実践編** では、`InMemorySaver` と `InMemoryStore` を実際に `create_agent` に統合し、2 ターン会話で二層記憶がどのように機能するかを実測ログとともに検証します。

### 参考リンク
- [LangChain Official Docs — Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangChain Official Docs — Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/factory.py)
