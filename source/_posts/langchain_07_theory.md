---
title: "LangChain v1 Part 7 理論編：Multi-Agent パターン比較（Subagents / Handoffs / Router / Skills）と MCP 統合アーキテクチャ"
date: 2026-05-21 14:45:42
categories: [AI]
tags: [LangChain, LLM, AI Agent, Multi-Agent, MCP, Architecture, Python]
lang: ja
description: "LangChain v1 における 4 つの主要 Multi-Agent パターン（Subagents / Handoffs / Router / Skills）の構造・モデル呼び出し回数・トークン効率を比較し、MCPAdapter を用いた Model Context Protocol (MCP) 統合アーキテクチャを理論面から徹底解説します。"
---

単一の AI エージェントにツールやドメイン知識を詰め込みすぎると、ある臨界点を超えた瞬間からツール選択の誤爆、コンテキストウィンドウの圧迫、プロンプト保守性の崩壊が急速に進行します。本稿（LangChain v1 シリーズ Part 7 理論編）では、この限界を突破するための **Multi-Agent アーキテクチャ 4 パターンの定量比較** と、外部ツールエコシステムを標準化する **MCP（Model Context Protocol）統合アーキテクチャ** を解説します。

---

## 1. なぜ Multi-Agent が必要なのか（そしていつ不要か）

まず大前提として、**すべての複雑なタスクに Multi-Agent が必要なわけではありません。** Part 6 で学んだ Context Engineering（動的プロンプトや動的ツールフィルタリング）を備えた単一エージェントで十分に解けるケースは多く、不要なマルチエージェント化はレイテンシとコストを悪化させます。

Multi-Agent が真価を発揮するのは、次の 3 つの課題に直面したときです。

1. **コンテキスト管理（Context Management）**：複数の専門領域ごとに数千トークンの手順書や多数のツールが存在し、単一プロンプトへ詰め込むと推論精度が低下する。
2. **分散開発（Distributed Development）**：SRE チーム、決済チーム、CS チームなど複数チームがそれぞれの専門エージェントを独立して開発・デプロイし、明確な境界で合成したい。
3. **並列化（Parallelization）**：複数のドメイン調査やサブタスクを同時に実行し、全体の応答時間（Wall-clock latency）を短縮したい。

---

## 2. Multi-Agent 4 大パターンの構造と特性比較

LangChain v1 は、代表的なマルチエージェント設計を **Subagents / Handoffs / Skills / Router** の 4 パターン（＋ LangGraph による **Custom workflow**）に整理しています。

{% mermaid %}
flowchart TB
    subgraph P1["1. Subagents (中央集権・コンテキスト分離)"]
        U1["User"] <--> SUP["Supervisor<br/>(統括エージェント)"]
        SUP <-->|"Tool Call"| SA1["Subagent A"]
        SUP <-->|"Tool Call"| SA2["Subagent B"]
    end

    subgraph P2["2. Handoffs (状態遷移・制御移譲)"]
        U2["User"] <--> A1["Agent A"]
        A1 -->|"Handoff Tool"| A2["Agent B"]
        A2 <--> U2
    end

    subgraph P3["3. Skills (単一エージェント・動的知識ロード)"]
        U3["User"] <--> AG["Single Agent"]
        AG <-->|"On-demand Load"| SK["Skill Prompts / Docs"]
    end

    subgraph P4["4. Router (分類ディスパッチ・結果統合)"]
        U4["User"] --> RT["Router<br/>(分類ステップ)"]
        RT --> RA["Agent A"]
        RT --> RB["Agent B"]
        RA --> SYN["Synthesizer"]
        RB --> SYN
        SYN --> U4
    end

    style SUP fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style A1 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style A2 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style AG fill:#fff8e1,stroke:#f57f17,color:#e65100
    style RT fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

### 2.1 各パターンの特徴と選定マトリクス

| パターン | 仕組み | 分散開発 | 並列実行 | マルチホップ | ユーザー直接対話 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Subagents** | 統括エージェント（Supervisor）が専門サブエージェントをツールとして呼び出す | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Handoffs** | ツール呼び出しで状態変数を更新し、アクティブなエージェントや設定を動的に切り替える | - | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Skills** | 単一エージェントが制御を保ったまま、必要な専門プロンプト／知識をオンデマンドで読み込む | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Router** | 分類ステップが入力を 1 つ以上の専門エージェントへ振り分け、結果を統合する | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | - | ⭐⭐⭐ |

> Subagents パターンでは、中央の統括エージェント（Supervisor）が専門サブエージェントを通常のツールとして呼び出します。サブエージェントはデフォルトでステートレス（Stateless）であり、独立したクリーンなコンテキストウィンドウで動作するため、メイン会話のコンテキスト肥大化（Context Bloat）を強力に防ぎます。

### 2.2 パフォーマンス定量比較（モデル呼び出し回数とトークン消費）

設計選定で最も重要なのは、**ワークロード別の LLM 呼び出し回数と総トークン消費量**のトレードオフです。LangChain 公式ベンチマークの 3 シナリオを比較します。

| パターン | One-shot（単発依頼） | Repeat request（同一依頼の 2 ターン連続） | Multi-domain（3 領域 × 2K トークン比較） |
| :--- | :--- | :--- | :--- |
| **Subagents** | 4 calls | 8 calls（4 + 4） | **5 calls / ~9K tokens** ✅ |
| **Handoffs** | **3 calls** ✅ | **5 calls（3 + 2）** ✅ | 7+ calls / ~14K+ tokens |
| **Skills** | **3 calls** ✅ | **5 calls（3 + 2）** ✅ | 3 calls / ~15K tokens |
| **Router** | **3 calls** ✅ | 6 calls（3 + 3） | **5 calls / ~9K tokens** ✅ |

この比較表から、実務で極めて重要な 3 つの設計指針が導き出されます。

1. **One-shot のオーバーヘッドと価値**
   > One-shot リクエストにおいて Subagents パターンは 4 回のモデル呼び出し（Supervisor の委譲判断 → Subagent のツール呼び出し → Subagent の要約生成 → Supervisor の最終回答）を要します。Handoffs や Skills（3 回）より 1 回多いですが、このオーバーヘッドこそが『中央集権的な制御と厳格なコンテキスト分離』の対価です。
2. **連続ターン（Repeat request）ではステートフル設計が強い**
   - **Handoffs** は 1 ターン目で切り替わった専門エージェントがアクティブ状態を維持し、**Skills** は会話履歴にロード済みスキルが残るため、2 ターン目は委譲／ロードを省略して 2 calls で完結します（呼び出し回数 40〜50% 削減）。
3. **複数ドメイン横断（Multi-domain）ではコンテキスト分離＋並列実行が圧倒的**
   - 3 領域（各 2,000 トークン）を同時に調査する場合、**Skills** はすべてのドキュメント（6K トークン）が単一履歴に蓄積されるため、以降の全推論でコンテキストが膨張し計 **~15K トークン** を消費します。一方、**Subagents** と **Router** は各専門エージェントが独立したコンテキストで並列調査して要約だけを返すため、**~9K トークン（約 40%〜67% 削減）** に抑えられます。**Handoffs** は本質的に逐次遷移（Sequential）であるため、並列調査には不向きです。

---

## 3. Subagents パターンの詳細設計判断

実務のエンタープライズ用途（SRE、データ分析、社内業務オーケストレーション）で最も採用率が高いのが **Subagents パターン** です。設計時には以下の 2 つの軸を明確に決定します。

### 3.1 Tool per agent vs Single dispatch tool

> サブエージェントのツール化には、専門エージェントごとに個別の `@tool` を定義する **Tool per agent** と、単一のディスパッチツールで名前指定する **Single dispatch tool** の 2 パターンがあります。入力スキーマや出力整形を細かく制御したい場合は Tool per agent が最適です。

| 軸 | Tool per agent | Single dispatch tool（`task` ツール） |
| :--- | :--- | :--- |
| **構成** | サブエージェントごとに専用の `@tool` 関数を用意 | 1 つの汎用ツール `task(agent_name, description)` でレジストリから起動 |
| **長所** | 引数スキーマ・コンテキスト受け渡し・出力加工を個別に厳密制御できる | エージェント数が 10 個以上でも統括側のツール定義が肥大化しない。チーム分散開発に強い |
| **短所** | エージェント増加に伴いツール定義が増える | 入出力が汎用文字列（規約ベース）になり、個別最適化しにくい |

### 3.2 同期（Sync）vs 非同期（Async / バックグラウンド）

- **同期実行（デフォルト）**：Supervisor がサブエージェントの完了を待機します。データ取得 → 分析 → 回答のように結果が次の推論に必須な場合に適しています。
- **非同期実行（Three-tool pattern）**：長時間かかる調査ジョブを別プロセスで起動し、ユーザーとの対話をブロックしない設計です。`start_job`（ジョブ ID 返却）、`check_status`、`get_result` の 3 ツールに分割して運用します。

### 3.3 Single dispatch tool（`task` ツール）の実装パターンとコンテキスト設計

エージェント数が 5〜10 を超える大規模システムや複数チームでの分散開発では、個別のツールを増やす代わりに **レジストリ＋単一ディスパッチツール** を採用します。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# 共通の LLM クライアント（環境変数から接続設定を読み込む）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

# 各チームが独立して保守する専門サブエージェント
metrics_agent = create_agent(
    model=llm,
    tools=[],  # 実際は Prometheus / Datadog ツール等を登録
    system_prompt="あなたはメトリクス分析の専門家です。",
)
incident_agent = create_agent(
    model=llm,
    tools=[],  # 実際は PagerDuty / Jira ツール等を登録
    system_prompt="あなたはインシデント履歴調査の専門家です。",
)

# 名前で引けるサブエージェントレジストリ
SUBAGENT_REGISTRY = {
    "metrics": metrics_agent,
    "incident": incident_agent,
}

# 単一の汎用ディスパッチツール（規約ベースで任意のサブエージェントを起動）
@tool
def dispatch_task(agent_name: str, task_description: str) -> str:
    """指定した専門サブエージェントに独立したコンテキストでタスクを委譲する。

    利用可能な agent_name:
    - metrics: サービスメトリクス（RPS / エラー率 / レイテンシ）の調査
    - incident: 過去の障害チケットやポストモーテムの検索
    """
    if agent_name not in SUBAGENT_REGISTRY:
        return f"Error: Unknown agent '{agent_name}'. Choose from {list(SUBAGENT_REGISTRY.keys())}."
    res = SUBAGENT_REGISTRY[agent_name].invoke(
        {"messages": [{"role": "user", "content": task_description}]}
    )
    # 途中の推論やツール結果は破棄し、最終回答のみ上位へ返却（コンテキスト分離）
    return str(res["messages"][-1].content)
```

さらに、Subagents における **Context Engineering** では以下の 3 層の制御が品質を左右します。

1. **Subagent Specs（発見性）**：ツール名と Docstring、またはシステムプロンプト列挙・Enum 制約・動的ディスカバリツール（Progressive Disclosure）により、Supervisor が正しい委譲先を迷わず選べるようにする。
2. **Subagent Inputs（入力設計）**：原則は `query` のみを渡してクリーンに保つが、ユーザー権限や会話の前提が必要な場合は `ToolRuntime` 経由で必要最小限の State / Runtime Context だけを明示的に注入する。
3. **Subagent Outputs（出力設計）**：デフォルトは最終メッセージの要約のみを返す。構造化データやソース URL などを上位の状態にも残したい場合は、`Command(update=...)` で明示的に親 State へ書き戻す設計にする。

---

## 4. MCP（Model Context Protocol）統合アーキテクチャ

マルチエージェントや専門ツールをスケールさせる上で避けて通れないのが、**「ツール実装の再利用性と標準化」** です。**Model Context Protocol（MCP）** は、アプリケーションが LLM にツールやコンテキストを提供するためのオープンプロトコルです。

LangChain v1（`langchain[mcp]>=1.4.0`）では、`langchain.mcp` 名前空間に **`MCPAdapter`** が導入されました（参考：[langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/adapter.py)）。内部的には **FastMCP** を基盤としており、トランスポート推論・プロトコルネゴシエーション・接続管理・認証（Bearer / OAuth 2.1）を抽象化します。

### 4.1 `MCPAdapter` の役割とトランスポート推論

`MCPAdapter` は接続先ターゲットの型からトランスポートを自動推論します。

{% mermaid %}
flowchart LR
    Agent["LangChain Agent<br/>create_agent(model, tools)"]
    Adapter["MCPAdapter<br/>(FastMCP 基盤)<br/>list_tools() で標準 Tool へ変換"]

    subgraph Transports["自動推論されるトランスポート"]
        T1["In-Memory<br/>FastMCP インスタンス<br/>(単体テスト向け)"]
        T2["stdio (Subprocess)<br/>pathlib.Path('server.py')<br/>(ローカルツール向け)"]
        T3["Streamable HTTP<br/>'https://.../mcp'<br/>(リモート MCP サーバー)"]
        T4["MCPConfig Dict<br/>{'mcpServers': {...}}<br/>(複数サーバー統合)"]
    end

    Agent <-->|"ToolCall / ToolMessage<br/>(Content Blocks & Artifact)"| Adapter
    Adapter <--> T1
    Adapter <--> T2
    Adapter <--> T3
    Adapter <--> T4

    style Agent fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Adapter fill:#fff8e1,stroke:#f57f17,color:#e65100
    style T1 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style T2 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style T3 fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    style T4 fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

セキュリティ上の重要な仕様として、**`str` 型のターゲットは `http://` または `https://` URL のみ許可**されます。ローカルスクリプトを stdio で起動したい場合は明示的に `pathlib.Path("server.py")` を渡す必要があります。これは、設定ファイルや LLM 出力由来の文字列が意図せずローカルファイルを実行してしまうサブプロセス起動脆弱性を防ぐための設計です。

### 4.2 MCP ツール出力の変換（Multimodal & Structured Artifact）

`MCPAdapter.list_tools()` で取得したツールは、LangChain の標準ツールと完全に透過的に扱えます。

- **マルチモーダル出力**：MCP サーバーが返す画像やファイルブロックは、LangChain 標準の `content_blocks`（`type="image"`, `type="file"`, `type="text"`）へ自動変換されます。
- **構造化出力（Structured Content）**：MCP ツールが構造化 JSON を返した場合、モデル向けテキストに混ぜてコンテキストを汚すのではなく、**`ToolMessage.artifact["structured_content"]`** に格納されます。これにより、アプリ側の後続処理は厳密な構造化データを受け取りつつ、LLM のトークン消費を抑えられます。

```python
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

async def run_mcp_agent() -> dict:
    # 環境変数から LLM 接続設定を読み込む
    llm = ChatOpenAI(
        model="MiniCPM-V-4.7-35B-A3B",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.0,
    )

    # 例: リモートの LangChain 公式ドキュメント MCP サーバーへ Streamable HTTP 接続
    # ローカルスクリプトを stdio 接続する場合は Path("metrics_server.py") を渡す
    async with MCPAdapter("https://docs.langchain.com/mcp") as adapter:
        # MCP サーバーが公開するツール一覧を LangChain 標準ツールへ自動変換
        mcp_tools = await adapter.list_tools()

        # 通常の create_agent へそのまま渡せる（Subagent のツールとしても組み込み可能）
        agent = create_agent(
            model=llm,
            tools=mcp_tools,
            system_prompt="MCP ツールを活用して最新の公式ドキュメントに基づき回答してください。",
        )
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "LangChain の短期記憶の追加方法を教えて"}]}
        )
```

### 4.3 Multi-Agent × MCP の組み合わせパターン

実務では **「Subagents パターン ＋ MCP」** の組み合わせが非常に強力です。たとえば、SRE 統括エージェント（Supervisor）の配下に「監視メトリクス専門 Subagent（Prometheus MCP サーバー接続）」「障害チケット専門 Subagent（Jira/Linear MCP サーバー接続）」を配置することで、以下のメリットを同時に享受できます。

- 各 MCP サーバーが提供する数十個のツール定義が Supervisor のコンテキストを汚染しない
- ツール実装は MCP サーバーとして言語非依存・再利用可能に保てる
- Supervisor は高レベルの意思決定と最終報告の統合だけに専念できる

---

## 5. まとめ

本稿では、LangChain v1 の Multi-Agent 設計と MCP 統合の理論を整理しました。

- **パターンの使い分け**：単発・連続対話の効率重視なら **Handoffs / Skills**、複数ドメインの並列処理・コンテキスト分離・チーム分散開発を重視するなら **Subagents / Router** を選択する。
- **Subagents の設計軸**：細かい入出力制御には **Tool per agent**、大規模レジストリには **Single dispatch tool** を使い分ける。
- **MCP 統合**：`MCPAdapter` により、インメモリ・stdio・Streamable HTTP の各種 MCP サーバーを標準ツールとしてシームレスにエージェントやサブエージェントへ組み込める。

次回の **Part 7 実践編** では、本稿で解説した **Subagent-as-Tool（Tool per agent）パターン** を実際に実装し、SRE メトリクス調査サブエージェントと統括エージェントの協調動作・メッセージ数・コンテキスト分離の効果を実測ログで検証します。
