---
title: "DeepAgents Part 1 理論編：Agent Harness の全体像（create_deep_agent・Planning / TodoList・デフォルト Middleware スタック）"
date: 2026-06-25 10:17:49
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, Agent Harness, Middleware, AI Agent, Architecture]
lang: ja
description: "LangChain エコシステムの最上位ハーネスである DeepAgents の全体設計を徹底解説。create_deep_agent の 4 本柱、v0.7 以降のオプトイン TodoListMiddleware、そして 14 層から成るデフォルト Middleware スタックの実行順序と置換ルールを詳解します。"
---

# DeepAgents Part 1 理論編：Agent Harness の全体像（`create_deep_agent`・Planning / TodoList・デフォルト Middleware スタック）

LangChain v1 では **Agent = Model + Harness** という設計原則が打ち出されました。しかし、実務の複雑なタスク（複数ファイルの調査・編集、数万トークンを超えるログ解析、専門エージェントへの並列委譲、破壊的操作前の人間承認）をこなすエージェントを `create_agent` から毎回ゼロから組み上げるのは多大なボイラープレートを伴います。

そこで登場したのが、LangChain と LangGraph の上に構築された「Batteries-included（全部入り）」なエージェント・ハーネス **DeepAgents**（[`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py)）です。

本シリーズでは、DeepAgents の内部設計と実戦投入の勘所を **理論編＋実践編のペア** で体系的に解説します。第 1 回となる本記事（Part 1 理論編）では、DeepAgents が提供する **Agent Harness の 4 本柱**、`create_deep_agent` の全体構造、v0.7 で大きく方針が変わった **Planning（`TodoListMiddleware`）**、そしてエージェントの振る舞いを決定づける **デフォルト Middleware スタックの実行順序** を深く掘り下げます。

---

## 1. DeepAgents とは何か：LangChain / LangGraph / DeepAgents の 3 層構造

DeepAgents は独立した LLM フレームワークではなく、LangChain エコシステムの 3 層アーキテクチャにおける **最上位の Harness 層** に位置します。

| レイヤー | 主な API / パッケージ | 責務と選定基準 |
| :--- | :--- | :--- |
| **DeepAgents（Harness 層）** | `deepagents.create_deep_agent` | 仮想ファイルシステム・コンテキスト自動圧縮・サブエージェント委譲・Skills / Memory・HITL を標準装備した実戦向けハーネス。 |
| **LangChain（Framework 層）** | `langchain.agents.create_agent` | Model + Tools + Middleware の最小エージェントループ。独自の軽量ハーネスをゼロから合成したい場合に選択。 |
| **LangGraph（Runtime 層）** | `StateGraph` / `CompiledStateGraph` | 永続チェックポイント（Checkpointer）・長期ストア（BaseStore）・割り込み再開（Interrupt）・Durable Execution を担う低レベルランタイム。 |

重要なのは、**`create_deep_agent()` が返すオブジェクトもまた LangGraph の `CompiledStateGraph` である** という点です。したがって、LangGraph が提供する `.invoke()`・`.stream()`・`InMemorySaver` / Postgres チェックポインタ・`Command(resume=...)` による中断再開がそのまま透過的に機能します。

### Agent Harness を構成する 4 本柱

DeepAgents のハーネスは、単なる「ツール呼び出しループ」を実務で壊れないレベルに引き上げるため、以下の **4 つの柱（Capabilities）** を統合しています。

{% mermaid %}
flowchart TB
    subgraph HARNESS["DeepAgents Agent Harness (create_deep_agent)"]
        direction LR
        subgraph EXEC["1. Execution Environment"]
            E1["Custom Tools / MCP"]
            E2["Virtual Filesystem<br/>(ls / read / write / edit / glob / grep / delete)"]
            E3["Permissions &<br/>Sandbox / QuickJS"]
        end
        subgraph CTX["2. Context Management"]
            C1["Memory (AGENTS.md)<br/>常時ロード"]
            C2["Skills (SKILL.md)<br/>段階的開示"]
            C3["Auto Offloading &<br/>Summarization / Caching"]
        end
        subgraph DEL["3. Delegation"]
            D1["Task Planning<br/>(TodoListMiddleware / Opt-in)"]
            D2["Subagents (task ツール)<br/>同期・非同期・Fork"]
        end
        subgraph STEER["4. Steering"]
            S1["Human-in-the-Loop<br/>(interrupt_on)"]
            S2["Model Refusal /<br/>PatchToolCalls 防御"]
        end
    end

    U["ユーザー要求 /<br/>Runtime Context"] --> HARNESS
    HARNESS --> LG["LangGraph Runtime<br/>(Checkpointer / BaseStore / Streaming)"]

    style HARNESS fill:#f0f7ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style EXEC fill:#ebf8ff,stroke:#3182ce,stroke-width:1.5px,color:#1a365d
    style CTX fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style DEL fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style STEER fill:#fff5f5,stroke:#c53030,stroke-width:1.5px,color:#742a2a
    style LG fill:#f7fafc,stroke:#4a5568,stroke-width:1.5px,color:#2d3748
{% endmermaid %}

> **DeepAgents の本質**: 「モデルに長いタスクを任せると途中でコンテキストが溢れ、作業計画を忘れ、ファイル操作やサブタスクの結果で履歴が汚染される」という典型的な失敗モードに対し、**ファイルシステム・圧縮・隔離・承認** を最初から組み込んだ標準実行基盤である。

---

## 2. `create_deep_agent` のシグネチャと設計パラメータ

[`deepagents/graph.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py) で定義されている `create_deep_agent` は、単一の関数呼び出しでこれらすべての機能を宣言的に配線します。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

# 環境変数からエンドポイントとAPIキーを読み込む（ハードコード厳禁）
load_dotenv()

llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
    max_tokens=512,
)

@tool
def lookup_policy(topic: str) -> str:
    """社内規定のトピックを検索して返す。"""
    return f"Policy for {topic}: 承認フローは2段階必要です。"

# 最小構成でも Filesystem / SubAgent / Summarization / PatchToolCalls が自動配線される
agent = create_deep_agent(
    model=llm,                         # 文字列 ("openai:gpt-...") または BaseChatModel
    tools=[lookup_policy],             # ドメイン固有ツールや MCP ツール
    system_prompt="正確かつ簡潔に日本語で回答してください。",
    # 以下は用途に応じて追加する主要オプション
    # backend=...,                     # 省略時は StateBackend()（スレッド内メモリ）
    # middleware=[...],                # 追加 Middleware（同名ならデフォルトを置換）
    # subagents=[...],                 # 専門サブエージェント定義
    # skills=["/skills/"],             # Progressive Disclosure 用スキルディレクトリ
    # memory=["/AGENTS.md"],           # 常時ロードする永続指示ファイル
    # permissions=[...],               # ファイルパス単位の allow/deny ルール
    # interrupt_on={"tool": True},     # HITL 承認ポイント
    # checkpointer=...,                # 会話スレッド永続化
    # store=...,                       # スレッド横断の長期ストレージ（BaseStore）
)
```

### システムプロンプトの自動組み立て順序

`create_deep_agent` に渡した `system_prompt` は、そのまま単独でモデルに渡されるわけではありません。ハーネス内部で以下の **8 段階の順序** でシステムメッセージが合成されます。

1. **ユーザー指定の `system_prompt`**（指定した場合、最先頭に配置）
2. **Base agent prompt**（DeepAgents 標準の行動指針。[`graph.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py) 定義）
3. **Memory prompt**（`memory=` 指定時：`AGENTS.md` 本文とメモリ更新ガイドライン）
4. **Skills prompt**（`skills=` 指定時：各 `SKILL.md` の YAML フロントマター一覧と利用方法）
5. **Virtual filesystem prompt**（`ls` / `read_file` / `write_file` / `edit_file` / `delete` / `glob` / `grep`、および Sandbox 時の `execute` の使い分けガイド）
6. **Subagent prompt**（`task` ツールによる委譲ルールと利用可能なサブエージェント一覧）
7. **ユーザー提供のカスタム Middleware プロンプト**
8. **Human-in-the-loop prompt**（`interrupt_on=` 指定時）

ここで見落としやすいのが **ベースライン・プロンプトのトークン量** です。デフォルトでは仮想ファイルシステムとサブエージェントのツールスキーマおよび説明プロンプトが毎ターン送信されます。もし読み取り専用の軽量エージェントを作りたい場合は、`FilesystemMiddleware(backend=..., tools=["read_file", "ls", "glob", "grep"])` や `HarnessProfile(excluded_tools=...)` で不要なツールを削ることで、毎ターンの入力トークンを大きく削減できます。

---

## 3. Task Planning と `TodoListMiddleware`（v0.7 での重要な仕様変更）

DeepAgents の紹介記事や初期バージョン（v0.6 以前）のコードでは、`create_deep_agent` を呼ぶだけで `write_todos` ツールが自動的に使える前提で書かれていることがよくあります。しかし **DeepAgents v0.7 以降、Task Planning（`TodoListMiddleware`）はデフォルト無効のオプトイン機能に変更されました**。

### なぜオプトインに変更されたのか

- 単純な 1〜2 ステップの質問応答やツール呼び出しに対して `write_todos` が常に露出していると、モデルが不要な計画作成に 1 ターン浪費するケースがある。
- 近年の高性能モデルは暗黙的に手順を管理できる場面が増えており、明示的なタスクリストが本当に効くのは「長時間の多段階タスク」「小型・軽量モデルに説明責任（accountability）を持たせたい場合」「フロントエンド UI に進捗チェックリストをストリーミング表示したい場合」に絞られる。

### オプトインによる有効化方法

Task Planning を有効にしたい場合は、`langchain.agents.middleware` から `TodoListMiddleware` を明示的にインポートして `middleware=` に渡します。

```python
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

