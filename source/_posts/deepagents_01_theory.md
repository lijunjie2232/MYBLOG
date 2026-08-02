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

# v0.7 以降で write_todos ツールを有効化する場合は明示的に TodoListMiddleware を追加する
agent = create_deep_agent(
    model=llm,
    tools=[lookup_policy],
    middleware=[TodoListMiddleware()],
)
```

`TodoListMiddleware` を追加すると、エージェントは `write_todos` ツールを通じて各タスクのステータスを `'pending'` → `'in_progress'` → `'completed'` と更新しながらグラフ状態（Agent State）に永続化できます。これにより、中断後の再開時やフロントエンドの進捗バー連携時にも「どこまで終わったか」が構造化データとして残ります。

---

## 4. デフォルト Middleware スタックの完全構造と実行順序

DeepAgents の最大の技術的価値は、**複数の Middleware を「壊れない順序」で厳密に合成している点** にあります。順序を 1 つ誤るだけで「プロンプトキャッシュが毎回無効化される」「中断再開時の宙ぶらりん `tool_calls` で API エラーになる」「要約後にスキル一覧が消える」といった深刻なバグが発生します。

`create_deep_agent` が構築するスタックには、最小構成の **Bare Stack（モデルのみ指定時）** と、全オプション有効時の **Full Stack（最大 14 層）** があります。

{% mermaid %}
flowchart TD
    IN["エージェント呼び出し<br/>(messages / state)"] --> M1["1. FilesystemMiddleware<br/>仮想FSツール・権限・大容量オフロード"]
    M1 --> M2["2. SubAgentMiddleware<br/>同期サブエージェント (task ツール)"]
    M2 --> M3["3. SummarizationMiddleware<br/>コンテキスト85%超過時の自動要約"]
    M3 --> M4["4. PatchToolCallsMiddleware<br/>中断・破損した tool_calls の修復"]
    M4 --> M5["5. AsyncSubAgentMiddleware (任意)<br/>非同期サブエージェント管理"]
    M5 --> M6["6. User Custom Middleware<br/>同名(.name一致)なら既存層を置換 / 新規はここへ挿入"]
    M6 --> M7["7. Harness Profile Extras<br/>モデル固有の拡張Middleware"]
    M7 --> M8["8. SkillsMiddleware (任意)<br/>SKILL.md フロントマター注入"]
    M8 --> M9["9. Prompt Caching Middleware<br/>Anthropic / Bedrock プレフィックスキャッシュ"]
    M9 --> M10["10. MemoryMiddleware (任意)<br/>AGENTS.md 注入 (キャッシュ後に配置!)"]
    M10 --> M11["11. HumanInTheLoopMiddleware (任意)<br/>interrupt_on による実行前承認"]
    M11 --> M12["12. UnsupportedContentMiddleware<br/>非対応マルチモーダルのテキスト置換"]
    M12 --> M13["13. ModelRefusalMiddleware<br/>モデル拒否応答の例外化"]
    M13 --> M14["14. Excluded-Tool Filtering<br/>プロファイル指定の除外ツール削除"]
    M14 --> MODEL["LLM 推論呼び出し"]

    style M1 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style M3 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style M4 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style M6 fill:#faf5ff,stroke:#6b46c1,stroke-width:2px,color:#44337a
    style M9 fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
    style M10 fill:#fff5f5,stroke:#c53030,stroke-width:1.5px,color:#742a2a
    style MODEL fill:#edf2f7,stroke:#2d3748,stroke-width:2px,color:#1a202c
{% endmermaid %}

### なぜこの順序でなければならないのか（設計上の 4 つの勘所）

1. **`PatchToolCallsMiddleware`（4 番目）が Prompt Caching（9 番目）より前にある理由**
   HITL 中断やネットワーク断で「`AIMessage(tool_calls=...)` の直後に対応する `ToolMessage` が欠損した履歴」が残ったまま LLM へ送ると、OpenAI / Anthropic API は即座に 400 エラーを返します。`PatchToolCallsMiddleware` がキャッシュ境界の確定前に不整合なツール呼び出しを修復することで、再開時の安定性とキャッシュ整合性の両方を守ります。
2. **`MemoryMiddleware`（10 番目）が Prompt Caching（9 番目）より後にある理由**
   `AGENTS.md` のようなメモリファイルは会話中にエージェント自身が書き換える可能性があります。もしメモリをキャッシュ対象プレフィックスの内側に入れてしまうと、メモリを 1 行更新しただけでシステムプロンプト全体のキャッシュが毎ターン無効化されてしまいます。あえてキャッシュ層の直後に配置することで、ベースプロンプトやツール定義のキャッシュヒット率を高く保ちます。
3. **ユーザー指定 `middleware=` の `.name` マッチによる In-Place 置換（v0.7+）**
   `create_deep_agent(..., middleware=[custom_mw])` に渡したインスタンスの `.name` がデフォルトスタック内の既存 Middleware（例：`SummarizationMiddleware` や `FilesystemMiddleware`）と一致する場合、重複追加ではなく **その位置で丸ごと置換（Override）** されます。一致しない新規 Middleware は必ず `PatchToolCallsMiddleware` の直後（6 番目）に挿入されます。
   - **注意**: 置換は「マージ」ではなく「完全な置き換え」です。たとえば `FilesystemMiddleware` を上書きする場合、`create_deep_agent` に渡した `backend=` や `permissions=` は自動継承されないため、カスタムインスタンス側へ明示的に渡す必要があります。
4. **Middleware の状態管理は「インスタンス属性の破壊的変更」を禁止**
   DeepAgents ではサブエージェントや並列ツール呼び出しが同時に走るため、カスタム Middleware 内で `self.count += 1` のようにインスタンス変数を書き換えると競合状態（Race Condition）が起きます。永続化・共有したいカウンタやフラグは必ず LangGraph の **Graph State（`DeepAgentState` サブクラス）** に格納する設計が推奨されています。

---

## 5. デフォルト Middleware の置換とツール露出の制御（v0.7+）

本番運用では「読み取り専用エージェントにしたい」「特定の組み込みツールをモデルから隠したい」といった要件が頻繁に発生します。DeepAgents v0.7 以降では、以下の 2 つの正規手段で安全にツール面を絞り込めます。

### パターン A：`FilesystemMiddleware` の `tools=` Allowlist による置換

`FilesystemMiddleware` は `tools` ホワイトリストを受け取ります。`middleware=` に渡すと `.name` 一致によりデフォルトの `FilesystemMiddleware` がその場で置換されます（**注**: `read_file` は必須であり、省略すると `ValueError` が送出されます）。

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

backend = StateBackend()

# 読み取り専用エージェント：write_file / edit_file / delete / execute をモデルから完全に隠蔽
# ※ デフォルト置換はマージではなく完全上書きのため、backend= を必ず明示する
readonly_agent = create_deep_agent(
    model=llm,
    backend=backend,
    middleware=[
        FilesystemMiddleware(
            backend=backend,
            tools=["read_file", "ls", "glob", "grep"],
        ),
    ],
)
```

### パターン B：`HarnessProfile` の `excluded_tools` によるツール非表示

ファイルシステムツール自体を一切使わせたくない場合でも、`FilesystemMiddleware` を `excluded_middleware` で削除することは仕様上拒否されます（オフロードや内部スキャフォールディングに必須のため）。代わりに `HarnessProfile` の `excluded_tools` を使い、Middleware を稼働させたままモデルへのツール露出だけを消します。

```python
from deepagents import HarnessProfile, register_harness_profile

# モデル固有のハーネスプロファイルで、不要な組み込みツールをモデルから隠す
register_harness_profile(
    "anthropic:claude-sonnet-4-6",
    HarnessProfile(
        excluded_tools=frozenset(
            {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"}
        ),
    ),
)
```

なお、自動追加される `general-purpose` サブエージェントはメインエージェントのデフォルト Middleware 上書きを継承しますが、`subagents=` で明示定義した宣言的サブエージェントは自動継承しません。サブエージェント側にも同じ制限を課したい場合は、サブエージェント辞書内の `middleware` フィールドへ個別に渡す必要があります。

---

## 6. まとめと次回の実践編に向けて

本記事（Part 1 理論編）では、DeepAgents の全体アーキテクチャを以下の 3 点から整理しました。

- DeepAgents は LangChain / LangGraph の上に立つ **Agent Harness** であり、Execution Environment・Context Management・Delegation・Steering の 4 本柱を `create_deep_agent` 1 つで提供する。
- **Task Planning（`TodoListMiddleware` / `write_todos`）は v0.7 以降オプトイン** となり、必要な場合のみ `middleware=[TodoListMiddleware()]` で明示的に組み込む設計へ進化した。
- **14 層のデフォルト Middleware スタック** は、要約・破損ツール修復・プロンプトキャッシュ・メモリ更新の相互干渉を防ぐために厳密な順序で設計されており、`.name` 一致による In-Place 置換で柔軟にカスタマイズできる。

次回の **Part 1 実践編** では、OpenAI 互換エンドポイント上の MoE モデル **`MiniCPM-V-4.7-35B-A3B`** を使って実際に `create_deep_agent` を動かし、在庫確認ツール（`check_inventory`）の呼び出しトレース、メッセージ数 4 件の内訳、そして「なぜ `write_todos` が呼ばれなかったのか」を実測データから検証します。
