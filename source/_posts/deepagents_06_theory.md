---
title: DeepAgents Part 6 理論編：Permissions・Human-in-the-Loop（interrupt_on）と Sandboxes による実行環境隔離
date: 2026-07-31 04:45:37
categories:
  - AI
tags:
  - DeepAgents
  - LangGraph
  - LangChain
  - Human-in-the-Loop
  - Permissions
  - Sandboxes
  - AI Agent
lang: ja
description: DeepAgents の安全制御アーキテクチャを体系解説。宣言的ファイル権限（FilesystemPermission）、Human-in-the-Loop（interrupt_on と 4 種の Decision）、さらに Sandbox Backend による OS・ネットワーク隔離とシークレット管理までを網羅します。
---

自律型 AI エージェントをローカル試作から本番環境へ持ち込む際、最大の障壁となるのが**「エージェントの暴走・誤操作・プロンプトインジェクションをどう封じ込めるか」**というセキュリティ設計です。DeepAgents は単一の防御策に頼るのではなく、**3層の多層防御（Defense in Depth）**として以下を提供しています。

1. **Permissions（宣言的ファイルシステム権限）**：ビルトインのファイルツールに対するパス単位の `allow` / `deny` / `interrupt` 制御
2. **Human-in-the-Loop（`interrupt_on`）**：破壊的・外部副作用を伴うツール実行の直前でグラフを一時停止し、人間の承認・編集・拒否を介在させる仕組み
3. **Sandboxes（実行環境隔離）**：任意シェルコマンド（`execute`）やコード実行をホスト OS から完全に隔離されたコンテナ／マイクロ VM に閉じ込めるアーキテクチャ

本記事では、DeepAgents（[langchain-ai/deepagents](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py)）におけるこれら3層の仕組みと境界条件を理論面から徹底解説します。

---

## 1. 多層防御アーキテクチャの全体像

DeepAgents における安全制御は、「ツールが呼ばれる前」「ツールが実行される瞬間」「実行される物理・仮想環境」の3段階で機能します。

{% mermaid %}
flowchart TD
    U["ユーザー入力 / 外部コンテキスト"] --> LLM["LLM 推論ノード (model)"]
    LLM -->|"ツール呼び出し提案 (tool_calls)"| L1["第1層: FilesystemPermission 評価<br/>(First-Match-Wins ルール判定)"]
    L1 -->|"mode = deny"| DENY["即座に拒否エラー返却<br/>(ツール未実行)"]
    L1 -->|"mode = allow / カスタムツール"| L2["第2層: HumanInTheLoopMiddleware<br/>(interrupt_on / mode=interrupt / when 条件)"]
    L1 -->|"mode = interrupt"| L2
    L2 -->|"要承認 (Interrupt 発火)"| CP[("Checkpointer<br/>thread_id 単位で状態永続化")]
    CP --> HUMAN["人間のレビュー<br/>approve / edit / reject / respond"]
    HUMAN -->|"Command(resume=...)"| L2
    L2 -->|"reject"| REJ["拒否フィードバックを ToolMessage 化<br/>(ツール未実行で LLM へ返却)"]
    L2 -->|"approve / edit / 承認不要"| L3["第3層: 実行環境 (Backend / Sandbox)"]
    L3 -->|"通常ツール / Host"| HOST["ホスト環境 (シークレット保持可)"]
    L3 -->|"SandboxBackendProtocol"| SBX["隔離コンテナ / MicroVM<br/>(execute ツール実行・シークレット非保持)"]
    DENY --> LLM
    REJ --> LLM
    HOST --> LLM
    SBX --> LLM

    style L1 fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style L2 fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style L3 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style CP fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    style REJ fill:#ffebee,stroke:#c62828,color:#b71c1c
{% endmermaid %}

それぞれの層には明確な守備範囲と限界があります。特に**「Permissions はビルトインのファイルツールのみに効き、Sandbox の `execute` には効かない」**といった境界を正しく理解することが本番設計の要です。

---

## 2. Permissions：宣言的ファイルシステムアクセス制御

`deepagents>=0.5.2` で導入された `permissions` 引数を使うと、ビルトインのファイルシステムツール（`ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, `delete`）に対して宣言的なアクセス制御ルールを適用できます。

### 2.1 `FilesystemPermission` のデータ構造と評価規則

各ルールは `FilesystemPermission` オブジェクトとして定義し、以下の3フィールドを持ちます。

| フィールド | 型 | 役割と対象ツール |
| --- | --- | --- |
| `operations` | `list["read" \| "write"]` | `"read"` は `ls`, `read_file`, `glob`, `grep` に適用。`"write"` は `write_file`, `edit_file`, `delete` に適用 |
| `paths` | `list[str]` | glob パターン（例：`["/workspace/**"]`）。再帰マッチ `**` や選択 `{a,b}` をサポート |
| `mode` | `"allow" \| "deny" \| "interrupt"` | 一致時の挙動。デフォルトは `"allow"`。`"interrupt"` は `deepagents>=0.6.8` で利用可能 |

ルール評価には極めて重要な原則があります。

> `FilesystemPermission` のルールは宣言順に評価される **First-Match-Wins（最初に一致したルールが勝つ）** 方式を採用しており、どのルールにも一致しない操作はデフォルトで **許可（permissive default）** される。

したがって、特定のワークスペース配下だけに隔離し、さらに機密ファイル（`.env` など）を除外したい場合は、**「狭い拒否 → 許可範囲 → 包括的な拒否（`/**`）」** の順で記述する必要があります。

```python
import os
from langchain_openai import ChatOpenAI
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver

# 環境変数からエンドポイントと API キーを取得（ハードコード禁止）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

# 宣言順が重要：具体的なルールを先に、包括的な拒否ルールを最後に配置する
permissions = [
    # 1. 機密ファイルは読み書きともに完全拒否
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/.env", "/workspace/credentials/**"],
        mode="deny",
    ),
    # 2. /secrets 配下への書き込みは人間の承認を要求（interrupt モード）
    FilesystemPermission(
        operations=["write"],
        paths=["/secrets/**"],
        mode="interrupt",
    ),
    # 3. 作業ディレクトリ配下のみ読み書きを許可
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    # 4. 上記以外のすべてのパスへのアクセスをデフォルト拒否に変更
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(root_dir="/tmp/agent_root", virtual_mode=True),
    permissions=permissions,
    # mode="interrupt" を使う場合は Checkpointer が必須
    checkpointer=InMemorySaver(),
)
```

### 2.2 Permissions の適用境界とサブエージェント継承

設計上見落としやすい制約が4つあります。

1. **カスタムツール・MCP ツールは対象外**：`permissions` はビルトインのファイルツールのみを保護します。独自ツールや MCP 経由のファイル操作を制御したい場合は、Backend Policy Hooks やツール側のバリデーション、あるいは後述の `interrupt_on` を使います。
2. **Sandbox Backend には直接適用不可**：Sandbox は `execute` ツールで任意のシェルコマンド（`cat /etc/passwd` など）を実行できるため、パス単位の権限チェックだけでは回避されてしまいます。`CompositeBackend` でデフォルトに Sandbox を置き、`/memories/` ルートに `StoreBackend` を割り当てている場合、`/memories/**` に閉じたルールは許可されますが、ルート外や `/**` を含むルールは **`NotImplementedError`** を送出します。
3. **ディレクトリ削除の All-or-Nothing 判定**：`delete` がディレクトリを対象とする場合、対象とすべての配下パスについて `write` 権限を検査し、1つでも拒否される場合は操作全体を拒否します（部分削除を防ぎます）。一方、通常ファイルの削除は `write_file` / `edit_file` と同じく完全一致の First-Match-Wins で評価されます（`deepagents>=0.7.3`）。
4. **サブエージェントでの上書き仕様**：サブエージェントはデフォルトで親の `permissions` を継承しますが、サブエージェント定義に `permissions` を明示すると**親のルールを完全に置換**します（マージではありません）。監査用サブエージェントを完全読み取り専用にする際などに活用できます。

---

## 3. Human-in-the-Loop（`interrupt_on`）の内部メカニズム

ファイル操作以外の危険なカスタムツール（本番 DB テーブル削除、外部メール送信、決済 API 実行など）を制御するのが **`interrupt_on`** パラメータです。

`create_deep_agent` に `interrupt_on` を渡すと、内部の Middleware スタックに `HumanInTheLoopMiddleware` が組み込まれます。さらに同じスタックには `PatchToolCallsMiddleware` も存在し、割り込みやキャンセルによって応答のない `tool_calls` が履歴に残った場合でも、次のステップ前にメッセージ履歴を自動修復します。

> `interrupt_on` を設定して実行を中断・再開する場合、エージェントの状態（メッセージ履歴・保留中のツール呼び出し・割り込みメタデータ）を保持する **Checkpointer** と、中断前と再開時で同一の **`thread_id`** を指定することが絶対条件となる。

### 3.1 `interrupt_on` の設定形式と条件付き割り込み（`when`）

`interrupt_on` には `{ツール名: 設定}` の辞書を渡します。

- `True`：デフォルト設定で割り込みを有効化（`approve`, `edit`, `reject`, `respond` が許可）
- `False`：そのツールでは割り込みを無効化
- `InterruptOnConfig`（辞書）：`allowed_decisions` で許可する判断種別を絞り込み、さらに `when` 述語関数（`langchain>=1.3.3`）で**引数に応じた条件付き割り込み**を実装可能

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

# 環境変数から LLM を初期化
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
