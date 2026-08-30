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
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

def is_outside_workspace(request: ToolCallRequest) -> bool:
    """/workspace/ 配下以外への書き込み時のみ割り込みを発火する述語関数。"""
    path = request.tool_call["args"].get("file_path", "")
    return not path.startswith("/workspace/")

agent = create_deep_agent(
    model=llm,
    interrupt_on={
        # 破壊的ツールは approve / reject のみ許可（引数編集を禁止）
        "delete_production_table": {"allowed_decisions": ["approve", "reject"]},
        # write_file はワークスペース外への書き込み時のみ一時停止
        "write_file": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "when": is_outside_workspace,
        },
        # 読み取り専用ツールは割り込み不要
        "read_file": False,
    },
    checkpointer=InMemorySaver(),
)
```

`when` が `False` を返したツール呼び出しは自動的に承認され、割り込みバッチ（`action_requests`）にも含まれないため、レビュアーの承認疲れ（Approval Fatigue）を防げます。

### 3.2 4種類の Decision Type と使い分け

割り込みが発火すると、`invoke()` の戻り値（または `updates` ストリーム）に `__interrupt__` キーが現れます。そのペイロードには以下の2つが格納されます。

- `action_requests`：承認待ちのツール呼び出しリスト（`name`, `args`, `description`）
- `review_configs`：各ツールに対して許可されている操作リスト（`action_name`, `allowed_decisions`）

人間（または承認 UI）は `action_requests` と**同じ順序・同じ件数**で `decisions` リストを組み立て、`Command(resume={"decisions": decisions})` でグラフを再開します。

| Decision Type | 挙動 | 典型的なユースケース | 注意点 |
| --- | --- | --- | --- |
| `approve` | エージェントが提案した元の引数のままツールを実行する | 確認済みメールの送信、承認済みテーブル削除 | 副作用が即座に発生する |
| `edit` | `edited_action: {"name": ..., "args": ...}` で引数を書き換えてから実行する | 送信先アドレスや対象パスを安全な値へ修正 | 大幅な引数変更はモデルに再試行を誘発するため控えめにする |
| `reject` | **ツールを一切実行せず**、拒否理由 `message` をツール結果としてモデルへ返す | 危険な削除操作の却下と代替方針の指示 | `message` に「再試行禁止」や次の行動を明記すると暴走を防げる |
| `respond` | ツールを実行せず、人間の回答を合成ツール結果としてモデルへ返す | `ask_user` のような質問ツールへの直接回答 | **副作用ツールの拒否には絶対に使わない**（成功結果と誤認されるため） |

> `reject` を返した場合、対象ツール関数は一度も実行されないままスキップされ、指定した `message`（省略時はデフォルトの拒否メッセージ）がエージェントへフィードバックされる。これにより、モデルは拒否理由を踏まえてユーザーへ安全に回答を返すことができる。

### 3.3 中断〜再開のライフサイクル（シーケンス図）

{% mermaid %}
sequenceDiagram
    participant App as アプリケーション / UI
    participant Graph as DeepAgent Graph
    participant HITL as HumanInTheLoopMiddleware
    participant CP as Checkpointer
    participant Tool as 危険ツール本体

    App->>Graph: invoke(messages, config={thread_id: "t1"})
    Graph->>HITL: model が tool_calls を出力
    HITL->>CP: 現在状態と action_requests を保存
    HITL-->>App: __interrupt__ を返却して一時停止
    Note over App: レビュアーがツール名・引数を確認
    App->>Graph: invoke(Command(resume={"decisions": [{"type": "reject", "message": "..."}]}), config={thread_id: "t1"})
    Graph->>CP: thread_id="t1" のチェックポイントを復元
    CP-->>HITL: 保留中のツール呼び出しを復帰
    alt decision.type == "approve" / "edit"
        HITL->>Tool: ツールを実行（edit 時は修正後引数）
        Tool-->>Graph: ToolMessage（実行結果）
    else decision.type == "reject"
        Note over HITL,Tool: ツール本体は呼び出されない！
        HITL-->>Graph: 拒否 message を含む ToolMessage を注入
    end
    Graph-->>App: 最終応答メッセージ（AI が拒否理由を反映）
{% endmermaid %}

### 3.4 複数ツール同時呼び出しとサブエージェント割り込み

- **複数ツールの一括バッチ化**：LLM が1ターンで2つ以上の要承認ツールを同時に呼んだ場合、それらは1つの `__interrupt__` の `action_requests` 配列にまとめて格納されます。`Command(resume={"decisions": [...]})` では、配列の順序と1対1に対応した決定を返す必要があります。
- **サブエージェントからの割り込み**：サブエージェント定義にも独自の `interrupt_on` を設定でき、親の設定を上書きします。サブエージェント内で割り込みが発生しても、親の `invoke()` / `stream(stream_mode=["updates"])` の `__interrupt__` として透過的に浮上します（割り込み検知だけなら `subgraphs=True` も不要です）。また、カスタムツール内部から LangGraph の `interrupt()` プリミティブを直接呼び出すことも可能です。

---

## 4. Sandboxes：コード実行環境の完全隔離

コーディングエージェントやデータ分析エージェントのように、パッケージのインストール・テスト実行・スクリプト実行を自律的に行わせる場合、ホスト OS 上で直接シェルを叩く `LocalShellBackend` や `FilesystemBackend` を本番環境にデプロイしてはなりません。DeepAgents では **Sandbox Backend** を差し込むことで、`execute` ツールと隔離されたコンテナ／VM 環境を提供します。

### 4.1 2つの統合パターン比較

| 観点 | Agent in Sandbox パターン | Sandbox as Tool パターン（推奨） |
| --- | --- | --- |
| エージェント本体の配置 | コンテナ／VM 内部 | ホスト／Agent Server 側 |
| LLM API キーの配置 | **Sandbox 内部（漏洩リスク大）** | **Sandbox 外部（安全）** |
| コード更新の手軽さ | イメージ再ビルドが必要 | 即座に更新可能 |
| Sandbox 障害時の影響 | エージェント状態ごと消失 | 状態はホスト側 Checkpointer に保全 |
| レイテンシ | 低（ローカル実行） | ツール呼び出しごとにネットワーク往復 |

DeepAgents の公式推奨は **Sandbox as Tool パターン**です。プロバイダ側（LangSmith Sandboxes, Daytona, E2B, Modal, Runloop, Vercel Sandbox 等）は最小限の `execute()` メソッドだけを実装すれば、`BaseSandbox` 基底クラスがその上で `read_file`, `write_file`, `edit_file`, `delete`, `ls`, `glob`, `grep` をすべて自動構築します。さらに、`SandboxBackendProtocol` を実装していないバックエンドでは `execute` ツール自体がモデルへのツール一覧から自動的に除外されます。

### 4.2 ライフサイクル設計とファイル転送の2プレーン

本番運用では、Sandbox のスコープを以下の2種類から選択します。

- **Thread-scoped（デフォルト）**：会話（`thread_id`）ごとに新しい Sandbox を作成し、同一スレッド内の後続ターンでは再利用。アイドル TTL（例：`idle_ttl_seconds=3600`）で自動破棄します。データ分析ボットなど毎回クリーンな環境が欲しい用途に適します。
- **Assistant-scoped**：同一アシスタント（`assistant_id`）の全スレッドで1つの Sandbox を共有。巨大なリポジトリのクローンや依存パッケージを保持したいコーディング支援向きですが、状態肥大化を防ぐためスナップショットによる定期リセットや TTL 管理が不可欠です。

また、Sandbox とのファイル操作には **2つのプレーン** がある点を明確に区別します。

1. **Agent Filesystem Tools（エージェント側プレーン）**：LLM が推論中に呼ぶ `read_file`, `write_file`, `execute` 等。
2. **File Transfer APIs（アプリケーション側プレーン）**：ホストのアプリコードが呼ぶ `upload_files()` / `download_files()`。プロバイダのネイティブ転送 API を使い、実行前のソースコード・スキル・データ投入（Seeding）や、実行後の成果物回収（Artifact Retrieval）を行います。

### 4.3 最重要セキュリティ原則：Sandbox にシークレットを入れない

Sandbox はホスト OS を守りますが、**コンテキストインジェクション（プロンプトインジェクション）**と**ネットワーク経由のデータ持ち出し（Exfiltration）**単体では防げません。もし Sandbox 内の環境変数やファイルに API キーや DB パスワードを置くと、攻撃者に誘導されたエージェントが `env` を読み出して外部へ送信できてしまいます。

そのため、DeepAgents では以下の2方式を推奨しています。

1. **ホスト側ツールへの分離（第一選択）**：認証が必要な API 呼び出しはホスト側で動くカスタムツールとして定義し、エージェントには認証情報を一切見せない。
2. **Sandbox Auth Proxy の利用**：Sandbox からのアウトバウンド HTTP リクエストをプロキシでインターセプトし、宛先ホストに応じて `Authorization` ヘッダ等を自動注入する。これにより Sandbox 内のコード・環境変数・ログのいずれにも生シークレットが現れません。

---

## 5. まとめ

- **Permissions（`FilesystemPermission`）**は、First-Match-Wins と Permissive Default の原則に基づき、ビルトインファイルツールをパス単位で `allow` / `deny` / `interrupt` 制御する。
- **Human-in-the-Loop（`interrupt_on`）**は、`HumanInTheLoopMiddleware` と Checkpointer（+ 同一 `thread_id`）を組み合わせ、`approve` / `edit` / `reject` / `respond` による安全な承認ゲートを提供する。
- **Sandboxes** は `Sandbox as Tool` パターンと Auth Proxy（またはホスト側ツール分離）により、任意コード実行の柔軟性とシークレット保護を両立する。

次回の実践編では、実際に危険なカスタムツール `delete_production_table` を定義し、`interrupt_on` による一時停止ペイロードの構造と、`reject` フィードバックで安全に実行を阻止する一連のフローを実機出力とともに検証します。
