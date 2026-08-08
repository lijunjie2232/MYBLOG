---
title: "DeepAgents Part 2 理論編：仮想ファイルシステムと Backends（StateBackend・FilesystemBackend・StoreBackend・CompositeBackend ルーティング）"
date: 2026-07-05 11:38:01
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, Virtual Filesystem, Backends, CompositeBackend, Architecture]
lang: ja
description: "DeepAgents の中核インフラである仮想ファイルシステムとプラガブル Backends を徹底解説。BackendProtocol の設計、StateBackend・FilesystemBackend・StoreBackend の使い分け、namespace ファクトリ、そして CompositeBackend によるプレフィックスルーティングの仕組みを詳解します。"
---

# DeepAgents Part 2 理論編：仮想ファイルシステムと Backends（`StateBackend`・`FilesystemBackend`・`StoreBackend`・`CompositeBackend` ルーティング）

DeepAgents が単なるチャットボットや単純な ReAct エージェントと決定的に異なる点の一つが、**「すべてのエージェントに標準で仮想ファイルシステム（Virtual Filesystem）を与える」** というアーキテクチャです。

ファイルシステムは単にコードを読み書きするためだけの道具ではありません。DeepAgents 内部では、**大容量ツール結果の自動オフロード（`/large_tool_results/`）**、**会話履歴の退避（`/conversation_history/`）**、**Skills（`SKILL.md`）**、**Memory（`AGENTS.md`）**、そして **サブエージェント間のデータ受け渡し** まで、あらゆる機構がこの仮想ファイルシステムの上に構築されています。

本記事（Part 2 理論編）では、この仮想ファイルシステムを裏側で支える **`BackendProtocol`** と、用途別に選べる組み込み Backends（`StateBackend`・`FilesystemBackend`・`StoreBackend`・`CompositeBackend`）の設計理論を徹底解説します。

---

## 1. 仮想ファイルシステムと `BackendProtocol` の抽象化

DeepAgents の `FilesystemMiddleware` は、モデルに対して以下の統一されたファイル操作ツール群を公開します。

| ツール名 | 役割・主な仕様 |
| :--- | :--- |
| `ls` | 指定ディレクトリ内のファイル一覧とメタデータ（`size`, `modified_at`, `is_dir`）を取得。 |
| `read_file` | 行番号付きでファイル内容を取得。巨大ファイル向けに `offset` / `limit` のページネーションに対応。画像・PDF・音声・動画のマルチモーダル返却にも対応。 |
| `write_file` | 新規ファイルの作成、または既存ファイルの上書き。 |
| `edit_file` | 厳密な文字列一致による部分置換（`replace_all` モード対応）。 |
| `delete` | ファイルまたはディレクトリの再帰削除（`deepagents>=0.7`）。非対応バックエンドでは自動的にモデルから隠蔽される。 |
| `glob` | パターンマッチによるファイル検索（例：`**/*.py`）。 |
| `grep` | ファイル内容の正規表現・文字列検索（ファイル名のみ／前後行付き／件数カウント）。 |
| `execute` | シェルコマンド実行（`SandboxBackendProtocol` 実装バックエンドおよび `LocalShellBackend` のみ自動追加）。 |

### `BackendProtocol` の設計原則：例外を投げず構造化 Result を返す

これらすべてのツールは、背後に差し替え可能な **`BackendProtocol`**（[`deepagents/backends/protocol.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py)）を介して動作します。カスタムバックエンド（S3 や DB など）を実装する際も、次の 6〜7 メソッドを実装するだけで全ツールが自動的に機能します。

- `ls(path) -> LsResult`
- `read(file_path, offset=0, limit=2000) -> ReadResult`
- `write(file_path, content) -> WriteResult`
- `edit(file_path, old_string, new_string, replace_all=False) -> EditResult`
- `glob(pattern, path=None) -> GlobResult`
- `grep(pattern, path=None, glob=None) -> GrepResult`
- `delete(file_path) -> DeleteResult`（任意）

ここで重要な設計原則が **「失敗時に Python 例外を送出せず、必ず `error` フィールドを持つ構造化 Result 型（例：`ReadResult(error="Error: File '/x' not found")`）を返す」** という点です。例外でグラフ全体をクラッシュさせる代わりに、構造化されたエラーメッセージを `ToolMessage` としてモデルに返すことで、エージェント自身が「パスを間違えたので `ls` で確認し直そう」と自己修復できるようになります。

---

## 2. 組み込み Backends の比較と選定指針

DeepAgents は実行環境と永続化スコープに応じて、複数の組み込み Backend を提供しています。

| Backend | 保存先 | 永続スコープ | 主な用途・強み | 注意点 |
| :--- | :--- | :--- | :--- | :--- |
| **`StateBackend`**（デフォルト） | LangGraph Graph State (`files`) | スレッド内（Checkpointer 経由でターン跨ぎ永続） | 作業用スクラッチパッド、大容量ツール出力の自動オフロード先、サブエージェント間共有 | 別スレッドからは参照不可。グラフ実行外からの直接操作は不可。 |
| **`FilesystemBackend`** | ホストのローカルディスク | 永続（OS ファイルシステム） | ローカル開発 CLI、CI/CD パイプライン、永続ボリューム | **必ず `virtual_mode=True` を指定**。単独利用すると内部データもディスクに混ざる。Web API には不向き。 |
| **`StoreBackend`** | LangGraph `BaseStore`（InMemory / Postgres / Redis 等） | **スレッド横断**（永続） | ユーザー設定・長期記憶（`/memories/`）・組織ポリシーの保存 | **`namespace` ファクトリ指定が必須**。`store=` は `create_deep_agent` 側へ渡す。 |
| **`CompositeBackend`** | 複数 Backend の合成ルーター | パスごとに混在可能 | スレッド内作業（`StateBackend`）＋長期記憶（`StoreBackend`）＋実プロジェクト（`FilesystemBackend`）の共存 | 最長プレフィックス一致で振り分け。子 Backend へ渡す際にルートプレフィックスが剥がれる。 |
| **`LocalShellBackend`** / **Sandbox** | ホストシェル / 隔離コンテナ | 環境依存 | `execute` ツールによるテスト実行・依存インストール・CLI 操作 | `LocalShellBackend` は隔離なし（開発限定）。本番は Sandbox を利用。 |

{% mermaid %}
flowchart LR
    TOOLS["FilesystemMiddleware<br/>(ls / read / write / edit / grep / glob)"] --> ROUTER["CompositeBackend<br/>(最長プレフィックス一致ルーター)"]

    ROUTER -->|"/workspace/*<br/>(プロジェクト実ファイル)"| FS["FilesystemBackend<br/>(root_dir + virtual_mode=True)<br/>ローカルディスク"]
    ROUTER -->|"/memories/*<br/>(ユーザー長期記憶)"| ST["StoreBackend<br/>(namespace=lambda rt: ...)<br/>LangGraph BaseStore"]
    ROUTER -->|"デフォルト経路<br/>(/notes.txt, /large_tool_results/* 等)"| STATE["StateBackend<br/>LangGraph Graph State<br/>(スレッド内スクラッチパッド)"]

    style TOOLS fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style ROUTER fill:#faf5ff,stroke:#6b46c1,stroke-width:2px,color:#44337a
    style FS fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style ST fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style STATE fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

---

## 3. 各 Backend の内部動作とセキュリティ設計

### 3.1 `StateBackend`：デフォルトの作業領域とサブエージェント共有

`create_deep_agent` で `backend=` を省略した場合、内部的には `StateBackend()` が自動的に選択されます。ファイルは LangGraph のグラフ状態辞書（`state["files"]`）に格納され、Checkpointer が設定されていれば同一 `thread_id` の複数ターンにわたって保持されます。

特筆すべきは、**`StateBackend` は親エージェント（Supervisor）とサブエージェント間で共有される** 点です。サブエージェントが調査結果を `/reports/summary.md` に書き込んで終了しても、そのファイルはグラフ状態に残り、親エージェントや別のサブエージェントが即座に `read_file` で読み取れます。

### 3.2 `FilesystemBackend`：`virtual_mode=True` が不可欠な理由

`FilesystemBackend(root_dir=...)` はホスト OS の実ファイルを直接読み書きします。ここで絶対に忘れてはならないのが **`virtual_mode=True`** です。

- **`virtual_mode=False`（デフォルト）**: `root_dir` を設定していても、モデルが `../../.env` や `/etc/passwd` のような相対トラバーサル・絶対パスを指定すると、セキュリティ制限が機能しません。
- **`virtual_mode=True`**: すべてのパスを `root_dir` 配下の仮想ルート `/` として正規化し、`..`・`~`・`root_dir` 外の絶対パスおよび危険なシンボリックリンク越えをブロックします。

さらに重要な実務上のベストプラクティスとして、**`FilesystemBackend` は極力 `CompositeBackend` でラップして使う** ことが公式に推奨されています。DeepAgents は大容量ツール出力（`/large_tool_results/`）や会話履歴（`/conversation_history/`）などの内部データを自動的に **デフォルト Backend** へ書き込むため、`FilesystemBackend` を単独のデフォルトにすると、プロジェクトの Git リポジトリ直下にエージェント内部の一時ファイルが混入してしまうからです。

### 3.3 `StoreBackend` と `namespace` ファクトリ設計（v0.5+ の新 API）

スレッドを跨いでユーザー設定や学習済みナレッジを永続化したい場合は、`StoreBackend` を LangGraph の `BaseStore`（開発時は `InMemoryStore`、本番は Postgres / Redis 等）と組み合わせて使います。

ここで **v0.5 以降の重要な API 刷新** を押さえておく必要があります。

- **旧 API（非推奨）**: `backend=lambda rt: StoreBackend(rt)` のように Backend ファクトリ関数を渡し、`BackendContext` を受け取る形式。
- **新 API（現行標準）**: Backend インスタンスを直接渡し、ランタイム情報（`get_store()`, `get_runtime()`）は内部で自動解決する。`StoreBackend` には **キーワード専用引数 `namespace`** として `NamespaceFactory = Callable[[Runtime], tuple[str, ...]]` を渡す。

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

# 1. LangGraph の BaseStore を用意（本番では PostgresStore 等へ差し替え）
store = InMemoryStore()

# 2. StoreBackend にはキーワード専用引数 namespace を必ず指定してテナント/ユーザー隔離を行う
#    ※ store インスタンス自体は StoreBackend ではなく create_deep_agent(..., store=store) に渡す
backend = CompositeBackend(
    default=StateBackend(),
    routes={
        # ユーザー単位の長期記憶（Runtime の context や server_info から動的に名前空間を決定）
        "/memories/": StoreBackend(
            namespace=lambda rt: ("user", "demo"),
        ),
    },
)

agent = create_deep_agent(
    model="openai:gpt-4.1",
    backend=backend,
    store=store,  # LangGraph ランタイムが StoreBackend へ自動供給する
)
```

`Runtime` オブジェクトからは以下の 3 系統のメタデータを参照でき、マルチテナント環境での厳密なデータ隔離に活用できます。

- `rt.context`: `context_schema` で定義し、`invoke(..., context=...)` で渡したユーザー独自コンテキスト（例：`user_id`, `org_id`）
- `rt.server_info`: Agent Server 稼働時の認証ユーザー・アシスタント ID（例：`rt.server_info.user.identity`）
- `rt.execution_info`: 実行スレッド・ラン・チェックポイント ID（例：`rt.execution_info.thread_id`）

また、`namespace` の各要素には英数字・ハイフン・アンダースコア・ドット・`@`・`+`・`:`・`~` のみが許可され、ワイルドカード（`*`, `?`）は Glob インジェクション防止のため拒否されます。

---

## 4. `CompositeBackend` のプレフィックスルーティングとパス変換の仕組み

`CompositeBackend` は、単一の仮想ファイルシステムツリーの配下に複数の Backend をマウント（Mount）するルーターです。Linux のファイルシステムにおけるマウントポイントとほぼ同じ感覚で理解できます。

{% mermaid %}
sequenceDiagram
    autonumber
    participant Agent as エージェント (write_file)
    participant Comp as CompositeBackend<br/>routes={"/memories/": StoreBackend}
    participant Sub as StoreBackend<br/>namespace=("user", "demo")
    participant Store as LangGraph BaseStore

    Agent->>Comp: write("/memories/pref.txt", "lang=ja")
    Note over Comp: 最長一致ルート "/memories/" にヒット<br/>プレフィックス "/memories" を除去 → "/pref.txt"
    Comp->>Sub: write("/pref.txt", "lang=ja")
    Sub->>Store: put(namespace=("user", "demo"), key="/pref.txt", value=...)
    Store-->>Sub: OK
    Sub-->>Comp: WriteResult(path="/pref.txt")
    Note over Comp: 呼び出し元へ返すパスや ls/grep 結果には<br/>元のプレフィックス "/memories/" を復元
    Comp-->>Agent: WriteResult(path="/memories/pref.txt")
{% endmermaid %}

### ルーティングの 3 つの重要ルール

1. **最長プレフィックス一致（Longest Prefix Match）**
   `routes={"/memories/": backend_a, "/memories/projects/": backend_b}` のようにネストしたルートを定義した場合、より長いプレフィックス（`"/memories/projects/"`）が優先されます。
2. **委譲時のプレフィックス除去（Prefix Stripping）と集約時の復元**
   エージェントが `/memories/pref.txt` へ書き込むと、`CompositeBackend` はマッチしたルートプレフィックスを取り除いた **`/pref.txt`** を子バックエンド（`StoreBackend` や `FilesystemBackend`）へ渡します。一方、エージェントが `ls("/")`・`glob`・`grep` を実行した際には、全マウント先の結果を集約し、元のプレフィックス（`/memories/pref.txt`）を復元して返します。
   - **実務上の重要ポイント**: アプリケーション側の Python コードから `store.search(("user", "demo"))` のように `BaseStore` を直接参照すると、保存されているキーは `/memories/pref.txt` ではなく **`/pref.txt`** になっています（この挙動は次回の Part 2 実践編で実測確認します）。
3. **権限（`permissions=`）との連携**
   `create_deep_agent(..., permissions=[FilesystemPermission(...)])` で宣言的なアクセス制御を設定すると、権限チェックは `CompositeBackend` のルーティング **前** に仮想パスに対して評価されます（上から順に First-match-wins）。これにより「`/policies/**` は全組織共有の `StoreBackend` にマウントしつつ、エージェントからの書き込み（`"write"`）だけを拒否して読み取り専用にする」といった構成が数行で実現できます。

---

## 5. 宣言的 Permissions とカスタム Backend / Policy Hook

エンタープライズ用途では「特定ディレクトリだけ読み取り専用にしたい」「S3 や社内 API を仮想ファイルシステムとして見せたい」「書き込み前に PII チェックや監査ログを挟みたい」といった要件が出てきます。DeepAgents はこれらを 2 つのレイヤーでサポートします。

### 5.1 `FilesystemPermission` による宣言的アクセス制御

`permissions=` に渡したルールは上から順に評価され（First-match-wins）、どのルールにも一致しない操作は許可されます。

```python
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

# /memories/ はユーザー単位で読み書き可、/policies/ は組織共有かつ書き込み禁止にする構成
agent = create_deep_agent(
    model="openai:gpt-4.1",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/policies/": StoreBackend(
                namespace=lambda rt: (rt.context.org_id,),
            ),
        },
    ),
    permissions=[
        # /policies/** への write / edit / delete を拒否（read は許可）
        FilesystemPermission(
            operations=["write"],
            paths=["/policies/**"],
            mode="deny",
        ),
    ],
)
```

※ 注意：宣言的 Permissions は組み込みファイルシステムツールに対してのみ適用されます。`SandboxBackend` や `LocalShellBackend` の `execute` ツール（任意シェル実行）には適用されないため、シェル実行時はコンテナ隔離や HITL と併用してください。

### 5.2 Policy Wrapper によるカスタム検証フック

パス単位の allow/deny を超えた独自ロジック（監査ログ出力・レート制限・機密パターン検査など）を挟む場合は、`BackendProtocol` をラップする `PolicyWrapper` を実装します。失敗時は例外を投げず、`WriteResult(error=...)` を返すのが鉄則です。

```python
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)


class PolicyWrapper(BackendProtocol):
    """任意の Backend に禁止プレフィックスや監査ロジックを後付けするラッパー。"""

    def __init__(self, inner: BackendProtocol, deny_prefixes: list[str] | None = None):
        self.inner = inner
        # 末尾スラッシュを正規化してプレフィックス誤一致を防止
        self.deny_prefixes = [
            p if p.endswith("/") else p + "/" for p in (deny_prefixes or [])
        ]

    def _is_denied(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.deny_prefixes)

    def ls(self, path: str) -> LsResult:
        return self.inner.ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.inner.read(file_path, offset=offset, limit=limit)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self.inner.grep(pattern, path, glob)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.inner.glob(pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:
        # 禁止パスへの書き込みは例外ではなく構造化エラーで返し、モデルに別経路を選ばせる
        if self._is_denied(file_path):
            return WriteResult(error=f"Writes are not allowed under {file_path}")
        return self.inner.write(file_path, content)

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        if self._is_denied(file_path):
            return EditResult(error=f"Edits are not allowed under {file_path}")
        return self.inner.edit(file_path, old_string, new_string, replace_all)
```

---

## 6. まとめと次回の実践編に向けて

本記事（Part 2 理論編）では、DeepAgents の仮想ファイルシステムと Backends のアーキテクチャを解説しました。

- 全ファイルツールは **`BackendProtocol`** を介して動作し、失敗時は例外ではなく構造化 Result の `error` を返すことでモデルの自己修復を促す。
- スレッド内の **`StateBackend`**、ローカルディスクの **`FilesystemBackend`（`virtual_mode=True` 必須）**、スレッド横断の **`StoreBackend`（キーワード専用 `namespace` 必須）** を **`CompositeBackend`** で束ねるのが王道パターンである。
- `CompositeBackend` は最長プレフィックス一致で子バックエンドへ委譲する際、ルートプレフィックスを剥がして保存し、`ls` / `grep` / `glob` 時に復元する。

次回の **Part 2 実践編** では、`default=FilesystemBackend(virtual_mode=True)` と `routes={"/memories/": StoreBackend(namespace=...)}` を組み合わせた `CompositeBackend` を実際に構築し、1 回の指示で「実ディスク（`/notes.txt`）」と「LangGraph Store（`/memories/pref.txt` → キー `/pref.txt`）」へ経路分割される様子を実測検証します。
