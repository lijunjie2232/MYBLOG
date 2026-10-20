---
title: DeepAgents ソースコード解析 Part 2：BackendProtocol と各種 Backend 実装の内部構造
date: 2026-09-22 21:44:52
categories: [AI]
tags: [Python, LLM, AI Agent, DeepAgents, LangGraph, ソースコード解析]
lang: ja
description: DeepAgents のストレージ・実行抽象化層 BackendProtocol と SandboxBackendProtocol を解析。StateBackend の Pregel 内部チャネル直読み書き、FilesystemBackend の仮想ルートと symlink 対策、StoreBackend の名前空間検証とページネーション、CompositeBackend の最長一致ルーティングとバッチ処理までソースコードから紐解く。
---

# DeepAgents ソースコード解析 Part 2：`BackendProtocol` と各種 Backend 実装の内部構造

DeepAgents では、エージェントが扱うファイル（作業ファイル、メモリ、スキル定義、巨大ツール出力の退避先など）の保存先を **Backend** として完全に抽象化しています。これにより、ローカルディスク・LangGraph のインメモリ State・永続的な `BaseStore`・リモートサンドボックスを、まったく同じツールインターフェース（`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`）で操作できます。

Part 2 では、[`backends/`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/) 配下の 5 つのコアモジュールを読み解きます。

- [`protocol.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py)：`BackendProtocol` / `SandboxBackendProtocol` と型安全な結果データクラス
- [`state.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/state.py)：`StateBackend`（Pregel の `CONFIG_KEY_READ` / `CONFIG_KEY_SEND` による状態直結）
- [`filesystem.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/filesystem.py)：`FilesystemBackend`（`virtual_mode` のパストラバーサル防御と `O_NOFOLLOW`）
- [`store.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/store.py)：`StoreBackend`（クロススレッド永続化・名前空間バリデーション・自動ページネーション）
- [`composite.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/composite.py)：`CompositeBackend`（最長プレフィックス一致ルーティングと横断検索・バッチ分割）

---

## 1. `BackendProtocol` と `SandboxBackendProtocol` の階層設計

[`protocol.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py) が定義するクラス階層と結果オブジェクトの関係は以下の通りです。

{% mermaid %}
classDiagram
    class BackendProtocol {
        <<ABC>>
        +ls(path) LsResult
        +read(file_path, offset, limit) ReadResult
        +write(file_path, content) WriteResult
        +edit(file_path, old_string, new_string, replace_all) EditResult
        +delete(file_path) DeleteResult
        +grep(pattern, path, glob, max_count) GrepResult
        +glob(pattern, path) GlobResult
        +upload_files(files) list~FileUploadResponse~
        +download_files(paths) list~FileDownloadResponse~
    }

    class SandboxBackendProtocol {
        <<ABC>>
        +id str
        +execute(command, timeout) ExecuteResponse
        +aexecute(command, timeout) ExecuteResponse
    }

    class StateBackend {
        +CONFIG_KEY_READ / CONFIG_KEY_SEND
    }
    class FilesystemBackend {
        +cwd: Path
        +virtual_mode: bool
    }
    class StoreBackend {
        +_store: BaseStore
        +_namespace: NamespaceFactory
    }
    class CompositeBackend {
        +default: BackendProtocol
        +sorted_routes: list
    }

    BackendProtocol <|-- SandboxBackendProtocol
    BackendProtocol <|-- StateBackend
    BackendProtocol <|-- FilesystemBackend
    BackendProtocol <|-- StoreBackend
    BackendProtocol <|-- CompositeBackend
{% endmermaid %}

### 1.1 なぜ `grep` / `glob` が `SandboxBackendProtocol` ではなく `BackendProtocol` にあるのか

`protocol.py` の docstring には重要な設計判断が記されています。

1. `StateBackend` や `StoreBackend` にはシェルプロセスが存在しないため、Pure Python で `grep` / `glob` を提供する必要がある。
2. シェルを持つサンドボックス環境でも、単なる `execute("grep ...")` のラッパーではない。**正規表現ではなくリテラル一致**を強制し、構造化された `GrepResult` / `GlobResult`、`max_count` による上限打ち切り、さらに Middleware 層の `permissions` チェックを通す契約になっている。

### 1.2 `ReadResult.__post_init__` によるページネーション不変条件の強制

[`ReadResult`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py#L203-L275) は、Backend が返却する行範囲メタデータの整合性をコンストラクタ時点で厳密に検証します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py
@dataclass
class ReadResult:
    error: str | None = None
    file_data: FileData | None = None
    total_lines: int | None = None
    start_line: int | None = None      # 1-indexed 開始行
    end_line: int | None = None        # 1-indexed 終了行
    next_offset: int | None = None     # 0-indexed 次回読み込み開始オフセット
    no_lines_requested: bool = False   # limit <= 0 でファイル未検査のショートサーキットを示すフラグ

    def __post_init__(self) -> None:
        # start_line と end_line は必ずペアで設定されるか、両方 None でなければならない
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("ReadResult.start_line and end_line must be set together or both left unset")
        # limit <= 0 で未検査の場合、エラーやページネーション情報を混在させない
        if self.no_lines_requested and (
            self.error is not None or self.start_line is not None
            or self.next_offset is not None or self.total_lines is not None
        ):
            raise ValueError("ReadResult.no_lines_requested ...")

        # ウィンドウが存在する場合の数値整合性チェック
        if self.start_line is not None and self.end_line is not None:
            if self.start_line < 1 or self.end_line < self.start_line:
                raise ValueError(...)
            if self.total_lines is not None and self.total_lines < self.end_line:
                raise ValueError(...)
            # end_line は 1-indexed、next_offset は 0-indexed なので必ず next_offset == end_line になる
            # Backend が誤った next_offset を返して未読行をサイレントにスキップするバグを防ぐ
            if self.next_offset is not None and self.next_offset != self.end_line:
                raise ValueError(...)
```

また、非同期メソッド（`als`, `aread`, `agrep` など）はデフォルトで `asyncio.to_thread()` に委譲されますが、`agrep` だけは [`ASYNC_GREP_TIMEOUT`（35 秒）](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/protocol.py#L558-L596) のタイムアウト付きラッパーを持ち、ハングした検索がイベントループを永遠に待たせないよう保護されています。

---

## 2. `StateBackend`：Pregel 内部チャネルを使った Read-Your-Writes

[`StateBackend`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/state.py) は `create_deep_agent` のデフォルト Backend であり、ファイルを LangGraph のグラフ状態（`files` チャネル）に保存します。

興味深いのは、`StateBackend()` の初期化時にグラフ状態やランタイムを一切受け取らず、**呼び出し時に LangGraph の `get_config()` から Pregel の内部コールバックを取り出している点**です。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/state.py
def _read_files(self) -> dict[str, Any]:
    config = self._get_config()
    # LangGraph Pregel が注入する内部読み取り関数を取得
    read = config["configurable"][CONFIG_KEY_READ]
    # fresh=True を渡すことで、同一 superstep 内の未コミット書き込みもリデューサ経由で反映された最新値を読む
    # （例: 1 回のツール実行内で write した直後に read しても自分の書き込みが見える = Read-Your-Writes）
    fresh = True
    return read("files", fresh) or {}

def _send_files_update(self, update: dict[str, Any]) -> None:
    config = self._get_config()
    # LangGraph Pregel のチャネル送信関数を取得
    send = config["configurable"][CONFIG_KEY_SEND]
    # files チャネルは dict マージ用リデューサを持つため、変更・削除したキーの差分だけを送ればよい
    send([("files", update)])
```

### 削除（`delete`）の表現方法

`files` チャネルが差分マージ型リデューサであるため、[`StateBackend.delete`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/state.py#L254-L278) は対象パスと配下のプレフィックス（`base + "/"`）に一致する全キーに対し、値として `None`（トゥームストーン）を送出します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/state.py
base = file_path.rstrip("/")
prefix = base + "/"
to_delete = [key for key in files if key == base or key.startswith(prefix)]
if not to_delete:
    return DeleteResult(error=f"Error: File '{file_path}' not found")

# None を送ることで files チャネルのリデューサが該当キーを削除する
self._send_files_update(dict.fromkeys(to_delete, None))
return DeleteResult(path=file_path)
```

---

## 3. `FilesystemBackend`：`virtual_mode` とセキュリティガード

[`FilesystemBackend`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/filesystem.py) はホスト OS の実ファイルシステムへ直接アクセスします。デフォルトで `virtual_mode=True` が有効になっており、`root_dir` を仮想ルート `/` に見立てて以下の防御を実装しています。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/filesystem.py
def _resolve_path(self, key: str) -> Path:
    if self.virtual_mode:
        vpath = key if key.startswith("/") else "/" + key
        # パストラバーサル（..）およびホームディレクトリ展開（~）を明示的にブロック
        if ".." in vpath or vpath.startswith("~"):
            raise ValueError("Path traversal not allowed")
        full = (self.cwd / vpath.lstrip("/")).resolve()
        try:
            # シンボリックリンク解決後の実パスも root_dir（self.cwd）配下にあることを検証
            full.relative_to(self.cwd)
        except ValueError:
            raise ValueError(f"Path:{full} outside root directory: {self.cwd}") from None
        # シンボリックリンクの循環参照（ELOOP）を検知
        _raise_if_symlink_loop(full)
        return full

    # virtual_mode=False の場合はホスト絶対パスをそのまま許可（ローカル開発 CLI 向け）
    path = Path(key)
    if path.is_absolute():
        _raise_if_symlink_loop(path)
        return path
    resolved = (self.cwd / path).resolve()
    _raise_if_symlink_loop(resolved)
    return resolved
```

さらにファイル読み込み（[`read` L470](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/filesystem.py#L470)）では、通常の `open()` ではなく低レベルの `os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))` を使用してシンボリックリンク差し替え（TOCTOU）を抑止し、動画ファイルに対しては `MAX_VIDEO_INPUT_BYTES` でメモリ枯渇を防止しています。また `grep` 実行時は `shutil.which("rg")` で ripgrep をキャッシュ探索し、存在しない場合のみ Pure Python フォールバックへ切り替える高速化も行っています。

---

## 4. `StoreBackend`：クロススレッド永続化と名前空間の安全性

[`StoreBackend`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/store.py) は LangGraph の `BaseStore`（PostgreSQL やメモリ上の KV ストア）にファイルを永続化し、スレッドを跨いだ長期メモリ（例：ユーザー設定や `/memories/`）を実現します。

### 4.1 名前空間インジェクション防止と NUL バイト対策

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/store.py
# 許可する文字を英数字とユーザーID等で一般的な記号に限定し、'*' や '?' によるワイルドカード注入を遮断
_NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")

def _encode_upload(content: bytes) -> tuple[str, str]:
    # Postgres JSONB バックエンドは NUL バイト (\x00) を含むテキストを保存できないため、
    # UTF-8 として妥当でも \x00 を含む場合はバイナリ扱い（base64）にフォールバックする
    if b"\x00" not in content:
        try:
            return content.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            pass
    return base64.standard_b64encode(content).decode("ascii"), "base64"
```

### 4.2 自動ページネーションとネイティブ Async

`BaseStore.search()` はデフォルトで件数上限があるため、[`_search_store_paginated`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/store.py#L251-L296) が `page_size=100` 単位で `offset` を進めながら全件取得します。また `StateBackend` と異なり、`StoreBackend` は `aread` / `awrite` で `store.aget()` / `store.aput()` を直接呼ぶネイティブ非同期実装をオーバーライドしており、スレッドプールを介さず非同期 I/O の性能を引き出します。

---

## 5. `CompositeBackend`：パスプレフィックスによるハイブリッドルーティング

実運用のエージェントでは「一時作業ファイルは `StateBackend`（またはサンドボックス）、長期記憶は `StoreBackend`、スキル定義は `FilesystemBackend`」のように複数ストレージを組み合わせたいケースが多々あります。これを実現するのが [`CompositeBackend`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/composite.py) です。

{% mermaid %}
flowchart LR
    REQ["ツール呼び出し<br/>read / write / ls / grep / glob"] --> ROUTER["_route_for_path()<br/>sorted_routes を最長一致で走査"]
    ROUTER -- "/memories/note.md<br/>-> strip: /note.md" --> B1["StoreBackend<br/>(永続メモリ)"]
    ROUTER -- "/skills/foo/SKILL.md<br/>-> strip: /foo/SKILL.md" --> B2["FilesystemBackend<br/>(ローカル読み取り)"]
    ROUTER -- "/workspace/app.py<br/>-> 未一致" --> B3["default: StateBackend<br/>(または SandboxBackend)"]

    B1 -- "返却パスに /memories を再付与" --> MERGE["結果マージ & ソート<br/>(ls '/', grep '/', glob '/')"]
    B2 -- "返却パスに /skills を再付与" --> MERGE
    B3 --> MERGE

    style REQ fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style ROUTER fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style MERGE fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

### 5.1 最長一致ルーティングと境界チェック（`_route_for_path`）

初期化時に `self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)` と**プレフィックス長降順**にソートしておくことで、`/memories/private/` と `/memories/` のような入れ子ルートを正しく解決します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/backends/composite.py
def _route_for_path(
    *,
    default: BackendProtocol,
    sorted_routes: list[tuple[str, BackendProtocol]],
    path: str,
) -> tuple[BackendProtocol, str, str | None]:
    for route_prefix, backend in sorted_routes:
        prefix_no_slash = route_prefix.rstrip("/")
        # ルートディレクトリそのもの（例: "/memories"）が指定された場合はバックエンド側の "/" に変換
        if path == prefix_no_slash:
            return backend, "/", route_prefix

        # "/memories_backup" が "/memories" に誤マッチしないよう、末尾 '/' を付けて境界を強制
        normalized_prefix = route_prefix if route_prefix.endswith("/") else f"{route_prefix}/"
        if path.startswith(normalized_prefix):
            suffix = path[len(normalized_prefix) :]
            backend_path = f"/{suffix}" if suffix else "/"
            return backend, backend_path, route_prefix
    # どのルートにも一致しなければ default バックエンドへ
    return default, path, None
```

### 5.2 横断検索（`grep` / `glob`）とバッチ転送の最適化

- **グローバル `max_count` の予算配分（`grep`）**：`path="/"` で全バックエンドを横断検索する際、`_remaining_grep_budget(max_count, len(all_matches))` で残り件数を計算し、上限に達した時点で後続ルートをショートサーキット（`truncated=True`）します。
- **ルートアンカー付き `glob` のスキップ**：`/*.py` のように先頭 `/` でルート直下に固定されたパターンは、マウント先のサブパス（`/memories/foo.py` 等）には原理上マッチしないため、`_route_glob_pattern` が `None` を返して不要なバックエンド呼び出しをスキップします。
- **`upload_files` / `download_files` のバックエンド別バッチ化**：複数ファイルの一括転送では、`backend_batches` にターゲット Backend ごとにグルーピングして 1 回ずつ呼び出し、元のインデックス順（`orig_idx`）に復元して返却します。
- **`execute` は常に `self.default` へ委譲**：コマンド実行はパス単位でルーティングできないため、`isinstance(self.default, SandboxBackendProtocol)` を確認してデフォルトバックエンドに転送します。

---

## まとめ

DeepAgents の Backend 層は、単なる CRUD インターフェースではなく、エージェント運用で実際に踏むエッジケース（Read-Your-Writes、シンボリックリンク脱出、Postgres JSONB の NUL バイト制約、横断検索の予算制御）をプロトコルと各実装に丁寧に封じ込めています。

次回 Part 3 では、これらの Backend を LLM 向けの安全なツール群へと変換し、巨大ツール結果の自動退避（Offloading）と切り詰め（Truncation）を担う **`FilesystemMiddleware`** を徹底解析します。
