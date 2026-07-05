---
title: "LangGraph Part 3 理論編：Persistence と Checkpointer（Thread・Super-step スナップショット・InMemorySaver / SqliteSaver / PostgresSaver）"
date: 2026-06-07 15:08:43
categories: [AI]
tags: [LangGraph, Persistence, Checkpointer, SqliteSaver, PostgresSaver, StateSnapshot, 理論]
lang: ja
description: "LangGraph の永続化基盤（Persistence）の核心である Checkpointer を、Thread・Super-step スナップショット・Pending Writes・Durability Modes・InMemorySaver / SqliteSaver / PostgresSaver の構造まで体系的に解説する。"
---

# LangGraph Part 3 理論編：Persistence と Checkpointer の内部アーキテクチャ

ここまでの Part 1・Part 2 では、単発の `graph.invoke(...)` の中で State がノード間をどう流れるかを解説してきました。しかし、実運用の AI エージェントには「複数ターンにまたがる会話記憶」「途中でプロセスが落ちた際の再開」「人間の承認待ち（Human-in-the-Loop）」「過去の特定ステップへ巻き戻す Time Travel」が不可欠です。

これらすべてを統一的に支えるのが LangGraph の **Persistence（永続化）レイヤー** です。本記事では、短期・スレッド単位の永続化を担う **Checkpointer** の内部モデルから、ストレージ実装（`InMemorySaver` / `SqliteSaver` / `PostgresSaver`）、シリアライズと暗号化までを体系的に解説します。

---

## 1. Persistence の二大システム：Checkpointer vs Store

LangGraph の永続化レイヤーは、スコープの異なる 2 つの補完的な仕組みに分かれています。

| 比較軸 | Checkpointer | Store |
| --- | --- | --- |
| **永続化対象** | グラフの State スナップショット（`StateSnapshot`） | アプリケーション定義の Key-Value データ |
| **スコープ** | **単一のスレッド（Thread-scoped）** | **複数スレッド横断（Cross-thread）** |
| **記憶の種類** | 短期記憶（セッション・会話コンテキスト） | 長期記憶（ユーザー嗜好・事実・共有知識） |
| **主な用途** | 会話継続、Human-in-the-Loop、Time Travel、耐障害性 | プロファイル記憶、社内ポリシー、横断ナレッジ |
| **アクセス方法** | `config={"configurable": {"thread_id": "..."}}` | ノード引数 `store` / `runtime.store` から直接読み書き |

本 Part 3 では、グラフ実行の基盤となる **Checkpointer** に焦点を当てます（`Store` の実践は後の Part で扱います）。

---

## 2. Checkpointer のコア概念：Thread・Checkpoint・Pending Writes

グラフを `builder.compile(checkpointer=...)` でコンパイルすると、Pregel ランタイムは **Super-step 境界ごとに State のスナップショットを自動保存** するようになります。

### 2.1 Thread（`thread_id`）

**Thread** とは、一連のグラフ実行（Run）にわたって蓄積されるチェックポイント群を束ねる一意の識別子です。Checkpointer を有効にしたグラフを呼び出す際は、必ず `config` の `configurable` 内に `thread_id` を指定します。

```python
# Checkpointer は thread_id を主キーとしてチェックポイントの保存・復元を行う
config = {"configurable": {"thread_id": "session-2026-001"}}
```

### 2.2 Checkpoint（`StateSnapshot`）と Super-step 境界

> **チェックポイントの定義**：ある時点におけるスレッドのグラフ状態スナップショットを **Checkpoint** と呼び、Python API では `StateSnapshot` オブジェクトとして表現されます。チェックポイントは **Super-step の境界ごと** に生成され、各スナップショットは直前のチェックポイントへのリンク（`parent_config`）を保持する不変の履歴チェーンを形成します。

`StateSnapshot` が持つ主要フィールドは以下の通りです。

| フィールド | 型 | 役割 |
| --- | --- | --- |
| `values` | `dict` | その時点の各 State チャネルの値 |
| `next` | `tuple[str, ...]` | 次に実行予定のノード名タプル。空 `()` はグラフ完了を意味する |
| `config` | `dict` | `thread_id`、`checkpoint_ns`、`checkpoint_id` を含む識別設定 |
| `metadata` | `dict` | `source`（`"input"` / `"loop"` / `"update"`）、`writes`、`step`（Super-step 番号） |
| `created_at` | `str` | ISO 8601 形式の生成タイムスタンプ |
| `parent_config` | `dict \| None` | 親チェックポイントの config（先頭チェックポイントでは `None`） |
| `tasks` | `tuple[PregelTask, ...]` | 当該ステップのタスク情報（`id`, `name`, `error`, `interrupts` 等） |

### 2.3 二層永続化：`checkpoints` と `checkpoint_writes`（Pending Writes）

LangGraph の耐障害性を支えているのが、Checkpointer 内部の **2 つのテーブル抽象化** です。

1. **`checkpoints` テーブル（Super-step 単位）**：Super-step 完了時に `channel_values`・`channel_versions`・`versions_seen` と親 ID を 1 行として記録します。Time Travel や `get_state_history()` が参照する正規のスナップショットです。
2. **`writes`（`checkpoint_writes`）テーブル（ノード／タスク単位）**：Super-step の実行中に各ノードが完了するたび、その出力 `(task_id, channel, value)` を先行記録（Pending Writes）します。もし同一 Super-step 内の並列ノード B が例外で失敗しても、先に成功したノード A の書き込みは保全されているため、再開時にノード A を再実行せずに済みます。

{% mermaid %}
flowchart LR
    subgraph T1["Turn 1（START -> chat -> END）"]
        CP0["step: -1<br/>source: input<br/>next: ('__start__',)"]
        CP1["step: 0<br/>source: loop<br/>next: ('chat',)"]
        CP2["step: 1<br/>source: loop<br/>next: ()"]
        CP0 -->|"parent_config"| CP1
        CP1 -->|"parent_config"| CP2
    end

    subgraph T2["Turn 2（同一 thread_id で継続）"]
        CP3["step: 2<br/>source: input<br/>next: ('__start__',)"]
        CP4["step: 3<br/>source: loop<br/>next: ('chat',)"]
        CP5["step: 4<br/>source: loop<br/>next: ()"]
        CP2 -->|"parent_config"| CP3
        CP3 -->|"parent_config"| CP4
        CP4 -->|"parent_config"| CP5
    end

    PW[("writes テーブル<br/>ノード単位の Pending Writes")] -.->|"Super-step 中の障害復旧"| CP1
    PW -.->|"Super-step 中の障害復旧"| CP4

    style CP0 fill:#eceff1,stroke:#546e7a,color:#263238
    style CP1 fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style CP2 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style CP3 fill:#eceff1,stroke:#546e7a,color:#263238
    style CP4 fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style CP5 fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    style PW fill:#fffde7,stroke:#f9a825,stroke-dasharray: 5 5,color:#e65100
{% endmermaid %}

### 2.4 `checkpoint_ns`（チェックポイント名前空間）

親グラフのチェックポイントは `checkpoint_ns == ""`（空文字）を持ちます。サブグラフが呼び出された場合は `"node_name:uuid"`（入れ子では `"outer:uuid|inner:uuid"`）のように独立した名前空間が付与され、親子の状態が衝突しないよう分離されます。

---

## 3. スレッドの継続・検査・更新 API

既存スレッドに対して `invoke()` を呼ぶとき、**何を渡すか**によって実行モードが明確に分かれます。

> **スレッド継続の重要原則**：同じ `thread_id` を再利用するだけでは「中断の再開」にはなりません。通常の入力辞書を渡すと「保存済み State へ Reducer で新入力をマージした上で `START` から新規ターンを開始」します。エラー復旧や履歴リプレイには `None`、割り込み再開には `Command(resume=...)` を使い分けます。

| 目的 | 呼び出し方 | ランタイムの挙動 |
| --- | --- | --- |
| **新規ターン入力** | `graph.invoke(input_dict, config)` | 保存済み State へ入力を Reducer 適用し、`START` から新規実行 |
| **エラー後の復旧** | `graph.invoke(None, config)` | 最新チェックポイントを読み込み、未完了タスクのみ再実行 |
| **`interrupt()` 応答** | `graph.invoke(Command(resume=val), config)` | 割り込みの戻り値として `val` を注入し、中断ノードを先頭から再実行 |
| **過去時点のリプレイ** | `graph.invoke(None, snapshot.config)` | 特定 `checkpoint_id` 以降のノードを再実行 |

また、状態検査と手動更新には以下のメソッドを使います。

- `graph.get_state(config)`：最新（または `checkpoint_id` 指定）の `StateSnapshot` を 1 件取得。
- `graph.get_state_history(config)`：指定スレッドの全 `StateSnapshot` を**最新順（降順）**のイテレータで取得。
- `graph.update_state(config, values, as_node=...)`：元のチェックポイントを破壊せず、Reducer を通して更新した**新しいフォークチェックポイント**を作成。
- `checkpointer.list(None)`：全スレッド横断でチェックポイントを列挙（`{}` ではなく必ず `None` を渡す）。

---

## 4. Durability Modes（永続化の一貫性モード）

LangGraph では、実行時のレイテンシとクラッシュ耐性のトレードオフを制御する 3 段階の **Durability Mode** を `graph.invoke(..., durability="sync")` のように指定できます。

| モード | 永続化タイミング | 特徴とトレードオフ |
| --- | --- | --- |
| `"exit"` | グラフ終了時（正常・エラー・interrupt）のみ | 最高速だが、実行途中のプロセスクラッシュで中間状態が失われる |
| `"async"` | 次ステップと並行して非同期書き込み | 性能と耐久性のバランスが良い（クラッシュ直前の数ミリ秒のみ喪失リスク） |
| `"sync"` | 次ステップ開始前に同期コミット完了を待機 | 最高の一貫性・耐障害性（ステップごとにストレージ I/O 待ちが発生） |

---

## 5. Checkpointer バックエンドとシリアライザ・暗号化

すべての公式・カスタム Checkpointer は共通インターフェース **`BaseCheckpointSaver`**（[langgraph/checkpoint/base](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py)）を実装し、`put` / `put_writes` / `get_tuple` / `list` / `delete_thread`（および非同期版 `aput` 等）を提供します。

{% mermaid %}
flowchart TB
    G["CompiledStateGraph<br/>(Pregel ランタイム)"] --> IF["BaseCheckpointSaver 契約<br/>put / put_writes / get_tuple / list / delete_thread"]
    IF --> SER["SerializerProtocol<br/>JsonPlusSerializer (ormsgpack/JSON)<br/>または EncryptedSerializer (AES)"]

    SER --> M["InMemorySaver<br/>RAM 保存（テスト・開発用）"]
    SER --> S["SqliteSaver / AsyncSqliteSaver<br/>ローカル SQLite ファイル（検証・単一ノード用）"]
    SER --> P["PostgresSaver / AsyncPostgresSaver<br/>PostgreSQL（本番・分散・高可用性）"]

    style G fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style IF fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
    style SER fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
    style M fill:#eceff1,stroke:#546e7a,color:#263238
    style S fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style P fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
{% endmermaid %}

### バックエンドの選定基準

1. **`InMemorySaver`（`langgraph-checkpoint` 同梱）**
   RAM 上に保持するためセットアップ不要・最速ですが、**プロセス再起動で全チェックポイントが消失**します。ユニットテストやプロトタイピング専用です。
2. **`SqliteSaver` / `AsyncSqliteSaver`（`langgraph-checkpoint-sqlite`）**
   ローカルの SQLite ファイルへ永続化します。プロセスを跨いだ状態復元や SQL による直接検査ができ、ローカル開発や単一サーバーの軽量ワークフローに最適です。
3. **`PostgresSaver` / `AsyncPostgresSaver`（`langgraph-checkpoint-postgres`）**
   コネクションプールとトランザクション管理を備えた本番環境向け実装です。初回利用時には `checkpointer.setup()` を呼んでテーブルとインデックスを作成します。

### シリアライズと保存時暗号化（Encryption at Rest）

デフォルトの **`JsonPlusSerializer`** は `ormsgpack` と JSON を用いて LangChain の Message オブジェクトや `datetime`、`Enum` 等を高速にシリアライズします。Pandas DataFrame など未対応のオブジェクトを扱う場合は `JsonPlusSerializer(pickle_fallback=True)` が利用できます。さらに個人情報（PII）を含む会話ログを保護したい場合は、環境変数 `LANGGRAPH_AES_KEY` を利用する **`EncryptedSerializer.from_pycryptodome_aes()`** を `serde=` 引数へ渡すだけで透過的に AES 暗号化できます。

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# 環境変数 LANGGRAPH_AES_KEY を自動で読み込む AES 暗号化シリアライザを生成
serde = EncryptedSerializer.from_pycryptodome_aes()

# SqliteSaver へ暗号化シリアライザを注入することで、DB 内の全チェックポイントを暗号化保存する
conn = sqlite3.connect("encrypted_checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn, serde=serde)
```

### 本番運用で意識すべきスケーリング課題

- **`thread_id` の長さ制限**：`PostgresSaver` 等ではカラム長制限があるため、`thread_id` は 255 文字以内（UUIDv7 等）に収めます。
- **チェックポイントの肥大化対策**：長い会話では毎 Super-step で全チャネル値が保存されるためストレージが増加します。定期的なプルーニング（古いスレッドの `delete_thread()`）や、増分のみを保存するベータ機能 `DeltaChannel`（`langgraph>=1.2`）の検討が有効です。

---

## 6. まとめ

本記事では、LangGraph の永続化の中核である Checkpointer について以下を整理しました。

- **Checkpointer vs Store**：短期（Thread スコープ）と長期（Cross-Thread スコープ）の明確な役割分担。
- **二層永続化モデル**：Super-step 境界の `checkpoints` テーブルと、ノード単位の障害復旧を支える `writes`（Pending Writes）テーブル。
- **スレッド操作 API と Durability Modes**：`invoke(input / None / Command(resume=...))` の使い分け、`get_state_history()`、`sync` / `async` / `exit` モード。
- **実装選定とセキュリティ**：`InMemorySaver` → `SqliteSaver` → `PostgresSaver` の段階的採用と `EncryptedSerializer`。

次回の **Part 3 実践編** では、`SqliteSaver` と `MessagesState` を使い、`MiniCPM-V-4.7-35B-A3B` による 2 ターンの会話永続化と、生成される **6 つのチェックポイント履歴（`step -1` 〜 `step 4`）** の詳細検査を実測データ付きで解説します。

---

### 参考リンク
- [Persistence（公式ドキュメント）](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Checkpointers（公式ドキュメント）](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph ソースコード（BaseCheckpointSaver）](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py)
