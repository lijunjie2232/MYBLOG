---
title: "LangGraph Part 3 実践編：SqliteSaver による会話状態の永続化とチェックポイント履歴の検査"
date: 2026-06-12 01:49:54
categories: [AI]
tags: [LangGraph, Checkpointer, SqliteSaver, MessagesState, Persistence, MiniCPM, 実践]
lang: ja
description: "LangGraph Part 3 理論編に基づき、SqliteSaver と MessagesState・MiniCPM-V-4.7-35B-A3B を組み合わせて 2 ターン会話の永続化と 6 チェックポイント履歴の検査を実装・実測する。"
---

# LangGraph Part 3 実践編：`SqliteSaver` による会話状態の永続化とチェックポイント履歴の検査

本記事では、[LangGraph Part 3 理論編](/2026/06/07/langgraph_03_theory/) で解説した **Checkpointer・Thread・Super-step スナップショット** の仕組みを、ローカル SQLite 永続化バックエンド **`SqliteSaver`** と **`MiniCPM-V-4.7-35B-A3B`** を使って実践検証します。

1 ターン目で「私のプロジェクト名は Project Orion です」と伝え、2 ターン目で同じ `thread_id` を指定して「私のプロジェクト名は何でしたか？」と質問したとき、グラフがどのように SQLite から過去の状態を復元し、最終的に**なぜ 6 個のチェックポイント（`latest_step = 4`）が生成されるのか**を実測データから完全に解き明かします。

---

## 1. 理論編の振り返りと実践シナリオ

Part 3 理論編では、チェックポイントの生成規則とスレッド継続の原則について次のように整理しました。

> **チェックポイントの定義**：ある時点におけるスレッドのグラフ状態スナップショットを **Checkpoint** と呼び、Python API では `StateSnapshot` オブジェクトとして表現されます。チェックポイントは **Super-step の境界ごと** に生成され、各スナップショットは直前のチェックポイントへのリンク（`parent_config`）を保持する不変の履歴チェーンを形成します。

> **スレッド継続の重要原則**：同じ `thread_id` を再利用するだけでは「中断の再開」にはなりません。通常の入力辞書を渡すと「保存済み State へ Reducer で新入力をマージした上で `START` から新規ターンを開始」します。

今回の実践ラボでは、組み込みの **`MessagesState`**（`messages: Annotated[list[AnyMessage], add_messages]`）と 1 ノード構成（`START -> chat -> END`）のグラフに `SqliteSaver` を接続し、2 ターンの会話を通して SQLite 上のチェックポイントチェーンを検証します。

{% mermaid %}
flowchart LR
    subgraph Turn1["Turn 1：プロジェクト名の記憶"]
        S0["step -1 (input)<br/>next: ('__start__',)"] --> S1["step 0 (loop)<br/>HumanMsg #1 反映<br/>next: ('chat',)"]
        S1 --> S2["step 1 (loop)<br/>AIMsg #1 追記<br/>next: ()"]
    end

    subgraph Turn2["Turn 2：同一 thread_id で想起"]
        S3["step 2 (input)<br/>新入力受付<br/>next: ('__start__',)"] --> S4["step 3 (loop)<br/>HumanMsg #2 連結<br/>next: ('chat',)"]
        S4 --> S5["step 4 (loop)<br/>AIMsg #2 ('Project Orion')<br/>next: ()"]
    end

    S2 -->|"SQLite から自動復元"| S3

    style S0 fill:#eceff1,stroke:#546e7a,color:#263238
    style S1 fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style S2 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style S3 fill:#eceff1,stroke:#546e7a,color:#263238
    style S4 fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style S5 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
{% endmermaid %}

---

## 2. 完全な実装コード（`SqliteSaver` + `MessagesState`）

事前パッケージとして `langgraph-checkpoint-sqlite` が必要です。API キーとベース URL は必ず環境変数（`BASE_URL`, `API_KEY`）から取得します。

```python
import json
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# .env から環境変数をロード（シークレットのハードコードを避ける）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"

def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """MiniCPM-V-4.7-35B-A3B 用の ChatOpenAI クライアントを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,        # 使用モデル名
        base_url=BASE_URL,       # 環境変数から読み込んだエンドポイント URL
        api_key=API_KEY,         # 環境変数から読み込んだ API キー
        temperature=temperature, # 再現性確保のため 0.0
        max_tokens=256,          # 応答トークン上限
        timeout=30,              # タイムアウト秒数
        max_retries=1,           # クライアント側リトライ回数
    )

def run_lab3() -> dict:
    llm = make_llm(temperature=0.0)

    # 1. 一時ディレクトリに検証用の SQLite ファイルパスを準備（毎回クリーン状態で検証）
    db_path = os.path.join(tempfile.gettempdir(), "lg_lab3_checkpointer.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)

    # 2. 会話ノード：MessagesState 内の全メッセージ履歴を LLM へ渡し、新規応答のみを返す
    def chat_node(state: MessagesState) -> dict:
        resp = llm.invoke(state["messages"])
        # MessagesState は add_messages Reducer を持つため、新規 AIMessage 1 件だけ返せば追記される
        return {"messages": [resp]}

    # 3. SqliteSaver.from_conn_string() はコンテキストマネージャとして開き、安全に DB 接続を管理する
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        # compile(checkpointer=checkpointer) で永続化レイヤーをグラフへ注入
        graph = (
            StateGraph(MessagesState)
            .add_node("chat", chat_node)
            .add_edge(START, "chat")
            .add_edge("chat", END)
            .compile(checkpointer=checkpointer)
        )

        # 4. 同一スレッドを識別するための configurable.thread_id を定義
        cfg = {"configurable": {"thread_id": "sqlite-thread-01"}}

        # Turn 1：プロジェクト名を記憶させる（この時点の履歴が SQLite へ永続化される）
        graph.invoke(
            {"messages": [HumanMessage(content="私のプロジェクト名は Project Orion です。覚えてください。")]},
            config=cfg,
        )

        # Turn 2：同じ thread_id で別の質問を送信（過去履歴は自動ロードされ add_messages で結合される）
        r2 = graph.invoke(
            {"messages": [HumanMessage(content="私のプロジェクト名は何でしたか？短く答えてください。")]},
            config=cfg,
        )

        # 5. 当該スレッドに蓄積された全チェックポイント履歴（最新順＝降順）を取得して検査
        history = list(graph.get_state_history(cfg))

        return {
            "reply": r2["messages"][-1].content,       # Turn 2 の最新 AI 応答テキスト
            "checkpoint_count": len(history),          # 2 ターンで生成された総チェックポイント数
            "latest_step": history[0].metadata.get("step"),  # 最新スナップショットの step 番号
        }

if __name__ == "__main__":
    output = run_lab3()
    print(json.dumps(output, ensure_ascii=False, indent=2))
```

---

## 3. 実測実行結果と「6 チェックポイント・`latest_step: 4`」の理由

上記のコードを `MiniCPM-V-4.7-35B-A3B` で実行した実測出力（実行時間 `3.01 秒`）は以下の通りです。

```json
{
  "reply": "Project Orion",
  "checkpoint_count": 6,
  "latest_step": 4
}
```

### なぜ 2 ターンの会話で `checkpoint_count` が 6、`latest_step` が 4 になるのか？

1 ノード（`START -> chat -> END`）のグラフを 2 回 `invoke()` しただけなのに、チェックポイントが 2 個ではなく **6 個** 生成され、しかも最後の `step` が 6 ではなく **4** になっています。この内訳を理解すると、LangGraph のチェックポイント設計が腑に落ちます。

| `history` インデックス | `metadata["step"]` | `metadata["source"]` | `next` | 保存されている状態の意味 |
| --- | --- | --- | --- | --- |
| `history[5]`（最古） | `-1` | `"input"` | `("__start__",)` | Turn 1 開始直前の初期入力受付チェックポイント |
| `history[4]` | `0` | `"loop"` | `("chat",)` | Turn 1 の `HumanMessage` が State へ反映され `chat` 実行待ち |
| `history[3]` | `1` | `"loop"` | `()` | Turn 1 の `chat` 完了（`AIMessage` 追記済み・Turn 1 終了） |
| `history[2]` | `2` | `"input"` | `("__start__",)` | Turn 2 の新規入力受付チェックポイント（`step` が連番継続） |
| `history[1]` | `3` | `"loop"` | `("chat",)` | Turn 2 の `HumanMessage` が `add_messages` で連結され `chat` 実行待ち |
| `history[0]`（最新） | `4` | `"loop"` | `()` | Turn 2 の `chat` 完了（`"Project Orion"` が追記され Turn 2 終了） |

ポイントは 3 つあります。

1. **初回 `step` は `-1` から始まる**：`START`（内部名 `__start__`）へ入力を受け付ける瞬間が `step: -1`（`source: "input"`）として記録されるため、1 ターンあたり **3 チェックポイント**（`-1, 0, 1`）が作られます。
2. **2 ターン目は `step: 2` から連番で継続する**：同一の `thread_id` を使うと、Turn 1 の最終ステップ（`step: 1`）を `parent_config` として引き継ぎ、Turn 2 で `step: 2, 3, 4` の 3 チェックポイントが積み上がります。結果として合計 `3 + 3 = 6` 個、最新ステップは `4` になります。
3. **`add_messages` による自動コンテキスト復元**：Turn 2 の `invoke()` では質問 1 通しか渡していませんが、ランタイムが SQLite の `step: 1` スナップショットから Turn 1 の 2 通（Human + AI）を復元し、`add_messages` Reducer で Turn 2 の HumanMessage を末尾へ連結した合計 3 通を `chat_node` へ渡したため、LLM は正確に `"Project Orion"` と回答できました。

{% mermaid %}
sequenceDiagram
    participant App as アプリケーション
    participant Graph as CompiledStateGraph
    participant DB as SqliteSaver (SQLite DB)
    participant LLM as MiniCPM-V-4.7-35B-A3B

    App->>Graph: Turn 1 invoke("Project Orion...", thread_id="sqlite-thread-01")
    Graph->>DB: step -1 (input) & step 0 (HumanMsg #1) を保存
    Graph->>LLM: [HumanMsg #1]
    LLM-->>Graph: AIMsg #1 ("承知しました...")
    Graph->>DB: step 1 (HumanMsg #1, AIMsg #1) を保存
    Graph-->>App: Turn 1 完了

    App->>Graph: Turn 2 invoke("何でしたか？", thread_id="sqlite-thread-01")
    Graph->>DB: thread_id="sqlite-thread-01" の最新(step 1)をロード
    DB-->>Graph: [HumanMsg #1, AIMsg #1]
    Note over Graph: add_messages Reducer で HumanMsg #2 を連結<br/>step 2 (input) & step 3 (loop) を保存
    Graph->>LLM: [HumanMsg #1, AIMsg #1, HumanMsg #2]
    LLM-->>Graph: AIMsg #2 ("Project Orion")
    Graph->>DB: step 4 (全4メッセージ) を保存
    Graph-->>App: Turn 2 完了 ("Project Orion")

    App->>Graph: get_state_history(cfg)
    Graph->>DB: thread_id の全履歴を降順取得
    DB-->>App: 6 件の StateSnapshot（step 4 -> 3 -> 2 -> 1 -> 0 -> -1）
{% endmermaid %}

---

## 4. 実践的なチェックポイント履歴検査レシピ

デバッグや運用監視で `get_state_history()` や `checkpointer.list(None)` を使う際の定番スニペットを示します。

```python
# 1. 単一スレッドの全チェックポイントを最新順に表示する
for snap in graph.get_state_history(cfg):
    step = snap.metadata.get("step")
    source = snap.metadata.get("source")
    cp_id = snap.config["configurable"]["checkpoint_id"]
    msg_count = len(snap.values.get("messages", []))
    print(f"step={step:>2} | source={source:<6} | next={snap.next} | msgs={msg_count} | id={cp_id[:8]}...")

# 2. 特定の条件（例：chat ノード実行直前）のスナップショットを抽出する
before_chat = [s for s in graph.get_state_history(cfg) if s.next == ("chat",)]

# 3. DB 内の全スレッド横断でチェックポイントを列挙する（必ず None を渡す）
all_thread_ids = {
    cp_tuple.config["configurable"]["thread_id"]
    for cp_tuple in checkpointer.list(None)
}
```

---

## 5. 注意点・ハマりどころ（Pay Attention）

`SqliteSaver` をはじめとする Checkpointer 運用で頻出する落とし穴を 5 点まとめます。

1. **`SqliteSaver.from_conn_string(db_path)` は必ず `with` 文で開く**
   `SqliteSaver.from_conn_string()` は直接インスタンスではなくコンテキストマネージャを返します。`checkpointer = SqliteSaver.from_conn_string(db_path)` のように `with` なしで渡すと `TypeError` やコネクションリークの原因になります。常駐アプリで自前の接続を保持したい場合は `sqlite3.connect(db_path, check_same_thread=False)` を `SqliteSaver(conn)` へ直接渡してください。
2. **Checkpointer 付きグラフで `thread_id` を渡し忘れると即エラーになる**
   `compile(checkpointer=...)` したグラフに対して `config` なし（または `thread_id` なし）で `invoke()` を呼ぶと `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id` が発生します。
3. **`get_state_history(cfg)` は「最新が先頭（降順）」である**
   リストの末尾 `history[-1]` が最古（`step: -1`）、先頭 `history[0]` が最新（今回の `step: 4`）です。インデックスを逆に参照すると意図しない過去状態を最新と誤認してしまいます。
4. **全スレッド一覧取得では `checkpointer.list({})` ではなく `checkpointer.list(None)` を渡す**
   `SqliteSaver` は `None` 以外の `config`（空辞書 `{}` を含む）を「特定スレッドのフィルタ指定」と解釈するため、`thread_id` 欠落エラーになります。全スレッドを走査したい場合は必ず `None` を指定してください。
5. **環境ごとの Checkpointer 選定と `PostgresSaver.setup()` の呼び忘れ**
   `InMemorySaver` はプロセス終了でデータが消えるため本番禁止です。ローカル検証は `SqliteSaver`、本番の複数ワーカー環境では `PostgresSaver`（または `AsyncPostgresSaver`）を使用し、初回起動時に必ず `checkpointer.setup()` でテーブル初期化を行ってください。

---

## 6. まとめ

本記事では、`SqliteSaver` と `MessagesState` を用いて `MiniCPM-V-4.7-35B-A3B` との 2 ターン会話を永続化し、生成された 6 つのチェックポイント（`step -1` 〜 `step 4`）の内部構造を実測データで検証しました。

Checkpointer によって Super-step 単位の完全な状態履歴が永続化されたことで、次なる高度な機能――**Streaming・耐障害性（`RetryPolicy` / `CachePolicy`）・Human-in-the-Loop（`interrupt()`）・Time Travel（履歴フォーク）**――を自在に構築できる土台が整いました。次回の Part 4 以降でこれらの発展機能をさらに深掘りしていきます。
