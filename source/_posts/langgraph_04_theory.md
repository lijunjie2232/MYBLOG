---
title: "LangGraph Part 4 理論編：Streaming（values / updates / messages / custom / debug）と Fault Tolerance（RetryPolicy・CachePolicy・冪等性）"
date: 2026-06-14 05:28:38
categories:
  - AI
tags:
  - LangGraph
  - Streaming
  - Fault Tolerance
  - RetryPolicy
  - CachePolicy
  - AI Agent
lang: ja
description: "LangGraphにおける5大ストリーミングモード（values / updates / messages / custom / debug）とStreamPart v2形式、さらに本番運用に不可欠なRetryPolicy・CachePolicy・タイムアウト・冪等性設計の理論を徹底解説します。"
---

# はじめに

LLM を組み込んだステートフルなエージェントを本番環境へ投入する際、機能的な正しさだけでは不十分です。実運用では次の 2 つの非機能要件がユーザー体験（UX）とシステム信頼性を大きく左右します。

1. **Streaming（リアルタイム可視化）**：LLM の応答生成や複数ステップのツール呼び出しは数秒〜数十秒のレイテンシを伴います。処理完了まで無応答にするのではなく、トークン・状態差分・カスタム進捗を逐次クライアントへ届ける仕組みが不可欠です。
2. **Fault Tolerance（耐障害性と高速化）**：外部 API の一時的な 5xx エラー、ネットワーク断、LLM のレートリミット、あるいは重い計算の重複実行に耐えるため、リトライ制御（`RetryPolicy`）、ノードキャッシュ（`CachePolicy`）、タイムアウト制御、そして「再実行されても壊れない冪等性（Idempotency）」の設計が求められます。

本記事（Part 4 理論編）では、LangGraph の公式アーキテクチャに基づき、Streaming サブシステムと Fault Tolerance メカニズムの内部設計を体系的に解説します。

---

# 1. Streaming アーキテクチャとモード体系

LangGraph のコンパイル済みグラフ（`Pregel` ランタイム）は、同期版の `graph.stream(...)` と非同期版の `graph.astream(...)` を提供します。重要な設計思想は、**「同じグラフ定義のまま、呼び出し時の `stream_mode` を切り替えるだけで異なる粒度のイベントを購読できる」** 点にあります。

## 1.1 主要な Stream Modes の比較

LangGraph は用途に応じて以下のストリーミングモードを備えています（複数同時指定も可能です）。

| Stream Mode | 出力されるデータ（`data`） | 主なユースケース | Checkpointer 要否 |
| :--- | :--- | :--- | :--- |
| `values` | 各 Superstep 完了後の**State 全体のスナップショット** | UI 全体を再描画する、完全な状態履歴を監視する | 不要 |
| `updates` | 各ノードが返した**State の差分更新のみ**（`{node_name: delta}`） | どのノードが何を書き換えたか軽量に追跡する | 不要 |
| `messages` | `(message_chunk, metadata)` の 2 要素タプル | LLM のトークン単位ストリーミング（ノード内で `.invoke()` していても自動捕捉） | 不要 |
| `custom` | `get_stream_writer()` でノード・ツール内から送出した任意データ | 進捗率（%）、リトライ試行回数、中間ログなどの独自イベント配信 | 不要 |
| `checkpoints` | `get_state()` と同形式のチェックポイントイベント | 永続化された各ステップの確定状態を購読する | 必要 |
| `tasks` | タスクの開始・終了・結果・エラーイベント | ノード実行ライフサイクルの監視 | 必要 |
| `debug` | `checkpoints` と `tasks` に追加メタデータを統合した全情報 | 実行経路の詳細なデバッグ・トラブルシューティング | 推奨 |

> **ポイント（`messages` モードの仕組み）**：ノード関数の中で LLM を `llm.stream()` ではなく `llm.invoke()` で呼び出していても、`stream_mode="messages"` を指定してグラフを実行すれば、LangGraph のコールバック機構が LLM のトークンチャンクを自動的に横取りしてストリームへ流してくれます。

## 1.2 Streaming データフローの全体像

以下の図は、Pregel ランタイム内でノード実行時に各種ストリームイベントがどのように生成・多重化され、クライアントへ届くかを示したものです。

{% mermaid %}
flowchart LR
    subgraph Runtime["Pregel 実行ランタイム"]
        Node["ノード関数 / ツール実行"]
        LLM["LLM 呼び出し（invoke / ainvoke）"]
        Writer["get_stream_writer()"]
        Reducer["State Channel / Reducer 更新"]
        CP["Checkpointer 保存"]
    end

    subgraph Modes["stream_mode 多重化チャネル"]
        M_Msg["messages: (token, metadata)"]
        M_Cust["custom: 任意ペイロード"]
        M_Upd["updates: ノード差分"]
        M_Val["values: State 全スナップショット"]
        M_Dbg["debug / checkpoints / tasks"]
    end

    Client["呼び出し元（stream / astream）"]

    Node --> LLM --> M_Msg
    Node --> Writer --> M_Cust
    Node --> Reducer
    Reducer --> M_Upd
    Reducer --> M_Val
    Reducer --> CP --> M_Dbg

    M_Msg --> Client
    M_Cust --> Client
    M_Upd --> Client
    M_Val --> Client
    M_Dbg --> Client

    style Runtime fill:#eef6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    style Modes fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
    style Client fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,color:#78350f
{% endmermaid %}

## 1.3 Stream 出力形式：v1 タプル形式と v2 `StreamPart` 形式

LangGraph のストリーミング出力には 2 つのフォーマットが存在します。

- **v1（従来のデフォルト）**：指定したオプションによって戻り値の構造が変化します。単一モードなら生データ `chunk`、複数モード（例：`stream_mode=["custom", "updates"]`）なら `(mode, chunk)` タプル、さらに `subgraphs=True` を併用すると `(namespace, mode, chunk)` タプルになります。
- **v2（`version="v2"`、LangGraph 1.1+）**：オプションの組み合わせに関わらず、すべてのチャンクが統一された `StreamPart` 辞書（`{"type": ..., "ns": ..., "data": ...}`）で返されます。`type` フィールドによる型ナローイング（`ValuesStreamPart`, `UpdatesStreamPart`, `MessagesStreamPart`, `CustomStreamPart` 等）が効くため、大規模アプリでは v2 の採用が推奨されます。

```python
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END

# 環境変数からエンドポイントとAPIキーを取得（ハードコード禁止）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

class StreamState(TypedDict):
    topic: str
    answer: str

def research_node(state: StreamState) -> dict:
    # ノード内部からカスタムストリームへ任意データを送出するライターを取得
    writer = get_stream_writer()
    writer({"stage": "start", "progress": 10, "detail": f"{state['topic']}を調査中"})

    # .invoke() で呼んでも stream_mode="messages" 指定時はトークンが自動配信される
    resp = llm.invoke(f"「{state['topic']}」を1文で解説してください。")

    writer({"stage": "done", "progress": 100})
    return {"answer": resp.content}

graph = (
    StateGraph(StreamState)
    .add_node("research", research_node)
    .add_edge(START, "research")
    .add_edge("research", END)
    .compile()
)

# 複数モードを同時購読（version="v2" で統一的な StreamPart 辞書を受け取る例）
for part in graph.stream(
    {"topic": "LangGraph Streaming", "answer": ""},
    stream_mode=["updates", "messages", "custom"],
    version="v2",
):
    if part["type"] == "custom":
        # get_stream_writer() から送られた進捗イベント
        print(f"[CUSTOM] {part['data']}")
    elif part["type"] == "messages":
        # (AIMessageChunk, metadata) のタプル。langgraph_node でノード絞り込みも可能
        msg_chunk, metadata = part["data"]
        if msg_chunk.content and metadata.get("langgraph_node") == "research":
            print(msg_chunk.content, end="", flush=True)
    elif part["type"] == "updates":
        # ノード完了時の差分State
        print(f"\n[UPDATE] {part['data']}")
```

### 特定ノード・特定 LLM 呼び出しのフィルタリングと除外

- **ノード単位の絞り込み**：`messages` モードの `metadata["langgraph_node"]` を参照すれば、並列実行される複数ノードのうち「最終回答ノードのトークンだけ UI に表示する」といった制御が容易に書けます。
- **内部推論トークンの除外**：裏側で分類やメモ生成を行う LLM だけストリームから隠したい場合は、`model.with_config({"tags": ["nostream"]})` を付与することで `messages` ストリームから除外できます。
- **サブグラフの透過配信**：`create_agent` などのコンパイル済みグラフをノードとして埋め込んだ場合、親グラフ側で `subgraphs=True` を指定しないと内部の LLM トークンやノード更新はストリームに現れません。

---

# 2. Fault Tolerance（耐障害性）とノードキャッシュの理論

エージェントが外部 API・データベース・LLM を呼び出す限り、一時的な障害（Transient Failure）は必ず発生します。LangGraph はノード単位の耐障害メカニズムとして、**RetryPolicy**・**TimeoutPolicy**・**Error Handler**・**CachePolicy** を合成可能な形で提供しています（実装の詳細は [langgraph/pregel/_retry.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_retry.py) 等を参照）。

## 2.1 `RetryPolicy`：指数バックオフと例外フィルタリング

`StateGraph.add_node(..., retry_policy=RetryPolicy(...))` を設定すると、ノード実行中に例外が送出された際、ランタイムが自動的に再試行を行います。

| パラメータ | 型 | デフォルト値 | 役割 |
| :--- | :--- | :--- | :--- |
| `max_attempts` | `int` | `3` | 初回実行を含めた最大試行回数（`3` ならリトライは最大 2 回） |
| `initial_interval` | `float` | `0.5` | 初回リトライまでの待機秒数 |
| `backoff_factor` | `float` | `2.0` | リトライごとに待機時間を乗算する指数バックオフ係数 |
| `max_interval` | `float` | `128.0` | 待機秒数の上限 |
| `jitter` | `bool` | `True` | 複数ワーカーの同時再試行（Thundering Herd）を防ぐランダムジッター |
| `retry_on` | `Exception` 型 / タプル / 判定関数 | `default_retry_on` | リトライ対象とする例外の指定 |

### `default_retry_on` の重要な仕様

デフォルトの `default_retry_on` は「バグ由来の決定論的エラーはリトライしても無駄」という原則に基づき、**`ValueError`、`TypeError`、`KeyError`（`LookupError`）、`RuntimeError`、`OSError` などの標準例外をリトライ対象外**にしています。一方、ネットワーク系や HTTP ライブラリの **5xx ステータスエラー**、および非同期ノードの `NodeTimeoutError` はデフォルトでリトライ対象になります。

したがって、独自の検証やビジネス例外（あるいはテスト用の `ValueError`）をリトライさせたい場合は、明示的に `retry_on=ValueError` やカスタム判定関数を渡す必要があります。

```python
from langgraph.types import RetryPolicy, default_retry_on

def custom_retry_policy_fn(exc: BaseException) -> bool:
    """特定の業務例外は即座に失敗させ、それ以外は標準ルールに委譲する判定関数"""
    if isinstance(exc, PermissionError):
        # 認証・認可エラーは再試行しても回復しないため即失敗
        return False
    return default_retry_on(exc)

# 指数バックオフ + ジッター付きの堅牢なリトライ設定
api_retry = RetryPolicy(
    max_attempts=4,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
    retry_on=custom_retry_policy_fn,
)
```

さらに、ノード引数に `runtime: Runtime` を受け取れば、`runtime.execution_info.node_attempt`（1 始まりの試行回数）を参照して「1 回目はプライマリ API、2 回目以降はフォールバック API に切り替える」といった高度な制御も可能です。

## 2.2 `CachePolicy`：高コストノードの結果再利用

同じ入力に対して繰り返し重い処理（外部検索、埋め込み計算、決定論的な LLM 要約など）が走る場合、ノードに `CachePolicy` を付与し、コンパイル時にキャッシュバックエンド（`InMemoryCache` や `SqliteCache`）を渡すことでノード実行そのものをスキップできます。

- **`key_func`**：ノード入力からキャッシュキーを生成する関数（デフォルトは入力の pickle ハッシュ）。
- **`ttl`**：キャッシュの有効期限（秒）。省略すると無期限で保持されます。

> **キャッシュヒット時の挙動**：キャッシュにヒットするとノード関数本体は一切実行されず、キャッシュされた State 更新差分がそのまま返されます。なお、`stream_mode="updates"` で観察すると、キャッシュヒット時には `{'__metadata__': {'cached': True}}` のようなメタデータが付与されます。

## 2.3 リトライ・キャッシュ・エラーハンドリングの実行ライフサイクル

ノード呼び出し時に `CachePolicy`、`RetryPolicy`、タイムアウト、および `error_handler` がどの順序で評価されるかを整理すると、以下のようになります。

{% mermaid %}
flowchart TD
    Start["Superstep 開始：ノード呼び出し"] --> CacheCheck{"CachePolicy 有効かつ<br/>キャッシュヒット？"}
    CacheCheck -->|"ヒット（Cached）"| ReturnCached["キャッシュ済み差分を即返却<br/>（ノード本体・LLM呼び出しをスキップ）"]
    CacheCheck -->|"ミス / キャッシュ無効"| ExecNode["ノード関数を実行<br/>（attempt = 1, 2, ...）"]

    ExecNode --> Outcome{"実行結果"}
    Outcome -->|"成功"| SaveCache["CachePolicy があれば結果を保存"]
    SaveCache --> Commit["State 更新をコミット<br/>（Checkpointer へ永続化）"]

    Outcome -->|"例外発生 / NodeTimeoutError"| Discard["失敗試行の未コミット書き込みを破棄"]
    Discard --> RetryCheck{"RetryPolicy に一致し<br/>attempt < max_attempts？"}
    RetryCheck -->|"Yes"| Backoff["指数バックオフ + Jitter 待機"]
    Backoff --> ExecNode
    RetryCheck -->|"No（リトライ上限到達 or 対象外）"| ErrHandler{"error_handler 設定あり？"}
    ErrHandler -->|"Yes"| Recover["補償処理 / Command(goto=...) で回復"]
    ErrHandler -->|"No"| RaiseErr["例外をグラフ呼び出し元へ送出"]

    style ReturnCached fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    style Commit fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    style Backoff fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    style RaiseErr fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
{% endmermaid %}

---

# 3. リトライ境界と冪等性（Idempotency）の設計原則

LangGraph の耐障害性を理解するうえで最も重要な事実が 1 つあります。それは、**「LangGraph のリトライはノード関数の先頭からの再実行であり、At-Least-Once（少なくとも 1 回）の実行保証である」** という点です。

- **グラフ内部の状態（State）**：チェックポイントは Superstep の境界でのみ書き込まれます。ノードの途中で例外が起きた場合、その失敗試行による State 書き込みはすべて破棄されるため、グラフ内部の整合性は自動的に保たれます。
- **外部副作用（Side Effects）**：決済 API 呼び出し、メール送信、外部 DB への `INSERT` などはチェックポイント境界の外にあります。仮に外部 API が成功した直後にネットワーク断やプロセス障害が起きると、リトライ時に**同じ外部処理が 2 回実行される**危険があります。

### 冪等キー（Idempotency Key）の正しい設計

外部副作用を安全にするには、外部プロバイダへ渡す「冪等キー」を**副作用の実行前に永続化済みの State から決定論的に導出する**必要があります。

```python
import uuid
from typing import TypedDict

class OrderState(TypedDict):
    order_id: str
    payment_key: str
    charged: bool

# ❌ 悪い例：決済ノード内で毎回 uuid4() を生成すると、リトライ時に別キーになり二重課金される
# ✅ 良い例 1：業務ID（order_id）など既に永続化済みの値から決定論的にキーを導出する
def charge_with_business_key(state: OrderState) -> dict:
    # 何度リトライされても同一の order_id からは同じ冪等キーが生成される
    idempotency_key = f"charge:{state['order_id']}"
    # external_payment_api.charge(idempotency_key=idempotency_key)
    return {"payment_key": idempotency_key, "charged": True}

# ✅ 良い例 2：業務IDが無い場合は、前段の独立ノードでキーだけ先に生成・チェックポイント確定させる
def reserve_payment_key_node(state: OrderState) -> dict:
    # 既にキーがあれば再利用し、無ければ新規発行して Superstep 境界で永続化する
    return {"payment_key": state.get("payment_key") or str(uuid.uuid4())}
```

また、`langgraph>=1.2` では非同期ノード向けに `TimeoutPolicy(run_timeout=120, idle_timeout=30)`（処理全体の上限時間と、進捗イベントが途絶えた場合のアイドル上限時間の分離）や、`set_node_defaults(...)` によるグラフ全体のポリシー一括設定もサポートされています。

---

# まとめ

- **Streaming** は `values`（全体状態）、`updates`（差分）、`messages`（LLM トークン）、`custom`（`get_stream_writer()` による任意進捗）、`debug`（全実行メタデータ）の 5 大モードを柔軟に組み合わせられ、`version="v2"` で統一的な `StreamPart` 形式として扱える。
- **Fault Tolerance** は `RetryPolicy`（指数バックオフ・`default_retry_on`）、`CachePolicy`（`InMemoryCache` / `SqliteCache` + `ttl`）、`TimeoutPolicy` の 3 層でノードの信頼性と速度を底上げする。
- **冪等性設計** は At-Least-Once 実行を前提とし、永続化済みの State から再現可能な冪等キーを用いるか、副作用ノードを分離することが本番運用の鉄則である。

次回の「Part 4 実践編」では、意図的に 1 回目で失敗するノードを構築し、`get_stream_writer()` によるカスタムイベント配信、`RetryPolicy` の自動回復、そして `CachePolicy` によるミリ秒未満の高速化を実測データ付きで検証します。
