---
title: "LangGraph Part 4 実践編：get_stream_writer カスタムストリームと RetryPolicy / CachePolicy の耐障害・高速化検証"
date: 2026-06-17 11:34:55
categories:
  - AI
tags:
  - LangGraph
  - Streaming
  - RetryPolicy
  - CachePolicy
  - Python
  - AI Agent
lang: ja
description: "LangGraph Part 4 実践編。意図的に失敗するノードに対して get_stream_writer のカスタムストリーム、RetryPolicy による自動リトライ回復、CachePolicy による 0.87ms のキャッシュヒット高速化を実測検証します。"
---

# はじめに

前回の理論編（Part 4 理論編）では、LangGraph のストリーミングモード体系と、ノード単位の耐障害性（`RetryPolicy`）・高速化（`CachePolicy`）の仕組みを整理しました。本記事（実践編）では、これら 3 つの機能を **1 つの実行可能な実験ラボ（Lab 4）** に統合し、実際の LLM（`MiniCPM-V-4.7-35B-A3B`）を用いて動作とレイテンシを実測検証します。

---

# 1. 理論編の振り返りと実験設計

まず、今回の検証の前提となる理論編の重要ポイントを確認します。

> **理論編より引用（Streaming 多重化とカスタムイベント）**：
> LangGraph のコンパイル済みグラフ（`Pregel` ランタイム）は、同期版の `graph.stream(...)` と非同期版の `graph.astream(...)` を提供します。重要な設計思想は、**「同じグラフ定義のまま、呼び出し時の `stream_mode` を切り替えるだけで異なる粒度のイベントを購読できる」** 点にあります。

> **理論編より引用（`default_retry_on` と `RetryPolicy`）**：
> デフォルトの `default_retry_on` は「バグ由来の決定論的エラーはリトライしても無駄」という原則に基づき、**`ValueError`、`TypeError`、`KeyError`（`LookupError`）、`RuntimeError`、`OSError` などの標準例外をリトライ対象外**にしています。（中略）したがって、独自の検証やビジネス例外（あるいはテスト用の `ValueError`）をリトライさせたい場合は、明示的に `retry_on=ValueError` やカスタム判定関数を渡す必要があります。

> **理論編より引用（`CachePolicy` のキャッシュヒット挙動）**：
> キャッシュにヒットするとノード関数本体は一切実行されず、キャッシュされた State 更新差分がそのまま返されます。

これらを踏まえ、本ラボでは以下のシナリオを検証します。

1. 1 回目の呼び出しで意図的に `ValueError` を発生させる不安定なノード（`flaky_node`）を作成する。
2. ノード先頭で `get_stream_writer()` を使い、`{"event": "attempt", "count": N}` というカスタム進捗イベントを送出する。
3. `add_node()` に `retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, retry_on=ValueError)` と `cache_policy=CachePolicy(ttl=60)` を同時適用し、`compile(cache=InMemoryCache())` でコンパイルする。
4. 1 回目の実行を `stream_mode=["custom", "updates"]` で購読し、リトライ発生時のカスタムイベントと自動回復後の更新イベントを観察する。
5. 2 回目に同じ入力で `graph.invoke(...)` を呼び出し、LLM を叩かずにキャッシュから即座に応答が返る時間（ms）と、`attempts["count"]` が増えないことを確認する。

{% mermaid %}
sequenceDiagram
    participant Caller as 呼び出し元
    participant Graph as CompiledStateGraph (InMemoryCache)
    participant Node as flaky_node
    participant LLM as ChatOpenAI (MiniCPM-V)

    Note over Caller,LLM: 【フェーズ1】1回目実行（stream_mode=["custom", "updates"]）
    Caller->>Graph: graph.stream(query, stream_mode=["custom", "updates"])
    Graph->>Node: Attempt 1 開始（キャッシュミス）
    Node-->>Caller: custom: {"event": "attempt", "count": 1}
    Node--xGraph: raise ValueError("Transient simulated failure")
    Note over Graph: RetryPolicy(retry_on=ValueError) が検知し 0.1s 待機

    Graph->>Node: Attempt 2 開始
    Node-->>Caller: custom: {"event": "attempt", "count": 2}
    Node->>LLM: invoke("LangGraphのRetryPolicyとは？を1文で...")
    LLM-->>Node: 回答テキスト返却
    Node-->>Graph: {"answer": "..."}
    Graph->>Graph: InMemoryCache に結果を保存（TTL=60s）
    Graph-->>Caller: updates: {"flaky": {"answer": "..."}}

    Note over Caller,LLM: 【フェーズ2】2回目実行（同一入力で invoke）
    Caller->>Graph: graph.invoke(query)
    Graph-->>Caller: キャッシュヒット（0.87ms / Node・LLM未実行）
{% endmermaid %}

---

# 2. 実装コード（Lab 4 全文）

API キーやエンドポイントは必ず環境変数（`BASE_URL`, `API_KEY`）から読み込みます。

```python
import json
import os
import time
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.cache.memory import InMemoryCache
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy

# .env から環境変数をロード（認証情報のハードコードを避ける）
load_dotenv()

def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント向けの ChatOpenAI クライアントを生成"""
    return ChatOpenAI(
        model="MiniCPM-V-4.7-35B-A3B",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=temperature,
        max_tokens=256,
        timeout=30,
        max_retries=1,
    )

def run_lab4():
    llm = make_llm()
    # ノードが実際に何回実行されたかを記録するカウンタ
    attempts = {"count": 0}

    class FaultState(TypedDict):
        query: str
        answer: str

    def flaky_node(state: FaultState) -> dict:
        # 1. カスタムストリームライターを取得
        writer = get_stream_writer()

        # 2. 試行回数をインクリメントし、即座にクライアントへ進捗を通知
        attempts["count"] += 1
        writer({"event": "attempt", "count": attempts["count"]})

        # 3. 初回試行のみ一時的障害をシミュレートして ValueError を送出
        if attempts["count"] == 1:
            raise ValueError("Transient simulated failure on attempt 1")

        # 4. 2回目以降は正常に LLM を呼び出して結果を返す
        resp = llm.invoke(f"{state['query']}を1文で答えてください。")
        return {"answer": resp.content}

    # グラフ構築：RetryPolicy と CachePolicy をノードに同時付与
    graph = (
        StateGraph(FaultState)
        .add_node(
            "flaky",
            flaky_node,
            # ValueError は default_retry_on では対象外のため明示的に retry_on=ValueError を指定
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_interval=0.1,
                retry_on=ValueError,
            ),
            # 同一入力の結果を 60 秒間キャッシュする
            cache_policy=CachePolicy(ttl=60),
        )
        .add_edge(START, "flaky")
        .add_edge("flaky", END)
        # CachePolicy を有効化するには compile() に cache バックエンドを渡すことが必須
        .compile(cache=InMemoryCache())
    )

    # フェーズ1：custom と updates の 2 モードを同時ストリーミング
    events = []
    for mode, chunk in graph.stream(
        {"query": "LangGraphのRetryPolicyとは？", "answer": ""},
        stream_mode=["custom", "updates"],
    ):
        events.append((mode, chunk))

    # フェーズ2：同一入力で再実行し、キャッシュヒットによる高速化を計測
    t0 = time.perf_counter()
    cached_res = graph.invoke({"query": "LangGraphのRetryPolicyとは？", "answer": ""})
    cached_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "attempts": attempts["count"],
        "stream_event_count": len(events),
        "custom_events": [c for m, c in events if m == "custom"],
        "cached_call_ms": cached_ms,
        "answer": cached_res["answer"],
    }

if __name__ == "__main__":
    result = run_lab4()
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

# 3. 実測結果とメカニズム解剖

上記コードを実際に実行した際の出力（`lab4_streaming_fault_tolerance`、全体所要時間 `1.31s`）は以下の通りです。

```json
{
  "status": "ok",
  "elapsed_s": 1.31,
  "output": {
    "attempts": 2,
    "stream_event_count": 3,
    "custom_events": [
      {
        "event": "attempt",
        "count": 1
      },
      {
        "event": "attempt",
        "count": 2
      }
    ],
    "cached_call_ms": 0.87,
    "answer": "LangGraphのRetryPolicyは、ノードの実行中にエラーが発生した際に、指定された条件（回数や間隔など）に基づいて自動的に処理を再試行する機能です。"
  }
}
```

この実測データから、3 つの重要な内部挙動を読み解くことができます。

### ① 失敗試行でも `custom` イベントはリアルタイムに届き、`updates` は成功時のみ届く

`stream_event_count` が **`3`** になっている点に注目してください。内訳は以下の通りです。

1. `("custom", {"event": "attempt", "count": 1})` — Attempt 1 の先頭で即時送出（直後に `ValueError` で失敗）
2. `("custom", {"event": "attempt", "count": 2})` — リトライ待機（`0.1s`）後、Attempt 2 の先頭で即時送出
3. `("updates", {"flaky": {"answer": "LangGraphのRetryPolicyは..."}})` — Attempt 2 が成功した後に 1 回だけ送出

`get_stream_writer()` によるカスタムイベントはノード実行中に即時フラッシュされるため、ユーザー画面に「API 接続に失敗しました。再試行中（2/3）...」といった親切なステータスを表示できます。一方で、失敗した Attempt 1 の状態書き込みは破棄されるため、`updates` イベントは整合性の取れた成功結果の 1 回しか流れません。

### ② キャッシュヒット時は `0.87ms`（約 1,500 倍高速化）かつノード本体が一切動かない

- 1 回目の実行（リトライ + LLM 推論含む）は **`1.31秒（1310ms）`** かかっています。
- 2 回目の同一入力呼び出しは **`0.87ms`** で完了しています。
- さらに `attempts` の最終値は **`2`** のまま（3 に増えていない）です。これはキャッシュヒット時に `flaky_node` 自体がスキップされ、外部 LLM API も一切呼ばれていないことの明確な証拠です。

{% mermaid %}
flowchart LR
    subgraph Run1["1回目実行：1,310 ms"]
        A1["Attempt 1<br/>custom送出 → ValueError"] --> W1["0.1s Backoff"]
        W1 --> A2["Attempt 2<br/>custom送出 → LLM推論 → 成功"]
        A2 --> C1["Cache保存 (TTL=60s)"]
    end

    subgraph Run2["2回目実行：0.87 ms（約1,500倍高速）"]
        H1["Cacheキー照合"] --> H2["HIT! ノード本体スキップ"]
        H2 --> H3["即時 State 返却<br/>(attempts=2 のまま)"]
    end

    C1 -.->|"同一入力ハッシュ"| H1

    style Run1 fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#7c2d12
    style Run2 fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#064e3b
{% endmermaid %}

---

# 4. 注意点・ハマりどころ（Pay Attention）

実運用で Streaming・RetryPolicy・CachePolicy を組み合わせる際、以下の落とし穴に注意してください。

1. **`default_retry_on` は `ValueError` や `OSError` をリトライしない**
   - テストや独自バリデーションで `ValueError` を投げても、デフォルト設定では即座にクラッシュします。独自例外をリトライ対象にしたい場合は `retry_on=(MyTransientError, ValueError)` や判定関数を明示指定してください。
2. **`CachePolicy` だけではキャッシュは動かない（`compile(cache=...)` が必須）**
   - `add_node(..., cache_policy=CachePolicy(ttl=60))` を書いても、`.compile(cache=InMemoryCache())`（永続化したい場合は `SqliteCache` 等）を渡し忘れるとキャッシュは機能しません。
3. **キャッシュキーはデフォルトで「ノード入力 State 全体」のハッシュ**
   - State 内に毎回変わるタイムスタンプや `request_id` が含まれていると、実質同じ質問でもキャッシュミスし続けます。その場合は `CachePolicy(ttl=60, key_func=lambda state: state["query"])` のように `key_func` を明示定義して、キャッシュ対象としたいフィールドだけをキーにしてください。
4. **リトライ境界の外部副作用とカスタムストリームの重複**
   - リトライが起きるとノード関数は先頭から再実行されます。本ラボでも `custom` イベントが 2 回送出されたように、ノード内に「メール送信」「課金」「ログ追記」などの非冪等な処理があるとリトライのたびに重複実行されます。外部書き込みには必ず永続化済み State 由来の冪等キー（Idempotency Key）を付与してください。
5. **非同期コード + Python 3.11 未満での `get_stream_writer()` 制約**
   - Python 3.10 以前の `async def` ノードでは、`contextvars` の伝播制約により `get_stream_writer()` が動作しません。Python 3.11+ を使うか、ノード引数に `writer: StreamWriter` を明示的に受け取る設計にしてください。

---

# まとめ

本実践編では、`get_stream_writer()`、`RetryPolicy`、`CachePolicy` を 1 つのグラフ上で連携させ、以下を実証しました。

- `stream_mode=["custom", "updates"]` により、リトライ中の途中経過（Attempt 1 → Attempt 2）をユーザーへリアルタイム通知しつつ、最終的な State 更新は成功時のみ安全に受け取れる。
- `RetryPolicy(retry_on=ValueError)` によって一時障害から自動回復し、`InMemoryCache` + `CachePolicy(ttl=60)` によって 2 回目の同一クエリを `1.31s` → `0.87ms` へと劇的に高速化できた。

次回（Part 5）では、エージェントの実行途中で人間の承認を待つ **Interrupts（Human-in-the-Loop）** と、過去のチェックポイントへ遡って別ルートを試す **Time Travel（Replay / Fork）** の理論と実践に踏み込みます。
