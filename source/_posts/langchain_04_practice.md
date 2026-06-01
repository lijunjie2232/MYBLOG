---
title: "LangChain v1 Part 4 実践編：InMemorySaver と InMemoryStore を統合したパーソナライズ記憶エージェントの構築"
date: 2026-05-13 13:02:55
categories: [AI]
tags: [LangChain, LangGraph, AI Agent, Memory, InMemorySaver, InMemoryStore, Python]
lang: ja
description: "LangChain v1 の InMemorySaver（短期記憶）と InMemoryStore（長期記憶）を create_agent に統合し、ToolRuntime 経由でユーザープロフィールを参照しながらマルチターン会話を継続する実践ラボを実測ログ付きで解説します。"
---

# LangChain v1 Part 4 実践編：`InMemorySaver` と `InMemoryStore` を統合したパーソナライズ記憶エージェントの構築

本記事では、Part 4 理論編で整理した「短期記憶（Checkpointer）× 長期記憶（Store）」の二層アーキテクチャを、実際に動く Python コードとして実装・検証します。

理論編では、両者の役割分担について次のように整理しました。

> 設計原則：短期記憶（Checkpointer）は「いま進行中の会話と実行状態の忠実なスナップショット」を担い、長期記憶（Store）は「セッションが終わっても残すべき構造化・検索可能な知識」を担います。この二つを混同しないことが、スケーラブルなエージェント設計の第一歩です。

また、実行時のデータパスについても次の挙動を予測しました。

> キャッシュ効果としての二層連携：一度長期記憶（Store）からツール経由で読み出された情報は、そのスレッドの短期記憶（`messages` 内の `ToolMessage` / `AIMessage`）へ取り込まれます。その結果、同一スレッドの後続ターンでは、わざわざ Store へ再問い合わせしなくても LLM が短期記憶から文脈を即座に参照できます。

この理論が実際の LLM 推論（`MiniCPM-V-4.7-35B-A3B`）と LangChain v1.5.2 のランタイム上で本当にその通りに動くのか、実測結果をもとに解剖していきましょう。

---

## 1. 実験シナリオとアーキテクチャ構成

今回の実践ラボ（Lab 4）では、以下の 2 ターン構成でパーソナライズ記憶エージェントを動かします。

1. **事前シード（長期記憶）**: `InMemoryStore` の Namespace `("users", "alice")`, Key `"profile"` に、Alice のプロフィール JSON を格納しておく。
2. **Turn 1（長期記憶の参照 → 短期記憶への取り込み）**: ユーザーが「私のプロフィールを確認して、好きな技術スタックを覚えておいてください。」と依頼。エージェントが `lookup_profile` ツール経由で Store を読み取り、回答する。
3. **Turn 2（短期記憶のみでの文脈継続）**: 同じ `thread_id="thread-lc-04"` を指定して「さっき確認した私の名前と役割は何でしたか？」と質問。Checkpointer が復元した履歴だけで正しく答えられるか、そして最終的なメッセージ数（`history_len`）がいくつになるかを確認する。

{% mermaid %}
flowchart LR
    subgraph Seed["1. 事前準備"]
        S0["store.put(('users','alice'), 'profile', JSON)"]
    end

    subgraph T1["2. Turn 1 (thread-lc-04)"]
        H1["HumanMessage<br/>プロフィール確認依頼"]
        A1["AIMessage<br/>tool_calls: lookup_profile"]
        TM1["ToolMessage<br/>Store から JSON 取得"]
        A2["AIMessage<br/>Alice / ML Engineer / LangGraph を回答"]
        H1 --> A1 --> TM1 --> A2
    end

    subgraph T2["3. Turn 2 (同じ thread-lc-04)"]
        CP["InMemorySaver が<br/>4メッセージを自動復元"]
        H2["HumanMessage<br/>名前と役割を再確認"]
        A3["AIMessage<br/>ツール再呼出なしで即答"]
        CP --> H2 --> A3
    end

    Seed -.->|"runtime.store.get()"| TM1
    A2 -->|"Checkpoint 保存 (len=4)"| CP

    style Seed fill:#fef3c7,stroke:#d97706,color:#78350f
    style T1 fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style T2 fill:#dcfce7,stroke:#16a34a,color:#14532d
{% endmermaid %}

---

## 2. 実装コード：二層記憶エージェントの構築

API キーやエンドポイントはハードコードせず、`.env` から `os.getenv("BASE_URL")` および `os.getenv("API_KEY")` で読み込みます。

```python
import json
import os
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# .env から接続先 URL と API キーを読み込む（認証情報のハードコード厳禁）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(model: str = MODEL_NAME, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント向けの ChatModel を生成する。"""
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=30,
        max_retries=1,
    )


def run_lab4():
    llm = make_llm()

    # 1. 短期記憶（Thread スコープ）の Checkpointer を用意
    checkpointer = InMemorySaver()

    # 2. 長期記憶（Cross-Thread スコープ）の Store を用意し、初期プロフィールを登録
    store = InMemoryStore()
    store.put(
        ("users", "alice"),  # 階層 Namespace: ユーザー単位で分離
        "profile",           # ドキュメント Key
        {
            "name": "Alice",
            "role": "ML Engineer",
            "favorite_stack": "LangGraph",
        },
    )

    # 3. ToolRuntime 経由で長期記憶 Store にアクセスするツールを定義
    #    ※ runtime 引数はフレームワークが自動注入するため、LLM の引数スキーマには露出しない
    @tool
    def lookup_profile(runtime: ToolRuntime) -> str:
        """長期記憶ストアからユーザープロフィールを検索する。"""
        # configurable から安全に user_id を取り出す（LLM の推測に委ねない）
        user_id = runtime.config.get("configurable", {}).get("user_id", "alice")
        item = runtime.store.get(("users", user_id), "profile")
        return json.dumps(item.value, ensure_ascii=False) if item else "not found"

    # 4. checkpointer と store の両方を create_agent に渡して二層記憶を有効化
    agent = create_agent(
        model=llm,
        tools=[lookup_profile],
        checkpointer=checkpointer,
        store=store,
        system_prompt="短期記憶（会話履歴）と長期記憶ツールを活用して簡潔に日本語で回答してください。",
    )

    # 5. thread_id（短期記憶のキー）と user_id（長期記憶のキー）を分離して指定
    cfg = {"configurable": {"thread_id": "thread-lc-04", "user_id": "alice"}}

    # Turn 1: 長期記憶ツールを呼び出してプロフィールを確認させる
    r1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "私のプロフィールを確認して、好きな技術スタックを覚えておいてください。",
                }
            ]
        },
        config=cfg,
    )

    # Turn 2: 同じ thread_id で直前の会話を参照できるか検証する
    r2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "さっき確認した私の名前と役割は何でしたか？",
                }
            ]
        },
        config=cfg,
    )

    return {
        "turn1": r1["messages"][-1].content,
        "turn2": r2["messages"][-1].content,
        "history_len": len(r2["messages"]),
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    output = run_lab4()
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[OK {elapsed}s]")
    print(json.dumps(output, indent=2, ensure_ascii=False))
```

---

## 3. 実測出力結果（Real Execution Output）

上記コードを実際に実行した際の結果（`langchain_labs.json` の `lab4_memory_store` 記録）は以下の通りです。2 ターン合計の所要時間は **4.24 秒** でした。

```json
{
  "status": "ok",
  "elapsed_s": 4.24,
  "output": {
    "turn1": "プロフィールを確認しました。\n\n*   **名前**: Alice\n*   **役割**: ML Engineer\n*   **好きな技術スタック**: LangGraph\n\n「LangGraph」を好きな技術スタックとして覚えておきました。",
    "turn2": "さっき確認したプロフィールによると、あなたの名前は **Alice**、役割は **ML Engineer**（機械学習エンジニア）でした。",
    "history_len": 6
  }
}
```

---

## 4. なぜこの結果になるのか？ `history_len: 6` の内部構造を解剖

注目すべきは、`turn1` と `turn2` の両方が正確に回答されたことに加え、**`history_len` が厳密に `6` になっている点**です。メッセージ配列 `r2["messages"]` のインデックス 0〜5 を追うと、二層記憶の協調メカニズムが手にとるように分かります。

{% mermaid %}
flowchart TD
    M0["[0] HumanMessage (Turn 1)<br/>私のプロフィールを確認して..."]
    M1["[1] AIMessage (Turn 1)<br/>tool_calls: [lookup_profile()]"]
    M2["[2] ToolMessage (Turn 1)<br/>{'name':'Alice','role':'ML Engineer','favorite_stack':'LangGraph'}"]
    M3["[3] AIMessage (Turn 1)<br/>プロフィールを確認しました..."]
    M4["[4] HumanMessage (Turn 2)<br/>さっき確認した私の名前と役割は何でしたか？"]
    M5["[5] AIMessage (Turn 2)<br/>さっき確認したプロフィールによると、Alice / ML Engineer でした。"]

    M0 --> M1 --> M2 --> M3
    M3 -.->|"InMemorySaver がスナップショット保存 & 次ターンで復元"| M4
    M4 --> M5

    style M0 fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style M1 fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style M2 fill:#fef3c7,stroke:#d97706,color:#78350f
    style M3 fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style M4 fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style M5 fill:#dcfce7,stroke:#16a34a,color:#14532d
{% endmermaid %}

1. **Turn 1 で 4 メッセージが生成・永続化される（`[0]`〜`[3]`）**
   - ユーザーの依頼（`[0]`）に対し、LLM は長期記憶ツール `lookup_profile` の呼び出しを決定（`[1]`）。
   - LangGraph のツール実行ノードが `ToolRuntime` を自動注入し、`runtime.store.get(("users", "alice"), "profile")` を実行。結果を `ToolMessage`（`[2]`）として状態へ追加。
   - LLM が `ToolMessage` を読んで最終回答（`[3]`）を生成。この時点で `InMemorySaver` に 4 件のメッセージが `thread-lc-04` として保存される。
2. **Turn 2 ではツールを再呼び出しせず 2 メッセージだけ増加（`[4]`〜`[5]`）**
   - Turn 2 の `invoke()` が同じ `thread_id="thread-lc-04"` を受け取ると、`InMemorySaver` が即座に `[0]`〜`[3]` をロードし、新しい質問（`[4]`）を末尾に追加。
   - LLM は `[2]` の `ToolMessage` と `[3]` の回答をすでにコンテキスト内に持っているため、**余計なツール呼び出し（`lookup_profile` の再実行）を行わずに** 直接 `AIMessage`（`[5]`）を返した。
   - 仮に Turn 2 でもツールを再呼び出ししていれば、`history_len` は `8` になっていたはずです。`history_len: 6` という数字は、**「長期記憶から取得した情報が短期記憶にキャッシュされ、無駄なツール実行とレイテンシを省けた」**ことの客観的な証明になっています。

---

## 5. 注意点・ハマりどころ（Pay Attention）

本番システムへ二層記憶を導入する際に頻発する落とし穴を 4 つ整理します。

1. **`thread_id` を渡し忘れると Checkpointer が機能しない（エラーになる）**
   - `checkpointer` を指定したエージェントに対して `config={"configurable": {"thread_id": ...}}` を省略して `invoke()` を呼ぶと、LangGraph はどのスレッドへ保存すべきか判断できず `ValueError` を送出します。テストコードでも必ず `thread_id` を明示してください。
2. **`user_id` を LLM のツール引数にしてはいけない**
   - もし `lookup_profile(user_id: str)` のように LLM に `user_id` を渡させる設計にすると、プロンプトインジェクション等で「`bob` のプロフィールを見せて」と誘導された際に他人の Namespace を読み出せてしまいます。今回の実装のように、`runtime.config["configurable"]["user_id"]`（または `context_schema` の `runtime.context.user_id`）から**アプリケーション側で信頼できる識別子を注入**してください。
3. **`InMemorySaver` / `InMemoryStore` は単一プロセス限定の揮発メモリ**
   - FastAPI 等の Web サーバーを複数ワーカー（`uvicorn --workers 4` 等）で起動すると、リクエストごとに別プロセスのメモリを参照してしまい記憶が途切れます。本番環境では必ず `PostgresSaver` / `PostgresStore` や Redis バックエンドに切り替えてください。
4. **長期記憶の「更新」は明示的なツール設計が必要**
   - 今回の Turn 1 で LLM は「『LangGraph』を好きな技術スタックとして覚えておきました」と回答していますが、これはあくまで**短期記憶（会話履歴）に残っただけ**です。新しいスレッド（`thread-lc-05` 等）でもユーザーが発言した新しい嗜好を記憶させたい場合は、別途 `save_profile` のような `runtime.store.put(...)` を呼ぶ書き込みツールを提供する必要があります。

---

## まとめ

本実践編では、`InMemorySaver` と `InMemoryStore` を `create_agent` に統合し、実測ログ（`elapsed_s: 4.24s`, `history_len: 6`）を通じて以下を実証しました。

- `ToolRuntime` を使うことで、LLM に内部構造や `user_id` を露出させずに長期記憶 `Store` を安全に検索できる。
- `Checkpointer` がスレッド内の全メッセージ（`ToolMessage` 含む）を永続化するため、後続ターンでは不要なツール再実行なしに文脈を維持できる。

次回の **Part 5 理論編** では、こうしたエージェントループの前後・呼び出し箇所へ横断的関心事（計測、回数制限、フォールバック、ガードレール等）を差し込む **Middleware アーキテクチャ** を深掘りします。
