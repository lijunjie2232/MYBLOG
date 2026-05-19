---
title: "LangChain v1 Part 2 実践編：safe_calculator と ToolRuntime によるユーザー属性取得・ゼロ除算リカバリの実証"
date: 2026-05-07 03:29:36
categories: [AI]
tags: [LangChain, LangChain v1, ToolRuntime, Pydantic, Error Handling, Python]
lang: ja
description: "Pydantic args_schema を使った safe_calculator と ToolRuntime でユーザーIDを安全に注入する get_user_tier を実装し、並列実行時のゼロ除算リカバリと6メッセージの遷移を実測データで検証します。"
---

# LangChain v1 Part 2 実践編：`safe_calculator` と `ToolRuntime` によるユーザー属性取得・ゼロ除算リカバリの実証

前回の **Part 2 理論編** では、Pydantic `args_schema` による厳格なツール入力契約、`ToolRuntime` によるモデル非公開コンテキストの注入、そして多層エラーハンドリング戦略を解説しました。

本記事（Part 2 実践編）では、理論編の設計をそのままコードへ落とし込んだ **Lab 2** を `MiniCPM-V-4.7-35B-A3B` 上で実行し、**「ユーザー属性の確認」「ゼロ除算（エラー系）」「正常除算（正常系）」の 3 タスクを 1 プロンプトで同時に依頼したときの挙動**を実測データで検証します。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、`ToolRuntime` によるスキーマ隠蔽と第 2 層（ツール内部）の Soft Error 設計について以下のように述べました。

> ツールの引数に `runtime: ToolRuntime` を宣言すると、LangGraph の `ToolNode` が以下の 2 つを自動的に行います。
> 1. **スキーマからの隠蔽（Schema Hiding）**: モデルへ送信する JSON Schema から `runtime` 引数を完全に除外する。モデルからは「引数なしのツール」または「ビジネス引数だけのツール」に見える。
> 2. **実行時インジェクション（Runtime Injection）**: ツール実行直前に、現在のグラフ状態・設定・ストア・ツール呼び出し ID を束ねた `ToolRuntime` オブジェクトを自動注入する。

> ゼロ除算（`b == 0`）、サポート外の演算子、対象レコード未検出といった**想定内の業務エラー**は、Python の例外をそのまま送出せず、`"Error: Division by zero is not allowed."` のように**原因が明確なエラー文字列**として返します。これにより、同一バッチで並列実行されている他のツール呼び出し（例: `100 / 4` や `get_user_tier`）の結果を無駄にせず、モデルへすべての結果を揃えて渡すことができます。

今回の実践ラボ（Lab 2）では、この 2 点が実際のエージェントループ上で狙い通り機能するかを確認します。

1. ユーザープロンプトに `u-2026-sre` という ID を一切書かずに、`config={"configurable": {"user_id": "u-2026-sre"}}` だけで `get_user_tier` が正しいユーザー情報を取得できるか。
2. `100 / 0` のエラー文字列返却によってエージェントがクラッシュせず、並列実行された `100 / 4` の結果およびプラン情報と統合して自然な日本語回答を返せるか。

---

## 2. 実験アーキテクチャと実装コード

1 回のユーザーリクエストから 3 つのツール呼び出しが分岐・合流するフローを以下に示します。

{% mermaid %}
flowchart LR
    U["User<br/>プラン確認 + 100÷0 + 100÷4"] --> M["Model Node<br/>(3件の tool_calls を並列発行)"]
    CFG["config.configurable<br/>{user_id: 'u-2026-sre'}"] -.->|"ToolRuntime 経由で注入<br/>(LLM からは不可視)"| T1

    M -->|"Call #1 ()"| T1["get_user_tier(runtime)"]
    M -->|"Call #2 (100, 0, div)"| T2["safe_calculator"]
    M -->|"Call #3 (100, 4, div)"| T3["safe_calculator"]

    T1 -->|"user_id=u-2026-sre, tier=Enterprise"| MERGE["Model Node<br/>(3件の ToolMessage を統合)"]
    T2 -->|"Error: Division by zero..."| MERGE
    T3 -->|"25.0"| MERGE
    MERGE --> OUT["最終回答 (日本語)"]

    style M fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style CFG fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style T1 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style T2 fill:#fff5f5,stroke:#c53030,stroke-width:1.5px,color:#742a2a
    style T3 fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
    style MERGE fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
{% endmermaid %}

### 実装コード（Lab 2）

```python
import json
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 環境変数からエンドポイントとAPIキーを読み込む
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(model: str = MODEL_NAME, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI互換エンドポイント用のモデルインスタンスを作成する。"""
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=30,
        max_retries=1,
    )


def run_lab2():
    llm = make_llm()

    # 1. Pydantic args_schema による厳格な入力定義
    # Field の description がそのまま JSON Schema としてモデルに提示される
    class CalcInput(BaseModel):
        a: float = Field(description="左辺の数値")
        b: float = Field(description="右辺の数値")
        op: str = Field(description="演算子: add, sub, mul, div")

    # 2. 第2層（ツール内部）の Soft Error ガードを備えた計算ツール
    @tool("safe_calculator", args_schema=CalcInput)
    def safe_calculator(a: float, b: float, op: str) -> str:
        """安全な四則演算ツール。ゼロ除算時はエラーメッセージを返す。"""
        if op == "add":
            return str(a + b)
        if op == "sub":
            return str(a - b)
        if op == "mul":
            return str(a * b)
        if op == "div":
            # ゼロ除算を ZeroDivisionError で落とさず、明確なエラー文字列として返却する
            if b == 0:
                return "Error: Division by zero is not allowed."
            return str(a / b)
        return f"Error: Unsupported op {op}"

    # 3. ToolRuntime を用いたコンテキスト注入ツール
    # runtime 引数はモデル向けスキーマから自動除外され、実行時に ToolNode が注入する
    @tool
    def get_user_tier(runtime: ToolRuntime) -> str:
        """ランタイムコンテキストから現在のユーザーIDとプランを取得する。"""
        cfg = runtime.config.get("configurable", {})
        user_id = cfg.get("user_id", "guest")
        # 本番ではここで user_id をキーに DB や課金サービスを参照する
        return f"user_id={user_id}, tier=Enterprise"

    agent = create_agent(
        model=llm,
        tools=[safe_calculator, get_user_tier],
        system_prompt="ツールを使って計算やユーザー確認を行い、日本語で簡潔に答えてください。",
    )

    # プロンプトには user_id を書かず、config.configurable 経由で安全に渡す
    res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "私のプランを確認し、100を0で割った結果と100を4で割った結果を計算してください。",
                }
            ]
        },
        config={"configurable": {"user_id": "u-2026-sre"}},
    )

    return {
        "message_count": len(res["messages"]),
        "tool_messages": [m.content for m in res["messages"] if m.type == "tool"],
        "final": res["messages"][-1].content,
    }


if __name__ == "__main__":
    print(json.dumps(run_lab2(), indent=2, ensure_ascii=False))
```

---

## 3. 実測出力の解析（実行時間: 4.35 秒）

実際に Lab 2 を実行して得られた出力が以下です（実行時間 `elapsed_s: 4.35`、ステータス `ok`）。

```json
{
  "message_count": 6,
  "tool_messages": [
    "user_id=u-2026-sre, tier=Enterprise",
    "Error: Division by zero is not allowed.",
    "25.0"
  ],
  "final": "あなたのプランは **Enterprise** です。\n\n計算結果は以下の通りです：\n- 100を0で割る：エラー（ゼロ除算は許可されていません）\n- 100を4で割る：**25.0**"
}
```

### なぜこの挙動になったのか？

実測ログから、3 つの重要なメカニズムが裏付けられました。

{% mermaid %}
sequenceDiagram
    participant U as User
    participant M as Model Node
    participant T as Tools Node (ToolRuntime 注入)

    U->>M: [1] HumanMessage: プラン確認 + 100÷0 + 100÷4
    M->>T: [2] AIMessage (tool_calls: 3件同時発行)
    Note over T: get_user_tier に config.configurable.user_id を注入
    T-->>M: [3] ToolMessage: user_id=u-2026-sre, tier=Enterprise
    Note over T: safe_calculator(100, 0, div) は例外を投げず文字列返却
    T-->>M: [4] ToolMessage: Error: Division by zero is not allowed.
    T-->>M: [5] ToolMessage: 25.0
    Note over M: 3件の観察結果を1回の推論で統合
    M-->>U: [6] AIMessage: Enterprise / ゼロ除算エラー / 25.0
{% endmermaid %}

1. **なぜ `message_count` がちょうど 6 件なのか？**
   - 内訳は `[1] HumanMessage` + `[2] AIMessage (3 件の tool_calls)` + `[3..5] ToolMessage × 3` + `[6] 最終 AIMessage` = **合計 6 メッセージ** です。
   - モデルが 3 つの独立したツール呼び出し（`get_user_tier`, `safe_calculator(100, 0, "div")`, `safe_calculator(100, 4, "div")`）を 1 ターンにまとめて並列発行したため、LLM 呼び出しは 2 回（合計 4.35 秒）で完了しています。
2. **`ToolRuntime` がプロンプト非依存で安全に動作した理由**
   - `tool_messages[0]` に `"user_id=u-2026-sre, tier=Enterprise"` が返っています。ユーザーの発話には `u-2026-sre` が一切含まれていないにもかかわらず、`ToolNode` が `config={"configurable": {"user_id": "u-2026-sre"}}` を `runtime.config` として注入したため、モデルが `user_id` を捏造する余地なく正確な属性を取得できました。
3. **ゼロ除算エラーからの完全なリカバリ**
   - `tool_messages[1]` の `"Error: Division by zero is not allowed."` は例外ではなく通常の `ToolMessage` として返却されました。その結果、同一バッチの `tool_messages[2]`（`"25.0"`）も正常に処理され、モデルは最終回答で「エラー（ゼロ除算は許可されていません）」と「25.0」を明確に区別して日本語で説明できています。

### モデルに送信される JSON Schema の違い（`args_schema` vs `ToolRuntime`）

なぜ `get_user_tier` に `user_id` を推測されずに済んだのかは、LangChain がモデルへ渡すツールスキーマを比較すると一目瞭然です。

| ツール名 | Python 関数シグネチャ | LLM へ送信される `parameters.properties` |
| :--- | :--- | :--- |
| `safe_calculator` | `(a: float, b: float, op: str)` + `args_schema=CalcInput` | `{"a": {"type": "number", ...}, "b": {"type": "number", ...}, "op": {"type": "string", ...}}` |
| `get_user_tier` | `(runtime: ToolRuntime)` | `{}`（空オブジェクト — `runtime` は完全除外） |

このように、`ToolRuntime` は単なる便利ヘルパーではなく、**LLM のプロンプト空間（信頼境界の外側）とサーバー側の実行コンテキスト（信頼境界の内側）を構造的に分離するセキュリティ境界**として機能しています。

---

## 4. 注意点・ハマりどころ（Pay Attention）

`args_schema` と `ToolRuntime` を実務導入する際は、以下の 4 点に注意してください。

1. **予約語 `runtime` / `config` の衝突**
   - 理論編で強調した通り、`runtime` と `config` は LangChain v1 のツール引数における予約名です。たとえば「コンテナの設定（`config`）を渡すツール」を作りたい場合、引数名を `config: dict` にすると `RunnableConfig` 注入と衝突してランタイムエラーになります。必ず `app_config: dict` や `settings: dict` のように別名を付けてください。
2. **`args_schema` に `runtime` フィールドを書かない**
   - Pydantic の `args_schema` と `runtime: ToolRuntime` を同一ツールで併用する場合、`args_schema` 側には**モデルに渡させたいビジネス引数だけ**を定義します。`runtime` を `BaseModel` に含めてしまうとスキーマ検証エラーになります。
3. **インポート元の違いに注意**
   - `ToolRuntime` は `from langchain.tools import ToolRuntime` からインポートします（`langchain_core.tools` ではありません）。一方で `@tool` デコレータは `langchain.tools` と `langchain_core.tools` のどちらからもインポート可能です。
4. **Soft Error と Hard Error の境界線**
   - ゼロ除算や「対象データなし」のような業務エラーは Soft Error（文字列返却）にしてモデルへ判断を委ねるべきですが、認証トークン失効や DB 接続断のような**リトライ不可能なインフラ障害**まで闇雲に握りつぶすと、モデルが何度も同じツールを呼び直して無限ループに陥る危険があります。インフラ障害は `@wrap_tool_call` Middleware 側でリトライ回数上限（または `ModelCallLimitMiddleware`）と組み合わせて制御してください。

---

## 5. まとめ

Part 2 実践編では、`safe_calculator`（Pydantic `args_schema` + Soft Error）と `get_user_tier`（`ToolRuntime` 注入）を組み合わせ、**プロンプトに秘密情報を含めずにユーザー属性を取得し、ゼロ除算エラーを含む 3 並列ツール呼び出しを 4.35 秒・6 メッセージで安全に完走する**様子を実証しました。

次回の **Part 3 理論編** では、エージェントの出力をアプリケーションへ安全に統合するための **Structured Output（`ToolStrategy` vs `ProviderStrategy`）** と、UX を劇的に改善する **Streaming（`updates` / `messages` / `custom`）および Event Streaming（`stream_events v3`）** の設計理論を解説します。
