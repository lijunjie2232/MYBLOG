---
title: "LangChain v1 Part 5 実践編：カスタム計測 Middleware と ModelCallLimitMiddleware / ModelFallbackMiddleware の組み合わせ実践"
date: 2026-05-17 14:12:25
categories: [AI]
tags: [LangChain, LangGraph, AI Agent, Middleware, Observability, Fallback, Python]
lang: ja
description: "LangChain v1 の AgentMiddleware を継承したカスタム計測・監査 Middleware を実装し、組み込みの ModelCallLimitMiddleware・ModelFallbackMiddleware と合成して実行。実測ログからフック実行順序とミリ秒単位のオーバーヘッドを検証します。"
---

# LangChain v1 Part 5 実践編：カスタム計測 Middleware と組み込み Middleware の合成実践

Part 5 理論編では、LangChain v1 の Middleware が Node-style（`before_model` / `after_model`）と Wrap-style（`wrap_model_call` / `wrap_tool_call`）の 2 系統から成り、複数の Middleware を重ねた際に「オニオンモデル」で合成されることを解説しました。

理論編の要点を振り返ります。

> 使い分けの鉄則：状態（`state["messages"]` 等）の検査や順次的な書き換え・ジャンプ判定には **Node-style** を使い、例外捕捉・リトライ・フォールバック・実行時間計測・`ModelRequest` の動的書き換えには **Wrap-style** を使います。

> 順序設計の具体例：計測・監査 Middleware（`TimingAndAuditMiddleware`）を先頭に置き、内側に `ModelCallLimitMiddleware` や `ModelFallbackMiddleware` を配置すると、**フォールバック発生時の切り替えコストも含めたトータルの実測レイテンシ**を外側から一貫して観測できます。

本実践編（Lab 5）では、この 3 層 Middleware スタックを実際に構築して実行し、出力された `audit_log` と実測レイテンシから、フレームワーク内部のフック実行順序とオーバーヘッドを定量的に検証します。

---

## 1. 構築する 3 層 Middleware スタックの設計

今回のラボでは、`create_agent` の `middleware` 引数へ以下の順序で 3 つの Middleware を合成します。

1. **`TimingAndAuditMiddleware`（カスタム / 最外層）**: `before_model` で入力メッセージ数、`wrap_model_call` でモデル呼び出し実測時間（ms）、`after_model` で応答メッセージ種別と文字数を `audit_log` に記録する。
2. **`ModelCallLimitMiddleware`（組み込み / 中間層）**: `run_limit=3, exit_behavior="end"` を設定し、1 回の `invoke()` 内でモデル呼び出しが 3 回を超えたら暴走を防いで安全終了する。
3. **`ModelFallbackMiddleware`（組み込み / 最内層）**: プライマリモデル（`MiniCPM-V-4.7-35B-A3B`）が失敗した場合に備え、代替モデル（`Hy4-flash`）への自動フォールバックを待機させる。

{% mermaid %}
flowchart LR
    REQ(["invoke() 入力"]) --> M1_B["1. Timing.before_model<br/>msgs=1 を記録"]
    M1_B --> M2_B["2. CallLimit.before_model<br/>run_count < 3 を確認"]
    M2_B --> M1_W["3. Timing.wrap_model_call<br/>t0 = perf_counter()"]

    subgraph FallbackLayer["4. ModelFallback.wrap_model_call"]
        PRI["Primary Model<br/>MiniCPM-V-4.7-35B-A3B"]
        FB["Fallback Model<br/>Hy4-flash (障害時のみ)"]
        PRI -.->|"Exception 発生時"| FB
    end

    M1_W --> PRI
    PRI --> M1_W_END["5. Timing.wrap_model_call 完了<br/>経過時間 (ms) を記録"]
    M1_W_END --> M2_A["6. CallLimit.after_model<br/>run_count += 1 (逆順)"]
    M2_A --> M1_A["7. Timing.after_model<br/>type=ai, len=10 を記録 (逆順)"]
    M1_A --> RES(["最終応答返却"])

    style FallbackLayer fill:#fef3c7,stroke:#d97706,color:#78350f
    style M1_B fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style M1_W fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style M1_W_END fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style M1_A fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style M2_B fill:#dcfce7,stroke:#16a34a,color:#14532d
    style M2_A fill:#dcfce7,stroke:#16a34a,color:#14532d
{% endmermaid %}

---

## 2. 実装コード：カスタム計測 Middleware ＋ 組み込み Middleware 合成

認証情報は `.env` から `os.getenv("BASE_URL")` および `os.getenv("API_KEY")` で読み込みます。

```python
import json
import os
import time
from typing import Any
from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_openai import ChatOpenAI

# 環境変数からエンドポイントと API キーを読み込む（ハードコード厳禁）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"
FALLBACK_MODEL = "Hy4-flash"


def make_llm(model: str = MODEL_NAME, temperature: float = 0.0) -> ChatOpenAI:
    """指定モデル名で ChatOpenAI インスタンスを生成する。"""
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=30,
        max_retries=1,
    )


def run_lab5():
    llm = make_llm()
    audit_log: list[str] = []

    # 3 種類のフック（Node-style 2種 + Wrap-style 1種）を 1 クラスに統合したカスタム Middleware
    class TimingAndAuditMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
            """モデル呼び出し直前に、現在のコンテキスト内メッセージ数を記録する。"""
            audit_log.append(f"before_model: msgs={len(state['messages'])}")
            return None  # 状態は変更せず通過させる

        def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
            """内側の Middleware（Fallback 含む）とモデル推論の総所要時間を計測する。"""
            t0 = time.perf_counter()
            resp = handler(request)
            dt = round((time.perf_counter() - t0) * 1000, 1)
            audit_log.append(f"wrap_model_call: {dt}ms")
            return resp

        def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
            """モデル応答が state に反映された直後、最新メッセージの種別と文字数を記録する。"""
            last = state["messages"][-1]
            audit_log.append(f"after_model: type={last.type}, len={len(str(last.content))}")
            return None

    # 3 つの Middleware を「観測（外側）→ 上限ガード（中間）→ 障害回避（内側）」の順で合成
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            TimingAndAuditMiddleware(),
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
            ModelFallbackMiddleware(make_llm(FALLBACK_MODEL)),
        ],
        system_prompt="日本語で1文で回答してください。",
    )

    res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Middlewareパターンの利点を一言で教えてください。",
                }
            ]
        }
    )

    return {
        "audit_log": audit_log,
        "final": res["messages"][-1].content,
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    output = run_lab5()
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[OK {elapsed}s]")
    print(json.dumps(output, indent=2, ensure_ascii=False))
```

---

## 3. 実測出力結果（Real Execution Output）

上記コードを実際に実行した際の結果（`langchain_labs.json` の `lab5_middleware` 記録）は以下の通りです。

```json
{
  "status": "ok",
  "elapsed_s": 0.39,
  "output": {
    "audit_log": [
      "before_model: msgs=1",
      "wrap_model_call: 381.5ms",
      "after_model: type=ai, len=10"
    ],
    "final": "処理の分離と再利用性"
  }
}
```

---

## 4. なぜこの結果になるのか？ 実測データの詳細分析

たった 3 行の `audit_log` と `elapsed_s: 0.39` の中にも、LangChain v1 Middleware の重要な設計特性が凝縮されています。

### 4.1 厳密なフック発火順序の証明

`audit_log` の並び順は **`before_model` → `wrap_model_call` → `after_model`** となっています。

1. `before_model: msgs=1`: 呼び出し直前の `state["messages"]` には `HumanMessage` 1 件だけが存在する（`system_prompt` は `state["messages"]` ではなく `ModelRequest.system_message` として渡されるため、`len(state["messages"])` にはカウントされない点にも注目してください）。
2. `wrap_model_call: 381.5ms`: 実際に `handler(request)` が内側の `ModelFallbackMiddleware` とプライマリモデル（`MiniCPM-V-4.7-35B-A3B`）を実行し終えた瞬間に記録されます。
3. `after_model: type=ai, len=10`: モデル応答が `AgentState` に追加された**後**に発火するため、`state["messages"][-1]` はすでに `AIMessage`（`type="ai"`）になっており、本文「`処理の分離と再利用性`」の文字数（ちょうど 10 文字）が正確に観測されています。

### 4.2 フレームワーク・オーバーヘッドの定量評価（`381.5ms` vs `0.39s`）

`invoke()` 全体の経過時間（`elapsed_s`）が **0.39 秒（約 390ms）** であったのに対し、純粋なモデル呼び出し区間（`wrap_model_call`）は **381.5ms** でした。

{% mermaid %}
flowchart LR
    T0["全体の実行時間<br/>約 390.0 ms (100%)"]
    LLM["wrap_model_call (LLM 推論 + 通信)<br/>381.5 ms (約 97.8%)"]
    OVH["LangGraph + 3層 Middleware 制御<br/>約 8.5 ms (約 2.2%)"]

    T0 --> LLM
    T0 --> OVH

    style T0 fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style LLM fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style OVH fill:#dcfce7,stroke:#16a34a,color:#14532d
{% endmermaid %}

つまり、LangGraph のステート管理および 3 層の Middleware（カスタム計測＋回数制限＋フォールバック）をすべて通過しても、**フレームワーク側のオーバーヘッドは 10ms 未満**に収まっています。本番環境で可観測性や安全装置を Middleware として重ねても、レイテンシ面でのペナルティはほぼ無視できることが実証されました。

### 4.3 異常系（上限到達・プライマリ障害）での振る舞い

今回の正常系実行では 1 回のモデル呼び出しで完了したため、`ModelCallLimitMiddleware` と `ModelFallbackMiddleware` は「待機・通過」として機能しましたが、異常時には次のように動作します。

- **`ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")`**: ツール呼び出しループが 3 回を超えて 4 回目のモデル呼び出しに入ろうとした瞬間、`before_model` フックが `jump_to="end"` を返して安全に実行を終了します（例外でプロセスを落としたい場合は `exit_behavior="error"` を指定）。
- **`ModelFallbackMiddleware(make_llm(FALLBACK_MODEL))`**: `MiniCPM-V-4.7-35B-A3B` へのリクエストがタイムアウトや 5xx エラーで失敗すると、内側の `wrap_model_call` が例外を捕捉し、同一の `ModelRequest` をそのまま `Hy4-flash` へ転送してリトライします。外側にある `TimingAndAuditMiddleware` は、このフォールバックを含めた全体の応答時間をシームレスに計測し続けます。

---

## 5. 注意点・ハマりどころ（Pay Attention）

実務でカスタム Middleware と組み込み Middleware を組み合わせる際にハマりやすいポイントを 4 つ挙げます。

1. **ミュータブルな共有リスト（`audit_log`）は並列実行・マルチリクエストで競合する**
   - 今回のラボは単発実行のためクロージャ内の `audit_log` リストへ追記していますが、本番の Web サーバーで同じ `agent` インスタンスを複数リクエストで共有する場合、インスタンス変数やグローバルリストへログを溜めると他ユーザーのデータと混線します。本番では構造化ロガー（`structlog` / OpenTelemetry Span）へ直接出力するか、Middleware 専用の `state_schema` に記録してください。
2. **`ModelCallLimitMiddleware` の `thread_limit` には `checkpointer` が必須**
   - `run_limit`（1 回の `invoke()` あたりの上限）は `checkpointer` なしでも動作しますが、スレッド累計の `thread_limit` を効かせるには必ず `checkpointer` と `thread_id` を設定する必要があります。
3. **同期フックしか実装していないクラスを `ainvoke()` で呼ぶ際の注意**
   - `AgentMiddleware` を継承して同期メソッド（`before_model` / `wrap_model_call`）だけを実装した場合、非同期アプリ（FastAPI の `async def` 内で `await agent.ainvoke(...)`）から呼び出すと非同期パスで期待通りに動かないケースがあります。非同期環境では `abefore_model` / `awrap_model_call` / `aafter_model` も併せて実装しましょう。
4. **`Middleware` の並び順を逆にすると観測範囲が変わる**
   - もし `ModelFallbackMiddleware` を `TimingAndAuditMiddleware` の**外側**に配置すると、プライマリモデルが失敗してフォールバックが発生した際、`TimingAndAuditMiddleware.wrap_model_call` が **2 回（失敗した 1 回目と成功した 2 回目）** 呼ばれることになります。「1 リクエスト単位の合計時間」を測りたいなら計測を外側、「モデル試行 1 回ごとの時間」を測りたいなら計測を内側に置くなど、目的に応じて配置を設計してください。

---

## まとめ

本実践編では、`AgentMiddleware` を使ったカスタム計測 Middleware と、組み込みの `ModelCallLimitMiddleware` / `ModelFallbackMiddleware` を 3 層に合成し、実測ログ（`elapsed_s: 0.39s`, `wrap_model_call: 381.5ms`, `after_model: type=ai, len=10`）から以下を確認しました。

- `before_model` → `wrap_model_call` → `after_model` のフック順序が理論通りに機能する。
- 3 層の Middleware を重ねても制御オーバーヘッドは 10ms 未満と極めて軽量であり、本番導入の障壁にならない。
- 観測・制限・フォールバックといった横断的関心事を、エージェント本体のプロンプトやツールから完全に分離できる。

次回の **Part 6** では、Middleware の応用として **Context Engineering・PII ガードレール（`PIIMiddleware`）・Human-in-the-Loop（`HumanInTheLoopMiddleware`）** による安全な本番エージェント制御へ進みます。
