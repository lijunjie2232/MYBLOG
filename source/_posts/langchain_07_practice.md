---
title: "LangChain v1 Part 7 実践編：Subagent-as-Tool パターンによる SRE メトリクス調査・統括マルチエージェント"
date: 2026-05-24 11:21:23
categories: [AI]
tags: [LangChain, LLM, AI Agent, Multi-Agent, Subagents, SRE, Python]
lang: ja
description: "LangChain v1 の Subagent-as-Tool パターンを用いて、メトリクス分析専門サブエージェントと統括エージェント（Supervisor）から成る SRE 健全性調査システムを実装・実行検証します。実測ログからコンテキスト分離とメッセージ構造を紐解きます。"
---

本稿（LangChain v1 シリーズ Part 7 実践編）では、前回の理論編で解説した **Subagents（Subagent-as-Tool）パターン** を実際にコードへ落とし込み、**「SRE 統括エージェント（Supervisor）が、メトリクス分析専門のサブエージェントへ調査を委譲し、健全性レポートを統合する」** マルチエージェントシステムを実装・実測検証します。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、Subagents パターンの本質とコスト構造について以下の 3 点を整理しました。

> Subagents パターンでは、中央の統括エージェント（Supervisor）が専門サブエージェントを通常のツールとして呼び出します。サブエージェントはデフォルトでステートレス（Stateless）であり、独立したクリーンなコンテキストウィンドウで動作するため、メイン会話のコンテキスト肥大化（Context Bloat）を強力に防ぎます。

> One-shot リクエストにおいて Subagents パターンは 4 回のモデル呼び出し（Supervisor の委譲判断 → Subagent のツール呼び出し → Subagent の要約生成 → Supervisor の最終回答）を要します。Handoffs や Skills（3 回）より 1 回多いですが、このオーバーヘッドこそが『中央集権的な制御と厳格なコンテキスト分離』の対価です。

> サブエージェントのツール化には、専門エージェントごとに個別の `@tool` を定義する **Tool per agent** と、単一のディスパッチツールで名前指定する **Single dispatch tool** の 2 パターンがあります。入力スキーマや出力整形を細かく制御したい場合は Tool per agent が最適です。

本実践ラボ（`lab7_multi_agent`）では、**Tool per agent** 方式を採用し、次の 3 点を実測ログから確認します。

1. `supervisor` 側のメッセージ履歴（`supervisor_msgs`）に、サブエージェント内部の生ツール呼び出し（`fetch_metrics`）が一切漏れず、**厳密なコンテキスト分離** が実現されているか
2. サブエージェントの最終回答が `ToolMessage` として統括エージェントへ返却され、理論通りの **計 4 回の LLM 呼び出し** で完結するか
3. 統括エージェントがサブエージェントの要約結果を踏まえ、ユーザー向けの最終レポートへ構造化して再構成できるか

---

## 2. システム構成と 4 回の LLM 呼び出しフロー

今回のマルチエージェント構成における制御フローとコンテキスト境界を Mermaid 図で整理します。

{% mermaid %}
sequenceDiagram
    autonumber
    participant User as ユーザー
    participant Sup as Supervisor Agent<br/>(統括コンテキスト)
    participant ToolWrap as @tool ask_metrics_specialist
    participant Sub as Metrics Subagent<br/>(分離コンテキスト)
    participant MetricTool as @tool fetch_metrics

    User->>Sup: "checkout-api の現在のメトリクスを調べて健全性を報告してください。"
    Note over Sup: LLM 呼び出し (1/4)<br/>専門エージェントへの委譲を決定
    Sup->>ToolWrap: ask_metrics_specialist(query="checkout-api ...")
    ToolWrap->>Sub: invoke({"messages": [{"role": "user", "content": query}]})
    Note over Sub: LLM 呼び出し (2/4)<br/>生メトリクス取得ツールを選択
    Sub->>MetricTool: fetch_metrics(service="checkout-api")
    MetricTool-->>Sub: "checkout-api: RPS=1250, ErrorRate=0.02%, P99=42ms"
    Note over Sub: LLM 呼び出し (3/4)<br/>数値を解釈し健全性評価を要約
    Sub-->>ToolWrap: 要約テキスト（最終メッセージのみ返却）
    ToolWrap-->>Sup: ToolMessage として要約を受領
    Note over Sup: LLM 呼び出し (4/4)<br/>最終レポートを Markdown で整形
    Sup-->>User: "## checkout-api 健全性報告 ..."
{% endmermaid %}

---

## 3. 実装コード（`lab7_multi_agent`）

以下は実際に検証を実行した完全な Python コードです。エンドポイント URL と API キーは環境変数（`BASE_URL`, `API_KEY`）から読み込みます。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# .env から接続設定を読み込む（API キーのハードコード厳禁）
load_dotenv()

def make_llm(model: str = "MiniCPM-V-4.7-35B-A3B", temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント向けの ChatOpenAI インスタンスを生成する。"""
    return ChatOpenAI(
        model=model,
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=temperature,
        max_tokens=512,
        timeout=30,
        max_retries=1,
    )

def run_lab7_multi_agent() -> dict:
    llm = make_llm()

    # 1. 下位ドメインツール: 生のメトリクスデータを取得する
    @tool
    def fetch_metrics(service: str) -> str:
        """サービスのメトリクスを取得する。"""
        return f"{service}: RPS=1250, ErrorRate=0.02%, P99=42ms"

    # 2. メトリクス分析専門サブエージェントを構築（ステートレス・独立コンテキスト）
    metrics_subagent = create_agent(
        model=llm,
        tools=[fetch_metrics],
        system_prompt="あなたはメトリクス分析専門のサブエージェントです。数値を簡潔に要約して返してください。",
    )

    # 3. Tool per agent パターン: サブエージェントを @tool 関数としてラップする
    @tool
    def ask_metrics_specialist(query: str) -> str:
        """メトリクス専門サブエージェントに調査を委譲する。"""
        # サブエージェントは毎回クリーンな messages 配列で起動される（コンテキスト分離）
        res = metrics_subagent.invoke({"messages": [{"role": "user", "content": query}]})
        # 途中の ToolCall / ToolMessage は捨て、最終要約メッセージのみを上位へ返す
        return str(res["messages"][-1].content)

    # 4. 統括エージェント（Supervisor）を構築
    supervisor = create_agent(
        model=llm,
        tools=[ask_metrics_specialist],
        system_prompt="あなたは統括エージェントです。必要に応じて専門サブエージェントを呼び出し、最終報告を日本語でまとめてください。",
    )

    # 5. ユーザーからの調査依頼を統括エージェントへ投入
    res = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "checkout-api の現在のメトリクスを調べて健全性を報告してください。",
                }
            ]
        }
    )

    return {
        "supervisor_msgs": len(res["messages"]),
        "subagent_tool_output": [m.content for m in res["messages"] if m.type == "tool"],
        "final": res["messages"][-1].content,
    }

if __name__ == "__main__":
    import json
    result = run_lab7_multi_agent()
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

## 4. 実測出力と詳細分析

上記コードを `MiniCPM-V-4.7-35B-A3B` に対して実行した際の実測結果（`langchain_labs.json` より抜粋、総実行時間 `19.42s`）は以下の通りです。

```json
{
  "status": "ok",
  "elapsed_s": 19.42,
  "output": {
    "supervisor_msgs": 4,
    "subagent_tool_output": [
      "checkout-api の現在のメトリクスは以下の通りです：\n\n- **リクエスト数 (RPS)**: 1,250\n- **エラー率**: 0.02%\n- **P99 レイテンシー**: 42ms\n\n**健全性評価**:\nエラー率が極めて低く（0.02%）、レイテンシーも42msと良好なため、サービスは**正常に稼働している**と判断できます。"
    ],
    "final": "## checkout-api 健全性報告\n\n### 現在のメトリクス\n- **リクエスト数 (RPS)**: 1,250\n- **エラー率**: 0.02%\n- **P99 レイテンシー**: 42ms\n\n### 健全性評価\n**✅ 正常に稼働しています**\n\nエラー率が極めて低く（0.02%）、P99 レイテンシーも42msと良好な水準です。現在のところ、checkout-api に異常は検出されていません。"
  }
}
```

### なぜこの挙動になったのか？（内部メカニズムの解説）

#### 1. `supervisor_msgs: 4` が証明する厳格なコンテキスト分離
統括エージェント側の最終メッセージ数は **ちょうど 4 件** です。その内訳は以下の通りです。

1. `HumanMessage`: `"checkout-api の現在のメトリクスを調べて健全性を報告してください。"`
2. `AIMessage`: `ask_metrics_specialist` の `tool_calls` を含む委譲リクエスト
3. `ToolMessage`: サブエージェントが生成した要約テキスト（`subagent_tool_output`）
4. `AIMessage`: 統括エージェントが Markdown 構造化した最終レポート（`final`）

注目すべきは、**サブエージェント内部で発生した `fetch_metrics` の呼び出し履歴や生の返却文字列 `"checkout-api: RPS=1250, ErrorRate=0.02%, P99=42ms"` が、統括エージェントの履歴に一切混入していない** 点です。今回のデモではサブエージェントのツール呼び出しは 1 回ですが、実運用の SRE 調査でサブエージェントがログ検索・メトリクス照会・トレース取得を 10 回繰り返して数万トークンの生ログを処理した場合でも、統括エージェントの履歴には **圧縮された最終要約の 1 メッセージだけ** が入ります。これがコンテキスト分離の最大の威力です。

#### 2. 役割分担と出力の段階的洗練
`subagent_tool_output` と `final` を比較すると、サブエージェントは「数値の抽出と専門的な一次評価」に集中し、統括エージェントは「見出し（`## checkout-api 健全性報告`）やステータスアイコン（`✅`）を備えたユーザー向けレポートへの統合」を担っていることが分かります。将来ここに「ログ専門サブエージェント」「インシデント履歴サブエージェント」を追加しても、統括エージェントは複数の `ToolMessage` を並列で受け取り、単一の統合レポートへまとめ上げることができます。

### コンテキスト境界の可視化

{% mermaid %}
flowchart TB
    subgraph SupCtx["Supervisor のコンテキストウィンドウ (messages=4)"]
        M1["1. HumanMessage<br/>checkout-api を調査して"]
        M2["2. AIMessage<br/>tool_call: ask_metrics_specialist"]
        M3["3. ToolMessage<br/>サブエージェントの要約結果"]
        M4["4. AIMessage<br/>## checkout-api 健全性報告"]
        M1 --> M2 --> M3 --> M4
    end

    subgraph SubCtx["Metrics Subagent の一時コンテキスト (呼出ごとに破棄)"]
        S1["1. HumanMessage (委譲クエリ)"]
        S2["2. AIMessage (tool_call: fetch_metrics)"]
        S3["3. ToolMessage (生メトリクス文字列)"]
        S4["4. AIMessage (専門評価・要約)"]
        S1 --> S2 --> S3 --> S4
    end

    M2 -.->|"クエリ文字列のみ渡す"| S1
    S4 -.->|"最終要約のみ返す"| M3

    style SupCtx fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style SubCtx fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

---

## 5. 応用：複数専門サブエージェントの並列委譲

Subagents パターンの真価は、複数ドメインを同時調査する場面で発揮されます。以下のように `ask_logs_specialist` を追加するだけで、統括エージェントは 1 回の推論ターンで **2 つのサブエージェントを並列 Tool Call** できます。

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def fetch_error_logs(service: str, minutes: int = 15) -> str:
    """指定サービスの直近エラーログを取得する。"""
    return f"{service} (last {minutes}m): 0 critical errors, 2 timeout warnings"

# ログ分析専門サブエージェント
logs_subagent = create_agent(
    model=make_llm(),
    tools=[fetch_error_logs],
    system_prompt="あなたはログ分析専門のサブエージェントです。異常の有無を簡潔に要約してください。",
)

@tool
def ask_logs_specialist(query: str) -> str:
    """ログ分析専門サブエージェントにエラーログ調査を委譲する。"""
    res = logs_subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return str(res["messages"][-1].content)

# Supervisor に両方の専門サブエージェントを登録（1ターンで並列委譲が可能）
sre_supervisor = create_agent(
    model=make_llm(),
    tools=[ask_metrics_specialist, ask_logs_specialist],
    system_prompt=(
        "あなたは SRE 統括エージェントです。メトリクスとログの調査が必要な場合は、"
        "対応する専門サブエージェントを並行して呼び出し、統合レポートを作成してください。"
    ),
)
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

Subagent-as-Tool パターンを本番運用する際の重要な注意点を 5 つ挙げます。

1. **ツール名と Docstring がルーティング精度のすべてを決める**
   - Supervisor はサブエージェントの内部プロンプトや下位ツールを見ることができません。Supervisor が判断材料にするのは、ラッパー関数（`ask_metrics_specialist`）の **関数名・引数説明・Docstring** だけです。役割が曖昧だったり複数サブエージェント間で説明が重複していると、誤委譲や二重呼び出しが頻発します。
2. **サブエージェントのステートレス性とコンテキスト欠落**
   - デフォルトでは `query: str` しかサブエージェントへ渡らないため、ユーザーとの過去ターンの前提（例：「さっき話したサービス」の指す名前）が欠落することがあります。Supervisor のプロンプトに「サブエージェントへ渡す `query` にはサービス名・時間範囲などの前提条件を自己完結して含めること」と明記するか、`ToolRuntime` を使って必要な State フィールドを明示的に注入してください。
3. **レイテンシの積み上がり（直列 4 回の LLM 呼び出し）**
   - 本ラボの実測でも、1 つのシンプルな調査に `19.42s` を要しています（4 回の LLM 呼び出しが直列に発生するため）。サブエージェント側にはより高速・軽量なモデル（例：mini / flash 系）を割り当て、Supervisor 側のみ高精度モデルを使う **モデル階層化** が本番のレイテンシ・コスト最適化に極めて有効です。
4. **サブエージェント内部での無限ループやタイムアウト伝播**
   - サブエージェントが下位ツールのエラーでループに陥ると、Supervisor 全体がハングします。サブエージェント側の `create_agent` にも必ず `ModelCallLimitMiddleware(run_limit=5, exit_behavior="end")` のような呼び出し回数上限を設定し、ラッパー `@tool` 内で例外をキャッチしてエラー要約文字列を返すフェイルセーフを入れてください。
5. **サブエージェントからの Human-in-the-Loop（割り込み）連携**
   - サブエージェント内部で危険ツールを実行し `HumanInTheLoopMiddleware` で一時停止させたい場合、サブエージェントを単なる `invoke()` で呼ぶだけでは親グラフのチェックポイントと連携しません。サブエージェント側にも Checkpointer 設計と LangGraph サブグラフ統合（または Supervisor 側での事前承認ゲート）を検討する必要があります。

---

## 7. まとめ

本稿では、LangChain v1 の **Subagent-as-Tool パターン** を用いた SRE 健全性調査マルチエージェントを実装し、実測データから以下を検証しました。

- 統括エージェントのメッセージ数は `4` 件に保たれ、下位ツール `fetch_metrics` の生出力が上位コンテキストを汚染しないこと
- `ask_metrics_specialist` がクリーンなコンテキストで一次分析を行い、統括エージェントが最終レポートへ統合する二段階の役割分担が自然に機能すること
- 本番導入では「明確なツール仕様」「自己完結した委譲クエリ」「モデル階層化」「呼び出し回数上限」の 4 点が安定運用の鍵となること

これで LangChain v1 シリーズ（Part 1〜Part 7：理論編・実践編 全 14 本）のコアアーキテクチャから本番運用パターン（Guardrails・HITL・Multi-Agent・MCP）までを一気通貫で網羅しました。ぜひご自身のプロダクト要件に合わせて、必要な Middleware や Multi-Agent パターンを組み合わせてみてください。
