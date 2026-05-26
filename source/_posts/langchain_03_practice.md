---
title: "LangChain v1 Part 3 実践編：ToolStrategy によるインシデント構造化抽出と updates/messages マルチモードストリーミング"
date: 2026-05-12 09:35:38
categories: [AI]
tags: [LangChain, LangChain v1, Structured Output, ToolStrategy, Streaming, Pydantic, Python]
lang: ja
description: "MiniCPM-V-4.7-35B-A3B に対して ToolStrategy で SRE インシデント報告書（IncidentReport）を構造化抽出し、さらに updates/messages マルチモードストリーミングで27チャンクの逐次生成を実測・解説します。"
---

# LangChain v1 Part 3 実践編：`ToolStrategy` によるインシデント構造化抽出と `updates`/`messages` マルチモードストリーミング

前回の **Part 3 理論編** では、構造化出力の 2 大戦略（`ProviderStrategy` と `ToolStrategy`）と、マルチモードストリーミング（`updates` / `messages` / `custom`）および Event Streaming v3 の設計を解説しました。

本記事（Part 3 実践編）では、OpenAI 互換エンドポイント上の `MiniCPM-V-4.7-35B-A3B` を使い、以下の 2 つを 1 つのラボ（**Lab 3**）で実証します。

1. **SRE 障害テキストからの構造化抽出**: `ToolStrategy(IncidentReport)` を使い、非構造テキストから `service` / `severity` / `root_cause` / `action_items` を Pydantic インスタンスとして抽出する。
2. **マルチモードストリーミング**: `stream_mode=["updates", "messages"]` を指定し、`messages` チャンネルから逐次トークン（チャンク）を受信・結合する。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、OpenAI 互換エンドポイントにおける `ToolStrategy` の重要性とマルチモード配信の形式について次のように整理しました。

> そのため、**OpenAI 互換エンドポイントでは `response_format=ToolStrategy(IncidentReport)` と明示指定するのが実務上の鉄則**です。これにより、モデルが普段通り得意とする Function Calling（Tool Calling）経路を使って構造化データを安全に抽出できます。

> デフォルト（v1 形式）で複数モードを指定すると、ループは `for mode, chunk in agent.stream(..., stream_mode=["updates", "messages"]):` のように **`(mode, payload)` タプル** を返します。

今回の Lab 3 では、次の 2 点を実測ログで確認します。

1. 非構造の障害ログ（`payment-gateway` の 503 エラー・DB コネクションプール枯渇）から、重大度 `P1` や配列形式の `action_items` まで含めて `res["structured_response"]` へ正確にマッピングされるか。
2. `stream_mode=["updates", "messages"]` のループ内で `mode == "messages"` のチャンクをフィルタし、空チャンクを除いたトークンが欠損なく 1 文の日本語として結合されるか。

---

## 2. 実験アーキテクチャと実装コード

Lab 3 の 2 つの実験（構造化抽出フェーズとマルチモードストリーミングフェーズ）のフローを以下に示します。

{% mermaid %}
flowchart TD
    subgraph P1["フェーズ1: ToolStrategy による構造化抽出"]
        LOG["非構造インシデントログ<br/>(payment-gateway / 503 / プール枯渇)"] --> A1["create_agent<br/>response_format=ToolStrategy(IncidentReport)"]
        A1 -->|"人工 Tool Call + Pydantic 検証"| SR["res['structured_response']<br/>IncidentReport インスタンス"]
    end

    subgraph P2["フェーズ2: updates / messages マルチモード配信"]
        Q["質問: ストリーミングの利点を1文で"] --> A2["stream_agent.stream<br/>stream_mode=['updates', 'messages']"]
        A2 -->|"mode == 'messages'"| CH["(AIMessageChunk, meta)<br/>非空 content を27チャンク収集"]
        A2 -->|"mode == 'updates'"| UP["ノード完了差分<br/>(ステップ進捗監視用)"]
        CH --> JOIN["''.join(chunks) で完全テキスト復元"]
    end

    style P1 fill:#f0f7ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style P2 fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
    style SR fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style JOIN fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
{% endmermaid %}

### 実装コード（Lab 3）

```python
import json
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# .env からエンドポイントとAPIキーを読み込む（ハードコード禁止）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(model: str = MODEL_NAME, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI互換エンドポイント向けの ChatOpenAI インスタンスを生成する。"""
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=30,
        max_retries=1,
    )


def run_lab3():
    llm = make_llm()

    # 1. 抽出対象の Pydantic スキーマを定義
    # 各フィールドの description が ToolStrategy 内部でツール引数スキーマに変換される
    class IncidentReport(BaseModel):
        service: str = Field(description="障害が発生したサービス名")
        severity: str = Field(description="重大度: P0, P1, P2")
        root_cause: str = Field(description="推定原因の要約")
        action_items: list[str] = Field(description="推奨アクションのリスト")

    # OpenAI 互換エンドポイントでも確実に動くよう ToolStrategy を明示指定する
    agent = create_agent(
        model=llm,
        tools=[],
        response_format=ToolStrategy(IncidentReport),
        system_prompt="インシデントログから構造化レポートを抽出してください。",
    )

    res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "決済API(payment-gateway)でDBコネクションプール枯渇により"
                        "503エラーが5分間発生。即時にプール上限を50→150へ引き上げ、"
                        "スロークエリを調査する必要がある。"
                    ),
                }
            ]
        }
    )
    # 検証済みの Pydantic インスタンスを辞書化して確認する
    structured = res["structured_response"].model_dump()

    # 2. マルチモードストリーミング（updates と messages を同時購読）
    stream_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="1文で簡潔に答えてください。",
    )
    chunks: list[str] = []

    # stream_mode にリストを渡すと (mode, chunk) のタプルが逐次 yield される
    for mode, chunk in stream_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "LangChainのストリーミングの利点を1文で教えてください。",
                }
            ]
        },
        stream_mode=["updates", "messages"],
    ):
        # messages モードでは (AIMessageChunk, metadata) のタプルが届く
        if mode == "messages":
            msg, meta = chunk
            # ツール呼び出し用や終了通知用の空チャンクを除外して本文トークンのみ収集
            if msg.content:
                chunks.append(msg.content)

    return {
        "structured_response": structured,
        "stream_chunk_count": len(chunks),
        "stream_text": "".join(chunks),
    }


if __name__ == "__main__":
    print(json.dumps(run_lab3(), indent=2, ensure_ascii=False))
```

---

## 3. 実測出力の解析（実行時間: 7.70 秒）

実際に Lab 3 を実行して得られた出力が以下です（2 フェーズ合計の実行時間 `elapsed_s: 7.7`、ステータス `ok`）。

```json
{
  "structured_response": {
    "service": "payment-gateway",
    "severity": "P1",
    "root_cause": "DBコネクションプールの枯渇",
    "action_items": [
      "プール上限を50から150へ引き上げる",
      "スロークエリの調査を実施する"
    ]
  },
  "stream_chunk_count": 27,
  "stream_text": "リアルタイムで結果を逐次表示できるため、ユーザー体験が向上し、長時間の処理でも待機ストレスを軽減できます。"
}
```

### なぜこの挙動になったのか？

{% mermaid %}
sequenceDiagram
    participant App as アプリケーション
    participant AG as create_agent
    participant LLM as MiniCPM-V-4.7-35B-A3B

    Note over App,LLM: フェーズ1: ToolStrategy による構造化抽出
    App->>AG: invoke(障害ログ)
    AG->>LLM: IncidentReport を仮想ツールとして送信
    LLM-->>AG: tool_calls=[IncidentReport(service="payment-gateway", severity="P1", ...)]
    Note over AG: Pydantic バリデーション成功 → structured_response に格納
    AG-->>App: res["structured_response"] (型安全な IncidentReport)

    Note over App,LLM: フェーズ2: updates + messages マルチモード配信
    App->>AG: stream(質問, stream_mode=["updates", "messages"])
    AG->>LLM: ストリーミング推論開始
    loop 27回の非空トークンチャンク
        LLM-->>AG: AIMessageChunk(content="...")
        AG-->>App: ("messages", (msg, meta))
    end
    AG-->>App: ("updates", {"model": {"messages": [最終 AIMessage]}})
{% endmermaid %}

1. **プロンプトに書いていない `severity: "P1"` まで推論・構造化された理由**
   - 入力ログには「重大度 P1」とは明記されていませんでしたが、`IncidentReport` スキーマの `severity: str = Field(description="重大度: P0, P1, P2")` という記述が `ToolStrategy` 経由でツールの JSON Schema に埋め込まれました。
   - モデルは「決済 API で 503 エラーが 5 分間発生」という文脈とスキーマ定義を照合し、妥当な `P1` を自律的に判定したうえで、2 つの推奨アクションも `list[str]` 配列へ綺麗に分解しています。
2. **`stream_chunk_count: 27` の意味と空チャンクフィルタリングの重要性**
   - 53 文字の日本語文が **27 個の非空トークンチャンク** に分割されてリアルタイム配信されました（平均 約 2 文字 / チャンク）。
   - `if msg.content:` のガードを入れているのは極めて重要です。OpenAI 互換ストリーミングでは、ロール宣言のみを含む先頭チャンクや `finish_reason="stop"` のみを運ぶ末尾チャンクで `msg.content == ""` が必ず発生するため、空チャンクを除外することで UI への無駄な再描画や配列膨張を防いでいます。
3. **`updates` と `messages` を同時指定しても本文が二重化しない理由**
   - `stream_mode=["updates", "messages"]` では、トークン配信の最後に `mode == "updates"` で完成済みの `AIMessage` が丸ごと 1 回届きます。コード側で `if mode == "messages":` と明確にチャンネルを分岐したため、最終テキストが二重に連結されることなく、`"".join(chunks)` でちょうど 1 文が復元されました。

---

## 4. 注意点・ハマりどころ（Pay Attention）

構造化出力とストリーミングを本番アプリへ導入する際は、以下の落とし穴に注意してください。

1. **`response_format=IncidentReport`（直接指定）が OpenAI 互換エンドポイントで失敗するケース**
   - `ChatOpenAI` クラスは既定で OpenAI 互換プロファイルを参照するため、スキーマクラスを直接渡すとネイティブの `ProviderStrategy`（`response_format: {type: "json_schema", ...}`）を試行し、バックエンドの推論サーバが非対応で 400 エラーを返すことがあります。独自エンドポイントでは必ず `ToolStrategy(IncidentReport)` を明示してください。
2. **構造化出力（`ToolStrategy`）とテキストストリーミングの相性**
   - `ToolStrategy` は内部的に Tool Calling の JSON 引数として構造化データを生成するため、`stream_mode="messages"` を流しても `msg.content`（通常テキスト）は空になり、代わりに `msg.tool_call_chunks` へ部分 JSON 断片が流れます。構造化出力エージェントの進捗を UI に見せたい場合は、`updates` モードでステップ完了を通知するか、Part 3 理論編で紹介した `stream_events(..., version="v3")` の `message.tool_calls` 投影を利用してください。
3. **サブグラフ内エージェントのストリーミング漏れ**
   - `create_agent` の戻り値（`CompiledStateGraph`）をさらに親の `StateGraph` にノードとして組み込んだ場合、親グラフ側で `stream_mode="messages"` だけを呼んでも内部エージェントのトークンは流れてきません。必ず `subgraphs=True` を指定するか、Event Streaming v3 の `stream.subagents` を使用してください。
4. **`severity` 等の列挙値は本番では `Literal["P0", "P1", "P2"]` にする**
   - 今回のラボでは `severity: str` として説明文で誘導しましたが、本番の障害チケット自動起票システム等では `Literal["P0", "P1", "P2"]` を指定すべきです。これにより万が一モデルが `"High"` などの不正値を返しても、`ToolStrategy` の `handle_errors=True` が Pydantic の `ValidationError` を捕捉し、自動で `"P0" / "P1" / "P2"` へ自己修正してくれます。

---

## 5. まとめ

Part 3 実践編では、`MiniCPM-V-4.7-35B-A3B` に対して **`ToolStrategy(IncidentReport)`** による型安全なインシデント構造化抽出と、**`stream_mode=["updates", "messages"]`** による 27 チャンクのマルチモードストリーミングを実証しました。

ここまでで、LangChain v1 の Core Primitives（`create_agent` / Models / Messages / Tools / `ToolRuntime` / Structured Output / Streaming）が一通り揃いました。

次回の **Part 4** では、エージェントに「会話の文脈を覚える力」と「セッションを跨いでユーザー知識を蓄積する力」を与える **Short-term Memory（Checkpointer）と Long-term Memory（Store）** の理論と実践へ進みます。
