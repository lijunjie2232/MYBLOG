---
title: "LangChain v1 Part 1 実践編：OpenAI 互換 MoE モデル（MiniCPM-V-4.7-35B-A3B）で構築する最小 SRE エージェントと content_blocks 検証"
date: 2026-05-03 15:48:30
categories: [AI]
tags: [LangChain, LangChain v1, MoE, SRE, Tool Calling, Content Blocks, Python]
lang: ja
description: "LangChain v1 の create_agent を使い、OpenAI 互換エンドポイント上の MiniCPM-V-4.7-35B-A3B で最小 SRE エージェントを構築。並列ツール呼び出しのメッセージ履歴と content_blocks の実測出力を詳しく解析します。"
---

# LangChain v1 Part 1 実践編：OpenAI 互換 MoE モデル（MiniCPM-V-4.7-35B-A3B）で構築する最小 SRE エージェントと `content_blocks` 検証

前回の **Part 1 理論編** では、LangChain v1 の核心である **Agent = Model + Harness** の設計思想と、`create_agent`・Models・Messages・`content_blocks` の仕組みを解説しました。

本記事（Part 1 実践編）では、OpenAI 互換 API エンドポイントでホストされた MoE（Mixture of Experts）モデル **`MiniCPM-V-4.7-35B-A3B`** を使い、複数リージョンのサーバー稼働状況を取得・比較する最小 SRE エージェントを実装します。実際にエージェントを実行して得られた生ログをもとに、**なぜメッセージ数が 5 件になるのか**、そして **`content_blocks` がどう正規化されるのか**を裏付けます。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、ハーネスの役割と並列ツール呼び出しについて次のように整理しました。

> **Agent = Model + Harness.**
> エージェントとは「タスクが完了するまでループ内でツールを呼び出し続けるモデル」であり、ハーネス（Harness）とは「そのループを取り巻くすべて（プロンプト、ツール、そしてモデルの振る舞いを制御するミドルウェア）」である。

> モデルが 1 ターンで複数の `tool_calls` を発行した場合（例: 2 つのリージョンのステータスを同時に照会）、Tools Node はそれらをすべて実行し、各 `tool_call["id"]` に 1 対 1 で対応する `ToolMessage` を生成して履歴へ追加する。

今回の実践ラボ（Lab 1）では、以下の 3 点を実測データで検証します。

1. **カスタム OpenAI 互換エンドポイントの接続**: `ChatOpenAI` に環境変数 `BASE_URL` / `API_KEY` を渡して `create_agent` に組み込む最小構成。
2. **並列 Tool Calling の挙動**: 「`ap-northeast-1` と `us-east-1` のステータスを確認して比較してください」という 1 つの依頼に対し、モデルが 2 つのツール呼び出しを 1 ターンに束ねて発行するか。
3. **`content_blocks` の正規化**: 文字列レスポンスを返す OpenAI 互換モデルに対して `res["messages"][-1].content_blocks` が標準の `TextContentBlock` リストを返すか。

---

## 2. 実験アーキテクチャと実装コード

今回の最小 SRE エージェントのデータフローを以下に示します。

{% mermaid %}
flowchart LR
    U["User<br/>ap-northeast-1 と us-east-1<br/>を比較して"] --> A["create_agent<br/>(MiniCPM-V-4.7-35B-A3B)"]
    A -->|"Call #1: ap-northeast-1"| T1["get_server_status"]
    A -->|"Call #2: us-east-1"| T2["get_server_status"]
    T1 -->|"HEALTHY (CPU 32%, 14ms)"| A
    T2 -->|"DEGRADED (CPU 89%, 210ms)"| A
    A --> OUT["最終 AIMessage<br/>& content_blocks 抽出"]

    style A fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style T1 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style T2 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style OUT fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

### 実装コード（Lab 1）

API キーやエンドポイント URL はコード内にハードコードせず、`.env` ファイル経由で `os.getenv("BASE_URL")` および `os.getenv("API_KEY")` から読み込みます。

```python
import json
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# .env から環境変数を読み込む（APIキーのハードコードは厳禁）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(model: str = MODEL_NAME, temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI互換エンドポイント向けの ChatOpenAI インスタンスを生成する。"""
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,      # カスタム推論エンドポイント
        api_key=API_KEY,        # 環境変数から取得したAPIキー
        temperature=temperature,  # ツール呼び出しの再現性を優先して 0.0
        max_tokens=512,         # 過剰出力による遅延・コスト増を防止
        timeout=30,             # ネットワークハング対策
        max_retries=1,          # 一時的エラーへの最小限リトライ
    )


def run_lab1():
    llm = make_llm()

    # @tool デコレータにより、引数型ヒントと docstring から JSON Schema が自動生成される
    @tool
    def get_server_status(region: str) -> str:
        """指定リージョンのサーバー稼働状況を取得する。"""
        statuses = {
            "ap-northeast-1": "HEALTHY (CPU 32%, latency 14ms)",
            "us-east-1": "DEGRADED (CPU 89%, latency 210ms)",
        }
        # 未知のリージョンが渡された場合でも例外ではなく安全な文字列を返す
        return statuses.get(region, f"UNKNOWN region: {region}")

    # create_agent で最小のハーネス（Model + Tools + System Prompt）を組み立てる
    agent = create_agent(
        model=llm,
        tools=[get_server_status],
        system_prompt="あなたはSREアシスタントです。ツールを使って簡潔に日本語で回答してください。",
    )

    # 辞書形式のメッセージは LangGraph/LangChain ハーネス内部で HumanMessage に正規化される
    res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "ap-northeast-1 と us-east-1 のステータスを確認して比較してください。",
                }
            ]
        }
    )

    # 全メッセージ履歴の型・本文・ツール呼び出し要求を抽出
    msgs = [
        {
            "type": m.type,
            "content": m.content,
            "tool_calls": getattr(m, "tool_calls", None),
        }
        for m in res["messages"]
    ]
    # LangChain v1 の標準化プロパティ content_blocks を検証
    last_blocks = res["messages"][-1].content_blocks

    return {
        "message_count": len(msgs),
        "messages": msgs,
        "content_blocks": last_blocks,
    }


if __name__ == "__main__":
    result = run_lab1()
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

## 3. 実測出力の解析（実行時間: 6.58 秒）

実際に上記スクリプトを実行して得られた JSON 出力が以下です（実行時間 `elapsed_s: 6.58`、ステータス `ok`）。

```json
{
  "message_count": 5,
  "messages": [
    {
      "type": "human",
      "content": "ap-northeast-1 と us-east-1 のステータスを確認して比較してください。",
      "tool_calls": null
    },
    {
      "type": "ai",
      "content": "",
      "tool_calls": [
        {
          "name": "get_server_status",
          "args": {
            "region": "ap-northeast-1"
          },
          "id": "chatcmpl-tool-a8ef84d3e37d1416",
          "type": "tool_call"
        },
        {
          "name": "get_server_status",
          "args": {
            "region": "us-east-1"
          },
          "id": "chatcmpl-tool-9161054d5c5252ec",
          "type": "tool_call"
        }
      ]
    },
    {
      "type": "tool",
      "content": "HEALTHY (CPU 32%, latency 14ms)",
      "tool_calls": null
    },
    {
      "type": "tool",
      "content": "DEGRADED (CPU 89%, latency 210ms)",
      "tool_calls": null
    },
    {
      "type": "ai",
      "content": "ap-northeast-1 と us-east-1 のサーバーステータスの比較結果は以下の通りです。\n\n*   **ap-northeast-1**: HEALTHY (CPU 32%, latency 14ms)\n*   **us-east-1**: DEGRADED (CPU 89%, latency 210ms)\n\nap-northeast-1 は正常な状態ですが、us-east-1 は負荷が高く（CPU 89%）、レイテンシも遅延（210ms）しているため、状態が劣化しています。",
      "tool_calls": []
    }
  ],
  "content_blocks": [
    {
      "type": "text",
      "text": "ap-northeast-1 と us-east-1 のサーバーステータスの比較結果は以下の通りです。\n\n*   **ap-northeast-1**: HEALTHY (CPU 32%, latency 14ms)\n*   **us-east-1**: DEGRADED (CPU 89%, latency 210ms)\n\nap-northeast-1 は正常な状態ですが、us-east-1 は負荷が高く（CPU 89%）、レイテンシも遅延（210ms）しているため、状態が劣化しています。"
    }
  ]
}
```

### なぜこの挙動になったのか？（メッセージ遷移の解剖）

出力から、LangChain v1 ハーネスと MoE モデルの協調動作が明確に読み取れます。

{% mermaid %}
sequenceDiagram
    participant U as User
    participant M as Model Node (MiniCPM-V-4.7-35B-A3B)
    participant T as Tools Node (create_agent)

    U->>M: [1] HumanMessage: 2リージョンを比較して
    Note over M: 2リージョンの情報が不足していると判断
    M->>T: [2] AIMessage (content="", tool_calls=2件並列発行)
    Note over T: id: a8ef84d3... と id: 9161054d... をディスパッチ
    T-->>M: [3] ToolMessage: HEALTHY (CPU 32%, latency 14ms)
    T-->>M: [4] ToolMessage: DEGRADED (CPU 89%, latency 210ms)
    Note over M: 2件の結果を統合・比較し tool_calls=[] で完了
    M-->>U: [5] AIMessage: 比較レポート (content_blocks=[{type: "text", ...}])
{% endmermaid %}

1. **なぜ `message_count` が 5 件なのか？**
   - もしモデルが逐次的に 1 リージョンずつ問い合わせていた場合、`Human -> AI(call1) -> Tool(1) -> AI(call2) -> Tool(2) -> AI(final)` の **6 メッセージ・LLM 呼び出し 3 回** になります。
   - 今回は `MiniCPM-V-4.7-35B-A3B` が 2 つのリージョン調査を独立したサブタスクと認識し、メッセージ `[2]` の `AIMessage` 内で **2 件の `tool_calls` を同時発行（Parallel Tool Calling）** しました。
   - その結果、`Tools Node` が 2 件の `ToolMessage`（`[3]`, `[4]`）を返し、LLM 呼び出しはわずか **2 往復（合計 5 メッセージ）** で完結しています。推論ラウンド数が 1 回減ることでレイテンシとトークン消費の両方が大幅に削減されます。
2. **メッセージ `[2]` の `content` が空文字 `""` である理由**
   - ツール呼び出しのみを行うターンでは、OpenAI 互換 API はテキスト本文を生成せず `tool_calls` 配列だけを返します。したがって `m.content` は `""` になり、フロントエンド等でチャット表示する際は「`m.content` が空で `m.tool_calls` が非空の `AIMessage` はツール実行中インジケータとして扱う」設計が必要です。
3. **`content_blocks` の正規化結果**
   - 理論編で述べた通り、次のように `content_blocks` が機能しています。
   > LangChain v1 のすべてのメッセージオブジェクトは、**`content_blocks` プロパティ**を実装しています。これにアクセスすると、元の `content`（文字列でもプロバイダ固有の辞書リストでも）が遅延評価によって**プロバイダ非依存の標準 TypedDict リスト**へと正規化されます。
   - 実際、最終メッセージの `content` はプレーンな文字列でしたが、`res["messages"][-1].content_blocks` は `[{"type": "text", "text": "..."}]` という厳密なリスト構造に変換されました。将来的に Anthropic の Extended Thinking や OpenAI の Reasoning モデルへ切り替えても、呼び出し元のパース処理を変更せずに済みます。

---

## 4. 注意点・ハマりどころ（Pay Attention）

実務で OpenAI 互換エンドポイントと `create_agent` を組み合わせる際、以下のポイントでつまずきやすいため注意してください。

1. **OpenAI 互換エンドポイントのモデル名検証エラーに注意**
   - `create_agent(model="openai:MiniCPM-V-4.7-35B-A3B", ...)` のように文字列ショートカットを使うと、環境変数 `OPENAI_BASE_URL` の設定漏れやプロバイダプロファイル解決で意図しないエンドポイントへ飛ぶことがあります。カスタムエンドポイントを使う場合は、本記事のように `ChatOpenAI(model=..., base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))` を明示的に組み立てて渡すのが安全です。
2. **`system_prompt` は `res["messages"]` に含まれない**
   - 実測出力の `message_count` が 5 件であることからも分かる通り、`create_agent(..., system_prompt=...)` で渡したシステムプロンプトはグラフ状態の `messages` 配列には永続化されず、Model Node がモデルを呼び出す直前に先頭へ付与されます。履歴の長さをカウントする際や監査ログを取る際は、`SystemMessage` が含まれていない前提で設計してください。
3. **Pydantic v2 のシリアライズ警告への対処**
   - 一部の OpenAI 互換レスポンスや LangChain 内部スキーマでは、実行時に Pydantic の `UserWarning` が出ることがあります。ログの可読性を保つため、必要に応じて `warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")` を設定するとクリーンな出力を維持できます。
4. **ツール内の未知入力に対する防御的実装**
   - `get_server_status` では `statuses.get(region, f"UNKNOWN region: {region}")` のように `KeyError` を投げず文字列を返しています。モデルがハルシネーションで存在しないリージョン名（例: `"tokyo"`）を渡した場合でも、エラー文字列を返せばモデル自身が次のターンで `"ap-northeast-1"` へ自己修正できます。

---

## 5. まとめ

Part 1 実践編では、`MiniCPM-V-4.7-35B-A3B` と `create_agent` を組み合わせた最小 SRE エージェントを実行し、**並列ツール呼び出しによる 5 メッセージ遷移**と **`content_blocks` の標準化** を実測データで確認しました。

次回の **Part 2 理論編** では、ツール設計をもう一段深掘りし、Pydantic `args_schema` による厳格な入力制約、`ToolRuntime` を使った安全なコンテキスト注入（モデルに秘密情報を見せない設計）、そしてツール実行エラーのリカバリ戦略を解説します。
