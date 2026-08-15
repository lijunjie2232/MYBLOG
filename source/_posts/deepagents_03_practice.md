---
title: "DeepAgents Part 3 実践編：大量ログ出力（dump_large_log）のハンドリング検証とオフロード閾値チューニングの勘所"
date: 2026-07-23 09:56:29
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, Context Engineering, Offloading, grep, Python]
lang: ja
description: "大量ログを返す dump_large_log ツールを使い、DeepAgents の自動オフロード機構を実測検証。600 行（約 36KB）でオフロードされなかった理由と、1,400 行（約 120KB）で /large_tool_results/ へ退避されエージェントが自律的に grep で ERROR 行を特定した実測ログを対比解説します。"
---

# DeepAgents Part 3 実践編：大量ログ出力（`dump_large_log`）のハンドリング検証とオフロード閾値チューニングの勘所

前回の **Part 3 理論編** では、DeepAgents がコンテキスト膨張を防ぐための 4 つの柱（Large Tool Result Offloading・Summarization・Overflow Recovery・Prompt Caching）を解説しました。

本記事（Part 3 実践編）では、大量のアプリケーションログを返すツール `dump_large_log` を実装し、OpenAI 互換 MoE モデル **`MiniCPM-V-4.7-35B-A3B`** と `StateBackend()` を使って **自動オフロードの境界条件** を実測検証します。具体的には、**600 行（約 36KB）ではオフロードが発動せずインライン保持された初回実験** と、**1,400 行（約 120KB）へ増量したことで実際に `/large_tool_results/` への退避と自律 `grep` 検索が発動した第 2 回実験** を正面から対比し、本番運用における閾値チューニングの勘所を明らかにします。

---

## 1. 理論編の振り返りと今回の検証仮説

理論編では、`FilesystemMiddleware` の自動オフロード条件と挙動を次のように整理しました。

> `FilesystemMiddleware` は、すべてのツール入出力を監視し、**デフォルト閾値 20,000 トークン（ASCII テキストで約 80KB 相当）** を超える巨大なペイロードを自動的に仮想ファイルシステムへ退避（Offload）します。

> 巨大なログや検索結果が返った瞬間、生のテキストを `ToolMessage` に入れる代わりにデフォルト Backend の `/large_tool_results/<tool_call_id>` へ保存します。`ToolMessage` には **保存先パスと先頭 10 行のプレビュー** だけが残され、モデルは以降 `grep` や `read_file(offset=..., limit=...)` を使って必要な行だけをオンデマンドに引き出します。

この仕様に基づくと、ツール出力サイズによって以下の 2 通りの挙動分岐が起きるはずです。

- **ケース A（600 行・約 36KB）**: 概算 ~9,000 トークン < 20,000 トークン閾値 → **オフロード非発動**。ログ全文が `ToolMessage` にインライン展開され、`state["files"]` は空（`[]`）のまま。
- **ケース B（1,400 行・約 120KB）**: 概算 ~30,000 トークン > 20,000 トークン閾値 → **オフロード発動**。ログ全文が `/large_tool_results/<tool_call_id>` に退避され、モデルが自力で `grep` を呼んで末尾の `ERROR` 行を抽出する。

{% mermaid %}
flowchart LR
    subgraph CASEA["ケース A: 600行 (約 36KB ≒ 9,000 tokens)"]
        A1["dump_large_log<br/>601 行出力"] --> A2{"20,000 tokens<br/>(≒80KB) 超過?"}
        A2 -->|"No"| A3["ToolMessage に全文インライン<br/>files_in_state = []<br/>grep 呼び出しなし"]
    end

    subgraph CASEB["ケース B: 1,400行 (約 120KB ≒ 30,000 tokens)"]
        B1["dump_large_log<br/>1,401 行出力"] --> B2{"20,000 tokens<br/>(≒80KB) 超過?"}
        B2 -->|"Yes!"| B3["/large_tool_results/<br/>chatcmpl-tool-954917c8055113ae<br/>へ自動退避"]
        B3 --> B4["ToolMessage はパス+先頭10行のみ<br/>→ モデルが自律的に grep を実行<br/>→ ERROR 行のみを特定・報告"]
    end

    style CASEA fill:#edf2f7,stroke:#4a5568,stroke-width:1.5px,color:#2d3748
    style CASEB fill:#e6fffa,stroke:#2c7a7b,stroke-width:2px,color:#234e52
    style A2 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style B2 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style B3 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style B4 fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

---

## 2. 実験コードの対比（600 行版 vs 1,400 行版）

まず初回検証（ケース A：600 行版）では、1 行あたり約 60 バイトのログを 600 行＋末尾に `ERROR` 行を 1 行生成しました。合計サイズは約 36KB です。

```python
# ケース A（600 行・約 36KB）: デフォルト閾値（約 80KB）を下回るためオフロードされない
@tool
def dump_large_log_600(service: str) -> str:
    """約 36KB のログ行を返すシミュレータ（デフォルト閾値未満）。"""
    lines = [
        f"2026-07-20T10:00:{i:02d}Z INFO {service} req={i} latency={10 + (i % 5)}ms"
        for i in range(600)
    ]
    lines.append(
        f"2026-07-20T10:10:00Z ERROR {service} CRITICAL_TIMEOUT db_pool_exhausted"
    )
    return "\n".join(lines)
```

これに対し、実際にデフォルト閾値（20,000 トークン ≒ 80KB）を確実に跨ぐよう 1 行あたり約 85 バイト × 1,400 行（合計 **約 120KB**）へ引き上げたのが、以下のケース B 実装です。

```python
import json
import os
import time
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

# Pydantic 互換性警告を抑止し、.env から接続情報を取得（ハードコード厳禁）
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
    max_tokens=512,
    timeout=40,
    max_retries=2,
)


def run_lab3_offload():
    t0 = time.perf_counter()

    # 約 120KB（1,401 行）のログを返すシミュレータ：20,000 トークン閾値を超過させる
    @tool
    def dump_large_log(service: str) -> str:
        """大量のログ行（約120KB）を返すシミュレータ。"""
        lines = [
            (
                f"2026-07-20T10:{(i // 60) % 60:02d}:{i % 60:02d}Z INFO {service} "
                f"req_id=req-{i:05d} latency={10 + (i % 7)}ms status=200 payload=ok"
            )
            for i in range(1400)
        ]
        # 1,401 行目（最末尾）に唯一の ERROR 行を埋め込む
        lines.append(
            f"2026-07-20T10:59:59Z ERROR {service} CRITICAL_TIMEOUT db_pool_exhausted active=150/150"
        )
        return "\n".join(lines)

    # StateBackend を使うことで、オフロードされたファイルは res["files"] に格納される
    agent = create_deep_agent(
        model=llm,
        tools=[dump_large_log],
        backend=StateBackend(),
        system_prompt=(
            "大量ログがオフロードされた場合は grep や read_file で "
            "ERROR 行だけを確認して簡潔に報告してください。"
        ),
    )

    res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "billing-api のログを取得して、ERROR 行の内容を報告してください。",
                }
            ]
        }
    )

    # グラフ状態 (StateBackend) に退避された仮想ファイルパスと ToolMessage プレビューを抽出
    files_in_state = list(res.get("files", {}).keys())
    tool_msgs = [str(m.content)[:220] for m in res["messages"] if m.type == "tool"]

    return {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "files_in_state": files_in_state,
        "tools_used": [
            tc["name"]
            for m in res["messages"]
            for tc in getattr(m, "tool_calls", [])
        ],
        "tool_msg_previews": tool_msgs,
        "final": res["messages"][-1].content,
    }


if __name__ == "__main__":
    print(json.dumps(run_lab3_offload(), indent=2, ensure_ascii=False))
```

---

## 3. 実測結果の対比と詳細解析

### ケース A（600 行・約 36KB）で起きたこと

600 行版では、`dump_large_log` の出力文字列長が約 36KB（約 9,000 トークン相当）であり、`FilesystemMiddleware` のデフォルト退避閾値（20,000 トークン ≒ 80KB）を下回りました。その結果、**オフロードは一切発動せず**、`files_in_state` は空 `[]`、`tools_used` は `["dump_large_log"]` のみとなり、モデルは約 36KB の生ログ全文をプロンプト内から直接目視スキャンして回答しました。

「600 行もあるのにオフロードされない！」と最初は驚くかもしれませんが、DeepAgents の文字数→トークン換算（約 4 文字/トークン基準）では 20,000 トークン ≒ 80,000 文字がボーダーラインであるため、仕様通りの挙動です。

### ケース B（1,400 行・約 120KB）の実測生出力

出力を約 120KB に引き上げたケース B の実測出力（`deepagents_labs.json`）は以下の通りです。

```json
{
  "lab3_context_offload": {
    "status": "ok",
    "elapsed_s": 63.33,
    "output": {
      "files_in_state": [
        "/large_tool_results/chatcmpl-tool-954917c8055113ae"
      ],
      "tools_used": [
        "dump_large_log",
        "grep"
      ],
      "tool_msg_previews": [
        "Tool result too large, the result of this tool call chatcmpl-tool-954917c8055113ae was saved in the filesystem at this path: /large_tool_results/chatcmpl-tool-954917c8055113ae\n\nYou can read the result from the filesystem",
        "/large_tool_results/chatcmpl-tool-954917c8055113ae"
      ],
      "final": "grep で ERROR 行を検索しました。結果は以下の通りです：\n\n**billing-api の ERROR 行:**\n\n```\n2026-07-20T10:59:59Z ERROR billing-api CRITICAL_TIMEOUT db_pool_exhausted active=150/150\n```\n\n**内容の概要:**\n- **タイムスタンプ:** 2026-07-20T10:59:59Z\n- **エラー内容:** `CRITICAL_TIMEOUT` - データベース接続プールが Exhausted（最大値 150/150 に達している）\n- **影響:** 接続プールがすべて使用中であり、新しい接続が取得できない状態\n\nこれがログ内に検出された唯一の ERROR 行です。"
    }
  }
}
```

### このトレースが示す 4 つの重要ポイント

1. **`files_in_state` に実退避ファイルが出現**
   `"/large_tool_results/chatcmpl-tool-954917c8055113ae"` が `StateBackend` の `files` 辞書に自動生成されました。ファイル名にはそのままツール呼び出しの `tool_call_id` が使われるため、同一スレッド内で複数回大容量ツールを呼んでも衝突しません。
2. **`ToolMessage` がポインタ＋プレビューへ自動置換**
   `tool_msg_previews[0]` に注目してください。120KB の生ログは履歴から完全に排除され、代わりに `"Tool result too large, the result of this tool call ... was saved in the filesystem at this path: /large_tool_results/..."` という軽量メッセージ＋先頭 10 行プレビューだけがモデルに届いています。
3. **モデルが自律的に `grep` を選択（`tools_used: ["dump_large_log", "grep"]`）**
   先頭 10 行プレビューには `INFO` 行しか載っておらず、目的の `ERROR` 行は 1,401 行目（最末尾）に隠れています。モデルは「`read_file` で最初から全行読むと無駄」と判断し、即座に組み込みの **`grep` ツール** を呼び出して 1,401 行の中から唯一の `ERROR` 行をピンポイント抽出しました。
4. **なぜ所要時間が `63.33s` かかったのか**
   Lab 1（2.7s）や Lab 2（7.23s）に比べて 63.33 秒と長いのは、次の 3 要因が重なったためです。
   - **LLM ラウンドトリップの増加**: ① `dump_large_log` 発行 → ② オフロード通知を読んで `grep` 発行 → ③ `grep` 結果から最終構造化レポート生成、の **3 回の推論ターン** が発生している。
   - **生成トークン量の増加**: 最終回答が箇条書き＋コードブロック付きの丁寧な日本語レポートになっている。
   - **共有推論エンドポイントのキュー待機**: OpenAI 互換エンドポイント側のレート制御・キューイングと長文生成が重なった影響。

{% mermaid %}
sequenceDiagram
    autonumber
    participant User as ユーザー
    participant Agent as MiniCPM-V-4.7-35B-A3B
    participant MW as FilesystemMiddleware
    participant State as StateBackend (files)

    User->>Agent: "billing-api のログを取得して ERROR 行を報告して"
    Agent->>MW: Turn 1: dump_large_log("billing-api")
    Note over MW: 出力 約 120KB > 20,000 tokens 閾値！
    MW->>State: 保存: /large_tool_results/chatcmpl-tool-954917c8055113ae
    MW-->>Agent: ToolMessage: "Tool result too large... saved at /large_tool_results/..." + 先頭10行
    Agent->>MW: Turn 2: grep(pattern="ERROR", path="/large_tool_results/...")
    MW->>State: 仮想FS内でパターン検索
    State-->>MW: 1,401行目の ERROR 行のみ返却
    MW-->>Agent: ToolMessage: ERROR 行のマッチ結果
    Agent-->>User: Turn 3: CRITICAL_TIMEOUT / db_pool_exhausted の詳細レポート (計 63.33s)
{% endmermaid %}

---

## 4. 注意点・ハマりどころ（Pay Attention）

実務で大容量ツール出力とコンテキスト管理を運用する際、以下の 4 点は必ずチューニング対象として検討してください。

### ① デフォルト 20,000 トークン（約 80KB）は 32K〜64K モデルには大きすぎる
今回のケース A（600 行・約 36KB）で実証した通り、デフォルトのオフロード閾値 20,000 トークンは、Claude や Gemini のような 200K〜1M クラスの大型コンテキストモデルを前提に設計されています。コンテキスト上限が **32K〜64K のローカル／MoE モデル** で 36KB（約 9,000 トークン）のログが 2〜3 回インラインに入ると、オフロードが一度も発動しないまま一気にコンテキストを圧迫します。
- **対策**: `FilesystemMiddleware` を上書きして閾値（`tool_token_limit_before_evict` 等）をモデル容量に合わせて引き下げるか、ツール側でページネーション・件数上限（`limit=50`）を設計する。

### ② 自前ホストモデルの `SummarizationMiddleware` フォールバック（170,000 トークン）問題
Part 3 理論編でも触れた通り、`ChatOpenAI(model="MiniCPM-V-4.7-35B-A3B", base_url=...)` のように LangChain の標準モデルプロファイルに未登録のモデルを使うと、`SummarizationMiddleware` は 85% 計算の基準となる `max_input_tokens` を取得できず、**フォールバック値 `170,000` トークン** を採用します。実上限が 32K のモデルでは、要約が走る前に API がコンテキスト超過エラーを返すため、必ず以下のように明示オーバーライドしてください。

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import SummarizationMiddleware

backend = StateBackend()

# プロファイル未登録の独自モデルでは trigger を明示指定するのが必須
agent = create_deep_agent(
    model=llm,
    backend=backend,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            backend=backend,
            trigger=("tokens", 20_000),  # 実コンテキスト上限の 60〜75% 程度に設定
            keep=("messages", 6),
        ),
    ],
)
```

### ③ 日本語テキストは「1 文字 ≒ 1〜1.5 トークン」になりやすい
ASCII の英語ログでは約 4 文字 ≒ 1 トークンですが、日本語を多く含む業務ドキュメントや JSON ではトークン密度が 3〜4 倍高くなります。「ファイルサイズは 30KB しかないのにトークン数が想定より多い」というズレが起きやすいため、日本語中心のワークロードでは閾値に余裕を持たせておくのが安全です。

### ④ `FilesystemBackend` をデフォルトにしていると `/large_tool_results/` がディスクに残る
Part 2 でも強調した通り、オフロード先は常に **デフォルト Backend** です。ローカルディスクを `default` にすると、実行のたびにプロジェクトディレクトリ直下に `large_tool_results/chatcmpl-tool-...` が溜まり続けます。実ディスクを使う場合でも、必ず `CompositeBackend(default=StateBackend(), routes={"/workspace/": FilesystemBackend(...)})` の構成を採り、オフロード一時ファイルはエフェメラルな `StateBackend` へ逃がしてください。

---

## 5. まとめ

本記事（Part 3 実践編）では、`dump_large_log` を用いて DeepAgents の自動オフロード機構を 2 つのサイズ条件で実測対比しました。

- **600 行（約 36KB）**: デフォルトの 20,000 トークン（約 80KB）閾値未満のためオフロードは発動せず、`ToolMessage` にインライン格納された。
- **1,400 行（約 120KB）**: 閾値を超過し、`FilesystemMiddleware` が自動的に `/large_tool_results/chatcmpl-tool-954917c8055113ae` へ退避。エージェントはパス通知を受け取って自律的に `grep` を実行し、1,401 行目の `CRITICAL_TIMEOUT / db_pool_exhausted` を正確に特定した（`tools_used: ["dump_large_log", "grep"]`・63.33 秒）。
- 自前ホストの OpenAI 互換モデルや日本語中心のワークロードでは、モデル容量に合わせて `SummarizationMiddleware` の `trigger` やオフロード閾値を明示チューニングすることが本番安定稼働の鍵となる。
