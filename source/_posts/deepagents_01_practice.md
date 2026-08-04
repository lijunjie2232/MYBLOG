---
title: "DeepAgents Part 1 実践編：create_deep_agent と MoE モデル（MiniCPM-V-4.7-35B-A3B）で動かす在庫確認ディープエージェント"
date: 2026-07-01 16:35:21
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, MoE, Tool Calling, Python, AI Agent]
lang: ja
description: "DeepAgents の create_deep_agent を OpenAI 互換 MoE モデル MiniCPM-V-4.7-35B-A3B で実測検証。在庫確認ツールの実行トレース（4メッセージ・2.7秒）と、v0.7 以降の TodoListMiddleware オプトイン仕様に伴う挙動の違い、ハマりどころを詳解します。"
---

# DeepAgents Part 1 実践編：`create_deep_agent` と MoE モデル（MiniCPM-V-4.7-35B-A3B）で動かす在庫確認ディープエージェント

前回の **Part 1 理論編** では、DeepAgents が LangChain / LangGraph の上に提供する **Agent Harness の 4 本柱**、v0.7 以降オプトインとなった **Task Planning（`TodoListMiddleware`）**、そして **14 層のデフォルト Middleware スタック** の設計思想を解説しました。

本記事（Part 1 実践編）では、OpenAI 互換エンドポイント上の MoE（Mixture of Experts）モデル **`MiniCPM-V-4.7-35B-A3B`** を用いて最小の DeepAgents（在庫確認エージェント）を実装・実行し、**実測ログ（所要時間 2.7 秒・メッセージ数 4 件）** の内訳と、プロンプトに `write_todos` と書いたにもかかわらず呼ばれなかった理由を技術的に裏付けます。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、DeepAgents の設計上の核心として次の 2 点を確認しました。

> **DeepAgents の本質**: 「モデルに長いタスクを任せると途中でコンテキストが溢れ、作業計画を忘れ、ファイル操作やサブタスクの結果で履歴が汚染される」という典型的な失敗モードに対し、**ファイルシステム・圧縮・隔離・承認** を最初から組み込んだ標準実行基盤である。

> **DeepAgents v0.7 以降、Task Planning（`TodoListMiddleware`）はデフォルト無効のオプトイン機能に変更されました**。Task Planning を有効にしたい場合は、`langchain.agents.middleware` から `TodoListMiddleware` を明示的にインポートして `middleware=` に渡します。

今回の実践ラボ（Lab 1）では、以下の 3 点を実測データで確認します。

1. **OpenAI 互換 MoE モデルとの接続**: `ChatOpenAI` に環境変数 `BASE_URL` / `API_KEY` を渡して `create_deep_agent` の Bare Stack（デフォルト Middleware 群）がエラーなく起動するか。
2. **カスタムツールと組み込みツールの共存**: `FilesystemMiddleware` や `SubAgentMiddleware` が多数の組み込みツール（`ls`, `read_file`, `write_file`, `task` 等）を同時に公開する状況でも、モデルが迷わず目的のカスタムツール `check_inventory` を選択できるか。
3. **v0.7 のオプトイン `TodoListMiddleware` の実挙動**: `system_prompt` に「必要に応じて `write_todos` で計画を立て〜」と指示した場合、デフォルト構成のエージェントはどう振る舞うか。

---

## 2. 実験アーキテクチャと実装コード（Lab 1）

今回の在庫確認ディープエージェントにおける、メッセージと Middleware の流れを図示します。

{% mermaid %}
sequenceDiagram
    autonumber
    actor User as ユーザー
    participant Harness as DeepAgents Bare Stack<br/>(FS / SubAgent / Summ / Patch)
    participant LLM as MiniCPM-V-4.7-35B-A3B<br/>(OpenAI 互換 API)
    participant Tool as check_inventory ツール

    User->>Harness: HumanMessage (#1)<br/>"SKU-9001の在庫を確認して、結果を1行で報告してください。"
    Harness->>LLM: 合成 System Prompt + 組み込みツール群 + check_inventory
    LLM-->>Harness: AIMessage (#2)<br/>tool_calls: [check_inventory(sku="SKU-9001")]
    Harness->>Tool: check_inventory("SKU-9001")
    Tool-->>Harness: ToolMessage (#3)<br/>{"sku": "SKU-9001", "stock": 42, "warehouse": "Tokyo-East"}
    Harness->>LLM: 履歴 (#1〜#3) を送信
    LLM-->>Harness: AIMessage (#4)<br/>"SKU-9001の在庫数は42個（東京東倉庫）です。"
    Harness-->>User: 最終応答 (合計 4 メッセージ / 2.70s)
{% endmermaid %}

### 実装コード

API キーや接続先 URL は決してソースコードに直書きせず、`.env` から `os.getenv("BASE_URL")` および `os.getenv("API_KEY")` 経由で取得します。

```python
import json
import os
import time
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

# Pydantic 互換性警告を抑止し、.env から環境変数を読み込む
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント向けの ChatOpenAI インスタンスを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,      # カスタム推論サーバのエンドポイント
        api_key=API_KEY,        # 環境変数から取得した API キー
        temperature=temperature,  # ツール選択の決定性を高めるため 0.0
        max_tokens=512,         # 過剰出力による遅延・タイムアウトを防止
        timeout=40,             # 推論サーバ遅延時のハング対策
        max_retries=2,          # 一時的な 429/5xx へのリトライ
    )


def run_lab1():
    llm = make_llm()

    # ドメイン固有の在庫照会ツールを定義（docstring がモデルのツール選択ヒントになる）
    @tool
    def check_inventory(sku: str) -> str:
        """SKUの在庫数を返す。"""
        return json.dumps({"sku": sku, "stock": 42, "warehouse": "Tokyo-East"})

    # create_deep_agent で Bare Stack 付きのディープエージェントを生成
    # ※ v0.7 以降、TodoListMiddleware は明示的に渡さない限り含まれない
    agent = create_deep_agent(
        model=llm,
        tools=[check_inventory],
        system_prompt="必要に応じて write_todos で計画を立て、ツールを使って簡潔に日本語で回答してください。",
    )

    t0 = time.perf_counter()
    res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "SKU-9001の在庫を確認して、結果を1行で報告してください。",
                }
            ]
        }
    )
    elapsed = round(time.perf_counter() - t0, 2)

    # 発行されたすべてのツール名と最終回答を抽出して返す
    return {
        "elapsed_s": elapsed,
        "msg_count": len(res["messages"]),
        "tools_used": [
            tc["name"]
            for m in res["messages"]
            for tc in getattr(m, "tool_calls", [])
        ],
        "final": res["messages"][-1].content,
    }


if __name__ == "__main__":
    print(json.dumps(run_lab1(), indent=2, ensure_ascii=False))
```

---

## 3. 実測出力の解析：なぜ 4 メッセージ・2.7 秒で完結したのか

実際に上記スクリプトを実行して得られた結果は以下の通りです（DeepAgents `v0.7.41` 環境で実測）。

```json
{
  "lab1_harness_basics": {
    "status": "ok",
    "elapsed_s": 2.7,
    "output": {
      "msg_count": 4,
      "tools_used": [
        "check_inventory"
      ],
      "final": "SKU-9001の在庫数は42個（東京東倉庫）です。"
    }
  }
}
```

### この出力から読み解ける 3 つの技術的事実

1. **メッセージ数 `msg_count: 4` の厳密な内訳**
   - `#1 HumanMessage`: ユーザー依頼（`"SKU-9001の在庫を確認して、結果を1行で報告してください。"`）
   - `#2 AIMessage`: 推論 1 回目の出力（`tool_calls=[{"name": "check_inventory", "args": {"sku": "SKU-9001"}, ...}]`）
   - `#3 ToolMessage`: `check_inventory` の実行結果 JSON（`{"sku": "SKU-9001", "stock": 42, "warehouse": "Tokyo-East"}`）
   - `#4 AIMessage`: 推論 2 回目の最終回答（`"SKU-9001の在庫数は42個（東京東倉庫）です。"`）
   つまり、余計なリトライやサブエージェント起動を一切発生させず、最短の **1 ツール呼び出し（2 LLM ターン）** で収束しています。

2. **組み込みツール群の中でも正確にカスタムツールを選択**
   `create_deep_agent` の Bare Stack は、`check_inventory` だけでなく `ls`・`read_file`・`write_file`・`edit_file`・`glob`・`grep`・`task` といった多数の組み込みツールスキーマを同時にモデルへ提示しています。MoE モデル（アクティブパラメータ 3B 級の `MiniCPM-V-4.7-35B-A3B`）であっても、ツールの `docstring` と引数名（`sku: str`）が明確であれば、ノイズに惑わされず最短パスでツールを選択できることが確認できました。

3. **なぜ `write_todos` は呼ばれなかったのか？**
   `system_prompt` に「必要に応じて `write_todos` で計画を立て〜」と書いたにもかかわらず、`tools_used` は `["check_inventory"]` のみでした。理由は 2 層あります。
   - **第 1 の理由（ハーネス仕様）**: 理論編で述べた通り、**DeepAgents v0.7 以降では `TodoListMiddleware` が Bare Stack から外れてオプトイン化** されています。そのため `middleware=[TodoListMiddleware()]` を渡していない本コードでは、そもそもモデルのツール定義に `write_todos` が存在しません。
   - **第 2 の理由（タスク粒度）**: 仮に `TodoListMiddleware()` を有効化していたとしても、単一 SKU の在庫照会のような 1 ステップタスクで計画ツールを呼ばせるのはレイテンシ（+1 ターン）の無駄になります。「プロンプトに書いたから呼ばれる」のではなく、**Middleware によってツールが登録されていること** と **タスクの複雑さが複数ステップであること** の両方が揃って初めて `write_todos` が機能します。

{% mermaid %}
flowchart LR
    Q["タスク入力<br/>SKU-9001の在庫確認"] --> C1{"TodoListMiddleware<br/>が登録済みか？"}
    C1 -->|"No (v0.7 デフォルト)"| D1["write_todos は非公開<br/>直接 check_inventory を実行"]
    C1 -->|"Yes (オプトイン)"| C2{"複数ステップの<br/>計画が必要か？"}
    C2 -->|"単一ステップ"| D2["計画をスキップし<br/>check_inventory を実行"]
    C2 -->|"複雑な多段階タスク"| D3["write_todos で計画作成<br/>→ 順次ツール実行 → 完了更新"]

    style C1 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style C2 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style D1 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style D2 fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
    style D3 fill:#faf5ff,stroke:#6b46c1,stroke-width:1.5px,color:#44337a
{% endmermaid %}

---

## 4. 注意点・ハマりどころ（Pay Attention）

実務で `create_deep_agent` を OpenAI 互換モデルや MoE モデルと組み合わせて運用する際、以下の 4 点は特にハマりやすいポイントです。

### ① v0.6 以前のサンプルコードとのギャップ（`TodoListMiddleware`）
古いブログ記事や v0.6 以前のドキュメントのまま `system_prompt` に「必ず最初に `write_todos` を呼ぶこと」と強く強制すると、**v0.7 環境では存在しないツールをモデルが呼ぼうとしてハルシネーションやエラーを引き起こす** ことがあります。計画機能を使いたい場合は必ず以下のように明示的に有効化してください。

```python
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

# write_todos を使わせる場合は必ず Middleware を明示追加する
agent_with_todos = create_deep_agent(
    model=llm,
    tools=[check_inventory],
    middleware=[TodoListMiddleware()],
)
```

### ② 組み込みツール群による入力トークン増大とツール誤選択
`create_deep_agent` は標準で仮想ファイルシステムとサブエージェント（`task`）のツール定義・プロンプトを自動注入します。今回の `MiniCPM-V-4.7-35B-A3B` では問題なく 2.7 秒で応答しましたが、コンテキスト長が短いモデルやツール選択精度が低い小型モデルでは、使わない組み込みツールがノイズやトークン課金の原因になります。
- ファイル書き込みが不要なら `FilesystemMiddleware(backend=StateBackend(), tools=["read_file", "ls", "glob", "grep"])` で読み取り専用に絞る。
- そもそもファイルツール自体が不要なら `HarnessProfile(excluded_tools=...)` でモデルへの露出を消す（**注**: `FilesystemMiddleware` 自体を `excluded_middleware` で消すことは仕様上拒否されるため、ツール露出の制限で制御する）。

### ③ OpenAI 互換エンドポイントの `model_profile` 未登録問題
`ChatOpenAI(model="MiniCPM-V-4.7-35B-A3B", base_url=...)` のように独自ホストのモデルを使う場合、LangChain 内部のモデルプロファイル表に `max_input_tokens` が登録されていないことがあります。この場合、`SummarizationMiddleware` はフォールバック値（170,000 トークン）を採用するため、実際のコンテキスト上限が 32K〜64K のモデルでは自動要約が発動する前にコンテキスト溢れが起きるリスクがあります（具体的なチューニング手法は Part 3 で詳しく扱います）。

### ④ `temperature=0.0` と `max_tokens` の適切な設定
DeepAgents は内部で JSON 形式のツール引数を多用するため、開発・検証フェーズでは `temperature=0.0` を基本にして再現性を確保してください。また、一部の推論サーバでは `max_tokens` 未指定時に極端に大きなデフォルト値が使われ、スループット低下やタイムアウトを招くことがあるため、用途に応じた上限値と `timeout` を必ず設定しておくのが安全です。

### ⑤ 本番投入に向けた最小チューニング構成（実戦テンプレート）
今回の Lab 1 をベースに、「複数 SKU の一括棚卸しのように計画が必要で、かつファイル改変は禁止したい」本番向け構成へ発展させる場合は、次のように `TodoListMiddleware` と読み取り専用 `FilesystemMiddleware` を組み合わせます。

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import TodoListMiddleware

backend = StateBackend()

# 計画機能(write_todos)を有効化しつつ、FSツールを読み取り専用に絞った実戦構成
production_inventory_agent = create_deep_agent(
    model=llm,
    tools=[check_inventory],
    backend=backend,
    middleware=[
        # write_file / edit_file / delete を隠してトークン節約と誤操作防止を両立
        FilesystemMiddleware(backend=backend, tools=["read_file", "ls", "glob", "grep"]),
        # v0.7+ で複数ステップの棚卸し計画を管理したい場合に明示追加
        TodoListMiddleware(),
    ],
    system_prompt="複数SKUの確認が必要な場合は write_todos で計画を立て、簡潔に日本語で報告してください。",
)
```

---

## 5. まとめ

本記事（Part 1 実践編）では、DeepAgents の `create_deep_agent` と `MiniCPM-V-4.7-35B-A3B` を組み合わせた在庫確認エージェントを実測検証しました。

- Bare Stack を備えたディープエージェントは、多数の組み込みツールが存在する状態でもカスタムツール `check_inventory` を正確に選択し、**4 メッセージ・2.70 秒** で完結した。
- DeepAgents v0.7 以降、`write_todos`（`TodoListMiddleware`）はオプトインであり、プロンプトに書くだけでは有効化されない点に注意が必要である。

次回の **Part 2 理論編** では、DeepAgents の中核インフラである **仮想ファイルシステムと Backends（`StateBackend`・`FilesystemBackend`・`StoreBackend`・`CompositeBackend`）** の設計とルーティング機構を徹底解説します。
