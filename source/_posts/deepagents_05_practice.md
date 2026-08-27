---
title: "DeepAgents Part 5 実践編：finops-analyst 専門サブエージェントへの task 委譲とコスト分析レポート生成"
date: 2026-07-30 04:17:52
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, AIAgent, Subagents, FinOps, Python]
lang: ja
description: "DeepAgents で専門サブエージェント finops-analyst を定義し、query_cost_db ツールを子側に隔離した状態で親エージェントから task 委譲を行う実装と実測ログを詳しく解説します。"
---

# はじめに

本記事は **DeepAgents Part 5 実践編** です。[Part 5 理論編](/2026/07/29/deepagents_05_theory/) で解説した **Subagents による Context Quarantine（コンテキスト隔離）** と **最小権限の専門サブエージェント設計** を、実際に動作する Python コードと実測ログで検証します。

理論編では、サブエージェントの本質と設計の要点を次のように整理しました。

> Subagents の本質は「Context Quarantine（コンテキスト隔離）」にある。大量のツール呼び出しや中間データ探索をサブエージェント側の独立したコンテキストに閉じ込め、親エージェント（Supervisor）には要約された最終結果だけを ToolMessage として返すことで、親のコンテキスト膨張を防ぐ。

> カスタムサブエージェントを設計する際は、description に「何をするか・いつ委譲すべきか」を具体的に記述し、tools を最小限（Least Privilege）に絞り、system_prompt で簡潔な返却フォーマットを明示することが、ルーティング精度とコンテキスト節約の鍵になる。

本実践編では、クラウドコスト照会ツール `query_cost_db` を**親エージェントには一切持たせず、`finops-analyst` サブエージェント専用に隔離**した構成を作り、以下を実証します。

1. 親エージェントが自分でコストを推測・捏造せず、`SubAgentMiddleware` が提供する `task` ツールで `finops-analyst` に委譲するか？
2. 親側の実行履歴（`tools_used`）には `query_cost_db` が一切現れず `["task"]` のみとなり、ツール実行の詳細がサブエージェント側へ完全に隔離されているか？

---

## 1. 検証アーキテクチャ：Supervisor と `finops-analyst` のツール分離

今回の構成では、親（Supervisor）と子（`finops-analyst`）の責務とツール権限を明確に分離します。

- **Supervisor（親エージェント）**: カスタムツールはゼロ。`SubAgentMiddleware` が自動付与する `task` ツールだけを使って専門サブエージェントへ委譲し、返ってきた要約をユーザー向けの日本語レポートに整える。
- **`finops-analyst`（専門サブエージェント）**: `query_cost_db` ツールのみを保持し、独立した隔離コンテキスト（`mode="isolated"`）内でコスト DB を照会し、合計と最大費目を 1 文で親に返却する。

{% mermaid %}
flowchart LR
    User["User\n'2026-07 のクラウドコストを分析してください。'"] --> Sup

    subgraph Parent["Supervisor コンテキスト (親)"]
        Sup["DeepAgent (Supervisor)\n保持ツール: task のみ\n(query_cost_db は非保持)"]
    end

    subgraph Child["隔離コンテキスト (finops-analyst / mode='isolated')"]
        Sub["finops-analyst\n保持ツール: query_cost_db のみ"]
        DB[("query_cost_db\n月次クラウドコスト照会")]
    end

    Sup -- "1. task(subagent_type='finops-analyst')" --> Sub
    Sub -- "2. query_cost_db('2026-07')" --> DB
    DB -- "3. 生データ返却" --> Sub
    Sub -- "4. 要約のみ返却 (ToolMessage)" --> Sup
    Sup -- "5. 最終レポート出力" --> User

    style Parent fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Child fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style DB fill:#e8f5e9,stroke:#2e7d32
{% endmermaid %}

---

## 2. 実装コード：`finops-analyst` 専門サブエージェントの定義と委譲

OpenAI 互換エンドポイント（`BASE_URL` / `API_KEY`）を環境変数から読み込み、`MiniCPM-V-4.7-35B-A3B` モデルで検証を実行する完全なコードです。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

# .env から接続先 URL と API キーを読み込む（ハードコード禁止）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """再現性確保のため temperature=0.0 で OpenAI 互換 LLM クライアントを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=40,
        max_retries=2,
    )


def run_subagents_lab() -> dict:
    """finops-analyst 専門サブエージェントへの task 委譲とコンテキスト隔離を検証する。"""
    llm = make_llm()

    # 1. サブエージェント専用のコスト照会ツールを定義
    #    親エージェントの tools には渡さず、finops-analyst だけに隔離する
    @tool
    def query_cost_db(month: str) -> str:
        """指定月のクラウドコストを返す。"""
        return f"{month}: Compute=$1,240, Storage=$310, Network=$95 (Total=$1,645)"

    # 2. SubAgent 辞書スペックで専門サブエージェントを定義
    #    - description: 親が task 委譲を判断するための具体的な責務説明
    #    - system_prompt: ツール利用手順と「1文で簡潔に返す」返却契約を明示
    #    - tools: 最小権限（query_cost_db のみ）
    finops_subagent = {
        "name": "finops-analyst",
        "description": "クラウドコストを調査・分析する専門サブエージェント",
        "system_prompt": (
            "query_cost_db を使ってコストを取得し、"
            "合計と最大費目を1文で報告してください。"
        ),
        "tools": [query_cost_db],
        "model": llm,
    }

    # 3. 親エージェント（Supervisor）を構築
    #    subagents を渡すと SubAgentMiddleware が自動で task ツールを提供する
    agent = create_deep_agent(
        model=llm,
        subagents=[finops_subagent],
        system_prompt=(
            "コスト分析は finops-analyst サブエージェントに task ツールで委譲し、"
            "結果を日本語で簡潔にまとめてください。"
        ),
    )

    # 4. 親エージェントへ分析を依頼
    res = agent.invoke(
        {"messages": [{"role": "user", "content": "2026-07 のクラウドコストを分析してください。"}]}
    )

    # 親側のメッセージ履歴から呼び出されたツール名のみを抽出
    tools_used = [
        tc["name"]
        for m in res["messages"]
        for tc in getattr(m, "tool_calls", [])
    ]
    return {
        "tools_used": tools_used,
        "final": res["messages"][-1].content,
    }


if __name__ == "__main__":
    output = run_subagents_lab()
    print("Parent tools_used:", output["tools_used"])
    print("Final answer:\n", output["final"])
```

---

## 3. 実測結果（Execution Output）

上記コードを実際に実行した結果（`deepagents_labs.json` の `lab5_subagents` 記録）は以下の通りです。

```json
{
  "lab5_subagents": {
    "status": "ok",
    "elapsed_s": 8.56,
    "output": {
      "tools_used": [
        "task"
      ],
      "final": "2026年7月のクラウドコスト分析結果は以下の通りです：\n\n**合計コスト: $1,645**\n\n**内訳:**\n- **Compute（計算リソース）: $1,240**（約75%）← 最大費目\n- **Storage（ストレージ）: $310**\n- **Network（ネットワーク）: $95**\n\n**主な洞察:**\nComputeコストが圧倒的に高く、インスタンスの最適化や闲置リソースの削除など、Compute分野での節約機会が最も期待できます。"
    }
  }
}
```

※上記は `MiniCPM-V-4.7-35B-A3B` からの生出力です（一部中国語簡体字「闲置」が混入していますが、数値・内訳・最大費目の特定はいずれも `query_cost_db` の実データと完全一致しています）。

---

## 4. なぜこの挙動になったのか？（コンテキスト隔離の実証）

実測ログから、理論編で解説した **Context Quarantine** が正確に機能していることが 3 つの事実から確認できます。

1. **親側の `tools_used` が `["task"]` の 1 件のみ**:
   親エージェントの `res["messages"]` には `query_cost_db` のツール呼び出しも生データも一切残っていません。親が知っているのは「`finops-analyst` に `task` を投げたこと」と「その最終要約が `ToolMessage` として返ってきたこと」だけです。
2. **正確な数値の伝播と構成比率の算出**:
   親エージェントは `query_cost_db` を直接呼べないにもかかわらず、合計 `$1,645`、Compute `$1,240`（約 75%）、Storage `$310`、Network `$95` という正確な数値をもとに、ユーザー向けのわかりやすいレポートと改善インサイトを生成しています。
3. **所要時間 8.56 秒の内訳**:
   単発ツール実行（Part 1 の 2.7 秒）より長い 8.56 秒を要しているのは、① 親 LLM の委譲判断 → ② 子 LLM の `query_cost_db` 呼び出し → ③ 子 LLM の結果要約 → ④ 親 LLM の最終レポート整形、という **計 4 回の推論ステップ** が同期的に直列実行されたためです。

この内部シーケンスを可視化すると以下のようになります。

{% mermaid %}
sequenceDiagram
    participant U as User
    participant Sup as Supervisor (親 LLM)
    participant MW as SubAgentMiddleware
    participant Sub as finops-analyst (子 LLM)
    participant DB as query_cost_db

    U->>Sup: "2026-07 のクラウドコストを分析してください。"
    Note over Sup: 自前のコスト照会ツールなし<br/>description から finops-analyst を選択
    Sup->>MW: tool_call: task(subagent_type="finops-analyst", description="2026-07 のクラウドコストを調査...")
    MW->>Sub: 隔離コンテキストを新規生成して起動

    Sub->>DB: tool_call: query_cost_db(month="2026-07")
    DB-->>Sub: "2026-07: Compute=$1,240, Storage=$310, Network=$95 (Total=$1,645)"
    Sub-->>MW: 要約テキスト（合計・内訳・最大費目）

    Note over MW: 子の全ツール履歴を破棄し<br/>最終メッセージのみを親へ返却
    MW-->>Sup: ToolMessage(content="...")
    Sup-->>U: 日本語の構造化コスト分析レポートを出力
{% endmermaid %}

---

## 5. 発展パターン：`response_format` による構造化 JSON 返却とモデル使い分け

本検証ではテキストで結果を返しましたが、後段のツール連携や自動アラート判定へ繋ぐ場合は、サブエージェントに **`response_format`** を設定すると親の `ToolMessage` が厳密な JSON 文字列になります。また、親と子で異なるモデルを割り当てることも可能です。

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

load_dotenv()


# サブエージェントの出力スキーマを Pydantic で厳密に定義
class CostSummary(BaseModel):
    month: str = Field(description="対象月 (YYYY-MM)")
    total_usd: float = Field(description="合計コスト (USD)")
    top_category: str = Field(description="最大費目のカテゴリ名")
    top_category_usd: float = Field(description="最大費目の金額 (USD)")
    recommendation: str = Field(description="コスト最適化の推奨アクション")


@tool
def query_cost_db(month: str) -> str:
    """指定月のクラウドコストを返す。"""
    return f"{month}: Compute=$1,240, Storage=$310, Network=$95 (Total=$1,645)"


# 親は全体調整用、子は分析用として同一または別モデルを構成可能
supervisor_llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

finops_structured_subagent = {
    "name": "finops-analyst",
    "description": "クラウドコストを調査し、構造化 JSON で合計・最大費目・推奨アクションを返す専門サブエージェント",
    "system_prompt": "query_cost_db でコストを取得し、指定されたスキーマに従って正確に回答してください。",
    "tools": [query_cost_db],
    "model": supervisor_llm,
    # 親の ToolMessage が常に CostSummary 準拠の JSON 文字列になる
    "response_format": CostSummary,
}

agent = create_deep_agent(
    model=supervisor_llm,
    subagents=[finops_structured_subagent],
    system_prompt="コスト分析は finops-analyst に委譲し、返却された JSON をもとに日本語で報告してください。",
)
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

Subagents を本番導入する際に現場でよく遭遇する落とし穴と対策を整理します。

1. **親エージェントが委譲せず自分で作業してしまう**:
   親が同じツールを持っている場合や、サブエージェントの `description` が曖昧な場合、親は `task` を呼ばずに自力で解決しようとします。対策は ① 専門ツールは子にのみ隔離する（今回の構成）、② `description` に「いつ委譲すべきか」を具体的に書く、③ 親の `system_prompt` に委譲方針を明記する、の 3 点です。
2. **サブエージェントが生データをそのまま返してコンテキストが膨張する**:
   せっかく隔離しても、サブエージェントが数百行のログや JSON を丸ごと最終回答に貼ってしまうと親のコンテキストが溢れます。子の `system_prompt` に「生データや中間出力は含めず、300 語以内で要約のみ返す。大きなデータは `/data/raw.txt` へ書き出してパスだけ返す」と明示しましょう。
3. **`tools` 指定は「追加」ではなく「完全置換（Override）」**:
   `SubAgent` 辞書に `"tools": [query_cost_db]` を指定すると、親が持つカスタムツールは一切継承されません（ファイルシステム等のミドルウェア提供ツールは別枠）。また `skills` もカスタムサブエージェントには自動継承されないため、必要な場合は子側にも明示指定が必要です。
4. **`general-purpose` サブエージェントの暗黙追加と無効化**:
   カスタムサブエージェントを 1 つ定義しただけでも、裏では常に `general-purpose` サブエージェントが自動追加されます。完全に無効化したい場合に `excluded_middleware` へ `SubAgentMiddleware` を入れても `ValueError` になるため、ハーネスプロファイル側で `GeneralPurposeSubagentProfile(enabled=False)` を設定するのが正規ルートです。
5. **Dynamic / Async 移行時のセキュリティとリソース注意**:
   - **Dynamic Subagents**: QuickJS の `eval` 内から `task()` が呼ばれるため、親の `interrupt_on` による個別承認はバイパスされます。承認が必要な場合は `eval` ツール自体をゲートしてください。
   - **Async Subagents**: ローカル開発（`langgraph dev`）で並列実行すると、ワーカー枠（親 1 + 子 N）が不足してキュー詰まりを起こしやすいため、`langgraph dev --n-jobs-per-worker 10` のようにワーカープールを拡張しておく必要があります。

---

## まとめ

本記事では、`finops-analyst` 専門サブエージェントへの `task` 委譲を実装・実測し、以下の挙動を確認しました。

- 親エージェントは `query_cost_db` を保持しないまま、自律的に `task` ツールで `finops-analyst` へ分析を委譲。
- 親の履歴には `tools_used: ["task"]` のみが記録され、ツール呼び出しと生データが子側の隔離コンテキスト内に完全に閉じ込められた（Context Quarantine の実証）。
- 8.56 秒で正確な内訳（合計 `$1,645`、Compute `$1,240` 約 75%）と改善インサイトを含む日本語レポートを生成。

次回の **Part 6** では、本番エージェントに不可欠な **Permissions（`FilesystemPermission`）と Human-in-the-Loop（`interrupt_on` / `Command(resume=...)`）** による安全制御を深掘りします。
