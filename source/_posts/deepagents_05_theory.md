---
title: "DeepAgents Part 5 理論編：Subagents によるコンテキスト隔離とタスク委譲（Static / General-Purpose / Forked / Dynamic / Async）"
date: 2026-07-29 20:18:48
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, AIAgent, Subagents, MultiAgent, ContextQuarantine]
lang: ja
description: "DeepAgents における Subagents の設計思想（Context Quarantine）と、Static・General-Purpose・Forked・Dynamic・Async の 5 つの委譲モデル、継承ルール、本番オーケストレーション戦略を理論面から徹底解説します。"
---

# はじめに

エージェントが Web 検索、大規模リポジトリのコード読解、データベース照会、ログ分析といった多段階タスクをすべて単一の会話コンテキスト内で実行すると、すぐに**コンテキスト汚染（Context Pollution）と膨張（Context Bloat）**が発生します。数十回に及ぶ中間ツール呼び出しのログや生データが履歴を埋め尽くし、最終的な意思決定の精度が著しく低下するためです。

DeepAgents はこの構造的な問題に対し、**Subagents（サブエージェント）** による階層的なタスク委譲とコンテキスト隔離をフレームワークの標準機能として組み込んでいます。

> Subagents の本質は「Context Quarantine（コンテキスト隔離）」にある。大量のツール呼び出しや中間データ探索をサブエージェント側の独立したコンテキストに閉じ込め、親エージェント（Supervisor）には要約された最終結果だけを ToolMessage として返すことで、親のコンテキスト膨張を防ぐ。

本記事（Part 5 理論編）では、DeepAgents が提供する **5 つのサブエージェント形態**（Static / General-Purpose / Forked / Dynamic / Async）の仕組みと選定基準を体系的に解説します。

---

## 1. DeepAgents の 5 つのサブエージェント形態

DeepAgents のサブエージェントは、実行モデル（同期 / 非同期）、コンテキスト共有（隔離 / フォーク）、オーケストレーション方式（ツール呼出 / インタープリタコード）の組み合わせによって 5 つの形態に整理できます。

| 形態 | 定義方法 | 実行方式 | コンテキスト | 主なユースケース |
| :--- | :--- | :--- | :--- | :--- |
| **1. Static Custom (`SubAgent` / `CompiledSubAgent`)** | `subagents=[{...}]` または `CompiledSubAgent(...)` | 同期（親がブロック） | 隔離（タスク記述のみ） | 専門ドメイン分析（FinOps、法務、セキュリティ監査）、専用ツール隔離 |
| **2. Default (`general-purpose`)** | 設定不要（自動付与） | 同期（親がブロック） | 隔離（親のツール・スキルを継承） | 特別な役割不要で、大量探索のコンテキストだけを隔離したい場合 |
| **3. Forked (`mode="fork"`)** | `SubAgent(..., mode="fork")` | 同期 / 非同期 | **親の会話履歴を継承**（Cache 再利用） | 親が調査済みの文脈を引き継いで修正パッチやポストモーテムを作成 |
| **4. Dynamic (`CodeInterpreterMiddleware`)** | QuickJS インタープリタから `task()` を呼出 | 同期並列（JS `Promise.all`） | 隔離（コード制御のファンアウト） | ディレクトリ全ファイルレビュー、トーナメント評価、収束ループ |
| **5. Async (`AsyncSubAgent`)** | `subagents=[AsyncSubAgent(...)]` | **非同期（即座に Task ID 返却）** | 隔離 / フォーク（独立スレッド永続化） | 長時間バッチ、対話継続しながらの裏実行、途中指示変更・キャンセル |

どの形態を選ぶべきかの意思決定フローを Mermaid 図にまとめます。

{% mermaid %}
flowchart TD
    Req["サブエージェント委譲が必要"] --> Long{"長時間実行・親は対話を\n継続したいか？"}
    Long -- "Yes" --> Async["5. AsyncSubAgent\n(start/check/update/cancel_async_task)"]
    Long -- "No (同期完了待ち)" --> FanOut{"多数アイテムへのループ・\n条件分岐ファンアウトが必要か？"}

    FanOut -- "Yes" --> Dynamic["4. Dynamic Subagents\n(QuickJS CodeInterpreter + task())"]
    FanOut -- "No" --> NeedHist{"親のここまでの会話履歴を\n引き継ぐ必要があるか？"}

    NeedHist -- "Yes" --> Forked["3. Forked Subagent\n(mode='fork' / Prompt Cache 再利用)"]
    NeedHist -- "No (クリーンな隔離)" --> Custom{"専用ツール・専用モデル・\n専門プロンプトが必要か？"}

    Custom -- "Yes" --> Static["1. Static Custom SubAgent\n(辞書定義 or CompiledSubAgent)"]
    Custom -- "No" --> GP["2. Default general-purpose\n(自動付与・親ツール/スキル継承)"]

    style Async fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style Dynamic fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style Forked fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style Static fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style GP fill:#f5f5f5,stroke:#616161,stroke-width:2px
{% endmermaid %}

---

## 2. 同期サブエージェントの内部構造：`SubAgentMiddleware` と `task` ツール

`create_deep_agent()` を呼ぶと、デフォルトで [`SubAgentMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py) が組み込まれ、親エージェントに **`task` ツール** が提供されます。親エージェントは 1 ターン中に複数の `task` ツールコールを並列発行することも可能です。

### 2.1 `SubAgent` 辞書スペックと継承ルール

カスタムサブエージェントは辞書（`SubAgent` 型）またはコンパイル済み LangGraph グラフ（`CompiledSubAgent`）で定義します。ここで極めて重要なのが**親からの継承ルール**です。

| フィールド | デフォルト（未指定時） | 指定時の挙動 |
| :--- | :--- | :--- |
| `name` / `description` | 必須 | 親が `task(subagent_type=...)` を選ぶ判断基準。ストリーミングや LangSmith の `lc_agent_name` メタデータにも使われる。 |
| `system_prompt` | `isolated` では必須 | **親から継承しない**。役割・ツール手順・返却フォーマットを明記する。 |
| `tools` | 親のツールを継承 | **完全置換（Override）**。指定すると親ツールは消え、指定した最小限のツール群のみを持つ。 |
| `model` | 親のモデルを継承 | 文字列（`"openai:gpt-4o"`）または `BaseChatModel` でタスク別モデル最適化が可能。 |
| `skills` | **継承しない**（空） | 必要な場合のみ専用パスを指定。**`general-purpose` のみ親の `skills` を自動継承する**。 |
| `middleware` | `isolated` は親カスタム MW 非継承 | サブエージェント固有の MW を追加・上書き可能。`fork` 時は親のカスタム MW を継承。 |
| `permissions` | 親の権限を継承 | 指定すると親の `FilesystemPermission` を**完全置換**する。 |
| `response_format` | 未指定（自由テキスト返却） | Pydantic モデル等を指定すると、親の `ToolMessage` に厳密な JSON 文字列として返却される。 |

> カスタムサブエージェントを設計する際は、description に「何をするか・いつ委譲すべきか」を具体的に記述し、tools を最小限（Least Privilege）に絞り、system_prompt で簡潔な返却フォーマットを明示することが、ルーティング精度とコンテキスト節約の鍵になる。

### 2.2 `general-purpose` の役割と無効化・モデル拒否ハンドリング

- **Default `general-purpose` サブエージェント**: ユーザーが `subagents` に何も渡さなくても自動的に追加されます。名前 `"general-purpose"` で定義を上書きできるほか、サブエージェント機能自体を完全にオフにしたい場合は、ハーネスプロファイルで `GeneralPurposeSubagentProfile(enabled=False)` を設定し `subagents` を空にします（`excluded_middleware` に `SubAgentMiddleware` を指定すると `ValueError` になります）。
- **Model Refusals（`deepagents>=0.7.33`）**: サブエージェント側のモデルが安全フィルタ等で応答を拒否した場合、親全体の実行をクラッシュさせず、「サブエージェントがモデル拒否により停止した」旨の短い `ToolMessage` を返し、親がタスクを言い換えるか続行できるように保護されます。

---

## 3. Forked Subagents（`mode="fork"`）と Prompt Cache 再利用

通常の `mode="isolated"` はクリーンな反面、「親がすでに 20 ターンかけて調査した PR の問題点をもとに、レビューコメントを起草する」といったタスクでは、親が膨大な文脈を `task` の引数テキストに手動で要約し直す必要があります。

そこで `deepagents>=0.7.13` で導入されたのが **Forked Subagents（`mode="fork"`）** です。

- **コンテキストの継承**: 親の有効な会話履歴（すでに圧縮＝Compaction 済みなら圧縮後の履歴）をそのまま引き継ぎ、末尾の `task` ツール呼び出しを取り除いた上で、継続プリアンブル＋タスク指示を新しい `HumanMessage` として追加します。
- **Prompt Cache の最適化（`deepagents>=0.7.35`）**: Fork 時に `system_prompt` を設定しても、親から継承したシステムプロンプト自体は書き換えず、**タスクメッセージ側に追加**します。これによりシステムプロンプト＋会話履歴のプレフィックスが親と完全一致し、Anthropic や OpenAI 等のプロンプトキャッシュを高確率で再利用できます。
- **制約**: Fork されたサブエージェントはさらなる `task` 呼び出し（再帰委譲）ができず、自力でタスクを完遂する必要があります。また `skills` パラメータの個別指定は拒否されます。

```python
from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents import create_agent

# 例 1: 既存の LangGraph グラフを CompiledSubAgent として組み込む
custom_analyzer_graph = create_agent(
    model="openai:gpt-4o",
    tools=[],
    system_prompt="データ分析に特化した専門グラフとして、統計的な異常値のみを簡潔に報告してください。",
)

compiled_subagent = CompiledSubAgent(
    name="data-analyzer",
    description="複雑なデータ分析パイプラインを実行するコンパイル済みサブエージェント",
    runnable=custom_analyzer_graph,
)

# 例 2: 親の PR レビュー履歴を引き継ぐ Forked Subagent（Prompt Cache を再利用）
comment_writer_fork = {
    "name": "comment-writer",
    "description": "親が診断済みの PR レビュー結果を引き継ぎ、具体的なレビューコメントを起草する",
    "mode": "fork",
    "system_prompt": "指摘済みの各問題に対し、修正コード案付きの簡潔なレビューコメントを作成してください。",
}

agent = create_deep_agent(
    model="openai:gpt-4o",
    subagents=[compiled_subagent, comment_writer_fork],
)
```

---

## 4. Dynamic Subagents と Async Subagents

より高度なオーケストレーションのために、DeepAgents は 2 つの拡張モードを備えています。

### 4.1 Dynamic Subagents（QuickJS インタープリタからの動的並列制御）

`langchain-quickjs` の `CodeInterpreterMiddleware()` を追加すると、エージェントは JavaScript インタープリタ内からグローバル関数 `task({ description, subagentType, responseSchema })` を呼び出せるようになります。

LLM が 1 ターンごとにツール呼び出しを逐次判断するのではなく、JS の `Promise.all()` や `while` ループで決定論的に制御できるため、以下の高度なパターン（Recursive Language Models: RLM を含む）が自然に実現します。

- **Fan-out and Synthesize**: `tools.glob()`（PTC 有効化時）で全ファイルを取得し、`Promise.all` で並列レビューして集約。
- **Classify and Act**: チケットを分類し、`SPECIALIST[ticket.category]` で専門サブエージェントへ動的ルーティング。
- **Adversarial Verification**: 1 パス目の検出結果を、独立した懐疑的 `verifier` サブエージェントへ送って誤検知を排除。
- **Tournament / Loop until Done**: 複数案をトーナメント形式で比較選定したり、新規検出が 0 件になるまで重複排除ループを回す。

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

# QuickJS インタープリタと PTC (glob) を有効化し、コードからの並列ファンアウトを許可
dynamic_agent = create_deep_agent(
    model="openai:gpt-4o",
    subagents=[
        {
            "name": "reviewer",
            "description": "認証・認可の脆弱性を行番号と深刻度付きでレビューする",
            "system_prompt": "指定ファイルの認証・認可の欠陥を特定し、行番号と深刻度を報告してください。",
        }
    ],
    # ptc=["glob"] により JS 内で tools.glob() → Promise.all(files.map(f => task(...))) が可能になる
    middleware=[CodeInterpreterMiddleware(ptc=["glob"])],
)
```

### 4.2 Async Subagents（非同期バックグラウンド実行と通知駆動）

同期 `task` ではサブエージェント完了まで親がブロックされます。数分かかる調査やコーディングを裏で走らせつつ、ユーザーと対話を続けたい場合は **`AsyncSubAgent`** を使用します。

`AsyncSubAgent` は [Agent Protocol](https://github.com/langchain-ai/agent-protocol) 準拠サーバー（LangSmith Agent Server 等）上で独立したスレッドとして動作し、`AsyncSubAgentMiddleware` が親に 5 つの管理ツールを提供します。

- `start_async_task`: 即座に Task ID（＝子スレッド ID）を返却して裏実行を開始。
- `check_async_task`: 最新ステータス・結果・エラー・承認待ち（`waiting`）割り込みを取得。
- `update_async_task`: 同一スレッド上で追加指示を送信（デフォルト `update_strategy="interrupt"`、待機実行なら `"enqueue"`）。
- `cancel_async_task`: 実行中タスクをキャンセル。
- `list_async_tasks`: 追跡中タスク一覧のライブステータスを並列取得。

特筆すべき設計として、タスクのメタデータは会話履歴ではなく専用の **`async_tasks` State チャネル** に保存されます。これにより、親の会話が長くなって要約（Compaction）が走っても Task ID が失われません。また、同一デプロイ内であれば **ASGI トランスポート**（ネットワーク遅延ゼロ）で通信し、完了・失敗時には Agent Server から親へ自動通知（`HumanMessage` + `TaskEvent` メタデータ）が届くため、親がポーリングし続ける無駄も排除されます。

```python
from deepagents import AsyncSubAgent, create_deep_agent
from langchain_core.messages import get_notification_metadata

# ハイブリッド構成: researcher は同一デプロイ内 ASGI、coder はリモート HTTP サーバーへ委譲
async_subagents = [
    AsyncSubAgent(
        name="researcher",
        description="複数ソースの調査と要約をバックグラウンドで実行するリサーチエージェント",
        graph_id="researcher",
        # url 省略時は同一 langgraph.json 内の ASGI トランスポート（遅延ゼロ）
    ),
    AsyncSubAgent(
        name="coder",
        description="独立スケールされたリモート環境で大規模コード修正を行うエージェント",
        graph_id="coder",
        url="https://coder-deployment.example.com",
        update_strategy="enqueue",  # 実行中タスクへの追指示を割り込みせずキュー積みする
    ),
]

supervisor = create_deep_agent(
    model="openai:gpt-4o",
    subagents=async_subagents,
)

# 親のメッセージ履歴から非同期サブエージェントの完了通知イベントだけを抽出するヘルパー
def inspect_async_notifications(messages: list) -> list[dict]:
    events = []
    for msg in messages:
        meta = get_notification_metadata(msg)
        if meta and meta.get("source") == "async_subagent":
            events.append(meta["data"])  # task_id, agent_name, status, run_id を含む
    return events
```

同期 `task` と非同期 `AsyncSubAgent` の実行フローの違いを比較します。

{% mermaid %}
sequenceDiagram
    participant U as User
    participant Sup as Supervisor (親)
    participant Sync as Sync SubAgent (task)
    participant Async as Async SubAgent (独立スレッド)

    rect rgb(227, 242, 253)
        Note over U,Sync: 同期委譲 (SubAgentMiddleware)
        U->>Sup: "2026-07 のコストを分析して"
        Sup->>Sync: task(subagent_type="finops-analyst")
        Note over Sup: 完了までブロック待機
        Sync->>Sync: 隔離コンテキスト内で複数ツール実行
        Sync-->>Sup: ToolMessage (要約結果のみ返却)
        Sup-->>U: 最終レポート回答
    end

    rect rgb(237, 231, 246)
        Note over U,Async: 非同期委譲 (AsyncSubAgentMiddleware)
        U->>Sup: "競合調査を裏で進めて"
        Sup->>Async: start_async_task(agent="researcher")
        Async-->>Sup: 即座に task_id を返却
        Sup-->>U: "調査を開始しました (ID: ...)。他に作業はありますか？"
        Note over Async: 独立スレッドでバックグラウンド実行
        Async-->>Sup: 完了通知 (HumanMessage + TaskEvent)
        Sup-->>U: 調査完了レポートを提示
    end
{% endmermaid %}

---

## 5. Runtime Context 伝播とベストプラクティス

`create_deep_agent(..., context_schema=Context)` で定義したランタイムコンテキスト（`user_id` や設定フラグ等）は、親の `invoke(..., context=...)` から**すべての同期サブエージェントとそのツールへ自動伝播**します。さらに、親とサブエージェントが同じツールを共有している場合は、ツール内から `runtime.config["metadata"]["lc_agent_name"]` を参照して「どのサブエージェントが呼んだか」で挙動を分岐できます。

```python
from dataclasses import dataclass
from langchain_core.tools import tool
from langchain.tools import ToolRuntime


@dataclass
class AppContext:
    user_id: str
    strict_audit: bool = False


@tool
def query_internal_metrics(query: str, runtime: ToolRuntime[AppContext]) -> str:
    """呼び出し元エージェント名と Runtime Context に応じてメトリクスを照会する。"""
    # どのサブエージェントからの呼び出しかを自動識別
    caller = runtime.config.get("metadata", {}).get("lc_agent_name", "supervisor")
    user_id = runtime.context.user_id
    mode = "strict" if runtime.context.strict_audit else "standard"
    return f"[{caller}] user={user_id} mode={mode} query={query}"
```

実務でサブエージェントを設計する際の 4 大原則は以下の通りです。

1. **明確な `description`**: 「何でも手伝う」ではなく、発動トリガーと責務境界を具体的に書く。
2. **最小権限のツールセット（Minimize Tool Sets）**: サブエージェントには業務に必須なツールだけを与え、誤操作とトークン消費を抑える。
3. **タスク別のモデル最適化**: 親のオーケストレーションと子の専門処理（長文読解・数値計算・高速分類）で最適なモデルを使い分ける。
4. **簡潔な返却契約**: サブエージェントの `system_prompt` で「生データは返さず、要約・数値・結論のみを返す（必要ならファイルへ書き出してパスだけ返す）」ルールを徹底する。

---

## まとめ

本記事では、DeepAgents の Subagents が実現する **Context Quarantine（コンテキスト隔離）** の理論と、5 つの委譲形態（Static / General-Purpose / Forked / Dynamic / Async）の設計指針を解説しました。

- 同期 `SubAgent` は `SubAgentMiddleware` と `task` ツールを通じて、最小限のツール・独立コンテキスト・厳格な返却要約を実現する。
- 文脈を引き継ぎたいときは `mode="fork"`、コード制御で大規模並列化したいときは `CodeInterpreterMiddleware`、長時間バックグラウンド実行したいときは `AsyncSubAgent` へとシームレスに拡張できる。

次回の **Part 5 実践編** では、クラウドコスト照会ツール `query_cost_db` を専門サブエージェント `finops-analyst` にのみ隔離し、親エージェントからの `task` 委譲とコスト分析レポート生成を実測ログ付きで検証します。
