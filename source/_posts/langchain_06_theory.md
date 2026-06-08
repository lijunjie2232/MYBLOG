---
title: "LangChain v1 Part 6 理論編：Context Engineering・Guardrails（PIIMiddleware）と Human-in-the-Loop（approve / edit / reject / respond）"
date: 2026-05-17 22:33:14
categories: [AI]
tags: [LangChain, LLM, AI Agent, Context Engineering, Guardrails, Human-in-the-Loop, Python]
lang: ja
description: "LangChain v1 における Context Engineering（Model / Tool / Life-cycle Context）、決定論的・モデルベース Guardrails（PIIMiddleware）、および Human-in-the-Loop（approve / edit / reject / respond・batched vs per_call モード）の設計理論を徹底解説します。"
---

プロトタイプとして動く AI エージェントを作ることは容易ですが、**本番環境で信頼性・安全性・コンプライアンスを満たすエージェントを運用すること**はまったく別の課題です。本稿（LangChain v1 シリーズ Part 6 理論編）では、本番品質のエージェントを支える 3 つの柱である **Context Engineering**、**Guardrails（`PIIMiddleware`）**、および **Human-in-the-Loop（`HumanInTheLoopMiddleware`）** のアーキテクチャと設計思想を体系的に解説します。

---

## 1. Context Engineering：なぜエージェントは本番で失敗するのか

LLM エージェントが意図しない動作を起こす原因は、大きく次の 2 つに集約されます。

1. 基盤となる LLM 自体の推論能力が不足している
2. **「適切なコンテキスト」が適切な形式で LLM に渡されていない**

実務で発生する障害の大半は 2 番目です。**Context Engineering** とは、「タスクを完遂するために必要な情報とツールを、適切なタイミング・適切な形式で LLM に供給する設計技術」を指します。LangChain v1 では、エージェントループ（Model call ⇄ Tool execution）の各段階を **Middleware** によって精密に制御できる設計になっています（参考：[langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py)）。

### 1.1 3 つのコンテキスト層と Transient / Persistent の境界

LangChain v1 は、制御対象となるコンテキストを以下の 3 層に分類しています。

| コンテキスト種別 | 制御対象 | 反映の性質 | 主な実装手段 |
| :--- | :--- | :--- | :--- |
| **Model Context** | LLM 呼び出しに渡す指示・履歴・ツール・モデル・出力スキーマ | **Transient（一時的）** | `@dynamic_prompt`, `@wrap_model_call` + `request.override(...)` |
| **Tool Context** | ツールが読み書きするセッション状態・長期記憶・静的設定 | **Persistent（永続的）** | `ToolRuntime`（`state`, `store`, `context`）, `Command(update=...)` |
| **Life-cycle Context** | モデル呼び出しとツール実行の「間」に介在する横断的処理 | **Persistent（永続的）** | `before_agent`, `before_model`, `after_model`, `after_agent` |

> Model Context への変更（`wrap_model_call` による `request.override(...)`）は**一時的（Transient）**であり単一の LLM 呼び出しのみに作用するのに対し、`before_model` / `after_model` などのライフサイクルフックや `PIIMiddleware` によるステート更新は**永続的（Persistent）**に `AgentState`（チェックポイント）へ書き込まれます。

この **Transient vs Persistent** の区別は極めて重要です。たとえば「今回の推論だけコンプライアンス規約を末尾に注入する」「権限のないユーザーには `delete_data` ツールを見せない」といった処理は `wrap_model_call` で一時的に上書きするのが適切です。一方で「古い会話履歴を要約して圧縮する（`SummarizationMiddleware`）」「ユーザー入力のメールアドレスをマスキングして履歴から消去する（`PIIMiddleware`）」といった処理は、後続ターンやチェックポイントにも反映させる必要があるため永続的更新を行います。

### 1.2 3 つのデータソース（Runtime Context / State / Store）

Middleware やツールが参照するデータソースも、スコープに応じて 3 つに明確化されています。

| データソース | 別名 | スコープ | 代表例 |
| :--- | :--- | :--- | :--- |
| **Runtime Context** | 静的設定 | 会話単位（不変） | `user_id`, 権限ロール（`admin`/`viewer`）, 環境（`production`）, DB接続情報 |
| **State** | 短期記憶 | 会話スレッド単位 | `messages`, 認証フラグ, アップロード済みファイル情報, ツール結果 |
| **Store** | 長期記憶 | スレッド横断 | ユーザー設定, 過去の好み, 抽出済みインサイト, フィーチャーフラグ |

{% mermaid %}
flowchart TB
    subgraph DataSources["データソース層"]
        RC["Runtime Context<br/>(静的設定: role / env)"]
        ST["State / Checkpointer<br/>(短期記憶: messages / 認証状態)"]
        SR["Store<br/>(長期記憶: ユーザー設定 / フラグ)"]
    end

    subgraph AgentLoop["LangChain v1 エージェントループ"]
        BA["before_agent / before_model<br/>(Persistent: 入力検査・PII置換・要約)"]
        WM["wrap_model_call<br/>(Transient: 動的プロンプト・ツール絞込・モデル切替)"]
        LLM["LLM 推論"]
        AM["after_model<br/>(Persistent: HITL interrupt・出力検証)"]
        TL["Tool 実行<br/>(ToolRuntime 経由の読み書き)"]
    end

    RC --> WM
    ST --> BA
    ST --> WM
    SR --> WM
    SR --> TL
    ST <--> TL

    BA --> WM --> LLM --> AM --> TL --> BA

    style RC fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style ST fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style SR fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    style WM fill:#fff8e1,stroke:#f57f17,color:#e65100
    style AM fill:#ffebee,stroke:#c62828,color:#b71c1c
{% endmermaid %}

### 1.3 Context Engineering の実践コードパターン

`wrap_model_call` と `request.override(...)` を使うことで、Runtime Context の権限や環境に応じた動的ツールフィルタリングと動的プロンプトをクリーンに分離できます。

```python
import os
from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)

# Runtime Context（会話全体で不変な静的設定）を dataclass で定義
@dataclass
class OpsContext:
    user_role: str        # 例: "admin" / "viewer"
    deployment_env: str   # 例: "production" / "staging"

# 1. 環境とロールに応じてシステムプロンプトを動的に構築（Transient）
@dynamic_prompt
def ops_dynamic_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env
    base = "あなたはクラウド運用支援エージェントです。"
    if role == "viewer":
        base += "\n読み取り専用権限です。変更操作は提案のみに留めてください。"
    if env == "production":
        base += "\n対象は本番環境です。破壊的操作には細心の注意を払ってください。"
    return base

# 2. 権限に応じて LLM に公開するツール集合を動的に絞り込む（Transient）
@wrap_model_call
def rbac_tool_filter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    role = request.runtime.context.user_role
    if role != "admin":
        # admin 以外には read_ で始まる安全なツールのみを公開する
        safe_tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=safe_tools)
    return handler(request)

# 環境変数からエンドポイントと認証情報を読み込む（ハードコード禁止）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)
```

---

## 2. Guardrails：多層防御と `PIIMiddleware`

Guardrails（ガードレール）は、エージェントの入出力やツール呼び出しを検証・フィルタリングし、PII 漏洩・プロンプトインジェクション・不適切コンテンツ・ポリシー違反を未然に防ぐ仕組みです。LangChain v1 では、すべてのガードレールを **Middleware の積層（Defense in Depth）** として構成します。

### 2.1 決定論的ガードレール vs モデルベースガードレール

| アプローチ | 実装手法 | 長所 | 短所・限界 | 適用箇所 |
| :--- | :--- | :--- | :--- | :--- |
| **決定論的（Deterministic）** | 正規表現、キーワード辞書、Luhn 検証、スキーマ検証 | 超低レイテンシ、低コスト、100% の再現性 | 文脈依存の巧妙な違反を見逃しやすい | `PIIMiddleware`, 禁止語フィルタ（`before_agent`） |
| **モデルベース（Model-based）** | 軽量 LLM や分類器によるセマンティック評価 | 微妙なニュアンスや間接的攻撃を検知可能 | レイテンシ・API コスト増、確率的ゆらぎ | 最終応答の安全性審査（`after_agent`） |

本番システムでは、**まず低コストな決定論的ガードレールで明確な違反を即座に遮断・無害化し、必要な場合のみモデルベースガードレールで意味的審査を行う多層構成**が鉄則です。

### 2.2 `PIIMiddleware` の仕組みと 4 つの戦略

`PIIMiddleware` は、メールアドレス（`email`）、クレジットカード番号（`credit_card`、Luhn チェック付き）、IP アドレス（`ip`）、MAC アドレス（`mac_address`）、URL（`url`）の 5 種類の組み込み PII に加え、任意の正規表現やカスタム検出関数（`detector`）をサポートします。

> `PIIMiddleware` は `strategy="redact"` で `[REDACTED_{PII_TYPE}]` へ置換し、`apply_to_input=True` によってユーザー入力が LLM やチェックポイントへ渡る前に機密情報を無害化します。決定論的ガードレールであるため低レイテンシかつ 100% の再現性で動作します。

| `strategy` | 動作 | 変換例 | 主な用途 |
| :--- | :--- | :--- | :--- |
| `"redact"`（デフォルト） | `[REDACTED_{PII_TYPE}]` に完全置換 | `alice@example.com` → `[REDACTED_EMAIL]` | 機密値を LLM やログに一切残したくない場合 |
| `"mask"` | 末尾数文字などを残して部分マスク | `5105-1051-0510-5100` → `****-****-****-5100` | カスタマーサポート等で下4桁の照合が必要な場合 |
| `"hash"` | 決定論的なハッシュ値へ置換 | `alice@example.com` → `a8f5f167...` | 値自体は隠しつつ同一人物の突合だけ行いたい場合 |
| `"block"` | PII 検出時に例外を送出して即時停止 | 例外送出（処理中断） | API キー混入など、処理続行自体が禁止される場合 |

さらに、適用ポイントも 3 つのブールフラグで独立制御できます。

- `apply_to_input=True`（デフォルト `True`）：モデル呼び出し前にユーザー入力を検査・サニタイズ
- `apply_to_output=True`（デフォルト `False`）：モデル出力後に AI メッセージを検査（`langchain>=1.3.2` ではストリーミング出力に対しても **末尾 128 文字のスライディングバッファ** を保持するストリーム変換器が働き、チャンク境界をまたぐ PII 漏洩＝Split-Boundary Leak を防止します）
- `apply_to_tool_results=True`（デフォルト `False`）：外部 DB や API から返却されたツール結果に含まれる PII を検査

```python
from langchain.agents.middleware import PIIMiddleware

# 複数の PII 種別に対して異なる戦略を多層適用する例
pii_middlewares = [
    # メールアドレスは入力段階で完全置換（[REDACTED_EMAIL]）
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    # クレジットカード番号は下4桁のみ残してマスク
    PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    # 自社形式の API キーはカスタム正規表現で検知し即時ブロック
    PIIMiddleware(
        "api_key",
        detector=r"sk-[a-zA-Z0-9]{32}",
        strategy="block",
        apply_to_input=True,
    ),
]
```

### 2.3 カスタムガードレールにおける `jump_to="end"`

独自の決定論的ガードレールを `before_agent` で実装する場合、`@hook_config(can_jump_to=["end"])` を宣言し、返り値に `"jump_to": "end"` を含めることで、**LLM を一度も呼び出さずに即座に安全な拒否応答を返して終了**できます。これにより、悪意ある入力に対するトークン消費とツール誤爆を完全にゼロへ抑え込めます。

---

## 3. Human-in-the-Loop（HITL）：危険操作に対する人間承認アーキテクチャ

本番環境へのデプロイ、データベースの削除、外部顧客へのメール送信といった**不可逆な副作用（Side Effects）を伴うツール実行**では、AI の自律判断だけで実行させてはなりません。LangChain v1 の `HumanInTheLoopMiddleware` は、LangGraph の **Interrupt（割り込み）** と **Checkpointer（状態永続化）** を組み合わせ、安全かつ再開可能な承認ワークフローを提供します。

> `HumanInTheLoopMiddleware` はツール実行前に `interrupt()` を発行してグラフ実行を一時停止し、LangGraph の永続化層（Checkpointer）へ状態を保存します。再開時は同じ `thread_id` と `Command(resume=...)` を渡すことで、`approve`・`edit`・`reject`・`respond` の4種類の意思決定を安全に反映できます。

### 3.1 4 種類の意思決定タイプ（Decision Types）

承認者（人間）は、一時停止されたツール呼び出しに対して次の 4 種類の判断を下せます。

| Decision Type | ツール実行 | 動作とエージェントへのフィードバック | 典型的なユースケース |
| :--- | :--- | :--- | :--- |
| **`approve`** | 実行する（元の引数） | モデルが提案した引数そのままツールを実行し、結果を返す | 妥当な本番デプロイやメール送信の承認 |
| **`edit`** | 実行する（修正後の引数） | 人間が修正した `edited_action`（ツール名・引数）で実行。モデルには「人間が引数を書き換えた」旨の通知（`edit_notice`）を付与 | デプロイ先を `production` から `staging` へ変更して実行 |
| **`reject`** | **実行しない** | ツール実行をスキップし、エラー状態の合成 `ToolMessage`（却下理由 `message` 付き）をモデルへ返す | 危険な `DELETE` クエリを拒否し、理由を伝えて代替案を促す |
| **`respond`** | **実行しない** | 人間の入力 `message` を**正常なツール結果**として合成 `ToolMessage` に格納して返す | `ask_user` のように人間の回答自体がツール出力となる確認プロンプト |

ここで極めて重要な設計ルールがあります。**副作用ツールの拒否に `respond` を使ってはいけません。** `respond` はモデルに対して「ツールが成功し、その戻り値が返ってきた」と伝えるため、モデルが「デプロイは完了した」と誤認する原因になります。拒否には必ず `reject` を使用してください。また、`langchain>=1.5.2` では `reject` と `respond` で生成された合成 `ToolMessage` に `source="hitl_interrupt_decision"` の通知メタデータが付与されるため、フロントエンド側で実ツール出力と明確に区別できます。

### 3.2 HITL 実行ライフサイクルと状態遷移

{% mermaid %}
sequenceDiagram
    autonumber
    participant User as ユーザー / アプリ
    participant Agent as Agent (Checkpointer付)
    participant PII as PIIMiddleware
    participant HITL as HumanInTheLoopMiddleware
    participant Tool as 危険ツール (deploy_release)

    User->>Agent: invoke(入力, config={"thread_id": "T1"})
    Agent->>PII: before_model (入力サニタイズ)
    PII-->>Agent: [REDACTED_EMAIL] へ永続置換
    Agent->>Agent: LLM 推論 → deploy_release 呼び出し提案
    Agent->>HITL: after_model / ツール実行前チェック
    HITL->>Agent: interrupt(HITLRequest) 発行 & チェックポイント保存
    Agent-->>User: __interrupt__ を含む中断レスポンス返却

    Note over User: 人間レビュアーが引数を確認し<br/>approve / edit / reject / respond を選択

    User->>Agent: invoke(Command(resume=decisions), config={"thread_id": "T1"})
    Agent->>HITL: 状態復元 & 意思決定の検証
    alt approve / edit
        HITL->>Tool: 承認/修正済み引数でツール実行
        Tool-->>Agent: 実行結果 (ToolMessage)
    else reject / respond
        HITL-->>Agent: ツール未実行で合成 ToolMessage を生成
    end
    Agent->>Agent: LLM 最終応答生成
    Agent-->>User: 最終レスポンス返却
{% endmermaid %}

### 3.3 条件付き割り込み（`when`）と 2 つの割り込みモード（`batched` vs `per_call`）

LangChain v1.3.3 以降では、すべての呼び出しを一律に止めるのではなく、引数に応じて動的に割り込みを判定する **`when` 述語** がサポートされています。たとえば「`SELECT` 文なら自動承認、`DELETE` / `UPDATE` なら承認待ち」といったきめ細かいポリシーが数行で書けます。

さらに `langchain>=1.4.6` からは、`interrupt_mode` に以下の 2 モードが用意されています。

| モード | 割り込み単位 | 再開時の指定方法 | 特徴・推奨度 |
| :--- | :--- | :--- | :--- |
| **`"batched"`**（現行デフォルト） | 1 回のモデルターンにつき 1 つの割り込みに複数ツールをまとめる | `Command(resume={"decisions": [...]})`（出現順のリスト） | `after_model` フックで動作。Lab 6 のようなシンプルな単一/一括承認に明快 |
| **`"per_call"`**（次世代推奨） | 承認対象ツール 1 件ごとに独立した割り込み ID を発行 | `Command(resume={interrupt_id: decision})`（ID マップ） | 各割り込みが JSON Schema（`response_schema`）を持ち、入力不正時は `ValidationError` で安全に再入力可能。非対象ツール（例: `read_data`）は待機せず即時実行 |

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest

# 本番環境（production）へのデプロイ時のみ人間承認を要求する条件述語
def is_production_deploy(request: ToolCallRequest) -> bool:
    return request.tool_call["args"].get("env") == "production"

# 条件付き割り込み＋per_call モードの設定例
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "deploy_release": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "when": is_production_deploy,
        },
        "read_status": False,  # 読み取り専用ツールは自動承認
    },
    description_prefix="本番操作の承認待ち",
    interrupt_mode="per_call",  # 既存の一括形式を使う場合は "batched"
)
```

---

## 4. まとめ

本稿では、本番運用の AI エージェントに不可欠な 3 つの設計理論を整理しました。

1. **Context Engineering**：`Transient`（単一呼び出しのみ）と `Persistent`（状態へ永続化）を明確に分け、3 つのデータソース（Runtime Context / State / Store）を Middleware 経由で制御する。
2. **Guardrails（`PIIMiddleware`）**：決定論的フィルタで機密情報を前段で確実にサニタイズし、必要に応じてモデルベース評価を後段に積む多層防御を構成する。
3. **Human-in-the-Loop**：Checkpointer と `thread_id` を前提に、ツール実行前に安全に一時停止し、`approve` / `edit` / `reject` / `respond` で人間の判断をグラフへ統合する。

次回の **Part 6 実践編** では、本稿で解説した `PIIMiddleware` と `HumanInTheLoopMiddleware` を組み合わせ、**メールアドレスの自動マスキングと本番デプロイ承認フロー**を実際に動作させ、中断ペイロードやチェックポイントの状態変化を実測ログとともに検証します。
