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

