---
title: "LangChain v1 Part 5 理論編：Middleware アーキテクチャ（before_model / wrap_model_call / after_model / wrap_tool_call のフック合成）"
date: 2026-05-15 22:08:33
categories: [AI]
tags: [LangChain, LangGraph, AI Agent, Middleware, Architecture, Python]
lang: ja
description: "LangChain v1 の中核拡張機構である Middleware アーキテクチャを徹底解説。Node-style（before_model / after_model）と Wrap-style（wrap_model_call / wrap_tool_call）の役割、複数 Middleware の実行順序（オニオンモデル）、状態更新と jump_to 制御、組み込み Middleware の合成原則を学びます。"
---

# LangChain v1 Part 5 理論編：Middleware アーキテクチャとフック合成

AI エージェントをプロトタイプから本番運用へ引き上げる段階になると、ビジネスロジック（ツールやシステムプロンプト）とは別の **横断的関心事（Cross-Cutting Concerns）** が急増します。

- すべてのモデル呼び出しのレイテンシ計測・監査ログ出力
- 無限ループやコスト爆発を防ぐ呼び出し回数制限（Rate / Call Limit）
- プライマリモデル障害時の自動リトライ・別モデルへのフォールバック
- 会話履歴の自動要約、PII（個人情報）マスキング、Human-in-the-Loop 承認

これらをツール関数や呼び出し元のラッパーに散在させると、コードは瞬く間に破綻します。LangChain v1 が導入した **Middleware アーキテクチャ** は、コンパイル済み LangGraph エージェントループの要所に標準化されたフックを差し込み、**独立した小さな責務を宣言的に合成（Compose）** するための仕組みです。

---

## 1. エージェントループと 2 種類のフック体系

LangChain v1 の Middleware は、独立した別ランタイムではなく、**`create_agent` が構築する LangGraph グラフの内部ノード・ハンドラとして直接組み込まれます**。提供されるフックは大きく **Node-style** と **Wrap-style** の 2 系統に分かれます。

{% mermaid %}
flowchart TD
    START(["invoke() 開始"]) --> BA["before_agent<br/>(実行開始時に1回)"]
    BA --> BM["before_model<br/>(各モデル呼出の前)"]

    subgraph ModelWrap["wrap_model_call (オニオン入れ子)"]
        WM1["Outer Middleware<br/>wrap_model_call"] --> WM2["Inner Middleware<br/>wrap_model_call"]
        WM2 --> LLM["ChatModel 実行"]
        LLM --> WM2
        WM2 --> WM1
    end

    BM --> WM1
    WM1 --> AM["after_model<br/>(各モデル応答の後)"]

    AM -->|"tool_calls あり"| ToolWrap
    AM -->|"tool_calls なし / jump_to='end'"| AA["after_agent<br/>(実行終了時に1回)"]

    subgraph ToolWrap["wrap_tool_call (オニオン入れ子)"]
        WT1["Outer wrap_tool_call"] --> WT2["Inner wrap_tool_call"]
        WT2 --> TOOL["Tool 実行"]
        TOOL --> WT2
        WT2 --> WT1
    end

    WT1 -->|"ToolMessage 追加後ループ"| BM
    AA --> END(["最終結果返却"])

    style ModelWrap fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    style ToolWrap fill:#fef3c7,stroke:#d97706,color:#78350f
    style BA fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style BM fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style AM fill:#f1f5f9,stroke:#64748b,color:#0f172a
    style AA fill:#f1f5f9,stroke:#64748b,color:#0f172a
{% endmermaid %}

### 1.1 Node-style フック（順次実行・状態遷移向き）

グラフ上の特定ポイントで **順次実行** され、エージェントの `AgentState` を読み取り、必要なら差分辞書（`dict`）を返して状態を更新します。

| フック名 | 実行タイミング | 典型的なユースケース |
| :--- | :--- | :--- |
| `before_agent` | `invoke()` 開始時に 1 回だけ | 初期バリデーション、入力 PII チェック、初期状態の注入 |
| `before_model` | **各**モデル呼び出しの直前 | メッセージのトリミング・要約、呼び出し上限チェック、監査ログ |
| `after_model` | **各**モデル応答の直後 | 応答ログ記録、出力ガードレール、カウンター加算、早期終了判定 |
| `after_agent` | エージェント完了時に 1 回だけ | 最終出力の検証、メトリクス集計の確定 |

### 1.2 Wrap-style フック（呼び出し制御・インターセプト向き）

モデル呼び出しやツール呼び出しそのものを **関数ラッパー（`handler(request)`）として包み込む** フックです。`handler` を **0 回（キャッシュ返却・ショートサーキット）、1 回（通常実行・計測・リクエスト書き換え）、複数回（リトライ・フォールバック）** のいずれで呼ぶかを完全に制御できます。

| フック名 | 対象 | 典型的なユースケース |
| :--- | :--- | :--- |
| `wrap_model_call` | 各 LLM 呼び出し | レイテンシ計測、動的モデル選択、動的システムプロンプト変更、リトライ、フォールバック |
| `wrap_tool_call` | 各ツール呼び出し | ツール実行時間の計測、ツールエラーのキャッチ・変換、指数バックオフ付きリトライ |

> **使い分けの鉄則**：状態（`state["messages"]` 等）の検査や順次的な書き換え・ジャンプ判定には **Node-style** を使い、例外捕捉・リトライ・フォールバック・実行時間計測・`ModelRequest` の動的書き換えには **Wrap-style** を使います。

---

## 2. 複数 Middleware の実行順序（オニオンモデル）

`create_agent(middleware=[M1, M2, M3])` のように複数の Middleware をリストで渡した場合、実行順序には **厳密な対称ルール** があります。ここを誤解すると「フォールバックが動く前に例外が握りつぶされる」「リトライ時間が計測に含まれない」といったバグに直結します。

{% mermaid %}
sequenceDiagram
    autonumber
    participant M1 as M1 (リスト先頭 / 外側)
    participant M2 as M2 (中間)
    participant M3 as M3 (リスト末尾 / 内側)
    participant Model as ChatModel

    Note over M1,M3: 1. before_model (先頭 → 末尾の正順)
    M1->>M2: M1.before_model() 完了
    M2->>M3: M2.before_model() 完了
    M3->>M1: M3.before_model() 完了

    Note over M1,Model: 2. wrap_model_call (外側が内側を包む入れ子構造)
    M1->>M2: M1.wrap_model_call() → handler()
    M2->>M3: M2.wrap_model_call() → handler()
    M3->>Model: M3.wrap_model_call() → handler()
    Model-->>M3: ModelResponse
    M3-->>M2: M3 後処理
    M2-->>M1: M2 後処理

    Note over M1,M3: 3. after_model (末尾 → 先頭の逆順)
    M3->>M2: M3.after_model() 完了
    M2->>M1: M2.after_model() 完了
{% endmermaid %}

### 実行順序の 3 原則

1. **`before_*` フックは正順（`M1 → M2 → M3`）**: リストの先頭から順に実行される。
2. **`wrap_*` フックは入れ子（`M1(M2(M3(target)))`）**: リストの**先頭が一番外側**となり、内側の Middleware やモデルが投げた例外を捕捉したり、内側全体の処理時間を計測したりできる。
3. **`after_*` フックは逆順（`M3 → M2 → M1`）**: リストの末尾から先頭へ巻き戻るように実行される。

> **順序設計の具体例**：計測・監査 Middleware（`TimingAndAuditMiddleware`）を先頭に置き、内側に `ModelCallLimitMiddleware` や `ModelFallbackMiddleware` を配置すると、**フォールバック発生時の切り替えコストも含めたトータルの実測レイテンシ**を外側から一貫して観測できます。また、エラーを捕捉したい Middleware（例：`ModelFallbackMiddleware`）は、エラーを発生させ得る Middleware（例：`ModelRefusalMiddleware`）より**外側（リストの前方）**に置く必要があります。

---

## 3. 実装スタイル：デコレータ vs `AgentMiddleware` クラス

LangChain v1 では、用途の規模に応じて 2 通りの実装スタイルを選べます。

### 3.1 デコレータベース（単一フック・軽量プロトタイプ向け）

`@before_model` / `@after_model` / `@wrap_model_call` / `@wrap_tool_call` / `@dynamic_prompt` などのデコレータで関数 1 つをそのまま Middleware にできます。

```python
import time
from collections.abc import Callable
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

@wrap_model_call
def simple_timer(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """モデル呼び出しの所要時間を標準出力へ記録する単一フック Middleware。"""
    t0 = time.perf_counter()
    response = handler(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[simple_timer] model call took {elapsed_ms:.1f}ms")
    return response
```

### 3.2 クラスベース（複数フック連携・設定注入・同期/非同期両対応）

`AgentMiddleware` を継承すると、1 つのクラス内で `before_model` / `wrap_model_call` / `after_model` / `wrap_tool_call` をまとめて実装でき、さらに以下のクラス属性による高度な拡張が可能になります。

- **`state_schema`**: Middleware 専用のカスタム状態フィールドを `AgentState` に自動マージする（例：呼び出し回数カウンタ）。
- **`tools`**: Middleware に付随するツールを自動登録する（例：To-do リスト管理ツール）。
- **`transformers`**: ストリーミング出力を変換・マスキングする `StreamTransformer` ファクトリを登録する。
- **非同期対応**: `abefore_model` / `awrap_model_call` / `aafter_model` / `awrap_tool_call` を定義すれば `ainvoke()` / `astream()` でもネイティブに動作する。

また、`wrap_model_call` 内で `request.override(...)` を使うと、実行時のコンテキストに応じてシステムプロンプトや使用モデルを動的に切り替えられます。同様に `wrap_tool_call` を使えば、ツール単位の計測や例外ガードを一元化できます。

```python
import time
from collections.abc import Callable
from typing import Any
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage
from typing_extensions import NotRequired

# Middleware 独自のカスタム状態スキーマを定義
class AuditState(AgentState):
    model_call_count: NotRequired[int]


class ProductionObservabilityMiddleware(AgentMiddleware[AuditState]):
    """動的コンテキスト注入・モデル/ツール計測・呼出カウントを統合したクラスベース Middleware。"""

    state_schema = AuditState

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 既存の system_message.content_blocks を壊さずに動的コンテキストを追記する
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append({"type": "text", "text": "回答は簡潔かつ根拠を明示してください。"})
        patched_request = request.override(system_message=SystemMessage(content=blocks))

        t0 = time.perf_counter()
        response = handler(patched_request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[model] elapsed={elapsed_ms:.1f}ms")
        return response

    def wrap_tool_call(self, request, handler):
        # ツール呼び出し単位の実行時間と例外を監視する
        tool_name = request.tool_call["name"]
        t0 = time.perf_counter()
        try:
            return handler(request)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[tool:{tool_name}] elapsed={elapsed_ms:.1f}ms")

    def after_model(self, state: AuditState, runtime) -> dict[str, Any] | None:
        # Node-style フックからカスタム状態フィールドを更新する
        return {"model_call_count": state.get("model_call_count", 0) + 1}
```

---

## 4. 状態更新・早期終了（`jump_to`）・`ExtendedModelResponse`

### 4.1 `jump_to` による制御フローのショートサーキット

Node-style フック（`before_model` / `after_model` 等）は、返り値辞書に `"jump_to"` を含めることで、エージェントの通常ループをバイパスできます。利用可能なジャンプ先は `'end'`（終了）、`'tools'`（ツールノード）、`'model'`（モデルノード）の 3 種で、`@hook_config(can_jump_to=[...])` で宣言しておく必要があります。

```python
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage

class SafetyStopMiddleware(AgentMiddleware):
    """メッセージ数が上限に達したらモデルを呼ばずに安全終了する Middleware。"""

    def __init__(self, max_messages: int = 40):
        super().__init__()
        self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        if len(state["messages"]) >= self.max_messages:
            return {
                "messages": [AIMessage(content="会話上限に達したため処理を終了します。")],
                "jump_to": "end",
            }
        return None
```

### 4.2 `wrap_model_call` からの状態更新：`ExtendedModelResponse`

通常、`wrap_model_call` は `ModelResponse` を返しますが、モデル呼び出し層からエージェント状態（`AgentState`）を更新したい場合は、`ExtendedModelResponse(model_response=..., command=Command(update={...}))` を返します。複数の Wrap Middleware がそれぞれ `ExtendedModelResponse` を返した場合、**内側 → 外側の順でリデューサーを通じて合成**され、外側のリトライでやり直した際は失敗した試行の `Command` が自動的に破棄される（Retry-safe）という堅牢な設計になっています。

---

## 5. 本番で頻用される組み込み Middleware（Built-in Middleware）

LangChain v1 は、本番運用で必須となる Middleware を標準で同梱しています。自前実装する前に組み込みの活用を検討するのが鉄則です。

| 組み込み Middleware | 主な役割 | 重要パラメータ |
| :--- | :--- | :--- |
| **`ModelCallLimitMiddleware`** | 1 回の実行（`run_limit`）またはスレッド累計（`thread_limit`）のモデル呼出回数を制限し、暴走・コスト超過を防ぐ | `run_limit`, `thread_limit`, `exit_behavior="end" \| "error"` |
| **`ModelFallbackMiddleware`** | プライマリモデルがエラーになった際、指定した代替モデルへ自動でフォールバックする | `first_model`, `*additional_models` |
| **`ModelRetryMiddleware`** | 一時的なネットワーク障害や 429/503 エラーに対して指数バックオフ＋ジッターで自動リトライ | `max_retries`, `backoff_factor`, `jitter`, `on_failure` |
| **`ToolCallLimitMiddleware`** | 全ツールまたは特定ツールの呼び出し回数を制限する | `tool_name`, `run_limit`, `thread_limit`, `exit_behavior` |
| **`ToolErrorMiddleware` / `ToolRetryMiddleware`** | ツール例外を安全なエラーメッセージへ変換、または指数バックオフで再試行する | `on_error`, `max_retries`, `retry_on` |
| **`SummarizationMiddleware`** | 会話履歴が閾値を超えたら自動要約し、直近コンテキストのみ保持する | `model`, `trigger`, `keep` |
| **`PIIMiddleware`** | メール・クレカ等の機密情報を検出・マスキング・ブロックする | `pii_type`, `strategy`, `apply_to_input`, `apply_to_output` |
| **`HumanInTheLoopMiddleware`** | 危険なツール実行の前に割り込み（Interrupt）を起こし人間の承認を待つ | `interrupt_on` |

> **合成の相乗効果**：たとえば「外側にカスタム計測 Middleware」「中間に `ModelCallLimitMiddleware`」「内側に `ModelFallbackMiddleware`」を重ねるだけで、**可視性（Observability）・コスト安全装置（Cost Guard）・可用性（High Availability）** の 3 要件を、ビジネスロジックへ一切手を触れずに実現できます。

次回の **Part 5 実践編** では、まさにこの 3 層構成の Middleware スタックを実装し、実測の監査ログとレイテンシ内訳からフックの実行順序とオーバーヘッドを検証します。

### 参考リンク
- [LangChain Official Docs — Middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [LangChain Official Docs — Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [LangChain Official Docs — Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/middleware/types.py)
