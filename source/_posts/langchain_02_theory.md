---
title: "LangChain v1 Part 2 理論編：ツール設計・Pydantic args_schema・ToolRuntime コンテキスト注入とエラーハンドリング戦略"
date: 2026-05-05 13:38:49
categories: [AI]
tags: [LangChain, LangChain v1, Tools, Pydantic, ToolRuntime, Error Handling]
lang: ja
description: "LangChain v1 におけるツール定義のベストプラクティス、Pydantic args_schema による厳格なスキーマ設計、ToolRuntime によるモデル非公開コンテキストの注入、および多層エラーハンドリング戦略を解説します。"
---

# LangChain v1 Part 2 理論編：ツール設計・Pydantic `args_schema`・`ToolRuntime` コンテキスト注入とエラーハンドリング戦略

エージェントの知能を決めるのが Model だとすれば、エージェントの「手足」と「実務能力」を決めるのが **Tools** です。しかし、実務でエージェントを運用すると、単に関数へ `@tool` を付けるだけではすぐに限界に直面します。

- モデルが不正な型や想定外の演算子を渡してくる（スキーマ設計の甘さ）。
- ユーザー ID や認証情報、会話履歴をツールで使いたいが、モデルのプロンプトや引数に露出させたくない（セキュリティと責務分離）。
- ゼロ除算や外部 API の一時障害でツールが例外を投げ、エージェント全体がクラッシュしてしまう（エラーリカバリの欠如）。

本記事（Part 2 理論編）では、LangChain v1 が提供する **Pydantic `args_schema`**、統一ランタイム注入機構 **`ToolRuntime`**、そして **多層エラーハンドリング戦略** の設計理論を体系的に解説します。

---

## 1. `@tool` と Pydantic `args_schema` による契約設計

LangChain v1 では、`@tool` デコレータ（[langchain-core/tools/convert.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/convert.py)）を使って任意の Python 関数をエージェント向けツールに変換できます。

### ツール名・説明文・型ヒントが LLM の行動を左右する

モデルがツールを呼び出す際に参照するのは、関数の実装コードではなく **JSON Schema（名前・説明文・引数定義）** だけです。したがって、以下の設計原則が精度に直結します。

1. **ツール名は `snake_case` を徹底する**: スペースや特殊文字を含む名前は一部プロバイダでエラーになるため、`safe_calculator` や `get_user_tier` のような英数字とアンダースコアの構成にします。
2. **docstring は「いつ使うか・何を返すか」を簡潔に書く**: 曖昧な説明はモデルの誤呼び出しを誘発します。
3. **複雑な引数は Pydantic `BaseModel` を `args_schema` に渡す**: 単純な型ヒントだけでもスキーマは生成されますが、`Field(description=...)` や `Literal[...]`、数値範囲制約（`ge`, `le`）を付与した Pydantic モデルを `args_schema` に指定することで、**モデルへの説明精度向上**と**実行前の自動バリデーション**を同時に実現できます。

```python
from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# Pydantic モデルで入力契約（型・許容値・説明文）を厳密に定義する
class CalcInput(BaseModel):
    a: float = Field(description="左辺の数値")
    b: float = Field(description="右辺の数値")
    # Literal を使うと JSON Schema の enum としてモデルに伝わる
    op: Literal["add", "sub", "mul", "div"] = Field(
        description="演算子: add, sub, mul, div"
    )


@tool("safe_calculator", args_schema=CalcInput)
def safe_calculator(a: float, b: float, op: str) -> str:
    """安全な四則演算ツール。ゼロ除算時はエラーメッセージを返す。"""
    if op == "add":
        return str(a + b)
    if op == "sub":
        return str(a - b)
    if op == "mul":
        return str(a * b)
    if op == "div":
        if b == 0:
            return "Error: Division by zero is not allowed."
        return str(a / b)
    return f"Error: Unsupported op {op}"
```

---

## 2. `ToolRuntime` によるコンテキスト注入の理論

実務のツールでは「現在ログインしているユーザーの ID」「会話の状態（State）」「長期記憶ストア（Store）」「実行設定（`RunnableConfig`）」などが必要になります。

もし `user_id: str` を通常のツール引数として公開してしまうと、**プロンプトインジェクションによって他人の `user_id` をモデルに渡させる攻撃**が成立してしまいます。また、モデルに不要な引数を推測させること自体がハルシネーションの温床です。

### 旧来の注入パターンから `ToolRuntime` への統合

LangChain v0.3 以前のエコシステムでは、`InjectedState`、`InjectedStore`、`InjectedToolCallId`、`get_runtime()` など複数の注入機構が乱立していました。LangChain v1 ではこれらを **`ToolRuntime`** という単一の明示的インターフェースに統合しています。

ツールの引数に `runtime: ToolRuntime` を宣言すると、LangGraph の `ToolNode` が以下の 2 つを自動的に行います。

1. **スキーマからの隠蔽（Schema Hiding）**: モデルへ送信する JSON Schema から `runtime` 引数を完全に除外する。モデルからは「引数なしのツール」または「ビジネス引数だけのツール」に見える。
2. **実行時インジェクション（Runtime Injection）**: ツール実行直前に、現在のグラフ状態・設定・ストア・ツール呼び出し ID を束ねた `ToolRuntime` オブジェクトを自動注入する。

{% mermaid %}
flowchart LR
    subgraph LLM_VIEW["LLM から見えるツール定義 (JSON Schema)"]
        S1["get_user_tier()"]
        S2["引数: なし<br/>(runtime は完全隠蔽)"]
    end

    subgraph RUNTIME["ToolNode 実行時インジェクション"]
        RT["ToolRuntime"]
        RT --- C1["state (短期記憶・履歴)"]
        RT --- C2["context (型付き不変コンテキスト)"]
        RT --- C3["config (RunnableConfig / configurable)"]
        RT --- C4["store (長期記憶 BaseStore)"]
        RT --- C5["tool_call_id / stream_writer"]
    end

    LLM_VIEW -->|"モデルが引数なしで呼出"| FN["get_user_tier(runtime)"]
    RUNTIME -->|"ハーネスが安全に注入"| FN
    FN --> RES["ToolMessage<br/>user_id=u-2026-sre, tier=Enterprise"]

    style LLM_VIEW fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style RUNTIME fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style FN fill:#fefcbf,stroke:#b7791f,stroke-width:2px,color:#744210
    style RES fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

### `ToolRuntime` が提供する 8 つの主要コンポーネント

| プロパティ | 役割 | 主なユースケース |
| :--- | :--- | :--- |
| `runtime.state` | 現在の会話状態（短期記憶 `AgentState`） | 直近のメッセージやカスタム状態フィールドの参照。 |
| `runtime.context` | 実行単位の不変コンテキスト（`context_schema` 指定時） | 型安全な `UserContext` によるパーソナライズや認可。 |
| `runtime.config` | 実行時の `RunnableConfig`（`configurable` 辞書等） | `user_id`、`thread_id`、コールバック、タグ、メタデータの取得。 |
| `runtime.store` | 長期記憶 `BaseStore`（`InMemoryStore` / `PostgresStore` 等） | スレッドを跨いだユーザー設定やナレッジの読み書き。 |
| `runtime.stream_writer` | カスタムストリームライター | 長時間ツールの進捗（例: `"10/100件完了"`）をリアルタイム送信。 |
| `runtime.tool_call_id` | 現在のツール呼び出しの一意な ID | `Command` で `ToolMessage` を手動構築する際の紐付け。 |
| `runtime.execution_info` | 実行 ID・スレッド ID・リトライ回数 | 冪等キー生成やリトライ時の挙動分岐。 |
| `runtime.server_info` | Agent Server 実行時の認証ユーザー・アシスタント ID | マネージド環境でのマルチテナント制御。 |

> **予約語に注意**: `runtime` と `config` は LangChain v1 のツール引数における**予約名**です。ビジネスロジックの引数名としてこれらを使うとランタイムエラーが発生するため、必ず `ToolRuntime` 注入専用として扱ってください。

### `runtime.config` と `runtime.context` の使い分け

- **`config={"configurable": {"user_id": ...}}` (`runtime.config`)**: `context_schema` を別途定義せずに軽量にメタデータを渡したい場合や、`thread_id` などの標準 LangGraph 設定と一緒に扱う場合に便利です。
- **`context=UserContext(...)` (`runtime.context`)**: `create_agent(..., context_schema=UserContext)` と組み合わせることで、IDE 補完や静的型チェック（`ToolRuntime[UserContext]`）が効く厳格な設計にしたい場合に推奨されます。

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool


# 1. 実行単位で不変なコンテキストを dataclass で型定義する
@dataclass
class UserContext:
    user_id: str
    tenant_id: str


# 2. ToolRuntime[UserContext] を引数に宣言すると、LLM向けスキーマからは完全隠蔽される
@tool
def get_tenant_quota(runtime: ToolRuntime[UserContext]) -> str:
    """現在のテナントのAPIクォータ残量を確認する。"""
    # IDE補完・静的型チェックが効く形で安全にテナント情報を参照できる
    tenant = runtime.context.tenant_id
    user = runtime.context.user_id
    return f"tenant={tenant}, user={user}, remaining_quota=4200"
```

### 旧来パターン（v0.3）から `ToolRuntime`（v1）への移行対照表

| 従来の注入手段（v0.3 以前） | LangChain v1 の統一表現（`ToolRuntime`） | 改善点 |
| :--- | :--- | :--- |
| `state: Annotated[dict, InjectedState]` | `runtime: ToolRuntime` → `runtime.state` | 引数増殖を防ぎ、1 つの引数で全ランタイム情報へアクセス可能。 |
| `store: Annotated[BaseStore, InjectedStore]` | `runtime: ToolRuntime` → `runtime.store` | 長期記憶ストアの参照を状態参照と統一。 |
| `tool_call_id: Annotated[str, InjectedToolCallId]` | `runtime: ToolRuntime` → `runtime.tool_call_id` | `Command` 返却時の `ToolMessage` 生成が簡潔になる。 |
| `get_runtime()` グローバル参照 | `runtime: ToolRuntime` | 明示的な依存注入（DI）となり、単体テスト時のモック差し替えが容易。 |

---

## 3. ツールの戻り値設計と状態更新（`Command` / `return_direct`）

LangChain v1 のツールは用途に応じて 4 種類の返却パターンを使い分けられます。

1. **文字列 (`str`)**: 人間やモデルがそのまま解釈できるテキスト。最も一般的で、自動的に `ToolMessage(content=...)` にラップされます。
2. **構造化オブジェクト (`dict` 等)**: 後段のモデル推論でフィールド単位の比較・計算を行わせたい場合に使用。シリアライズされて `ToolMessage` に格納されます。
3. **`Command` オブジェクト**: ツール実行と同時に `AgentState` のカスタムフィールドを書き換えたい場合に使用。`Command(update={"user_name": ..., "messages": [ToolMessage(..., tool_call_id=runtime.tool_call_id)]})` のように、**必ず対応する `ToolMessage` を同梱**する必要があります（欠落すると `ToolNode` が `ValueError` を送出）。なお、複数のツールが並列で同一カスタムフィールドを更新する可能性がある場合は、LangGraph の Reducer を定義して競合を解決します。
4. **`@tool(return_direct=True)`**: ツールの出力をそのまま最終回答として返し、再度の LLM 呼び出しをスキップしてループを即終了する最適化。定型照会などでレイテンシとコストを削減できます（ただし並列呼び出し時はバッチ内の**すべてのツール**が `return_direct=True` の場合のみ即終了します）。

```python
from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class SREAgentState(AgentState):
    """会話履歴に加えて現在の対象リージョンを保持するカスタム状態。"""
    active_region: str


@tool
def set_active_region(region: str, runtime: ToolRuntime[None, SREAgentState]) -> Command:
    """調査対象のデフォルトリージョンをエージェント状態へ保存する。"""
    # 状態更新とあわせて、現在の tool_call_id に紐付く ToolMessage を必ず返す
    return Command(
        update={
            "active_region": region,
            "messages": [
                ToolMessage(
                    content=f"Active region updated to {region}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

---

## 4. エラーハンドリングの多層防御戦略

本番エージェントにおいて「ツールがエラーを起こしたら例外でプロセス終了」は最悪の設計です。LLM エージェントの強みは、**エラーを観察（Observation）として受け取り、自律的にリカバリまたはユーザーへ的確に報告できること**にあります。

LangChain v1 では、エラーハンドリングを 3 つの層で設計します。

{% mermaid %}
flowchart TD
    CALL["モデルがツール呼び出しを発行"] --> L1["第1層: Pydantic args_schema 検証<br/>(型不一致・必須欠落を事前検出)"]
    L1 -->|"検証OK"| L2["第2層: ツール内部のドメイン例外ガード<br/>(ゼロ除算・404 等を 'Error: ...' 文字列で返却)"]
    L2 -->|"予期せぬ例外発生"| L3["第3層: @wrap_tool_call Middleware<br/>(全ツール横断で例外を捕捉し ToolMessage 化)"]
    L2 -->|"正常終了 / エラー文字列"| TM["ToolMessage として履歴へ追加"]
    L3 --> TM
    TM --> MODEL["Model Node へフィードバック<br/>(再試行・代替手段・ユーザーへの説明)"]

    style L1 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style L2 fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style L3 fill:#fff5f5,stroke:#c53030,stroke-width:1.5px,color:#742a2a
    style MODEL fill:#f0fff4,stroke:#2f855a,stroke-width:2px,color:#22543d
{% endmermaid %}

### 第 1 層：スキーマ検証（`args_schema`）
Pydantic モデルによる型・制約チェックで、明らかに不正な引数がビジネスロジックへ到達するのを防ぎます。

### 第 2 層：ツール内部のドメインエラー返却（Soft Error）
ゼロ除算（`b == 0`）、サポート外の演算子、対象レコード未検出といった**想定内の業務エラー**は、Python の例外をそのまま送出せず、`"Error: Division by zero is not allowed."` のように**原因が明確なエラー文字列**として返します。これにより、同一バッチで並列実行されている他のツール呼び出し（例: `100 / 4` や `get_user_tier`）の結果を無駄にせず、モデルへすべての結果を揃えて渡すことができます。

### 第 3 層：`@wrap_tool_call` Middleware による横断的例外捕捉
外部 API のタイムアウトや予期せぬ `RuntimeError` など、ツール内部で拾いきれない例外は Middleware の `@wrap_tool_call` で一括捕捉し、`ToolMessage` に変換してモデルへ返します。

```python
from collections.abc import Callable
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest


# 第3層: すべてのツール実行をラップし、未処理例外をモデルが理解可能な ToolMessage に変換する
@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """ツール実行中の予期せぬ例外を捕捉し、エージェントループのクラッシュを防ぐ。"""
    try:
        return handler(request)
    except Exception as e:
        # 例外メッセージを ToolMessage として返すことで、モデルが修正・謝罪・再試行を判断できる
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({e})",
            tool_call_id=request.tool_call["id"],
        )
```

---

## 5. まとめと実践編への橋渡し

Part 2 理論編の重要ポイントを振り返ります。

1. **Pydantic `args_schema`**: 型安全性と LLM 向けスキーマ記述を単一ソースで管理できる。
2. **`ToolRuntime` による安全な注入**: `runtime: ToolRuntime` はモデルへのスキーマから自動隠蔽されるため、ユーザー ID や権限情報をプロンプトインジェクションから守りつつツール内で参照できる。
3. **多層エラーハンドリング**: 想定内の業務エラーは明確なエラー文字列として返し、予期せぬ例外は `@wrap_tool_call` で `ToolMessage` 化することで、エージェントの自己リカバリ能力を最大限に引き出せる。

次の **Part 2 実践編** では、`safe_calculator`（Pydantic `args_schema` + ゼロ除算ガード）と `get_user_tier`（`ToolRuntime` による `configurable.user_id` 取得）を実装し、**1 回のユーザー指示で 3 つのツール呼び出し（属性取得・ゼロ除算エラー・正常除算）がどう並列実行・統合されるか**を実測ログで検証します。
