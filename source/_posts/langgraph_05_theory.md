---
title: "LangGraph Part 5 理論編：Interrupts（interrupt() / Command(resume=...)）による Human-in-the-Loop と Time Travel（Replay / Fork）"
date: 2026-06-18 02:08:06
categories:
  - AI
tags:
  - LangGraph
  - Interrupts
  - Human-in-the-Loop
  - Time Travel
  - Checkpointer
  - AI Agent
lang: ja
description: "LangGraphの動的割り込みinterrupt()とCommand(resume=...)によるHuman-in-the-Loop設計、絶対守るべき4つのルール、および過去チェックポイントからのTime Travel（Replay / Fork）の内部メカニズムを徹底解説します。"
---

# はじめに

自律型 AI エージェントを業務システムに組み込む際、「本番データベースのスキーマ変更」「高額決済の実行」「顧客へのメール一斉送信」といった不可逆な操作を完全に自律実行させるのは危険です。実務では、**「重要な判断の直前でエージェントを一時停止し、人間の承認や修正を受け取ってから再開する（Human-in-the-Loop: HITL）」** 仕組みが不可欠になります。

さらに、承認後に「もしあの時点で別の案を選んでいたらどうなったか？」を検証したり、誤った判断をしたステップへ戻ってやり直したりできる **Time Travel（Replay / Fork）** 機能があれば、エージェントのデバッグ性と監査性は飛躍的に高まります。

本記事（Part 5 理論編）では、LangGraph の **Interrupts（動的割り込み）** と **Time Travel（タイムトラベル）** の内部アーキテクチャと設計ルールを詳しく解説します。

---

# 1. Interrupts による動的 Human-in-the-Loop

従来の静的ブレークポイント（`interrupt_before` / `interrupt_after`）はグラフコンパイル時に固定のノード名を指定する必要がありました。これに対し、LangGraph の `interrupt()` 関数は**ノード内の任意の場所・任意の条件分岐の中で動的に一時停止を発火できる**強力な仕組みです。

## 1.1 `interrupt()` と `Command(resume=...)` のライフサイクル

`interrupt()` を機能させるには、次の 3 つが必須要件となります。

1. **Checkpointer の設定**：一時停止時のグラフ状態を保存するため、`compile(checkpointer=...)` が必須（本番では PostgreSQL / SQLite 等の永続 Checkpointer を使用）。
2. **`thread_id` の指定**：`config={"configurable": {"thread_id": "..."}}` が再開位置を指す永続カーソルとなる。
3. **JSON シリアライズ可能なペイロード**：`interrupt(payload)` に渡す値と、`Command(resume=value)` で返す値はシリアライズ可能である必要がある。

### 内部で何が起きているのか？

ノード内で `interrupt(payload)` が呼ばれると、LangGraph ランタイムは内部で**特殊な割り込み例外（`GraphInterrupt`）を送出**してノードの実行を即座に中断します。ランタイムはこの例外を捕捉すると、現在の State と割り込み情報をチェックポイントに保存し、呼び出し元へ制御を返します（デフォルトの `invoke()` では結果辞書の `__interrupt__` キー、`version="v2"` では `GraphOutput.interrupts`、`stream_events(..., version="v3")` では `stream.interrupts` にペイロードが現れます）。

その後、人間が UI 上で承認ボタンを押すなどして `graph.invoke(Command(resume=decision), config=config)` を呼び出すと、ランタイムはチェックポイントから状態を復元し、**`interrupt()` を呼んだノードの先頭から再実行**します。この再実行時に `interrupt()` に到達すると、今回は一時停止せず、`Command(resume=...)` で渡された `decision` をそのまま戻り値として返却し、後続の処理を続行します。

{% mermaid %}
sequenceDiagram
    participant App as アプリ / UI（人間）
    participant Graph as LangGraph ランタイム
    participant CP as Checkpointer
    participant Node as human_review_node

    App->>Graph: invoke(初期入力, thread_id="T1")
    Graph->>Node: ノード実行開始
    Node->>Graph: interrupt({"question": "承認しますか？"})
    Note over Node,Graph: 内部で特殊例外を送出し即座に一時停止
    Graph->>CP: 現在Stateと割り込み情報を永続保存
    Graph-->>App: __interrupt__ = [Interrupt(value=..., id=...)]

    Note over App: 人間が内容を確認（数秒〜数日待機可能）

    App->>Graph: invoke(Command(resume={"approved": True}), thread_id="T1")
    Graph->>CP: thread_id="T1" の最新チェックポイントを復元
    Graph->>Node: ノードの「先頭から」再実行
    Node->>Graph: interrupt(...) 到達 → resume値を即返却！
    Node-->>Graph: {"approved": True} を返却して完了
    Graph->>CP: 次ステップの状態を保存
    Graph-->>App: 最終結果を返却
{% endmermaid %}

## 1.2 型安全な入力検証：`response_schema` と並列 Interrupts

### `response_schema` による型付きバリデーション（`langgraph>=1.2.12`）

`interrupt(payload, response_schema=DecisionModel)` のように Pydantic モデル・`TypedDict`・dataclass を渡すと、`Interrupt.response_schema` に JSON Schema が自動生成されます。クライアント UI はこのスキーマから型付き入力フォームを自動描画でき、再開時には Pydantic の型変換・バリデーションを通過したオブジェクトが `interrupt()` の戻り値として得られます。

```python
from typing import Any
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt

class ApprovalDecision(BaseModel):
    """人間から受け取る承認レスポンスの厳密なスキーマ"""
    approved: bool
    reviewer_note: str | None = None

class SchemaState(TypedDict):
    decision: Any

def typed_approval_node(state: SchemaState) -> dict:
    # response_schema を指定すると Interrupt.response_schema に JSON Schema が公開される
    decision = interrupt(
        {"question": "本番デプロイを承認しますか？"},
        response_schema=ApprovalDecision,
    )
    # 再開時は Pydantic の ApprovalDecision インスタンスとして受け取れる
    return {"decision": decision}

graph = (
    StateGraph(SchemaState)
    .add_node("approve", typed_approval_node)
    .add_edge(START, "approve")
    .add_edge("approve", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "schema-demo-1"}}
paused = graph.invoke({"decision": None}, config)
# UI 側は paused["__interrupt__"][0].response_schema からフォームを自動生成可能

# 型不正な値を渡すと pydantic.ValidationError が発生し、同じ thread_id で再送できる
resumed = graph.invoke(
    Command(resume={"approved": True, "reviewer_note": "LGTM"}),
    config,
)
```

### 並列ブランチでの複数 Interrupts 同時再開

ファンアウトした複数ノードが同時に `interrupt()` を呼んだ場合、それぞれの `Interrupt` オブジェクトには一意の `id` が付与されます。再開時は `Command(resume={interrupt_id_1: ans1, interrupt_id_2: ans2})` のように ID マップを渡すことで、1 回の呼び出しで全ブランチへ正確に回答を配送できます。

```python
# 並列ノードが同時に停止した場合の再開パターン
# paused["__interrupt__"] に複数の Interrupt(value=..., id=...) が格納される
resume_map = {
    intr.id: f"承認済み: {intr.value}"
    for intr in paused["__interrupt__"]
}
# ID をキーにした辞書を渡すことで、各並列ブランチへ対応する値が返る
final_state = graph.invoke(Command(resume=resume_map), config)
```

> **重要**：`invoke()` や `stream()` の入力として渡してよい `Command` パターンは **`Command(resume=...)` のみ**です。`Command(update=..., goto=...)` はノード関数からの戻り値専用であり、通常のマルチターン会話の継続にはプレーンな入力辞書を渡してください。

---

# 2. 絶対に守るべき「Interrupts の 4 大ルール」

`interrupt()` が「特殊例外による中断」と「再開時のノード先頭からの再実行」で実装されているため、以下の 4 つのルールを破ると深刻なバグを引き起こします。

### ルール 1：`interrupt()` を包括的な `try / except Exception` で囲まない
`interrupt()` は内部例外を送出してランタイムに一時停止を通知します。`except Exception:` で握りつぶすと一時停止が機能しなくなります。エラー処理が必要な場合は、`interrupt()` の外に分離するか、具体的な例外クラス（例：`except NetworkError:`）だけを捕捉してください。

### ルール 2：1 つのノード内で `interrupt()` の呼び出し順序を動的に変えない
1 つのノードに複数の `interrupt()` がある場合、LangGraph はタスクの resume リストと**厳密なインデックス順**で対応付けます。`if` 条件で途中の `interrupt()` をスキップしたり、非決定論的な `while True` ループの中で `interrupt()` を呼んだりすると、再実行時にインデックスがずれて誤った値が返ります。入力バリデーションのループを行いたい場合は、ノード内ループではなく**グラフの条件付きエッジ（Conditional Edge）**で自ノードへ戻す設計にしてください。

### ルール 3：`interrupt()` にシリアライズ不能なオブジェクトを渡さない
関数、クラスインスタンス、DB コネクションなどをペイロードに含めると、Checkpointer への保存時にシリアライズエラーになります。必ず文字列・数値・真偽値・リスト・辞書などの JSON 互換データのみを渡してください。

### ルール 4：`interrupt()` より前に実行する副作用は必ず冪等（Idempotent）にする
再開時にノードは先頭から再実行されるため、`interrupt()` の手前にあるコードは最低 2 回実行されます。レコードの新規作成（`INSERT`）やリストへの追記を `interrupt()` の前に置くと重複データが発生します。**副作用は `interrupt()` の後ろに置くか、別ノードに分離する**のがベストプラクティスです。

```python
from typing import Literal, TypedDict
from langgraph.types import Command, interrupt

class ReviewState(TypedDict):
    proposal: str
    approved: bool

def safe_review_node(state: ReviewState) -> Command[Literal["execute", "cancel"]]:
    # ✅ ルール遵守：
    # 1. try/except Exception で囲まない
    # 2. interrupt() の前に非冪等な副作用（DB INSERT / API呼び出し等）を書かない
    # 3. ペイロードは JSON シリアライズ可能な単純な辞書にする
    decision = interrupt(
        {
            "question": "以下の提案を実行してよろしいですか？",
            "proposal": state["proposal"],
        }
    )

    # 再開後のみ実行されるルーティング処理
    if decision.get("approved"):
        return Command(update={"approved": True}, goto="execute")
    return Command(update={"approved": False}, goto="cancel")
```

---

# 3. Time Travel（Replay と Fork）のメカニズム

LangGraph の Checkpointer は、各 Superstep 完了時のスナップショットを単なる上書きではなく、**親チェックポイントへのポインタを持つ不変の履歴（DAG / 木構造）** として蓄積します。この構造により、過去の任意の時点へ戻る **Time Travel** が実現されます。

## 3.1 Replay（同一時点からの再実行）

`graph.get_state_history(config)` を呼ぶと、そのスレッドの全チェックポイントが**新しい順（降順）**で返されます。各 `StateSnapshot` には「次に実行予定のノード名」を示す `state.next` と、その時点固有の `checkpoint_id` を含む `state.config` が格納されています。

特定の過去スナップショットの `config` を指定して `graph.invoke(None, prior_snapshot.config)` を呼ぶと、**そのチェックポイントより前のノードは再実行せず（保存済み状態を利用）、それ以降のノードだけを再実行**します。

> **注意**：Replay はキャッシュ読み出しではなく「再実行」です。チェックポイント以降にある LLM 呼び出し、外部 API リクエスト、そして `interrupt()` はすべて改めて発火します。

## 3.2 Fork（状態改変による別世界線への分岐）

過去のチェックポイントに対して `graph.update_state(prior_config, values={...})` を呼ぶと、**元の履歴を上書き（ロールバック）するのではなく、その時点を親とする新しいチェックポイント（フォーク）を作成**し、新しい `fork_config` を返します。

続けて `graph.invoke(None, fork_config)` を実行することで、元の実行履歴を完全に保持したまま、別条件のパラレルワールド（What-If シナリオ）を探索できます。

{% mermaid %}
flowchart LR
    CP0["CP-0: START<br/>next=('draft',)"] --> CP1["CP-1: draft 完了<br/>proposal='DBインデックス追加'<br/>next=('review',)"]

    subgraph MainBranch["元のタイムライン（履歴は破壊されない）"]
        CP1 --> CP2["CP-2: review 一時停止<br/>interrupt発火"]
        CP2 -->|"Command(resume=True)"| CP3["CP-3: execute 完了<br/>EXECUTED: DBインデックス追加<br/>next=()"]
    end

    subgraph ForkBranch["フォークした別タイムライン（update_state）"]
        CP1 -.->|"update_state(proposal='レプリカ増設')"| CP1F["CP-1-Fork: 状態改変済み<br/>proposal='読み取りレプリカ増設'<br/>next=('review',)"]
        CP1F -->|"invoke(None, fork_cfg)"| CP2F["CP-2-Fork: review 再一時停止<br/>interrupt再発火"]
        CP2F -->|"Command(resume=True)"| CP3F["CP-3-Fork: execute 完了<br/>EXECUTED: 読み取りレプリカ増設<br/>next=()"]
    end

    style MainBranch fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    style ForkBranch fill:#fdf4ff,stroke:#c026d3,stroke-width:2px,color:#701a75
    style CP1 fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
{% endmermaid %}

### `as_node` パラメータとサブグラフの粒度

- **`as_node` の指定**：`update_state(config, values, as_node="draft")` を指定すると、「まるで `draft` ノードがその更新を返したかのように」Reducer を適用し、`draft` の後続ノードから実行を再開させます。通常は履歴から自動推論されますが、並列ブランチ直後（複数ノードが同時更新して曖昧な場合）や、特定ノードをスキップしたい場合に明示指定します。
- **サブグラフと Time Travel**：デフォルトでは親グラフがサブグラフ全体を 1 つの Superstep として扱うため、サブグラフ内部の途中ノードへはタイムトラベルできません。サブグラフ内部のステップ間でもフォークしたい場合は、サブグラフ側を `.compile(checkpointer=True)` として独立したチェックポイント履歴を持たせ、`graph.get_state(config, subgraphs=True)` からサブグラフ固有の `config` を取得します。

```python
# 1. as_node を明示して過去チェックポイントからフォークする例
history = list(graph.get_state_history(config))
target_cp = next(s for s in history if s.next == ("review",))

fork_config = graph.update_state(
    target_cp.config,
    values={"proposal": "代替案：キャッシュ層の追加"},
    # draft ノードが生成した更新として扱い、後続の review から再開させる
    as_node="draft",
)
# 入力 None でフォーク先から再開（review の interrupt() が再発火する）
fork_paused = graph.invoke(None, fork_config)

# 2. サブグラフ内部のチェックポイントからフォークする場合
#    サブグラフを .compile(checkpointer=True) でコンパイルしておく必要がある
parent_snapshot = parent_graph.get_state(config, subgraphs=True)
sub_config = parent_snapshot.tasks[0].state.config
sub_fork_cfg = parent_graph.update_state(sub_config, {"value": ["forked"]})
parent_graph.invoke(None, sub_fork_cfg)
```

---

# まとめ

- **`interrupt()` + `Command(resume=...)`** は、チェックポイントと内部例外メカニズムを活用し、ノードの任意の箇所で数時間〜数日単位の非同期な人間承認（HITL）を実現する。
- 再開時はノードの先頭から再実行されるため、**「`try/except Exception` 禁止」「呼び出し順序の固定」「JSON シリアライズ可能」「`interrupt()` 前の副作用の冪等化」** の 4 ルールを徹底する。
- **Time Travel（Replay / Fork）** は不変のチェックポイント木構造の上に成り立っており、`get_state_history()` と `update_state()` を使うことで、元履歴を壊さずに過去の任意地点から別シナリオを再実行できる。

次回の「Part 5 実践編」では、`draft -> review (interrupt) -> execute` の承認ワークフローを実際に動かし、一時停止・承認再開から、`draft` 直後のチェックポイントをフォークして別プランを実行する一連の流れを実測出力付きで検証します。
