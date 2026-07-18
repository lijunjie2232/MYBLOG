---
title: "LangGraph Part 5 実践編：承認フローの一時停止・再開と過去チェックポイントからの状態フォーク（Time Travel）"
date: 2026-06-18 07:50:51
categories:
  - AI
tags:
  - LangGraph
  - Interrupts
  - Time Travel
  - Human-in-the-Loop
  - Python
  - AI Agent
lang: ja
description: "LangGraph Part 5 実践編。interrupt() による承認フローの一時停止・Command(resume=...) 再開と、get_state_history / update_state を使った過去チェックポイントからの状態フォーク（Time Travel）を実測検証します。"
---

# はじめに

前回の理論編（Part 5 理論編）では、`interrupt()` と `Command(resume=...)` による動的 Human-in-the-Loop（HITL）の仕組みと、不変チェックポイント履歴を活用した Time Travel（Replay / Fork）の理論を学びました。

本記事（実践編）では、**「インフラ変更プランの起案（`draft`） → 人間による承認レビュー（`review`） → 実行（`execute`）」** という実務さながらの承認フローを構築し、以下の 2 段階を実際の実行コードと出力結果で検証します。

1. 最初のプラン（`DBインデックス追加`）を `interrupt()` で一時停止し、`Command(resume={"approved": True})` で承認・完了させる。
2. 完了後、`get_state_history()` から `draft` 完了直後のチェックポイントを探し出し、`update_state()` でプラン内容を「`読み取りレプリカ増設（フォーク版）`」へ書き換えてフォーク（Time Travel）し、別世界線として承認・実行まで完走させる。

---

# 1. 理論編の振り返りと実験設計

本ラボの設計に直結する理論編の要点を振り返ります。

> **理論編より引用（`interrupt()` と再開時の挙動）**：
> その後、人間が UI 上で承認ボタンを押すなどして `graph.invoke(Command(resume=decision), config=config)` を呼び出すと、ランタイムはチェックポイントから状態を復元し、**`interrupt()` を呼んだノードの先頭から再実行**します。この再実行時に `interrupt()` に到達すると、今回は一時停止せず、`Command(resume=...)` で渡された `decision` をそのまま戻り値として返却し、後続の処理を続行します。

> **理論編より引用（Fork による別世界線の生成）**：
> 過去のチェックポイントに対して `graph.update_state(prior_config, values={...})` を呼ぶと、**元の履歴を上書き（ロールバック）するのではなく、その時点を親とする新しいチェックポイント（フォーク）を作成**し、新しい `fork_config` を返します。続けて `graph.invoke(None, fork_config)` を実行することで、元の実行履歴を完全に保持したまま、別条件のパラレルワールド（What-If シナリオ）を探索できます。

> **理論編より引用（Time Travel 中の Interrupt 再発火）**：
> Replay はキャッシュ読み出しではなく「再実行」です。チェックポイント以降にある LLM 呼び出し、外部 API リクエスト、そして `interrupt()` はすべて改めて発火します。

今回の実験グラフの構造と、フォーク後の再開フローは以下の通りです。

{% mermaid %}
flowchart TD
    S(("START")) --> Draft["draft ノード<br/>proposal に 'Plan: ' を付与"]
    Draft --> CP_Target["★チェックポイント (next=('review',))<br/>ここを Time Travel の分岐点にする"]
    CP_Target --> Review["review ノード<br/>interrupt() で承認待ち一時停止"]
    Review -->|"Command(resume={'approved': True/False})"| Exec["execute ノード<br/>approved なら EXECUTED / 否なら CANCELLED"]
    Exec --> E(("END"))

    CP_Target -.->|"update_state(proposal='読み取りレプリカ増設')"| ForkNode["フォーク先チェックポイント<br/>(fork_cfg)"]
    ForkNode -.->|"invoke(None, fork_cfg)"| Review

    style CP_Target fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    style Review fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    style ForkNode fill:#fdf4ff,stroke:#c026d3,stroke-width:2px,color:#701a75
    style Exec fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
{% endmermaid %}

---

# 2. 実装コード（Lab 5 全文）

本ラボは制御フローとチェックポイント操作の純粋な挙動を高速・決定論的に検証するため、LLM 呼び出しを含めず純粋な Python ノードで構成しています（実務で LLM を組み込む場合も `os.getenv("BASE_URL")` / `os.getenv("API_KEY")` 経由のクライアントを `draft_node` 等に差し込むだけで同一の HITL・Time Travel 構造がそのまま機能します）。

```python
import json
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

def run_lab5():
    # 承認ワークフロー全体の状態スキーマ
    class ApprovalState(TypedDict):
        proposal: str
        approved: bool
        result: str

    def draft_node(state: ApprovalState) -> dict:
        """提案テキストを整形する起案ノード"""
        return {"proposal": f"Plan: {state['proposal']}"}

    def human_review_node(state: ApprovalState) -> dict:
        """
        人間の承認を待つレビューノード。
        interrupt() の前に非冪等な副作用を置かず、JSON互換の辞書のみを渡す。
        """
        decision = interrupt(
            {
                "question": "このプランを承認しますか？",
                "proposal": state["proposal"],
            }
        )
        # Command(resume={"approved": True}) の辞書がそのまま decision に入る
        return {"approved": bool(decision.get("approved", False))}

    def execute_node(state: ApprovalState) -> dict:
        """承認結果に応じて実行またはキャンセルを記録するノード"""
        if state["approved"]:
            return {"result": f"EXECUTED: {state['proposal']}"}
        return {"result": f"CANCELLED: {state['proposal']}"}

    # Interrupts と Time Travel には Checkpointer の指定が必須
    checkpointer = InMemorySaver()
    graph = (
        StateGraph(ApprovalState)
        .add_node("draft", draft_node)
        .add_node("review", human_review_node)
        .add_node("execute", execute_node)
        .add_edge(START, "draft")
        .add_edge("draft", "review")
        .add_edge("review", "execute")
        .add_edge("execute", END)
        .compile(checkpointer=checkpointer)
    )

    # thread_id がチェックポイント履歴の永続カーソルになる
    cfg = {"configurable": {"thread_id": "hitl-tt-01"}}

    # 1. 初回実行：draft を通過し、review ノードの interrupt() で一時停止する
    r1 = graph.invoke(
        {"proposal": "DBインデックス追加", "approved": False, "result": ""},
        config=cfg,
    )

    # 2. 承認して再開：同一 thread_id に対して Command(resume=...) を渡す
    r2 = graph.invoke(Command(resume={"approved": True}), config=cfg)

    # 3. Time Travel：履歴を取得し、'draft' 完了直後（次が 'review'）のチェックポイントを探す
    history = list(graph.get_state_history(cfg))
    after_draft = [h for h in history if h.next == ("review",)][0]

    # 4. Fork：過去チェックポイントの proposal を書き換えて新しい分岐点（fork_cfg）を生成
    fork_cfg = graph.update_state(
        after_draft.config,
        {"proposal": "Plan: 読み取りレプリカ増設（フォーク版）"},
    )

    # 5. フォーク先から再開（入力は None）：review ノードが再実行され、再び interrupt() で停止
    r3_pause = graph.invoke(None, config=fork_cfg)

    # 6. フォーク先の割り込みを承認して完走させる
    #    ※ 重要：元の cfg ではなく fork_cfg（または最新スレッド状態）を使う
    r3_final = graph.invoke(Command(resume={"approved": True}), config=fork_cfg)

    return {
        "first_interrupt": str(r1.get("__interrupt__", "")),
        "first_result": r2["result"],
        "fork_result": r3_final["result"],
        "total_checkpoints": len(history),
    }

if __name__ == "__main__":
    out = run_lab5()
    print(json.dumps(out, indent=2, ensure_ascii=False))
```

---

# 3. 実測結果とチェックポイント履歴の解剖

上記コードを実行した結果（`lab5_interrupt_timetravel`、所要時間 `0.01s`）は以下の通りです。

```json
{
  "status": "ok",
  "elapsed_s": 0.01,
  "output": {
    "first_interrupt": "[Interrupt(value={'question': 'このプランを承認しますか？', 'proposal': 'Plan: DBインデックス追加'}, id='ae66648a327d7951fafb3955dade1272', response_schema=None)]",
    "first_result": "EXECUTED: Plan: DBインデックス追加",
    "fork_result": "EXECUTED: Plan: 読み取りレプリカ増設（フォーク版）",
    "total_checkpoints": 5
  }
}
```

### ① `first_interrupt` の構造を詳しく見る

1 回目の `invoke()` が返した `r1["__interrupt__"]` には、次の `Interrupt` オブジェクトが格納されていました。

- **`value`**：`{'question': 'このプランを承認しますか？', 'proposal': 'Plan: DBインデックス追加'}` — `draft_node` が整形した後の `proposal` がそのまま人間への確認プロンプトとして表面化しています。
- **`id`**：`'ae66648a327d7951fafb3955dade1272'` — LangGraph が自動採番した一意の割り込み ID です（複数並列割り込み時に `Command(resume={id: value})` で対応付けるためのキーになります）。
- **`response_schema`**：`None` — 今回はスキーマを省略したため `None` になっています。Pydantic モデルを渡せばここに JSON Schema が入ります。

### ② なぜ `total_checkpoints` は `5` なのか？

フォーク実行前に取得した `history` の件数が `5` である理由は、Pregel の Superstep 境界ごとに以下の 5 つのチェックポイントが記録されるためです（`get_state_history()` は降順＝新しい順で返します）。

| 順序（新しい順） | step | `next`（次に実行予定のノード） | その時点のグラフ状態 |
| :--- | :---: | :--- | :--- |
| 1（最新） | `3` | `()`（完了） | `execute` 完了後（`result="EXECUTED: Plan: DBインデックス追加"`） |
| 2 | `2` | `('execute',)` | `review` 承認完了後（`approved=True`） |
| 3 | `1` | `('review',)` | **`draft` 完了直後（`after_draft` で抽出された分岐点）** |
| 4 | `0` | `('draft',)` | 初期入力投入直後（`proposal="DBインデックス追加"`） |
| 5（最古） | `-1` | `('__start__',)` | グラフ起動時の初期スナップショット |

### ③ フォーク実行時の「2 段階再開」に注目

コードのステップ 5〜6 で、フォーク後に 2 回 `invoke()` を呼んでいる点が極めて重要です。

{% mermaid %}
sequenceDiagram
    participant Caller as 呼び出し元
    participant Graph as CompiledStateGraph
    participant CP as InMemorySaver

    Caller->>CP: update_state(after_draft.config, proposal="読み取りレプリカ増設")
    CP-->>Caller: fork_cfg（新しい checkpoint_id を発行）

    Note over Caller,CP: ステップ5：過去時点から再開（入力=None）
    Caller->>Graph: invoke(None, config=fork_cfg)
    Graph->>Graph: review ノードを再実行 → interrupt() が再発火！
    Graph-->>Caller: r3_pause（一時停止状態）

    Note over Caller,CP: ステップ6：フォーク先の割り込みを承認
    Caller->>Graph: invoke(Command(resume={"approved": True}), config=fork_cfg)
    Graph->>Graph: review 完了 → execute 実行
    Graph-->>Caller: "EXECUTED: Plan: 読み取りレプリカ増設（フォーク版）"
{% endmermaid %}

フォーク後の `review` ノードは「元のタイムラインで承認済みだった事実」を引き継ぎません。プラン内容が「読み取りレプリカ増設」に変わっている以上、**人間の再承認を必ず求め直す**という安全設計になっているため、`invoke(None, config=fork_cfg)` で一度 `interrupt()` による一時停止に入り、続けて `Command(resume={"approved": True})` を渡すことで初めて `execute_node` が完了します。

---

# 4. 注意点・ハマりどころ（Pay Attention）

HITL と Time Travel を本番アプリへ組み込む際に頻発するトラブルと対策をまとめます。

1. **フォーク後の `invoke(None, fork_cfg)` で「終わらない（結果が空のまま）」と勘違いする**
   - 本ラボのように分岐点の先に `interrupt()` がある場合、`invoke(None, fork_cfg)` は最終結果ではなく**一時停止状態（`__interrupt__` 付き）**を返します。呼び出し元では必ず `if "__interrupt__" in result:`（または v2 の `result.interrupts`）を判定し、ループで再開処理を行う設計にしてください。
2. **フォーク後の再開時に古い `checkpoint_id` を握りっぱなしにする問題**
   - `update_state()` が返す `fork_cfg` には特定の `checkpoint_id` が固定で入っています。本ラボのように「フォーク直後の 1 回の割り込み」を再開するだけなら動作しますが、複数回の割り込みやマルチターン対話を続ける場合は、`thread_id` だけを指定した `{"configurable": {"thread_id": "hitl-tt-01"}}` で最新チェックポイントを辿るか、最新状態の `config` を使うよう意識してください（固定の古い `checkpoint_id` を渡し続けると毎回そこからリプレイされてしまいます）。
3. **`get_state_history()` の並び順と重複 `next` の選択**
   - `get_state_history()` は**新しい順（降順）**です。ループのあるグラフや複数回リトライされたグラフでは、`h.next == ("review",)` に一致するスナップショットが複数存在し得ます。`[0]`（最新）を選ぶのか `[-1]`（初回）を選ぶのか、あるいは `h.metadata["step"]` で厳密に絞り込むのかを明確にしてください。
4. **`interrupt()` を含むノード内での `try / except Exception` 事故**
   - ログ出力目的でノード全体を `try: ... except Exception as e: logger.error(e)` のように囲うと、`interrupt()` が投げる内部例外まで捕捉してしまい、一時停止が完全に壊れます。例外処理は `interrupt()` と分離するか、特定の業務例外のみを捕捉してください。
5. **本番環境での `InMemorySaver` 利用禁止**
   - 本ラボでは手軽さのため `InMemorySaver` を使いましたが、人間の承認は数時間〜数日後に届くことも珍しくありません。プロセス再起動でメモリ上のチェックポイントが消えると承認再開が不可能になるため、本番環境では必ず `SqliteSaver` や `PostgresSaver` などの永続 Checkpointer を使用してください。

---

# まとめ

本実践編では、`interrupt()` による承認一時停止・`Command(resume=...)` による再開、そして `get_state_history()` + `update_state()` による Time Travel（Fork）を一気通貫で検証しました。

- 初回実行では `draft` 後に `interrupt()` が発火し、`id='ae66648a327d7951fafb3955dade1272'` を持つ `Interrupt` ペイロードが返却され、承認再開により `EXECUTED: Plan: DBインデックス追加` まで完走した。
- その後、`total_checkpoints=5` の履歴から `next == ("review",)` の時点を抽出してフォークし、元の履歴を壊すことなく `EXECUTED: Plan: 読み取りレプリカ増設（フォーク版）` の別世界線を安全に再承認・実行できた。

次回（Part 6）では、大規模エージェントをモジュール化するための **Subgraphs（サブグラフ）** と、スレッドを跨いで長期記憶を共有する **Cross-Thread Store** を解説します。
