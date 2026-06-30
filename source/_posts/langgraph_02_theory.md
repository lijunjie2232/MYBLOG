---
title: "LangGraph Part 2 理論編：制御フローの設計（Conditional Edges・Cycles・Send による動的並列化と Command）"
date: 2026-05-30 22:44:58
categories: [AI]
tags: [LangGraph, ControlFlow, Send, Command, MapReduce, 並列処理, 理論]
lang: ja
description: "LangGraph における条件分岐（Conditional Edges）、循環ループ（Cycles）と再帰制限、Send による Map-Reduce 動的並列化、そして Command による状態更新とルーティングの統合設計を徹底解説する。"
---

# LangGraph Part 2 理論編：制御フローの設計（Conditional Edges・Cycles・`Send`・`Command`）

[LangGraph Part 1 理論編](/2026/05/27/langgraph_01_theory/) では、`StateGraph` の土台となる **Pregel BSP の Super-step** と **Channel / Reducer** を整理しました。しかし、実用的な AI エージェントは単純な一直線のパイプラインではなく、「ツール呼び出しの結果を見て再推論する循環ループ」「実行時に件数が決まる検索結果の並列処理」「状態更新と同時に次ノードを切り替えるハンドオフ」など、高度な制御フローを必要とします。

本記事では、LangGraph の Graph API が提供する 4 つの制御フロー機構――**Conditional Edges**、**Cycles（ループと再帰制御）**、**`Send`（動的 Fan-out / Map-Reduce）**、**`Command`（状態更新＋動的ルーティング）**――の理論と使い分けを体系的に解説します。

---

## 1. 条件分岐（Conditional Edges）と並列ブランチの同期

固定遷移（`add_edge`）だけでは、LLM の出力やツール実行結果に応じて経路を切り替えることができません。そこで使うのが **`add_conditional_edges`** です。

### 1.1 Conditional Edges の仕組み

`add_conditional_edges(source, path_fn, path_map)` は、`source` ノードの完了直後にルーティング関数 `path_fn(state)` を評価し、その戻り値に応じて次の Super-step でアクティブ化するノードを決定します。

- 戻り値には **単一のノード名（`str`）**、**終端（`END`）**、**複数ノード名のリスト（並列 Fan-out）**、または後述の **`Send` オブジェクトのリスト** を返せます。
- `path_fn` は **純粋なルーティング判断のみ** を担い、State を更新することはできません。

### 1.2 非対称ブランチの合流：`add_edge([...], target)` vs `defer=True`

並列ブランチ（Fan-out）を実行した後、合流ノード（Fan-in）で結果を統合する際、ブランチの長さが異なる（例：`A -> B -> B_2 -> D` と `A -> C -> D`）と合流タイミングが問題になります。LangGraph は 3 通りの合流セマンティクスを持ちます。

| 合流パターン | 書き方 | 実行タイミングと挙動 |
| --- | --- | --- |
| 個別エッジ | `add_edge("b_2", "d")` と `add_edge("c", "d")` | 各ブランチが完了した Super-step ごとに `d` が**複数回**発火する（長さが異なる場合） |
| リスト形式エッジ | `add_edge(["b_2", "c"], "d")` | 指定した**すべてのノードが完了するまで待機**し、`d` を 1 回だけ実行。ただし条件分岐で片方が未実行だと `d` は発火しない |
| 遅延実行（`defer=True`） | `add_node("d", d, defer=True)` | グラフ全体で他の未完了タスクがすべて終わるまで `d` を待機させ、選択されたブランチ群の完了後に 1 回だけ実行 |

並列実行時の同時実行数を制限したい場合は、`invoke` / `stream` のトップレベル設定（`configurable` の外）で `{"max_concurrency": 10}` を指定します。

---

## 2. Cycles（循環ループ）と再帰制御の設計

ReAct エージェントのように「LLM 推論 → ツール実行 → LLM 再推論」を繰り返すグラフでは、**循環エッジ（Cycle）**が不可欠です。循環グラフを安全に運用するには、必ず「終了条件」と「安全装置（Recursion Limit）」の両方を設計します。

{% mermaid %}
flowchart TD
    S(["START"]) --> R["reasoning_node<br/>LLM 推論・ツール判定"]
    R -->|"remaining_steps <= 2<br/>または終了条件達成"| FB["fallback / finalize<br/>ベストエフォート回答生成"]
    R -->|"ツール呼び出しが必要<br/>かつステップ残数あり"| T["tool_node<br/>外部ツール実行"]
    T -->|"結果を State へ追記"| R
    FB --> E(["END"])

    style S fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style R fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
    style T fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style FB fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
    style E fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
{% endmermaid %}

### `GraphRecursionError` と `RemainingSteps` の使い分け

LangGraph のランタイムは 1 回の実行あたりの最大 Super-step 数を **`recursion_limit`** で制限します（`config={"recursion_limit": 25}` のようにトップレベルキーで指定）。上限を超過すると `GraphRecursionError` が送出されます。

| アプローチ | 検知タイミング | 実装方法 | 制御フローへの影響 |
| --- | --- | --- | --- |
| **受動的（Reactive）** | 上限超過後 | グラフ外で `try / except GraphRecursionError` | 実行は即座に中断される |
| **能動的（Proactive）** | 上限到達前 | State に `remaining_steps: RemainingSteps` を宣言し条件分岐 | 上限手前でまとめノードへ遷移し、優雅に完了（Graceful Degradation）できる |

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps

class LoopState(TypedDict):
    # 推論履歴を蓄積するチャネル
    messages: Annotated[list[str], operator.add]
    # LangGraph ランタイムが自動注入する残り Super-step 数（マネージド値）
    remaining_steps: RemainingSteps

def route_loop(state: LoopState) -> Literal["reasoning", "fallback"]:
    # 残り 2 ステップ以下になったら無限ループを防ぎ、まとめノードへ安全に逃がす
    if state["remaining_steps"] <= 2:
        return "fallback"
    return "reasoning"
```

---

## 3. `Send` による動的並列化（Map-Reduce / Orchestrator-Worker）

通常のエッジや条件分岐の並列 Fan-out には 2 つの制約があります。
1. **トポロジの静性**：分岐先ノードの種類と個数がグラフ構築時（コンパイル時）に固定されている。
2. **共有 State の一括配布**：並列起動されるすべてのノードが同じ全体 State を受け取る。

しかし、実世界のエージェントでは「検索で見つかった N 件のドキュメントをそれぞれ並列要約したい」「プランナーが分解した M 個のサブタスクを同時実行したい」といった、**実行時までアイテム数が不明かつ各ワーカーに異なる入力を渡したい**ケースが頻出します。

> **`Send` の本質**：`Send(node_name, arg)` は、条件付きエッジ関数から複数返却することで、**同一のワーカーノードを実行時データ件数だけ動的に並列インスタンス化し、それぞれに独立した部分入力（専用スキーマ）を配布する Map-Reduce プリミティブ**です。

各ワーカーは自身の専用スキーマ（例：`WorkerState`）だけを受け取り、処理結果を全体 State の**集約用 Reducer 付きチャネル**（例：`Annotated[list[str], operator.add]`）へ書き戻します。同一 Super-step 内で全ワーカーが完了すると、バリア同期によってすべての結果がマージされ、次の Reduce ノードへ安全に引き渡されます。

{% mermaid %}
flowchart LR
    S(["START<br/>items: [A, B, C]"]) -.->|"Send('worker', {item: A})"| W1["worker #1<br/>WorkerState(item=A)"]
    S -.->|"Send('worker', {item: B})"| W2["worker #2<br/>WorkerState(item=B)"]
    S -.->|"Send('worker', {item: C})"| W3["worker #3<br/>WorkerState(item=C)"]

    W1 -->|"summaries += [res_A]"| B{"Super-step バリア<br/>operator.add で統合"}
    W2 -->|"summaries += [res_B]"| B
    W3 -->|"summaries += [res_C]"| B

    B --> R["reducer ノード<br/>Command(update=..., goto=END)"]
    R -.->|"Command.goto"| E(["END"])

    style S fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style W1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style W2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style W3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style B fill:#eceff1,stroke:#455a64,stroke-dasharray: 5 5,color:#263238
    style R fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
    style E fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
{% endmermaid %}

---

## 4. `Command` による状態更新と動的ルーティングの統合

従来、LangGraph では「ノードが State を更新し、エッジが遷移先を決める」という分離が基本でした。しかしマルチエージェントのハンドオフや分類器ノードでは、「LLM の判断結果で State を更新すると同時に、その判断に基づいて遷移先ノードも即座に決めたい」場面が多く存在します。

> **`Command` の設計思想**：`Command(update=..., goto=...)` をノードから返却することで、**1 つの関数内で「State の部分更新（`update`）」と「次の遷移先指定（`goto`）」をアトミックに統合**できます。これにより、遷移判断のためだけの冗長な一時フラグを State に増やす必要がなくなります。

### 4.1 `Command` の 4 つの主要パラメータ

- **`update`**：通常のノード返却値と同じく、Reducer を経由して適用される State の部分更新辞書。
- **`goto`**：次に実行するノード名（`str`）、`END`、ノード名リスト、または `Send` オブジェクト（リスト）。
- **`graph`**：サブグラフ内のノードから親グラフのノードへ直接ジャンプしたい場合に `Command.PARENT` を指定する（親と子で共有するキーを更新する場合は親スキーマに Reducer が必要）。
- **`resume`**：`interrupt()` で一時停止したグラフを再開する際に、`invoke(Command(resume=...), config)` として外部から注入する再開値。

### 4.2 型アノテーション `Command[Literal[...]]` の重要性

ノードが `Command` を返す場合、グラフビルダー側には `add_edge` や `add_conditional_edges` を書かないため、静的解析だけでは遷移先が分かりません。そこで戻り値型に **`Command[Literal["node_b", "node_c"]]`** のように遷移候補を明示します。LangGraph はこの型ヒントを読み取ってグラフ可視化（Mermaid 描画等）や構造チェックを行います。

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import END
from langgraph.types import Command

class ReviewState(TypedDict):
    score: int
    status: str

def evaluate_node(state: ReviewState) -> Command[Literal["retry_node", END]]:
    # 評価ロジックを実行し、State 更新と遷移先決定を 1 回の return で完結させる
    if state["score"] >= 80:
        return Command(
            update={"status": "approved"},  # 状態更新
            goto=END,                       # グラフ終了へ遷移
        )
    return Command(
        update={"status": "needs_revision"},
        goto="retry_node",                  # 再実行ノードへ遷移
    )
```

### 4.3 `Command` を扱う上での 3 つの厳格ルール

1. **同一ノードで `Command` と静的エッジ（`add_edge`）を混在させない**
   `Command(goto=...)` は動的エッジを**追加**する挙動であり、同じノードに `add_edge` も張られていると**両方の経路が同時実行**されてしまいます。
2. **ツールから `Command` を返す場合は `ToolMessage` を必ず含める**
   `@tool` 関数から `Command(update=...)` を返す場合、LLM プロバイダの規約上、`AIMessage(tool_calls=...)` の直後に対応する `tool_call_id` を持つ `ToolMessage` が必須です。`update` 内の `messages` に必ず `ToolMessage` を含め、`ToolNode` を通じて伝播させます。
3. **`invoke(Command(update=...))` を新ターンの開始に使わない**
   `invoke` / `stream` の入力に `Command` を渡すのは **`interrupt()` からの再開（`Command(resume=...)`）専用**です。通常の会話の次ターンを開始したいときに `Command(update=...)` を渡すと、`START` からではなく最新チェックポイントからの再開扱いとなり、終了済みスレッドでは何も動かず停滞してしまいます。新ターン開始には必ず通常の `dict` を渡してください。

---

## 5. Conditional Edges vs `Send` vs `Command` の使い分け指針

| 要件 | 推奨プリミティブ | 理由 |
| --- | --- | --- |
| 既存 State の値だけで単純に分岐したい | `add_conditional_edges` | ノードの責務とルーティングが明確に分離され、テストしやすい |
| 実行時に決まる N 件のタスクを並列処理したい | `Send`（Conditional Edges 経由） | 動的 Fan-out とワーカー専用スキーマ配布ができる唯一の手段 |
| State 更新と次ノード決定を 1 ステップで完結させたい | `Command(update=..., goto=...)` | ルーティング用の一時 State フィールドを排除できる |
| サブグラフから親グラフの別ノードへ直接ハンドオフしたい | `Command(..., graph=Command.PARENT)` | 通常のエッジでは表現できない階層横断遷移が可能 |

---

## 6. まとめ

本記事では、LangGraph の高度な制御フローを構成する理論要素を整理しました。

- **Conditional Edges & 並列合流**：`add_edge([...], target)` と `defer=True` による非対称ブランチの安全な同期。
- **Cycles & 再帰制御**：`recursion_limit` による保護と `RemainingSteps` による能動的なフォールバック設計。
- **`Send`**：実行時データ件数に応じた動的 Map-Reduce 並列化とワーカー専用スキーマ。
- **`Command`**：State 更新と遷移先決定の統合、および静的エッジとの混在禁止ルール。

次回の **Part 2 実践編** では、`Send` による 3 並列 LLM ワーカー実行（Map）と、`Command` による統合レポート生成＋終端遷移（Reduce）を `MiniCPM-V-4.7-35B-A3B` で実装・実測します。

---

### 参考リンク
- [Build sequences, branches, and loops（公式ドキュメント）](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow)
- [Use Send and Command（公式ドキュメント）](https://docs.langchain.com/oss/python/langgraph/graph-api-send-command)
- [LangGraph ソースコード（types.py: Send / Command）](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py)
