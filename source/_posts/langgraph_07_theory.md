---
title: "LangGraph Part 7 理論編：Functional API（@entrypoint / @task）・Pregel BSP 実行モデルと Workflows vs Agents 設計指針"
date: 2026-06-24 07:27:40
categories: [AI]
tags: [LangGraph, Functional API, Pregel, BSP, Workflows, Agents, Python, AI]
lang: ja
description: "LangGraph の低レベル実行基盤である Pregel BSP モデル（Actors と Channels）の仕組みを紐解き、Graph API と並ぶもう一つの高レベル抽象である Functional API（@entrypoint / @task）の決定的リプレイ規則、および Workflows vs Agents の 6 つの設計パターンを体系的に解説します。"
---

## はじめに：LangGraph の 3 層アーキテクチャ

Part 1〜6 では主に `StateGraph`（Graph API）を使ってエージェントやワークフローを構築してきました。最終章となる Part 7 では視点を一段深く掘り下げ、LangGraph 全体を支える **3 層アーキテクチャ** の全体像を完成させます。

1. **低レベル実行基盤：Pregel ランタイム**（Google の Pregel アルゴリズム／BSP モデルに基づく Actors & Channels エンジン）
2. **高レベル宣言的 API：Graph API（`StateGraph`）**（ノードとエッジ、共有 State スキーマでグラフを定義）
3. **高レベル手続き的 API：Functional API（`@entrypoint` / `@task`）**（通常の Python の `if` / `for` / 関数呼び出しのまま永続化・中断再開・ストリーミングを付与）

実は、`StateGraph.compile()` の戻り値も、`@entrypoint` でデコレートした関数の戻り値も、**内部的にはどちらも同一の `Pregel` インスタンス**です。本記事では、この Pregel BSP 実行モデルの仕組みから、Functional API の設計ルール、そして実務で最も重要な「Workflows vs Agents の設計指針」までを体系的に解説します。

---

## 1. Pregel BSP（Bulk Synchronous Parallel）実行モデル

LangGraph のランタイムである [`Pregel`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py) は、**Actors（`PregelNode`）** と **Channels** を組み合わせてグラフを実行します。実行は **Super-step（スーパーステップ）** と呼ばれる論理ステップの反復で構成され、各 Super-step は厳密に以下の 3 フェーズで進行します。

1. **Plan（計画）**：このステップで実行すべき Actor（ノード）を決定する。初回ステップでは入力チャネルを購読する Actor が選ばれ、2 ステップ目以降は直前のステップで更新されたチャネルを購読する Actor が選ばれる。
2. **Execution（実行）**：選ばれたすべての Actor を並列に実行する。**このフェーズ中、ある Actor が書き込んだチャネル更新は、他の Actor からは一切見えない（スナップショット隔離）**。全 Actor の完了、いずれかの失敗、またはタイムアウトで終了する。
3. **Update（更新）**：Execution フェーズで各 Actor が書き込んだ値を一括して Channels に反映し、必要に応じて Checkpointer へ永続化する。

この 3 フェーズを「実行対象となる Actor がなくなる」か「`recursion_limit`（最大ステップ数）に達する」まで繰り返します。

> **BSP モデルの最大の利点**：Execution フェーズでの更新が Update フェーズまで完全に隔離されるため、並列実行される複数ノード間でレースコンディションが起きず、Super-step 境界ごとに一貫性のあるチェックポイント（状態スナップショット）を安全に保存できます。

{% mermaid %}
flowchart LR
    subgraph SuperStep["1 Super-step (Pregel BSP サイクル)"]
        direction LR
        Plan["1. Plan フェーズ<br/>更新された Channel を購読する<br/>Actor (PregelNode) を抽出"]
        Exec["2. Execution フェーズ<br/>選択された Actor を並列実行<br/>(書き込みは他 Actor から不可視)"]
        Upd["3. Update フェーズ<br/>書き込みを Channel へ一括適用<br/>Checkpointer へ永続化"]
        Plan --> Exec --> Upd
    end

    Upd -- "購読 Actor あり" --> Plan
    Upd -- "購読 Actor なし / 終了条件" --> Done["実行完了 (Output Channels 返却)"]

    subgraph Channels["主要 Channel 型"]
        C1["LastValue<br/>(最新値で上書き)"]
        C2["EphemeralValue<br/>(1ステップ限り保持)"]
        C3["Topic<br/>(PubSub / 蓄積可)"]
        C4["BinaryOperatorAggregate<br/>(Reducer で逐次集約)"]
        C5["DeltaChannel (v1.2+)<br/>(差分のみ保存・読込時再構築)"]
    end

    Upd -. "更新ルールを委譲" .-> Channels

    style SuperStep fill:#eef6ff,stroke:#3b82f6,stroke-width:2px
    style Plan fill:#dbeafe,stroke:#2563eb
    style Exec fill:#fef3c7,stroke:#d97706
    style Upd fill:#dcfce7,stroke:#16a34a
    style Channels fill:#f8fafc,stroke:#64748b,stroke-dasharray: 5 5
{% endmermaid %}

### Pregel を支える 5 つの Channel 型

`StateGraph` で `TypedDict` の各フィールドを定義すると、コンパイル時に自動的に以下の Pregel Channel へ変換されます。

| Channel 型 | 挙動と用途 | `StateGraph` での対応 |
| --- | --- | --- |
| **`LastValue`** | 最後に書き込まれた値を保持（上書き）。デフォルトチャネル。 | Reducer なしの通常フィールド |
| **`EphemeralValue`** | 1 ステップのみ値を保持し、次ステップでクリア。 | 内部の `__start__` やノード間トリガー制御 |
| **`Topic`** | 複数値を PubSub 的に配送。`accumulate=True` で全ステップ分を蓄積。 | 複数出力の集約・サイクル制御 |
| **`BinaryOperatorAggregate`** | 現在値と新規更新に二項演算子（Reducer）を適用して集約。 | `Annotated[list[str], operator.add]` など |
| **`DeltaChannel`**（`langgraph>=1.2` Beta） | 蓄積済みの全体値ではなく**各ステップの差分（write）のみ**をチェックポイント保存し、読み込み時にバルク Reducer で再構築する。 | 長大化する `messages` 等のチェックポイント肥大化対策 |

#### `DeltaChannel` 設計時の 3 つの厳格なルール

`DeltaChannel` は長期間続くスレッドのストレージコストを劇的に下げられますが、書き込み時ではなく**読み込み（再構築）時に Reducer が実行される**ため、以下の制約を守る必要があります。

1. **結合法則（Associativity）を満たすバルク Reducer**：`reducer(reducer(state, [xs]), [ys]) == reducer(state, [xs, ys])` が成立する純粋関数でなければなりません。
2. **Reducer 内で非決定処理やミューテーションを行わない**：`uuid.uuid4()` や `datetime.now()` を Reducer 内で呼ぶと、リプレイのたびに異なる値が生成されます。ID 付与などのメタデータは必ずチャネル書き込み前のノード側で行います。
3. **`snapshot_frequency=K` による読み込み遅延の制御**：スナップショットなしでは `O(N)` ステップのリプレイが必要になるため、`DeltaChannel(my_reducer, snapshot_frequency=5)` のように K ステップごとの完全スナップショットを設定して読み込み遅延に上限を設けます。また、`DeltaChannel` を使ったスレッドは古い LangGraph バージョンへダウングレードできない点にも注意が必要です。

```python
from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict
from langgraph.channels import DeltaChannel


# 結合法則を満たすバルク Reducer（リスト結合の例）
# writes には 1 ステップ内で書き込まれた全要素が Sequence として一括で渡される
def list_bulk_reducer(state: list[Any], writes: Sequence[list[Any]]) -> list[Any]:
    result = list(state)  # 元の state を破壊せずコピーを作成
    for write in writes:
        result.extend(write)
    return result


class LongRunningThreadState(TypedDict):
    # 5 Super-step ごとに完全スナップショットを作成し、読み込み遅延を O(5) に抑える
    messages: Annotated[
        list[str],
        DeltaChannel(list_bulk_reducer, snapshot_frequency=5),
    ]
```

### 低レベル Pregel API と高レベル API の対応関係

通常は `StateGraph` や `@entrypoint` 経由で Pregel を利用しますが、低レベル API（`Pregel` と `NodeBuilder`）を直接書くと、Actors と Channels の関係がはっきりと見えます。

```python
from langgraph.channels import EphemeralValue, BinaryOperatorAggregate
from langgraph.pregel import Pregel, NodeBuilder

# Channel "a" を購読して 2 倍にし、"b" と "c" へ書き込む Actor
node1 = NodeBuilder().subscribe_only("a").do(lambda x: x + x).write_to("b", "c")

# Channel "b" を購読してさらに 2 倍にし、"c" へ書き込む Actor
node2 = NodeBuilder().subscribe_only("b").do(lambda x: x + x).write_to("c")


def join_reducer(current: str, update: str) -> str:
    """BinaryOperatorAggregate 用の二項集約関数。"""
    return f"{current} | {update}" if current else update


# Actors と Channels を直接配線して Pregel アプリを構築
app = Pregel(
    nodes={"node1": node1, "node2": node2},
    channels={
        "a": EphemeralValue(str),
        "b": EphemeralValue(str),
        "c": BinaryOperatorAggregate(str, operator=join_reducer),
    },
    input_channels=["a"],
    output_channels=["c"],
)
# app.invoke({"a": "foo"}) -> {"c": "foofoo | foofoofoofoo"}
```

同じ Pregel インスタンスでも、高レベル API の違いによって生成されるノードとチャネルの構造は大きく異なります。

- **`StateGraph.compile()`**：各ノードが `PregelNode` になり、State の各キーが `LastValue` や `BinaryOperatorAggregate`、さらに遷移制御用の `branch:...` チャネルが大量に自動生成される。
- **`@entrypoint`**：関数全体が **1 つの `PregelNode`** として登録され、チャネルは `__start__`（`EphemeralValue`）、`__end__`（`LastValue`）、`__previous__`（`LastValue`）のわずか 3 つだけになる。

---

## 2. Functional API（`@entrypoint` / `@task`）の理論

**Functional API** は、グラフ構造（ノード・エッジ・共有 State スキーマ）を明示的に定義することなく、**通常の Python 関数と制御フロー（`if` / `else` / `for` / `try`）に LangGraph のランタイム機能（永続化・HITL・ストリーミング・長期記憶）を付与する API** です。

### Graph API と Functional API の使い分け

| 比較観点 | Graph API (`StateGraph`) | Functional API (`@entrypoint` / `@task`) |
| --- | --- | --- |
| **推奨ユースケース** | **新規に構築する LangGraph エージェント／ワークフロー全般** | **既存コードや他フレームワーク（CrewAI / AutoGen / Claude Agent SDK 等）を最小改修でラップして LangSmith へデプロイしたい場合** |
| **制御フロー** | エッジ・条件分岐・`Send` / `Command` | 通常の Python 構文（`if` / `for` / 関数呼び出し） |
| **状態管理** | 共有 State スキーマ + Reducer | 関数ローカル変数 + `previous` パラメータ |
| **チェックポイント粒度** | 各 Super-step 終了ごと | 各 `@task` の完了結果および `@entrypoint` 終了時 |
| **静的グラフ可視化** | 対応（`draw_mermaid` / Studio グラフ表示） | 非対応（動的制御フローのため） |

### 2 つの構成要素：`@entrypoint` と `@task`

1. **`@entrypoint(checkpointer=..., store=...)`**
   - ワークフローの開始点を定義し、デコレートされた関数を `Pregel` インスタンスへ変換します。
   - **単一の位置引数**のみを入力として受け取ります（複数値を渡す場合は `dict` や `TypedDict` でまとめます）。
   - キーワード引数として `previous`（同一 `thread_id` での前回戻り値）、`config`（`RunnableConfig`）、`store`（`BaseStore`）、`writer`（カスタムストリーム用）、`runtime`（`Runtime`）を自動注入できます。戻り値と次回の `previous` を分けたい場合は `entrypoint.final(value=..., save=...)` を使用します。

2. **`@task`**
   - API 呼び出しや LLM 推論、ファイル I/O などの**離散的な作業単位**をラップします。
   - 呼び出すと即座にバックグラウンド実行を開始し、**Future ライクなオブジェクト**を返します。同期なら `.result()`、非同期なら `await` で結果を回収するため、複数タスクを先に起動してからまとめて `.result()` を呼ぶだけで簡単に並列実行できます。
   - タスクの戻り値はチェックポイントに保存されるため、**中断（`interrupt()`）やエラーからの再開時に再計算されずキャッシュから復元されます**。

> **決定的リプレイ（Deterministic Replay）の鉄則**：Functional API で `interrupt()` や障害から再開（Resume）すると、LangGraph は停止した行から直接再開するのではなく、**`@entrypoint` 関数の先頭からコードを再実行（リプレイ）します**。その際、完了済みの `@task` は再実行されずチェックポイントから結果が即座に返されます。したがって、**副作用（外部 API 書き込み・メール送信・ファイル出力）や非決定的な処理（LLM 呼び出し・乱数・現在時刻 `time.time()`）は、必ず `@task` の内部に閉じ込めなければなりません**。

### 再開とチェックポイントの 3 原則

1. **シリアライズ可能性（Serialization）**：`checkpointer` を有効にする場合、`@entrypoint` の入出力と `@task` の出力はすべて **JSON シリアライズ可能**（`dict`, `list`, `str`, `int`, `float`, `bool`, `None`）である必要があります。
2. **決定性（Determinism）**：`@entrypoint` 直下で `time.time()` や `random.random()` を使って `if` 分岐すると、リプレイ時に初回とは別の分岐へ入り、`interrupt()` のインデックス対応が壊れます。必ず `@task` 化して値を固定してください。
3. **冪等性（Idempotency）**：タスクの途中でプロセスがクラッシュした場合、未完了タスクは再開時に再実行されます。外部システムへの書き込みタスクは冪等性キー（Idempotency Key）を使うか、既存状態を確認してから書き込む設計にします。

```python
import time
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt

checkpointer = InMemorySaver()


# 良い設計：現在時刻の取得や外部書き込みを @task に閉じ込める
@task
def get_current_timestamp() -> float:
    """非決定的な壁時計読み取りをタスク化し、リプレイ時に同一値を復元する。"""
    return time.time()


@task
def write_audit_log(message: str) -> None:
    """副作用（ファイル書き込み）をタスク化し、再開時の二重実行を防止する。"""
    with open("audit.log", "a", encoding="utf-8") as f:
        f.write(message + "\n")


@entrypoint(checkpointer=checkpointer)
def safe_workflow(inputs: dict) -> dict:
    # リプレイ時も初回とまったく同じ timestamp が返り、分岐がズレない
    t1 = get_current_timestamp().result()
    branch = "slow" if (t1 - inputs["t0"]) > 1.0 else "fast"

    # 再開時にチェックポイント復元されるため、ログが二重に書かれない
    write_audit_log(f"branch={branch}").result()

    # 人間の承認待ちで一時停止
    approved = interrupt({"branch": branch, "question": "続行しますか？"})
    return {"branch": branch, "approved": approved}
```

---

## 3. Workflows vs Agents 設計指針

LangGraph でシステムを設計する際、最も重要なアーキテクチャ判断は**「どこまでを確定的な Workflow として組み、どこからを自律的な Agent に任せるか」**です。

- **Workflows（ワークフロー）**：事前に定義されたコードパスと順序に従って LLM やツールを呼び出すシステム。予測可能性・検証容易性・コスト制御に優れる。
- **Agents（エージェント）**：LLM 自身が状況を観察し、どのツールを何回呼び出すか・いつ終了するかを動的に決定するフィードバックループ型システム。未知の問題や開放的な探索に強い。

{% mermaid %}
flowchart LR
    W1["1. Prompt Chaining<br/>直列パイプライン + ゲート"]
    W2["2. Routing<br/>入力分類 → 専門ルート分岐"]
    W3["3. Parallelization<br/>並列サブタスク / 多数決"]
    W4["4. Orchestrator-Worker<br/>動的計画 → Send() 並列実行"]
    W5["5. Evaluator-Optimizer<br/>生成 ↔ 評価フィードバックループ"]
    A1["6. Autonomous Agent<br/>LLM ↔ ToolNode 自律ループ"]

    W1 --> W2 --> W3 --> W4 --> W5 --> A1

    style W1 fill:#ecfdf5,stroke:#10b981
    style W2 fill:#ecfdf5,stroke:#10b981
    style W3 fill:#eff6ff,stroke:#3b82f6
    style W4 fill:#eff6ff,stroke:#3b82f6
    style W5 fill:#fef3c7,stroke:#d97706
    style A1 fill:#fef2f2,stroke:#ef4444,stroke-width:2px
{% endmermaid %}

### 6 つの標準パターンと LangGraph 実装プリミティブ

| パターン | 分類 | 特徴と適用場面 | LangGraph での主要プリミティブ |
| --- | --- | --- | --- |
| **1. Prompt Chaining** | Workflow | 固定ステップで順次処理し、必要に応じて品質ゲートで分岐（翻訳・校正など） | `add_edge` + `add_conditional_edges` |
| **2. Routing** | Workflow | 構造化出力（`with_structured_output`）で入力を分類し専用ノードへ振り分け | `with_structured_output` + 条件エッジ |
| **3. Parallelization** | Workflow | 事前定義された複数の独立サブタスクを同時実行して統合（多観点スコアリング等） | 同一ノードからの複数 `add_edge`（Fan-out / Fan-in） |
| **4. Orchestrator-Worker** | Workflow | サブタスク数が実行時まで決まらない場合に計画ノードが動的にワーカーを生成 | `Send` API + `Annotated[list, operator.add]` |
| **5. Evaluator-Optimizer** | Workflow | 生成ノードと評価ノードを合格基準に達するまで反復させる | 循環エッジ + 構造化評価スキーマ |
| **6. Autonomous Agent** | Agent | LLM が `tool_calls` を出し続ける限り `ToolNode` を実行し続ける自律ループ | `MessagesState` + `bind_tools` + `ToolNode` + `ToolRuntime` |

### 実務における選定ガイドライン

> **設計の鉄則**：最初からフル自律型の Agent を作るべきではありません。まずは **Prompt Chaining や Routing などの単純な Workflow** で要件を満たせるか検討し、サブタスクが動的なら **Orchestrator-Worker**、品質の反復改善が必要なら **Evaluator-Optimizer** へと段階的に自律性を上げます。Agent ループは「解法ステップが事前に予測できない領域」に限定し、さらに `ToolRuntime` を使ってグラフ State や実行コンテキスト（`organization_id` 等）を LLM に改ざんされない形で安全にツールへ注入するのが堅牢な設計です。

---

## まとめ

本記事では、LangGraph の最下層から設計思想までを貫く 3 つの柱を解説しました。

- **Pregel BSP モデル**：Plan → Execution → Update の 3 フェーズと Channel 抽象（`LastValue`, `BinaryOperatorAggregate`, `DeltaChannel` 等）により、並列実行の安全性と Super-step 境界での一貫した永続化を実現している。
- **Functional API（`@entrypoint` / `@task`）**：既存の Python 制御フローを保ったまま Pregel の耐久実行・チェックポイント・並列化を利用できるが、「JSON シリアライズ」「決定性」「冪等性」の 3 原則を厳守する必要がある。
- **Workflows vs Agents**：自律性を上げるほどレイテンシ・コスト・予測困難性が増すため、問題構造に応じて 6 パターンの中から最小限の複雑さを選ぶのがベストプラクティス。

次回の **Part 7 実践編** では、Functional API（`@entrypoint` / `@task`）と `InMemorySaver` を使ってチェックポイント対応のコピーライティングワークフローを構築し、実測出力とともにその内部挙動を検証します。
