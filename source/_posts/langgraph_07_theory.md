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

