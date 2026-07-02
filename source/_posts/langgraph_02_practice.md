---
title: "LangGraph Part 2 実践編：Send による Map-Reduce 並列実行と Command による状態更新＋動的ルーティング"
date: 2026-06-05 20:12:19
categories: [AI]
tags: [LangGraph, Send, Command, MapReduce, MiniCPM, 並列処理, 実践]
lang: ja
description: "LangGraph Part 2 理論編に基づき、MiniCPM-V-4.7-35B-A3B を使って Send による動的 Map-Reduce 並列推論と Command による状態更新＋動的ルーティングを実装・実測する。"
---

# LangGraph Part 2 実践編：`Send` による Map-Reduce 並列実行と `Command` による状態更新＋動的ルーティング

本記事では、[LangGraph Part 2 理論編](/2026/05/30/langgraph_02_theory/) で解説した **`Send`（動的 Fan-out / Map-Reduce）** と **`Command`（状態更新＋動的ルーティング）** を、`MiniCPM-V-4.7-35B-A3B` を用いた実践コードで検証します。

入力リスト `["LangChain", "LangGraph", "DeepAgents"]` に対して、実行時の要素数に応じてワーカーノードを動的並列起動し、それぞれの特徴を 15 文字以内の日本語フレーズで並列要約（Map）した上で、`reducer` ノードが結果を統合しつつ `Command(update=..., goto=END)` でグラフを終了（Reduce）させます。

---

## 1. 理論編の振り返りと実践グラフの設計

Part 2 理論編では、`Send` と `Command` の本質を次のように定義しました。

> **`Send` の本質**：`Send(node_name, arg)` は、条件付きエッジ関数から複数返却することで、**同一のワーカーノードを実行時データ件数だけ動的に並列インスタンス化し、それぞれに独立した部分入力（専用スキーマ）を配布する Map-Reduce プリミティブ**です。

> **`Command` の設計思想**：`Command(update=..., goto=...)` をノードから返却することで、**1 つの関数内で「State の部分更新（`update`）」と「次の遷移先指定（`goto`）」をアトミックに統合**できます。

この 2 つを組み合わせた今回のグラフ構造は以下の通りです。

{% mermaid %}
flowchart LR
    START(["START<br/>items: ['LangChain',<br/>'LangGraph', 'DeepAgents']"])
    START -.->|"Send('worker', {item: 'LangChain'})"| W1["worker #1<br/>WorkerState"]
    START -.->|"Send('worker', {item: 'LangGraph'})"| W2["worker #2<br/>WorkerState"]
    START -.->|"Send('worker', {item: 'DeepAgents'})"| W3["worker #3<br/>WorkerState"]

    W1 -->|"summaries += [...]"| RED["reducer ノード<br/>MapReduceState を集約"]
    W2 -->|"summaries += [...]"| RED
    W3 -->|"summaries += [...]"| RED

    RED -.->|"Command(update={final_report}, goto=END)"| END(["END"])

    style START fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style W1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style W2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style W3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style RED fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
    style END fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
{% endmermaid %}

設計上の重要ポイントは次の 3 つです。

1. **親スキーマ（`MapReduceState`）とワーカー専用スキーマ（`WorkerState`）の分離**：`worker` ノードは全体リストではなく、自身が担当する単一要素 `{"item": ...}` だけを受け取ります。
2. **集約チャネル `summaries` への `operator.add` Reducer 付与**：3 つの並列ワーカーが同時に `summaries` へ書き込むため、Reducer が必須です。
3. **`reducer` ノードには静的エッジ `add_edge("reducer", END)` を張らない**：遷移先は `reducer_node` が返す `Command(goto=END)` に一本化します。

---

## 2. 完全な実装コード（`Send` + `Command`）

API 認証情報は環境変数（`BASE_URL`, `API_KEY`）から読み込みます。各行の役割を詳細コメントで示します。

```python
import json
import operator
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command

# .env から環境変数を読み込む（API キーのハードコードは厳禁）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"

def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """MiniCPM-V-4.7-35B-A3B クライアントを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,        # 推論モデル名
        base_url=BASE_URL,       # 環境変数から取得したエンドポイント URL
        api_key=API_KEY,         # 環境変数から取得した API キー
        temperature=temperature, # 決定的なフレーズ生成のため 0.0
        max_tokens=256,          # 出力トークン上限
        timeout=30,              # API タイムアウト（秒）
        max_retries=1,           # 一時的エラー時のクライアント側リトライ
    )

def run_lab2() -> dict:
    llm = make_llm(temperature=0.0)

    # 1. グラフ全体の State スキーマ（Map-Reduce の入力・中間集約・最終出力を保持）
    class MapReduceState(TypedDict):
        items: list[str]                                # Map 対象のアイテム一覧
        summaries: Annotated[list[str], operator.add]   # 並列ワーカーの出力を連結する Reducer 付きチャネル
        final_report: str                               # Reduce 後の最終統合レポート

    # 2. ワーカー専用の入力スキーマ（全体 State には存在しない単数キー item を定義）
    class WorkerState(TypedDict):
        item: str  # Send から個別に渡される担当アイテム 1 件

    # 3. 条件付きエッジ関数：items の件数だけ Send オブジェクトを生成して動的 Fan-out
    def fan_out(state: MapReduceState) -> list[Send]:
        return [Send("worker", {"item": it}) for it in state["items"]]

    # 4. Map ワーカーノード：1 件のアイテムだけを要約し、全体 State の summaries チャネルへ書き戻す
    def worker(state: WorkerState) -> dict:
        prompt = f"{state['item']}の特徴を15文字以内の日本語で1フレーズだけ答えてください。"
        resp = llm.invoke(prompt)
        # 全体 State の summaries キー（operator.add）へ 1 要素リストとして部分更新を返す
        return {"summaries": [f"{state['item']}: {resp.content.strip()}"]}

    # 5. Reduce ノード：集約結果を整形し、Command で状態更新と END へのルーティングを同時実行
    #    戻り値型 Command[Literal[END]] を明示することでグラフ可視化・検証に対応する
    def reducer_node(state: MapReduceState) -> Command[Literal[END]]:
        joined = " / ".join(state["summaries"])
        return Command(
            update={"final_report": f"統合結果 => {joined}"},  # State 更新
            goto=END,                                           # 動的ルーティングで終了
        )

    # 6. グラフの構築とコンパイル
    graph = (
        StateGraph(MapReduceState)
        .add_node("worker", worker)          # Map ワーカーノードを登録
        .add_node("reducer", reducer_node)   # Reduce ノードを登録
        # START から条件付きエッジで fan_out を実行（第 3 引数 ["worker"] は遷移先ヒント）
        .add_conditional_edges(START, fan_out, ["worker"])
        # すべての worker が完了したバリア後に reducer へ合流する
        .add_edge("worker", "reducer")
        # 注意：reducer は Command(goto=END) を返すため add_edge("reducer", END) は書かない
        .compile()
    )

    # 7. 3 件のアイテムを投入して Map-Reduce グラフを実行
    return graph.invoke({
        "items": ["LangChain", "LangGraph", "DeepAgents"],
        "summaries": [],
        "final_report": "",
    })

if __name__ == "__main__":
    output = run_lab2()
    print(json.dumps(output, ensure_ascii=False, indent=2))
```

---

## 3. 実測実行結果と Super-step 挙動の解析

`MiniCPM-V-4.7-35B-A3B` で実行した実際の出力結果（実行時間 `4.67 秒`）は以下の通りです。

```json
{
  "items": [
    "LangChain",
    "LangGraph",
    "DeepAgents"
  ],
  "summaries": [
    "LangChain: LLMアプリ開発フレームワーク",
    "LangGraph: 循環型AIエージェントフレームワーク",
    "DeepAgents: 自律的なエージェント連携"
  ],
  "final_report": "統合結果 => LangChain: LLMアプリ開発フレームワーク / LangGraph: 循環型AIエージェントフレームワーク / DeepAgents: 自律的なエージェント連携"
}
```

### なぜ 3 つのワーカーが競合せず、1 回で綺麗に統合されたのか？

実行時の内部タイムラインをシーケンス図で追ってみましょう。

{% mermaid %}
sequenceDiagram
    participant Caller as invoke()
    participant RT as Pregel ランタイム
    participant W1 as worker("LangChain")
    participant W2 as worker("LangGraph")
    participant W3 as worker("DeepAgents")
    participant RED as reducer_node

    Caller->>RT: items=["LangChain", "LangGraph", "DeepAgents"]
    Note over RT: START の条件付きエッジ fan_out() を評価<br/>3 つの Send タスクを同一 Super-step にスケジュール
    par Super-step 1（並列 Map 実行）
        RT->>W1: WorkerState(item="LangChain")
        RT->>W2: WorkerState(item="LangGraph")
        RT->>W3: WorkerState(item="DeepAgents")
        W1-->>RT: {"summaries": ["LangChain: LLMアプリ開発フレームワーク"]}
        W2-->>RT: {"summaries": ["LangGraph: 循環型AIエージェントフレームワーク"]}
        W3-->>RT: {"summaries": ["DeepAgents: 自律的なエージェント連携"]}
    end
    Note over RT: バリア同期：operator.add Reducer で 3 件のリストを統合
    RT->>RED: MapReduceState(summaries=[3件すべて])
    RED-->>RT: Command(update={"final_report": "統合結果 => ..."}, goto=END)
    Note over RT: Super-step 2：final_report を更新し END へ遷移
    RT-->>Caller: 最終 MapReduceState を返却
{% endmermaid %}

ここで注目すべき内部メカニズムは次の 3 点です。

1. **同一 Super-step での並列ディスパッチ**：`fan_out` が返した 3 つの `Send` オブジェクトは、すべて Super-step 1 にまとめてスケジュールされます。各ワーカーは全体 State の `items` ではなく、自身が受け取った `WorkerState` だけを参照するため、責務が完全に疎結合になります。
2. **バリア同期による自動 Fan-in**：3 つのワーカーがすべて完了するまで `reducer` は起動しません。完了時点で `operator.add` Reducer が 3 つの `["..."]` を 1 つのリストへ安全に結合するため、`reducer` は常に全ワーカーの結果が揃った状態の `state["summaries"]` を受け取れます。
3. **`Command` によるエッジ省略と明確な終端**：`reducer_node` は `Command(update={"final_report": ...}, goto=END)` を返しているため、`StateGraph` 定義に `add_edge("reducer", END)` を記述していなくても、ランタイムは `final_report` を更新した直後にグラフを正常終了させます。

---

## 4. 注意点・ハマりどころ（Pay Attention）

`Send` と `Command` は強力な反面、誤用するとデバッグしにくい挙動を引き起こします。以下の 5 点を必ず押さえてください。

1. **並列ワーカーが書き込むキーに Reducer を付け忘れると `InvalidUpdateError` になる**
   もし `summaries: list[str]` のように `Annotated[..., operator.add]` を付けずに 3 つの `Send` ワーカーから同時に `summaries` を返すと、同一 Super-step 内で Last-Write-Wins チャネルへ複数回の書き込みが発生し、LangGraph は競合を検知して `InvalidUpdateError: Can receive only one value per step` を送出します。並列 Fan-out 先が同じキーへ書き込む場合は、**必ず Reducer を定義**してください。
2. **`Command` を返すノードに `add_edge` を併用してはいけない**
   Part 2 理論編でも強調した通り、`Command(goto=...)` は動的エッジを**追加**します。たとえば `reducer` が `Command(goto="next_a")` を返しつつ、ビルダー側にも `.add_edge("reducer", "next_b")` が残っていると、意図せず `next_a` と `next_b` の両方が実行されます。
3. **並列 Super-step からの集約順序は保証されない**
   今回の実測では入力順に並びましたが、LLM の応答時間や非同期スケジューリング次第で、`summaries` に格納される順序は実行ごとに前後し得ます。出力順序を厳密に決定的にしたい場合は、ワーカーに `(index, item)` を渡して `{"summaries": [(idx, text)]}` のようにインデックス付きで集約し、`reducer` 側で `sorted(...)` してください。
4. **`Command[Literal[...]]` の型アノテーションを省略するとグラフ可視化からエッジが消える**
   `def reducer_node(state) -> Command:` のように遷移先リテラルを省略しても実行自体は可能ですが、`graph.get_graph().draw_mermaid()` 等で描画した際に遷移エッジが表示されなくなります。チーム開発や LangSmith Studio での可視化のため、常に `Command[Literal["node_x", END]]` を明記しましょう。
5. **`add_conditional_edges(START, fan_out, ["worker"])` の第 3 引数を忘れない**
   `Send` を返す条件付きエッジでは、第 3 引数に遷移候補ノード名リスト（`["worker"]`）を渡しておくことで、グラフ可視化時に `START -.-> worker` の破線エッジが正しく描画されます。

---

## 5. まとめ

本記事では、`Send` による実行時件数に応じた動的 Map-Reduce 並列実行と、`Command` による State 更新・ルーティングの統合を `MiniCPM-V-4.7-35B-A3B` で実装・検証しました。

次回の **Part 3 理論編** では、単発のグラフ実行を超えて「会話記憶・障害復旧・Human-in-the-Loop・Time Travel」を支える LangGraph 最重要基盤――**Persistence（永続化）と Checkpointer** の内部アーキテクチャを詳しく解説します。
