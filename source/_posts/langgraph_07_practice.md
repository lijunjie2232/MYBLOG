---
title: "LangGraph Part 7 実践編：@entrypoint と @task で構築するチェックポイント対応コピーライティングワークフロー"
date: 2026-06-25 04:02:27
categories: [AI]
tags: [LangGraph, Functional API, entrypoint, task, Checkpointer, Python, AI]
lang: ja
description: "LangGraph の Functional API（@entrypoint と @task）と InMemorySaver を使い、通常の Python 制御フローのままチェックポイント永続化に対応したコピーライティングワークフローを実装・検証します。"
---

## はじめに：Graph API を使わずに LangGraph の耐久実行を手に入れる

前回の **Part 7 理論編** では、LangGraph の実行基盤である **Pregel BSP モデル**、通常の Python 構文のまま LangGraph のランタイム機能を使える **Functional API（`@entrypoint` / `@task`）**、そして **Workflows vs Agents の設計指針** を体系的に整理しました。理論編では特に、Functional API の核となるリプレイ規則について次のように述べました。

> **決定的リプレイ（Deterministic Replay）の鉄則**：Functional API で `interrupt()` や障害から再開（Resume）すると、LangGraph は停止した行から直接再開するのではなく、**`@entrypoint` 関数の先頭からコードを再実行（リプレイ）します**。その際、完了済みの `@task` は再実行されずチェックポイントから結果が即座に返されます。したがって、**副作用（外部 API 書き込み・メール送信・ファイル出力）や非決定的な処理（LLM 呼び出し・乱数・現在時刻 `time.time()`）は、必ず `@task` の内部に閉じ込めなければなりません**。

本実践編では、`StateGraph` も `TypedDict` の State 定義も一切使わずに、**`@entrypoint` + `@task` + `InMemorySaver`** だけでチェックポイント対応のコピーライティングワークフロー（下書き生成 → レビュー整形）を構築し、実測出力と内部挙動を詳しく検証します。

---

## 1. ワークフロー設計と Pregel 内部マッピング

今回のハンズオンは、理論編で解説した 6 パターンのうち最も基本かつ実用的な **Prompt Chaining（直列ワークフロー）** を Functional API で実装した構成です。

{% mermaid %}
flowchart LR
    Input(["入力文字列<br/>'LangGraph Functional API'<br/>config: {thread_id: 'func-07'}"])

    subgraph Entrypoint["@entrypoint(checkpointer=InMemorySaver())<br/>copywriting_workflow(topic: str)"]
        direction LR
        T1["@task: generate_draft(topic)<br/>LLM でキャッチコピー生成<br/>→ .result() で確定・保存"]
        T2["@task: review_draft(draft)<br/>レビュー済みプレフィックス付与<br/>→ .result() で確定・保存"]
        T1 --> T2
    end

    CP[("InMemorySaver (Checkpointer)<br/>thread_id='func-07'<br/>各 @task の戻り値と<br/>__end__ / __previous__ を永続化")]
    LLM["ChatOpenAI<br/>(MiniCPM-V-4.7-35B-A3B)"]
    Out(["戻り値 dict<br/>{topic, draft, reviewed}"])

    Input --> T1
    T1 <--> LLM
    T1 -. "完了結果をチェックポイント保存" .-> CP
    T2 -. "完了結果をチェックポイント保存" .-> CP
    T2 --> Out

    style Entrypoint fill:#eef6ff,stroke:#3b82f6,stroke-width:2px
    style T1 fill:#dbeafe,stroke:#2563eb
    style T2 fill:#dcfce7,stroke:#16a34a
    style CP fill:#fef3c7,stroke:#d97706,stroke-width:2px
{% endmermaid %}

### なぜ 2 つの処理を別々の `@task` に分けるのか？

1. **LLM 呼び出し（非決定・高コスト）を独立したキャッシュ境界にする**：`generate_draft` を `@task` で包むことで、仮に後段の `review_draft` でエラーが起きたり、間に `interrupt()` による人間承認を挟んだりしても、再開時に LLM を再実行せずチェックポイントから初回の `draft` が復元されます。
2. **純粋な Python 関数としてテスト・再利用しやすい**：`@entrypoint` 自体は通常の関数呼び出しのように読めるため、既存のスクリプトや別フレームワークのパイプラインから移行する際の認知負荷を最小限に抑えられます。

---

## 2. 完全実装コード

環境変数 `BASE_URL` と `API_KEY` を読み込み、`@entrypoint` と `@task` を組み合わせてコピーライティングワークフローを実行する完全なコードです。

```python
import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

# 1. 環境変数から LLM 接続情報を読み込み（API キーはハードコードしない）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント向けの ChatOpenAI クライアントを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=256,
        timeout=30,
        max_retries=1,
    )


def run_functional_copywriting_demo() -> dict:
    llm = make_llm()
    # 2. スレッド単位のチェックポイントを保持する InMemorySaver を準備
    checkpointer = InMemorySaver()

    # 3. タスク 1：LLM を呼び出してキャッチコピー案を生成する
    # 非決定的な LLM 呼び出しを @task に閉じ込めることで、再開時に再実行を防ぐ
    @task
    def generate_draft(topic: str) -> str:
        prompt = f"「{topic}」の短いキャッチコピーを日本語で1つだけ作成してください。"
        return llm.invoke(prompt).content.strip()

    # 4. タスク 2：生成された下書きをレビュー済み形式に整形する
    # タスクの戻り値は JSON シリアライズ可能な str に限定する
    @task
    def review_draft(draft: str) -> str:
        return f"[REVIEWED] {draft}"

    # 5. エントリポイントの定義（この関数自体が Pregel インスタンスへ変換される）
    # 単一の位置引数 topic を受け取り、通常の Python 制御フローでタスクを連結する
    @entrypoint(checkpointer=checkpointer)
    def copywriting_workflow(topic: str) -> dict:
        # @task を呼ぶと即座に Future が返るため、.result() で完了を待機して値を取得
        draft = generate_draft(topic).result()
        reviewed = review_draft(draft).result()
        return {
            "topic": topic,
            "draft": draft,
            "reviewed": reviewed,
        }

    # 6. thread_id を指定してワークフローを実行
    config = {"configurable": {"thread_id": "func-07"}}
    out = copywriting_workflow.invoke("LangGraph Functional API", config=config)
    return out


if __name__ == "__main__":
    result = run_functional_copywriting_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 3. 実測実行結果と内部メカニズムの徹底解説

`MiniCPM-V-4.7-35B-A3B` を用いて実行した際の実測出力（`elapsed_s: 0.57s`）は以下の通りです。

```json
{
  "topic": "LangGraph Functional API",
  "draft": "「シンプルで柔軟なエージェント構築」",
  "reviewed": "[REVIEWED] 「シンプルで柔軟なエージェント構築」"
}
```

### 内部で何が起きているのか？（Pregel 実行とリプレイの仕組み）

`@entrypoint(checkpointer=checkpointer)` を付与した瞬間、`copywriting_workflow` は通常の Python 関数ではなく **`langgraph.pregel.Pregel` インスタンス**になります（ソースコード：[langgraph/func/__init__.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/func/__init__.py)）。

実際に `copywriting_workflow.nodes` と `copywriting_workflow.channels` を確認すると、次のように極めてシンプルな Pregel 構造になっています。

- **`nodes`**：`{'copywriting_workflow': <PregelNode>}` の 1 ノードのみ。
- **`channels`**：
  - `__start__`（`EphemeralValue`）：入力値 `"LangGraph Functional API"` を受け取る。
  - `__end__`（`LastValue`）：最終戻り値 `{"topic": ..., "draft": ..., "reviewed": ...}` を格納する。
  - `__previous__`（`LastValue`）：同一 `thread_id` での次回呼び出し時に `previous` 引数へ渡される値を保持する。

実行時（および中断・障害後のリプレイ時）のやり取りをシーケンス図で追ってみましょう。

{% mermaid %}
sequenceDiagram
    autonumber
    participant Caller as 呼び出し元
    participant Pregel as copywriting_workflow<br/>(Pregel)
    participant T1 as @task generate_draft
    participant T2 as @task review_draft
    participant CP as InMemorySaver<br/>(thread_id='func-07')

    Caller->>Pregel: invoke("LangGraph Functional API", config)
    Pregel->>CP: thread_id='func-07' の既存チェックポイントを確認
    CP-->>Pregel: 新規実行（キャッシュなし）

    Pregel->>T1: generate_draft("LangGraph Functional API") 起動 → Future 返却
    T1-->>Pregel: .result() 完了: "「シンプルで柔軟なエージェント構築」"
    Pregel->>CP: generate_draft の結果をチェックポイントへ保存

    Pregel->>T2: review_draft("「シンプルで柔軟なエージェント構築」") 起動 → Future 返却
    T2-->>Pregel: .result() 完了: "[REVIEWED] 「シンプルで柔軟なエージェント構築」"
    Pregel->>CP: review_draft の結果をチェックポイントへ保存

    Pregel->>CP: 最終結果 (__end__, __previous__) を保存
    Pregel-->>Caller: {"topic": "...", "draft": "...", "reviewed": "..."}

    Note over Pregel,CP: 仮に T1 と T2 の間に interrupt() があった場合：<br/>再開時は先頭からリプレイされるが、T1 は実行されず<br/>CP からキャッシュ済みの draft が即座に返される
{% endmermaid %}

このように、`@task` は単なるデコレータではなく、**「Pregel ランタイムに対して非同期／スレッドプール実行のタスクをスケジュールし、その戻り値を Checkpointer に記録・キャッシュ照合する境界」**として機能しています。

---

## 4. 実践拡張：並列ファンアウトと `interrupt()` による人間承認

Functional API の強みがさらに際立つのが、**「複数タスクの並列実行」**と**「人間承認（HITL）の差し込み」**です。Graph API のように `Send` や条件エッジを書き直す必要はなく、通常の Python コードとして数行追記するだけで実現できます。

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt


@task
def generate_variant(topic: str, tone: str) -> str:
    """トーン別にコピー案を生成する並列用タスク（簡略デモ）。"""
    return f"[{tone}] {topic} で開発を加速"


@entrypoint(checkpointer=InMemorySaver())
def parallel_copywriting_workflow(topic: str) -> dict:
    # 1. 複数タスクを先にすべて起動（Future のリストを作成 → ここで並列実行が始まる）
    tones = ["技術者向け", "経営層向け", "SNS向け"]
    futures = [generate_variant(topic, tone) for tone in tones]

    # 2. すべての並列タスクの結果を回収
    candidates = [f.result() for f in futures]

    # 3. 人間による採用案の選択を待機（ここで一時停止）
    # 再開時は先頭からリプレイされるが、3つの generate_variant はキャッシュ復元され再実行されない
    selected_index = interrupt(
        {
            "question": "採用するキャッチコピーの番号（0〜2）を選んでください",
            "candidates": candidates,
        }
    )

    return {
        "topic": topic,
        "selected": candidates[selected_index],
    }


cfg = {"configurable": {"thread_id": "func-parallel-01"}}
# 初回実行：並列生成後に interrupt で一時停止
paused = parallel_copywriting_workflow.invoke("LangGraph", config=cfg)
# 再開実行：0番の案を選択して完了（generate_variant は再計算されない）
final = parallel_copywriting_workflow.invoke(Command(resume=0), config=cfg)
```

---

## 5. 注意点・ハマりどころ（Pay Attention）

Functional API は書き始めの手軽さが魅力ですが、Graph API とは異なる落とし穴がいくつかあります。実務導入前に必ず以下の 5 点を確認してください。

1. **`@entrypoint` の入力引数は「単一の位置引数」に限定される**
   - `def workflow(topic: str, tone: str):` のように複数の位置引数を定義して `workflow.invoke(...)` するとエラーになります。複数の入力パラメータを渡したい場合は、必ず `dict` や `TypedDict`（例：`def workflow(payload: dict):`）にまとめて単一引数として受け取ってください。
2. **`@task` をワークフロー外部から直接呼び出してはいけない**
   - `@task` でデコレートされた関数は、`@entrypoint`・別の `@task`・または Graph API のノード内部（＝Pregel ランタイムのコンテキスト内）からのみ呼び出せます。通常の Python スクリプトやユニットテストから直接 `generate_draft("test")` を呼ぶと、実行コンテキストが存在せずエラーになります。ロジック単体をテストしたい場合は、プレーンな内部関数を分離して `@task` でラップするのが定石です。
3. **チェックポイント有効時の JSON シリアライズ制約**
   - `checkpointer` を指定した場合、`@entrypoint` の入出力およびすべての `@task` の戻り値は JSON シリアライズ可能でなければなりません。例えば `@task` から `llm.invoke(...)` の戻り値である `AIMessage` ではなく、接続オブジェクトやカスタムクラスをそのまま返すとチェックポイント保存時にシリアライズエラーが発生します。今回のように `.content.strip()` でプリミティブな `str` や `dict` へ変換して返すのが最も安全です。
4. **`@entrypoint` 直下に副作用や非決定処理を書くとリプレイで事故る**
   - 理論編でも強調した通り、`interrupt()` や障害復旧からの再開時には `@entrypoint` の先頭から再実行されます。`@entrypoint` 直下に `open("log.txt", "a")` やメール送信 API、`time.time()` による `if` 分岐を書いてしまうと、再開のたびに二重実行されたり、分岐先が変わって `interrupt()` のインデックス対応が壊れたりします。外部 I/O・乱数・現在時刻・LLM 呼び出しは例外なく `@task` の中に閉じ込めてください。
5. **`thread_id` の指定忘れと、静的グラフ可視化非対応のトレードオフ**
   - `checkpointer` 付き `@entrypoint` を実行する際は `config={"configurable": {"thread_id": "..."}}` が必須です。また、Functional API は制御フローが実行時まで確定しないため、Graph API のような `get_graph().draw_mermaid()` による構造可視化はできません。複雑なマルチエージェント遷移や可視化・Studio デバッグを重視する新規プロジェクトでは **Graph API（`StateGraph`）** を選び、既存コードのラップや簡潔なパイプラインには **Functional API** を選ぶのが公式推奨の住み分けです。

---

## まとめ（LangGraph Part 1〜7 シリーズ総括）

本記事では、`@entrypoint`・`@task`・`InMemorySaver` を用いてチェックポイント対応のコピーライティングワークフローを実装し、実測 `0.57s` での動作と、Pregel 内部における Future 解決・チェックポイントキャッシュ復元の仕組みを確認しました。

これで **LangGraph 全 7 パート（理論編＋実践編の計 14 記事）** が完結しました。

- **Part 1〜2**：`StateGraph` の基礎（State / Reducer / Node / Edge）と動的ルーティング（`Send` / `Command`）
- **Part 3〜4**：Checkpointer（`SqliteSaver`）による永続化、Streaming、および `RetryPolicy` / `CachePolicy` による耐障害性
- **Part 5**：`interrupt()` による Human-in-the-Loop と Time Travel（Fork）
- **Part 6**：Subgraphs（共有 State vs 独立 State）と Cross-Thread Store（長期記憶・Namespace 設計）
- **Part 7**：Pregel BSP 実行モデル、Functional API（`@entrypoint` / `@task`）、および Workflows vs Agents の設計指針

これらのプリミティブを適切に組み合わせることで、プロトタイプから本番運用まで耐えうる状態管理・耐障害性・観測性を備えた AI エージェント基盤を自在に設計できるようになります。ぜひご自身のプロジェクト要件に最適なパターン（Graph API / Functional API、Workflow / Agent）を選び、実践に活かしてみてください。
