---
title: "LangGraph Part 6 実践編：独立スキーマ Subgraph と InMemoryStore を組み合わせたポリシー参照型マルチ階層グラフ"
date: 2026-06-23 17:24:15
categories: [AI]
tags: [LangGraph, Subgraphs, InMemoryStore, Long-Term Memory, Python, AI]
lang: ja
description: "LangGraph の独立スキーマ Subgraph と InMemoryStore を組み合わせ、親グラフからサブグラフへ自動伝播された長期記憶（社内ポリシー）を参照して回答を生成するマルチ階層グラフの実装と実行結果を詳解します。"
---

## はじめに：独立スキーマ Subgraph と Cross-Thread Store の連携を検証する

前回の **Part 6 理論編** では、グラフのモジュール化を担う **Subgraphs** と、スレッドを横断して長期記憶を共有する **Cross-Thread Store（`BaseStore`）** の設計理論を学びました。理論編では特に、次の 2 つの重要ポイントを確認しました。

> **設計原則（Subgraph の境界分離）**：マルチエージェント構成で各サブエージェントの内部思考履歴やツール実行ログを親グラフの `messages` に混ぜたくない場合は、**独立 State スキーマ（ラッパーノード内からの `invoke`）** を選択します。これにより、親グラフには最終回答や構造化サマリだけを返し、コンテキストウィンドウの肥大化とスキーマ結合を防げます。

> **Store の自動伝播メカニズム**：親グラフを `parent.compile(store=store)` でコンパイルすると、親グラフ内のノードだけでなく、**親ノードから呼び出される Subgraph 内のノードへも同一の `BaseStore` インスタンスが自動的に注入されます**。ノード関数のシグネチャに `*, store: BaseStore` または `runtime: Runtime[Context]` を宣言するだけで、サブグラフ側のコンパイル時に `store` を再指定することなく長期記憶へアクセスできます。

本実践編では、この 2 つの原則を 1 つのハンズオンに統合し、**「親グラフ（ユーザー窓口）」→「独立スキーマ Subgraph（インフラ専門ワーカー）」→「`InMemoryStore`（社内デプロイポリシー）」** というポリシー参照型マルチ階層グラフを構築・実行して動作を検証します。

---

## 1. システム設計：State 分離と Namespace 設計

今回のラボで構築するアーキテクチャの全体像は以下の通りです。親グラフとサブグラフで State のキーを完全に分離しつつ、親グラフのコンパイル時に渡した `InMemoryStore` をサブグラフのワーカーノードへ自動伝播させます。

{% mermaid %}
flowchart LR
    User(["ユーザー入力<br/>user_question: '本番リリースの注意点は？'"])

    subgraph ParentGraph["ParentGraph (ParentState: user_question, final_answer)"]
        direction LR
        P_Start(("START")) --> CallSub["call_sub ノード<br/>State 変換ラッパー"]
        CallSub --> P_End(("END"))
    end

    subgraph SubGraph["Subgraph (SubState: sub_query, sub_answer)"]
        direction LR
        S_Start(("START")) --> SubWorker["sub_worker(state, *, store)<br/>ポリシー参照 + LLM 応答生成"]
        SubWorker --> S_End(("END"))
    end

    Store[("InMemoryStore<br/>NS: ('team', 'infra')<br/>Key: 'policy'<br/>Value: {rule: '本番変更は必ずカナリアデプロイを行う'}")]
    LLM["ChatOpenAI<br/>(MiniCPM-V-4.7-35B-A3B)"]

    User --> P_Start
    CallSub -- "入力変換<br/>{sub_query: user_question}" --> S_Start
    S_End -- "出力変換<br/>{final_answer: sub_answer}" --> CallSub
    Store -. "親の compile(store=store) から<br/>自動伝播 (store.get)" .-> SubWorker
    SubWorker <--> LLM

    style ParentGraph fill:#eef6ff,stroke:#3b82f6,stroke-width:2px
    style SubGraph fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    style CallSub fill:#dbeafe,stroke:#2563eb
    style SubWorker fill:#dcfce7,stroke:#15803d
    style Store fill:#fef3c7,stroke:#d97706,stroke-width:2px
{% endmermaid %}

### 設計上の 3 つのポイント

1. **スキーマの完全独立**：
   - 親グラフは `ParentState(user_question, final_answer)` を持ちます。
   - サブグラフは `SubState(sub_query, sub_answer)` を持ち、親グラフと 1 つもキーを共有しません。
2. **ラッパーノードによる明示的な変換**：
   - 親グラフの `call_sub` ノード内で `subgraph.invoke({"sub_query": state["user_question"], "sub_answer": ""})` を呼び出し、返却された `sub_out["sub_answer"]` を親の `final_answer` にマッピングします。
3. **階層タプル Namespace によるポリシー管理**：
   - `store.put(("team", "infra"), "policy", {"rule": "本番変更は必ずカナリアデプロイを行う"})` のように、`("組織/チーム階層", "サブドメイン")` のタプルで名前空間を切ります。単一ポリシーの参照には `O(1)` の `store.get()` を使用します。

---

## 2. 完全実装コード

環境変数 `BASE_URL` と `API_KEY` を `.env` から読み込み、独立スキーマ Subgraph と `InMemoryStore` を構築・実行する完全なスクリプトです。

```python
import json
import os
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

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


def run_policy_subgraph_demo() -> dict:
    llm = make_llm()

    # 2. Cross-Thread Store の初期化と社内ポリシーの事前登録
    # Namespace は階層タプル ("team", "infra")、キーは "policy" とする
    store = InMemoryStore()
    store.put(
        ("team", "infra"),
        "policy",
        {"rule": "本番変更は必ずカナリアデプロイを行う"},
    )

    # 3. サブグラフ側の State 定義（親グラフとはキーを一切共有しない）
    class SubState(TypedDict):
        sub_query: str
        sub_answer: str

    # 4. サブグラフのワーカーノード定義
    # キーワード専用引数 `*, store: BaseStore` を宣言すると、
    # 親グラフの compile(store=store) から自動的に同一 Store が注入される
    def sub_worker(state: SubState, *, store: BaseStore) -> dict:
        item = store.get(("team", "infra"), "policy")
        policy = item.value["rule"] if item is not None else "特段の社内ポリシーなし"

        prompt = (
            f"社内ポリシー「{policy}」を踏まえ、"
            f"「{state['sub_query']}」に1文で答えてください。"
        )
        resp = llm.invoke(prompt)
        return {"sub_answer": resp.content}

    # 5. サブグラフのコンパイル（ここには store を渡さなくてよい点に注目）
    subgraph = (
        StateGraph(SubState)
        .add_node("sub_worker", sub_worker)
        .add_edge(START, "sub_worker")
        .add_edge("sub_worker", END)
        .compile()
    )

    # 6. 親グラフ側の State 定義
    class ParentState(TypedDict):
        user_question: str
        final_answer: str

    # 7. 親グラフのラッパーノード：親 State → 子入力、子出力 → 親 State へ変換
    def call_sub(state: ParentState) -> dict:
        sub_out = subgraph.invoke(
            {
                "sub_query": state["user_question"],
                "sub_answer": "",
            }
        )
        return {"final_answer": sub_out["sub_answer"]}

    # 8. 親グラフのコンパイル時に store をバインドする
    parent = (
        StateGraph(ParentState)
        .add_node("call_sub", call_sub)
        .add_edge(START, "call_sub")
        .add_edge("call_sub", END)
        .compile(store=store)
    )

    # 9. 親グラフを実行して結果を返す
    return parent.invoke(
        {
            "user_question": "本番リリースの注意点は？",
            "final_answer": "",
        }
    )


if __name__ == "__main__":
    result = run_policy_subgraph_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 3. 実測実行結果と内部動作のトレース

実際に `MiniCPM-V-4.7-35B-A3B` を用いて上記ラボを実行した際の実測結果（`elapsed_s: 0.94s`）は以下の通りです。

```json
{
  "user_question": "本番リリースの注意点は？",
  "final_answer": "本番変更を行う際は、必ずカナリアデプロイを実施し、段階的にリリース範囲を広げながら異常監視を行うことが重要です。"
}
```

### 実行結果から読み取れる 2 つの事実

1. **社内ポリシーがサブグラフ経由で確実にプロンプトへ反映されている**：ユーザー質問は「本番リリースの注意点は？」という一般的な問いかけであり、「カナリアデプロイ」という単語は一切含まれていません。それでも回答に「必ずカナリアデプロイを実施し…」と組み込まれているのは、サブグラフの `sub_worker` が親グラフにバインドされた `InMemoryStore` から `("team", "infra") / "policy"` を正しく取得できた証拠です。
2. **親グラフの State はクリーンに保たれている**：出力辞書には `user_question` と `final_answer` のみが含まれており、サブグラフ内部の `sub_query` や `sub_answer` は親グラフのチャネルを汚染していません。

### なぜ `subgraph.compile()` に `store` を渡していないのに動くのか？

実装コードの 5 番目を見ると、`subgraph = StateGraph(SubState)...compile()` には `store` 引数を一切渡していません。にもかかわらず、`sub_worker(state, *, store)` に親の `InMemoryStore` が注入される内部メカニズムをシーケンス図で確認しましょう。

{% mermaid %}
sequenceDiagram
    autonumber
    participant Caller as 呼び出し元
    participant Parent as ParentGraph (Pregel)
    participant Ctx as LangGraph Runtime Context<br/>(contextvars)
    participant Wrap as call_sub ノード
    participant Sub as Subgraph (Pregel)
    participant Worker as sub_worker ノード
    participant Store as InMemoryStore

    Caller->>Parent: invoke({"user_question": "本番リリースの注意点は？", ...})
    Parent->>Ctx: 親の store / config / callbacks をコンテキストへ登録
    Parent->>Wrap: call_sub(parent_state) を実行
    Wrap->>Sub: subgraph.invoke({"sub_query": "...", "sub_answer": ""})
    Sub->>Ctx: 現在の実行コンテキストから親の store を自動継承
    Sub->>Worker: sub_worker(sub_state, store=親のStore) を注入呼び出し
    Worker->>Store: store.get(("team", "infra"), "policy")
    Store-->>Worker: Item(value={"rule": "本番変更は必ずカナリアデプロイを行う"})
    Worker-->>Sub: {"sub_answer": "本番変更を行う際は、必ずカナリアデプロイ..."}
    Sub-->>Wrap: {"sub_query": "...", "sub_answer": "..."}
    Wrap-->>Parent: {"final_answer": "本番変更を行う際は、必ずカナリアデプロイ..."}
    Parent-->>Caller: ParentState 最終結果を返却
{% endmermaid %}

LangGraph の Pregel ランタイム（[langgraph/pregel/main.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)）は、親グラフの実行中に `config` や `store`・`checkpointer` を Python の `contextvars` 経由で伝播させます。そのため、ラッパーノード関数の中から素の `subgraph.invoke(child_input)` を呼ぶだけで、サブグラフ側の Pregel ランタイムが「自分は親グラフの実行コンテキスト内から呼ばれた」と認識し、親の `store` を自動的に引き継ぎます。

---

## 4. 応用パターン：動的ポリシー更新と `Runtime[Context]` への発展

Cross-Thread Store の真価は、**「別スレッドや運用バッチから Store の内容を更新した瞬間、すべての親グラフ・サブグラフの次回実行に即座に反映される」**点にあります。さらに本番のマルチテナント構成では、チーム名やユーザー ID をハードコードせず `context_schema` + `Runtime[Context]` で渡す設計へ発展させます。

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


@dataclass
class TeamContext:
    """実行時に渡す不変コンテキスト（グラフ State には混ぜない）。"""
    team_id: str


class SubState(TypedDict):
    sub_query: str
    sub_answer: str


def sub_worker_with_runtime(state: SubState, runtime: Runtime[TeamContext]) -> dict:
    # Runtime 経由で team_id と store を型安全に取得
    team_id = runtime.context.team_id
    item = runtime.store.get(("team", team_id), "policy")
    rule = item.value["rule"] if item else "標準リリース手順に従う"
    return {"sub_answer": f"[{team_id} ポリシー適用] {rule} -> 質問: {state['sub_query']}"}


# 運用中に Store を更新すれば、新しいスレッドにも即座に新ポリシーが反映される
store = InMemoryStore()
store.put(("team", "infra"), "policy", {"rule": "ブルーグリーンデプロイ＋自動ロールバック必須"})
store.put(("team", "payments"), "policy", {"rule": "本番変更には SRE とセキュリティの二重承認が必須"})

graph = (
    StateGraph(SubState, context_schema=TeamContext)
    .add_node("sub_worker", sub_worker_with_runtime)
    .add_edge(START, "sub_worker")
    .add_edge("sub_worker", END)
    .compile(store=store)
)

# 同じグラフ定義のまま context を切り替えてチーム別ポリシーを参照
print(graph.invoke({"sub_query": "リリース手順", "sub_answer": ""}, context=TeamContext(team_id="payments")))
```

---

## 5. 注意点・ハマりどころ（Pay Attention）

Subgraphs と Store を実務に導入する際、特につまずきやすい 5 つのポイントを整理します。

1. **親グラフの `.compile(store=store)` 忘れによる `store=None` エラー**
   - サブグラフのノード引数に `*, store: BaseStore` を書いても、最上位の親グラフが `.compile(store=store)` されていなければ `store` は `None` として渡され、`store.get(...)` 呼び出し時に `AttributeError: 'NoneType' object has no attribute 'get'` で落ちます。単体テストでサブグラフだけを直接 `invoke` する場合は、テスト時のみサブグラフ側に `compile(store=test_store)` を渡すか、ファクトリ関数で `store` を注入できるようにしておきます。
2. **`store.get()` の戻り値は `Item | None` であり、直接 `.value` に触ると危険**
   - 指定した `(namespace, key)` が未登録の場合、`store.get()` は例外ではなく `None` を返します。ラボの最小コードでは事前 `put` を前提に簡略化していますが、本番コードでは必ず `if item is None:` のフォールバック（デフォルトポリシー適用など）を入れてください。
3. **独立スキーマ Subgraph でのキー変換漏れ・必須キー不足**
   - `TypedDict` ベースの独立スキーマでは、ラッパーノード `call_sub` がサブグラフ入力辞書を組み立てる際にキー名を 1 文字でも間違えると、サブグラフ内部で `KeyError` になります。また、サブグラフの返却値から親 State に戻すキーを書き忘れると、親グラフ側で値が更新されず無言で空文字のまま進行してしまうため、入出力マッピングのユニットテストを書くのが確実です。
4. **`InMemoryStore` の揮発性と `search()` の `limit`・並び順の違い**
   - `InMemoryStore` はプロセス終了で全データが消える開発・テスト専用実装です。本番では `PostgresStore` や `RedisStore` に差し替える必要があります。その際、`store.search()` はデフォルト `limit=10` で暗黙切り詰めされる点と、バックエンドによって返却順（挿入順 vs `updated_at` 降順）が異なる点に注意し、必要なら明示的にページネーションとクライアント側ソートを行ってください。
5. **Per-thread（`checkpointer=True`）Subgraph での並列呼び出し衝突**
   - 今回のように「毎回フレッシュな問い合わせ処理を行うサブグラフ」はデフォルトの Per-invocation（`checkpointer=None`）で問題ありませんが、サブグラフにマルチターン記憶を持たせようとして `checkpointer=True` に変更した途端、同一サブグラフの並列呼び出しでチェックポイント競合が起きます。永続化モードは「呼び出しごとに独立でよいか？」を基準に慎重に選んでください。

---

## まとめ

本記事では、Part 6 理論編の内容をもとに **独立スキーマ Subgraph + `InMemoryStore`** のポリシー参照型マルチ階層グラフを実装し、実測 `0.94s` で社内ポリシーを反映した回答生成が機能することを確認しました。

- 親グラフとサブグラフの State を分離することで、内部キーの汚染を防ぎながら疎結合なモジュール設計ができる。
- 親グラフの `.compile(store=store)` だけで、ラッパーノード経由で呼ばれるサブグラフのワーカーへも `BaseStore` が自動伝播される。
- `Runtime[Context]` と組み合わせることで、マルチテナント・複数チーム向けのポリシー切り替えへ自然に拡張できる。

次回の **Part 7 理論編** では、これまで使ってきた Graph API の裏側で動く **Pregel BSP 実行モデル** と、もう一つの高レベル抽象である **Functional API（`@entrypoint` / `@task`）**、さらに **Workflows vs Agents の設計指針** を解説します。
