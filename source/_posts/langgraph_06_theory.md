---
title: "LangGraph Part 6 理論編：Subgraphs（共有 State vs 独立 State）と Cross-Thread Store（長期記憶・Namespace 設計）"
date: 2026-06-21 13:02:25
categories: [AI]
tags: [LangGraph, Subgraphs, Store, Long-Term Memory, Multi-Agent, Python, AI]
lang: ja
description: "LangGraph における Subgraphs の2つの通信パターン（共有 State と独立 State）、チェックポイント永続化モード（Per-invocation / Per-thread / Stateless）、およびスレッド横断の長期記憶を提供する BaseStore の Namespace 設計とセマンティック検索の理論を体系的に解説します。"
---

## はじめに：大規模グラフのモジュール化とスレッド横断記憶

これまでの Part 1〜5 では、単一の `StateGraph` を中心に、条件分岐・Map-Reduce（`Send` / `Command`）・Checkpointer による永続化・Streaming・Fault Tolerance・Interrupts / Time Travel を学んできました。

しかし、実際のプロダクション向け AI エージェントを構築し始めると、すぐに次の 2 つの壁に直面します。

1. **グラフの肥大化と責務分離の壁**：1 つの巨大な `StateGraph` にすべてのノードと State キーを詰め込むと、状態スキーマが汚染され、チーム間での並行開発やコンポーネントの再利用が困難になります。
2. **スレッド（セッション）境界の壁**：Checkpointer は `thread_id` 単位で会話履歴や実行状態を保存するため、**別のスレッド（新しい会話や別ユーザー・別ワークフロー）から過去の知見や社内ポリシーを参照する仕組み**が別途必要になります。

LangGraph はこれら 2 つの課題に対して、**Subgraphs（サブグラフ）** と **Cross-Thread Store（`BaseStore`）** という直交する 2 つのプリミティブを提供しています。本記事では、両者の内部アーキテクチャと設計指針を理論面から整理します。

---

## 1. Subgraphs のアーキテクチャと 2 つの通信パターン

LangGraph における **Subgraph（サブグラフ）** とは、**別のグラフのノードとして組み込まれるコンパイル済みグラフ（`CompiledStateGraph` / `Pregel` インスタンス）** のことです。公式ドキュメントおよび実装（[langgraph/graph/state.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)）では、主に以下の 3 つのユースケースで Subgraph の活用が推奨されています。

- **マルチエージェントシステムの構築**：各専門エージェントを独立したサブグラフとしてカプセル化する。
- **ノード群の再利用**：複数の親グラフから共通の検索・検証・要約パイプラインを呼び出す。
- **分散開発（インターフェース分離）**：入力・出力スキーマの契約さえ守れば、別チームがサブグラフ内部のノード構成を独立して変更できる。

親グラフとサブグラフの通信方法は、**State スキーマを共有するか否か**によって明確に 2 パターンに分かれます。

| 通信パターン | 適用条件 | 組み込み方 | State の扱い |
| --- | --- | --- | --- |
| **パターン A：共有 State スキーマ** | 親と子で共通のキー（例：`messages`, `foo`）を持つ | コンパイル済みサブグラフを直接 `add_node("sub", subgraph)` に渡す | 親のチャネルから自動読み取りし、共通キーへの更新が親へ自動反映される（子専用キーはプライベートに保持） |
| **パターン B：独立 State スキーマ** | 親と子で共有キーがない、または明示的なデータ変換が必要 | 親のラッパーノード関数内で `subgraph.invoke(...)` を呼び出す | ラッパー関数が「親 State → 子入力」「子出力 → 親 State 更新」を明示的にマッピングする |

> **設計原則（Subgraph の境界分離）**：マルチエージェント構成で各サブエージェントの内部思考履歴やツール実行ログを親グラフの `messages` に混ぜたくない場合は、**独立 State スキーマ（ラッパーノード内からの `invoke`）** を選択します。これにより、親グラフには最終回答や構造化サマリだけを返し、コンテキストウィンドウの肥大化とスキーマ結合を防げます。

{% mermaid %}
flowchart LR
    subgraph PatternA["パターン A: 共有 State スキーマ (直接 add_node)"]
        direction TB
        PA_In["ParentState<br/>{foo: 'hi! foo'}"] --> PA_Sub["Subgraph (ノードとして登録)<br/>SubState: {foo, bar}"]
        PA_Sub --> PA_N1["sub_node_1<br/>bar = 'bar' (プライベート)"]
        PA_N1 --> PA_N2["sub_node_2<br/>foo = foo + bar (共有キー更新)"]
        PA_N2 --> PA_Out["ParentState<br/>{foo: 'hi! foobar'}"]
    end

    subgraph PatternB["パターン B: 独立 State スキーマ (ラッパーノード経由)"]
        direction TB
        PB_In["ParentState<br/>{user_question: '...'}"] --> PB_Wrap["call_sub (変換ラッパー)"]
        PB_Wrap -- "入力マッピング<br/>{sub_query: ...}" --> PB_Sub["Subgraph.invoke()<br/>SubState: {sub_query, sub_answer}"]
        PB_Sub -- "出力マッピング<br/>{final_answer: ...}" --> PB_Wrap
        PB_Wrap --> PB_Out["ParentState<br/>{final_answer: '...'}"]
    end

    style PatternA fill:#eef6ff,stroke:#3b82f6,stroke-width:2px
    style PatternB fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    style PA_Sub fill:#dbeafe,stroke:#2563eb
    style PB_Wrap fill:#dcfce7,stroke:#15803d
    style PB_Sub fill:#bbf7d0,stroke:#166534
{% endmermaid %}

### パターン A：共有 State スキーマの実装構造

親と子で少なくとも 1 つ以上のキー（以下の例では `foo`）が一致している場合、コンパイル済みサブグラフをそのまま `add_node` に渡せます。サブグラフ側だけに存在するキー（`bar`）はサブグラフ内部のプライベート状態として機能し、親グラフの State には漏れ出しません。

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# サブグラフの State：foo は親と共有、bar はサブグラフ専用（プライベート）
class SubgraphState(TypedDict):
    foo: str
    bar: str

def subgraph_node_1(state: SubgraphState) -> dict:
    # サブグラフ専用のプライベートキーを更新
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState) -> dict:
    # プライベートキー bar を使って、親との共有キー foo を更新
    return {"foo": state["foo"] + state["bar"]}

subgraph = (
    StateGraph(SubgraphState)
    .add_node("subgraph_node_1", subgraph_node_1)
    .add_node("subgraph_node_2", subgraph_node_2)
    .add_edge(START, "subgraph_node_1")
    .add_edge("subgraph_node_1", "subgraph_node_2")
    .add_edge("subgraph_node_2", END)
    .compile()
)

# 親グラフの State：foo のみを持つ
class ParentState(TypedDict):
    foo: str

def parent_node_1(state: ParentState) -> dict:
    return {"foo": "hi! " + state["foo"]}

# コンパイル済みサブグラフをそのままノードとして登録
parent_graph = (
    StateGraph(ParentState)
    .add_node("node_1", parent_node_1)
    .add_node("node_2", subgraph)
    .add_edge(START, "node_1")
    .add_edge("node_1", "node_2")
    .add_edge("node_2", END)
    .compile()
)
```

### パターン B：独立 State スキーマと多階層ネスト

親グラフとサブグラフがキーを一切共有しない場合、親グラフのノード関数の中で `subgraph.invoke()` を呼び出します。このパターンは **Parent → Child → Grandchild** のような多階層ネストにもそのまま拡張でき、各層は自身の State 以外のキーを参照できないため強いカプセル化が実現します。

さらに注目すべき点として、親ノード関数内から `subgraph.invoke(child_input)` を呼び出した場合でも、LangGraph のランタイム（Python の `contextvars` ベースの実行コンテキスト伝播）によって、**親グラフに設定された `checkpointer`・`store`・ストリーミングコールバック・`config` がサブグラフへ自動的に引き継がれます**。

---

## 2. Subgraph の永続化モード（Persistence）と Namespace 分離

サブグラフを使う際に最も見落とされがちなのが、**「サブグラフの内部状態を呼び出し間でどう保持するか」**という設計判断です。LangGraph ではサブグラフの `.compile(checkpointer=...)` 引数によって、以下の 3 つの永続化モードを切り替えられます。

| 機能・挙動 | Per-invocation（デフォルト） | Per-thread | Stateless |
| --- | --- | --- | --- |
| `.compile()` 指定 | `checkpointer=None`（省略時） | `checkpointer=True` | `checkpointer=False` |
| **Interrupts (HITL)** | 対応（親の Checkpointer を継承） | 対応 | 非対応 |
| **マルチターン記憶** | なし（呼び出しごとに初期化） | あり（同一スレッド内で蓄積） | なし |
| **別サブグラフの複数呼び出し** | 問題なし | 要注意（名前空間分離が必要） | 問題なし |
| **同一サブグラフの並列呼び出し** | 問題なし（呼び出しごとに固有 NS） | **非対応（チェックポイント衝突）** | 問題なし |
| **状態検査 (`get_state(subgraphs=True)`)** | 実行中／中断中のみ参照可 | 蓄積状態を常時参照可 | 非対応 |

> **重要な前提条件**：サブグラフの永続化機能（`interrupt()`、状態検査、Per-thread 記憶）を有効にするには、**必ず最上位の親グラフを Checkpointer 付きで `.compile(checkpointer=...)` している必要があります**。サブグラフ単体に別の Checkpointer インスタンスを渡すのではなく、親の Checkpointer を継承（`None` または `True`）させるのが LangGraph の標準設計です。

### Per-thread モードにおける 2 つの罠と対策

1. **同一サブグラフの並列呼び出し衝突**：`checkpointer=True` を設定したサブグラフは、同一スレッド内で同じチェックポイント名前空間を共有します。そのため、LLM がツール経由で同じサブグラフを並列に 2 回呼び出すと書き込み競合が発生します。並列実行を許容したいサブエージェントはデフォルトの **Per-invocation（`checkpointer=None`）** を使うべきです。
2. **複数種類の Per-thread サブグラフにおける Namespace 分離**：ラッパーノード内で複数の異なる Per-thread サブグラフを呼び出すと、デフォルトでは呼び出し順ベースの名前空間が割り当てられるため、呼び出し順序が変わると状態が混線します。これを防ぐには、各サブグラフを固有ノード名を持つ `StateGraph` に包むか、`add_node("unique_name", subgraph)` として静的に登録します。

### Subgraph の状態検査とストリーミング

- **状態検査**：`graph.get_state(config, subgraphs=True)` を使うと、現在中断中（または Per-thread で保持中）のサブグラフ状態に `state.tasks[0].state` 経由でアクセスできます。ただし、LangGraph がサブグラフを静的検出できる構成（`add_node` またはノード直下呼び出し）に限られ、ツール関数内部に隠蔽されたサブグラフの状態検査は対象外です（`interrupt()` の伝播自体はネストの深さに関わらず動作します）。
- **ストリーミング**：親グラフの `graph.stream(..., subgraphs=True, stream_mode="updates")` を指定すると、親と子の両方のイベントが名前空間 `ns`（ルートは `()`、サブグラフは `("node_2:<task_id>",)`）付きで出力されます。

---

## 3. Cross-Thread Store（`BaseStore`）の設計理論

Checkpointer が「**1 つのスレッド（会話・ジョブ）内の状態スナップショット**」を管理するのに対し、**Store（`BaseStore`）** は「**スレッドを横断して共有される長期記憶（Key-Value + ベクトル検索）**」を担います。

| 観点 | Checkpointer（短期・スレッド内記憶） | Store（長期・スレッド横断記憶） |
| --- | --- | --- |
| **スコープ** | 単一の `thread_id` に閉じた状態 | すべてのスレッド・ユーザー・サブグラフから参照可能 |
| **データ構造** | グラフの State スナップショット（Super-step ごと） | 階層 Namespace (`tuple[str, ...]`) + Key + JSON Dict (`Item`) |
| **検索機能** | `thread_id` / `checkpoint_id` による時系列取得 | Exact Get / Prefix Search / メタデータフィルタ / セマンティック検索 |
| **代表的な用途** | 会話履歴、HITL 中断再開、Time Travel | ユーザー嗜好、社内ポリシー、学習済み事実、ナレッジベース |
| **本番バックエンド** | `SqliteSaver`, `PostgresSaver` など | `PostgresStore`, `MongoDBStore`, `RedisStore`, `UpstashStore` など |

> **Store の自動伝播メカニズム**：親グラフを `parent.compile(store=store)` でコンパイルすると、親グラフ内のノードだけでなく、**親ノードから呼び出される Subgraph 内のノードへも同一の `BaseStore` インスタンスが自動的に注入されます**。ノード関数のシグネチャに `*, store: BaseStore` または `runtime: Runtime[Context]` を宣言するだけで、サブグラフ側のコンパイル時に `store` を再指定することなく長期記憶へアクセスできます。

{% mermaid %}
flowchart TB
    subgraph Thread1["Thread A (thread_id='t-001')"]
        P1["ParentGraph<br/>compile(checkpointer, store)"] --> S1["Subgraph<br/>sub_worker(state, *, store)"]
    end

    subgraph Thread2["Thread B (thread_id='t-002')"]
        P2["ParentGraph<br/>compile(checkpointer, store)"] --> S2["Subgraph<br/>sub_worker(state, *, store)"]
    end

    CP1[("Checkpointer<br/>Thread A 専用スナップショット")]
    CP2[("Checkpointer<br/>Thread B 専用スナップショット")]
    ST[("Cross-Thread Store (BaseStore)<br/>Namespace: ('team', 'infra') / key: 'policy'<br/>Namespace: ('user_1', 'memories') / key: uuid")]

    P1 <--> CP1
    P2 <--> CP2
    S1 -- "store 自動伝播<br/>get / search / put" --> ST
    S2 -- "store 自動伝播<br/>get / search / put" --> ST

    style Thread1 fill:#eef6ff,stroke:#3b82f6,stroke-width:2px
    style Thread2 fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style CP1 fill:#fef3c7,stroke:#d97706
    style CP2 fill:#fef3c7,stroke:#d97706
    style ST fill:#dcfce7,stroke:#16a34a,stroke-width:2px
{% endmermaid %}

### Namespace 設計の原則

`BaseStore` では、すべてのアイテムが **`(namespace: tuple[str, ...], key: str)`** の組で一意に識別されます。保存されたエントリは `Item` オブジェクトとして返され、`value`（JSON シリアライズ可能な `dict`）・`key`・`namespace`・`created_at`・`updated_at` 属性を持ちます。

実運用で破綻しない Namespace を設計する際の 4 つの原則を押さえておきましょう。

1. **階層は「広いスコープ → 狭いスコープ」の順で並べる**
   - 例：`("org_acme", "team_infra", "policies")` や `(user_id, "preferences")`。
   - `store.search(namespace_prefix)` は **前方一致（Prefix Match）** で動作するため、`("org_acme",)` を指定すれば配下の全チーム・全カテゴリのアイテムを横断検索でき、`("org_acme", "team_infra", "policies")` を指定すれば対象を厳密に絞り込めます。
2. **単一ドキュメント参照は `store.get(namespace, key)` を使う**
   - 社内ポリシーやユーザープロファイルのように固定キーで引けるものは `get()` を使うことで `O(1)`（SQL バックエンドでは `(namespace, key)` の複合主キー検索）で取得できます。
3. **列挙時の `limit` 切り詰めとバックエンド間の順序差異に注意する**
   - `store.search()` のデフォルト `limit` は `10` であり、上限を超えたアイテムはエラーを出さず暗黙的に切り捨てられます。全件走査には `limit` と `offset` を使ったページネーションが必要です。
   - また、`InMemoryStore` は挿入順（最新が末尾）で返すのに対し、`PostgresStore` は `updated_at` 降順（最新が先頭）で返します。順序に依存するロジックは必ずクライアント側で `item.updated_at` によるソートを行ってください。
4. **シリアライズ制約を守る**
   - `value` はプレーンな Python `dict`（JSON 互換）に限定し、カスタムクラスやソケットオブジェクトを格納してはいけません。

### セマンティック検索とフィールド単位のインデックス制御

`InMemoryStore` や `PostgresStore` に `index` 設定を与えると、`store.search(namespace, query="...", limit=3)` によるベクトル類似度検索が有効になります。

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# 埋め込みモデルを指定してセマンティック検索対応 Store を構築
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # 埋め込みプロバイダ
        "dims": 1536,                                              # ベクトル次元数
        "fields": ["rule", "$"],                                   # インデックス対象フィールド（"$" は全体）
    }
)

# 特定フィールドのみ埋め込み対象にする場合
store.put(
    ("team", "infra"),
    "deploy-policy-01",
    {"rule": "本番変更は必ずカナリアデプロイを行う", "owner": "SRE"},
    index=["rule"],  # owner は検索ノイズになるため埋め込みから除外
)

# 埋め込み不要なシステムメタデータは index=False でコスト削減
store.put(
    ("team", "infra"),
    "last-audit",
    {"checked_at": "2026-06-21"},
    index=False,
)
```

### ノードからの Store アクセス：2 つの注入スタイル

LangGraph のノード関数から Store にアクセスする方法は 2 通りあります。

1. **キーワード引数注入（`*, store: BaseStore`）**：シンプルに Store だけが必要な場合。Subgraph 内のワーカーノードでよく使われます。
2. **`Runtime[Context]` オブジェクト経由（`runtime: Runtime[Context]`）**：`StateGraph(State, context_schema=Context)` と組み合わせ、`runtime.context.user_id`（実行時コンテキスト）と `runtime.store`（長期記憶）を型安全に同時取得する場合。本番のマルチテナントアプリではこちらが推奨されます。

### カスタム Store 実装のコントラクト

独自のデータストア（DynamoDB や社内 API など）を LangGraph の Store として統合したい場合は、`langgraph.store.base.BaseStore` を継承し、以下の 5 つの非同期メソッド（および同期版）を実装します。

- `aput(namespace, key, value, index=None)`：単一アイテムの保存・上書き
- `aget(namespace, key)`：完全一致による取得（存在しなければ `None`）
- `adelete(namespace, key)`：単一アイテムの削除
- `asearch(namespace_prefix, *, query=None, filter=None, limit=10, offset=0)`：プレフィックス検索・フィルタ・ベクトル検索
- `alist_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)`：存在する名前空間の一覧取得

SQL バックエンドで実装する際は、`namespace TEXT[]` カラムに GIN インデックスを張り、`PRIMARY KEY (namespace, key)` とすることで、Prefix 検索と `O(1)` の Exact Get を両立できます。

---

## まとめ

本記事では、大規模・長期稼働する LangGraph アプリケーションに不可欠な **Subgraphs** と **Cross-Thread Store** の理論を整理しました。

- **Subgraphs** は「共有 State（直接 `add_node`）」と「独立 State（ラッパー関数経由）」の 2 方式があり、マルチエージェントの履歴分離には独立 State が有効。
- Subgraph の永続化は `checkpointer=None`（Per-invocation）を基本とし、マルチターン蓄積が必要な場合のみ `checkpointer=True`（Per-thread）＋名前空間分離を検討する。
- **Cross-Thread Store（`BaseStore`）** は Checkpointer と役割分担し、階層タプル Namespace によってスレッドやサブグラフの境界を越えた長期記憶を提供する。

次回の **Part 6 実践編** では、独立スキーマ Subgraph と `InMemoryStore` を実際に組み合わせ、親グラフからサブグラフへ自動伝播された社内ポリシーを参照して回答を生成するマルチ階層グラフをコードと実測出力付きで検証します。
