---
title: "LangGraph Part 1 理論編：StateGraph の核心理論（State・Reducer・Node・Edge・Super-step）と Thinking in Graph API"
date: 2026-05-27 09:46:26
categories: [AI]
tags: [LangGraph, StateGraph, Pregel, BSP, Reducer, AIエージェント, 理論]
lang: ja
description: "LangGraph の中核である StateGraph の設計思想を、Pregel BSP モデル・Super-step・Channel・Reducer・Node・Edge・Thinking in Graph API の観点から体系的に解説する。"
---

# LangGraph Part 1 理論編：StateGraph の核心理論と Thinking in Graph API

LLM アプリケーションが単発のプロンプト実行や線形チェーン（DAG）から、自律的なツール呼び出し・自己修正ループ・人間の承認待機（Human-in-the-Loop）を含む**ステートフルなエージェント**へと進化するにつれ、従来のパイプライン抽象化では制御フローと状態管理の破綻が避けられなくなりました。

**LangGraph** は、この課題を解決するために「エージェントワークフローを **共有状態（State）**・**ノード（Node）**・**エッジ（Edge）** から成る有向グラフ（循環を許容）として明示的にモデル化する」フレームワークです。本記事では、LangGraph の Graph API の基礎となる理論構造を、Google の Pregel 実行モデルから実践的な設計思考「Thinking in the Graph API」まで体系的に解説します。

---

## 1. Pregel BSP モデルと Super-step の実行メカニズム

LangGraph のグラフ実行ランタイム（[langgraph/pregel](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)）は、Google の大規模グラフ処理システム **Pregel** に着想を得た **BSP（Bulk Synchronous Parallel：バルク同期並列）モデル** とメッセージパッシングによって動作します。

> **核心原則**：LangGraph におけるグラフ実行は、離散的な **Super-step（スーパーステップ）** の反復として進行します。同じ Super-step にスケジュールされた複数ノードは並列実行され、逐次依存のあるノードは次の Super-step へ持ち越されます。

各 Super-step は以下の厳密なライフサイクルに従います。

1. **Plan（計画）**：入力エッジやチャネル更新によってメッセージを受信したノードが「アクティブ（Active）」となり、当該 Super-step の実行対象タスクとしてスケジュールされます。
2. **Execution（実行）**：アクティブなすべてのノードが、**同じ時点の State スナップショット**を読み取り、並列に処理を実行して部分更新（Partial Update）を返します。
3. **Update & Barrier（同期・更新）**：Super-step 内のすべてのノードが完了すると、バリア同期が行われ、各ノードの返却値がチャネルの Reducer を通じて一括で State にマージされます。この境界でチェックポイントの永続化が行われます。
4. **Termination（終了判定）**：次の Super-step にアクティブとなるノードが存在せず、配送待ちのメッセージもなくなった時点でグラフ実行が完了します。

{% mermaid %}
flowchart LR
    S(["START<br/>初期入力の投入"]) --> SS1["Super-step 1<br/>前処理ノード A"]
    SS1 --> B1{"バリア同期 &<br/>Reducer 適用"}
    B1 --> SS2A["Super-step 2<br/>並列ノード B"]
    B1 --> SS2B["Super-step 2<br/>並列ノード C"]
    SS2A --> B2{"バリア同期 &<br/>Reducer 適用"}
    SS2B --> B2
    B2 --> SS3["Super-step 3<br/>統合ノード D"]
    SS3 --> E(["END<br/>全ノード非アクティブ"])

    style S fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style SS1 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
    style SS2A fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style SS2B fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style SS3 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
    style B1 fill:#eceff1,stroke:#455a64,stroke-dasharray: 5 5,color:#263238
    style B2 fill:#eceff1,stroke:#455a64,stroke-dasharray: 5 5,color:#263238
    style E fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
{% endmermaid %}

### Super-step がもたらすトランザクション性

BSP モデルの最大の利点は、**Super-step 単位のトランザクション性**にあります。ある Super-step で並列実行されたブランチのうち 1 つでも例外を送出した場合、その Super-step 全体の State 更新はコミットされません。一方で Checkpointer を有効にしていれば、同一 Super-step 内で先に成功した兄弟ノードの出力は Pending Writes として保全され、再開時に無駄な再計算を回避できます。

---

## 2. State・Channel・Reducer の設計原理

LangGraph において **State** とは、すべてのノードが共有するアプリケーションの現在スナップショットです。内部的には、State スキーマの各キーが **Channel（チャネル）** と呼ばれる名前付きスロットに対応づけられます。

> **State 更新の基本契約**：ノードは State 全体をインプレースで書き換えるのではなく、自身が変更したいキーだけを含む**部分更新辞書（Partial Update）**を返却します。LangGraph は各チャネルに紐づく **Reducer** を介して、既存値（`left`）と新しい更新値（`right`）を安全にマージします。

### 2.1 スキーマ定義の選択肢

LangGraph の `StateGraph` は以下の 3 種類のスキーマ定義をサポートします。

| スキーマ形式 | 特徴 | 推奨ユースケース |
| --- | --- | --- |
| `TypedDict` | 実行時オーバーヘッドが最小。静的型チェックと相性が良い | ほとんどの LangGraph ワークフロー（第一選択） |
| `dataclass` | フィールドのデフォルト値を自然に定義できる | デフォルト値が必要な構造化ステート |
| Pydantic `BaseModel` | 実行時の厳密なバリデーションと型強制が可能 | 外部入力の厳格な検証が必須な境界スキーマ |

さらに、`StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)` のように**複数スキーマ**を指定することで、グラフ外部へ公開する入出力インターフェースと、内部ノード間だけで共有するプライベートチャネルを厳格に分離できます。

### 2.2 Reducer の数学的構造と動作

Reducer は次のシグネチャを持つ二項関数です。

$$\text{new\_value} = \text{reducer}(\text{left}=\text{current\_state}[k],\; \text{right}=\text{node\_update}[k])$$

- **デフォルト Reducer（Last-Write-Wins）**：`Annotated` で Reducer を指定しないキーは、新しい更新値で古い値が単純に上書きされます。
- **カスタム Reducer（集約・追記）**：`Annotated[list[str], operator.add]` のように型アノテーションへ関数を付与すると、複数ノードや複数ステップからの出力がリスト連結などで自動的に集約されます。
- **`add_messages` と `MessagesState`**：チャット履歴では単純な `operator.add` だけでなく、**メッセージ ID に基づくインプレース更新（上書き・修正）**や、`{"role": "user", "content": "..."}` といった辞書形式から `HumanMessage` 等への自動デシリアライズが必要です。これを提供する組み込み Reducer が `add_messages` であり、事前定義スキーマが `MessagesState` です。

{% mermaid %}
flowchart TB
    subgraph Current["現在の State（Channel 群）"]
        C1["topic: 'Pregel' (Default Reducer)"]
        C2["logs: ['step1'] (operator.add)"]
        C3["messages: [HumanMessage] (add_messages)"]
    end

    Node["ノード実行結果（Partial Update）<br/>{topic: 'BSP', logs: ['step2'], messages: [AIMessage]}"]

    subgraph Merge["Reducer 評価フェーズ"]
        R1["Last-Write-Wins<br/>'Pregel' -> 'BSP'"]
        R2["operator.add<br/>['step1'] + ['step2']"]
        R3["add_messages<br/>ID照合 & 追記/更新"]
    end

    subgraph Next["更新後の State スナップショット"]
        N1["topic: 'BSP'"]
        N2["logs: ['step1', 'step2']"]
        N3["messages: [HumanMessage, AIMessage]"]
    end

    C1 --> R1
    C2 --> R2
    C3 --> R3
    Node --> R1
    Node --> R2
    Node --> R3
    R1 --> N1
    R2 --> N2
    R3 --> N3

    style Current fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Node fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#e65100
    style Merge fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    style Next fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

### 2.3 Reducer のバイパスと `Overwrite`

集約型 Reducer（例：`operator.add`）を持つフィールドに対してノードが `{"errors": []}` を返しても、`["err1"] + [] == ["err1"]` となるため**フィールドはクリアされません**。リトライ間のエラーバッファ初期化などで Reducer を明示的にバイパスして値を置き換えたい場合は、`langgraph.types.Overwrite` を使用します。

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite

# State スキーマ定義：キーごとに異なる Reducer を宣言できる
class AgentState(TypedDict):
    # Reducer 未指定 -> デフォルトの Last-Write-Wins（上書き）
    current_topic: str
    # operator.add を指定 -> 各ノードが返すリストを末尾へ自動連結
    audit_logs: Annotated[list[str], operator.add]
    # add_messages を指定 -> メッセージ ID による更新・追記・辞書からの自動変換に対応
    messages: Annotated[list[AnyMessage], add_messages]
    # リトライ間でリセットしたいエラーログ用チャネル
    errors: Annotated[list[str], operator.add]

def reset_errors_node(state: AgentState) -> dict:
    # Overwrite([]) を返すことで operator.add をバイパスし、チャネルを空リストへ強制置換する
    return {
        "errors": Overwrite([]),
        "audit_logs": ["errors_cleared"],  # こちらは通常通り追記される
    }
```

---

## 3. Node・Edge・Compilation の構造と責務分離

LangGraph の設計美は「**Node が処理を行い、Edge が次の遷移先を決める**」という明確な責務分離にあります。

### 3.1 Node のシグネチャと実行単位（Unit of Work）

ノードは通常の同期関数（`def`）または非同期関数（`async def`）として定義します。内部的には LangChain の `RunnableLambda` でラップされ、バッチ処理・非同期実行・LangSmith トレーシングを自動的に獲得します。ノードは以下の引数を受け取れます。

1. `state`：現在のグラフ State（必須の第一引数）
2. `config`：`RunnableConfig`（`thread_id` や `tags`、`metadata["langgraph_step"]` 等を含む）
3. `runtime`：`Runtime[Context]`（State に含めたくない DB 接続やモデル設定などの依存性、および `store` や `stream_writer` へのアクセス）

ノードの粒度を設計する際は、**「ノード ＝ チェックポイント境界 かつ リトライ境界 かつ 分岐起点」**と捉えることが重要です。

| プリミティブ | 独自チェックポイント | 再開・リトライ時の挙動 | 選択すべき場面 |
| --- | --- | --- | --- |
| ノード内の通常関数 | なし（ノードの Super-step 内） | ノード全体と共に再実行される | 処理後に分岐がなく、再実行しても安全な純粋処理 |
| `@task`（Functional API 連携） | 親チェックポイントに結果を保存 | 完了済みタスクは保存済み結果を再利用 | 副作用や高コストな外部呼び出しをノード内で保護したい場合 |
| Subgraph | あり（親スレッドの名前空間配下） | サブグラフ内の失敗・中断ノードから再開 | 独立した State スキーマや再利用可能なサブワークフロー |
| Graph Node | あり（Super-step 境界ごとに保存） | そのノードから再開（先行ノードは再実行しない） | 条件分岐・`interrupt()`・Time Travel の境界にしたい処理 |

> **再実行と冪等性（Idempotency）の原則**：チェックポイントは Super-step 境界でのみ保存され、ノード関数の途中では保存されません。リトライや `interrupt()` からの再開時には**対象ノードが関数の先頭から再実行される**ため、外部への副作用（DB 書き込みや決済 API 呼び出し等）は冪等キーや Upsert で保護する必要があります。

### 3.2 Edge と Compilation

- **`START` / `END`**：グラフの仮想的な入口と終端を表す定数。
- **Normal Edge（`add_edge`）**：常に固定の次ノードへ遷移します。1 つのノードから複数の `add_edge` を張ると、遷移先ノード群は次の Super-step で並列実行されます。
- **Conditional Edge（`add_conditional_edges`）**：ルーティング関数が現在の State を評価し、次に実行すべきノード名（または `END`、複数ノード名のリスト、`Send` オブジェクト群）を動的に返します。
- **Compilation（`builder.compile(...)`）**：グラフ構造の静的検証（孤立ノードの検出やエッジの整合性確認）を行い、実行可能な `CompiledStateGraph` を生成します。Checkpointer・Store・Cache などのランタイム基盤もこの `compile()` 時に注入します。

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 最小の State 定義
class MinimalState(TypedDict):
    text: str  # デフォルト Reducer（上書き）

def append_suffix(state: MinimalState) -> dict:
    # 現在の State を読み取り、変更するキーだけを辞書で返す（インプレース変更は禁止）
    return {"text": f"{state['text']}-processed"}

# StateGraph ビルダーの構築とコンパイル
compiled_graph = (
    StateGraph(MinimalState)          # State スキーマをバインド
    .add_node("process", append_suffix)  # 名前付きノードとして登録
    .add_edge(START, "process")       # 仮想入口 START から process へ接続
    .add_edge("process", END)         # process 完了後に仮想終端 END へ接続
    .compile()                        # 構造検証を実行し、実行可能なグラフを生成
)
```

---

## 4. Thinking in the Graph API（5ステップ設計思考）

実際の業務エージェントを LangGraph で設計する際、いきなりコードを書き始めるのではなく、公式ガイドでも推奨されている **5 ステップの思考プロセス** を踏むことで保守性の高いグラフが構築できます。

1. **Step 1：ワークフローを離散ステップへ分解する**
   メール対応エージェントを例にとると、`Read Email` → `Classify Intent` →（`Doc Search` / `Bug Track`）→ `Draft Reply` →（`Human Review` / `Send Reply`）のように、1 ノード 1 責務の単位へ分解し、固定遷移（実線）と動的遷移（破線）を図示します。
2. **Step 2：各ステップの性質を 4 分類する**
   - **LLM ステップ**：意図分類・要約・回答生成（静的プロンプト方針＋動的 State 入力）
   - **Data ステップ**：検索・DB 参照（リトライ戦略・キャッシュ適用候補）
   - **Action ステップ**：メール送信・チケット起票（キャッシュ禁止・冪等性必須）
   - **User Input ステップ**：人間の承認・修正（`interrupt()` による一時停止）
3. **Step 3：Raw State の原則で State を設計する**
   > **Raw State の原則**：State には整形済みのプロンプト文字列ではなく**生データ（Raw Data）**のみを格納し、プロンプトの組み立ては各ノード内でオンデマンドに行います。これにより、プロンプトテンプレートの変更が State スキーマへ波及せず、デバッグ時にも各ノードが受け取った構造化データを明確に追跡できます。
4. **Step 4：エラーを「誰が回復するか」で分類する**
   - 一時的障害（ネットワーク遅延・レート制限）→ **システム**（`RetryPolicy`）
   - LLM で回復可能なエラー（ツール引数ミス・パース失敗）→ **LLM**（エラーを State に書き戻して再推論ループへ）
   - 情報不足・承認待ち → **人間**（`interrupt()`）
   - 想定外のバグ → **開発者**（そのまま例外を送出して検知）
5. **Step 5：ノード粒度のトレードオフを見極める**
   ノードを細分化しすぎると Super-step 数とチェックポイント I/O が増大し、逆に巨大すぎるノードはリトライ時の再計算コストとデバッグ難易度を悪化させます。「分岐点・外部 I/O 境界・人間の介入点」を基準に切るのがベストプラクティスです。

---

## 5. まとめ

本記事では、LangGraph Part 1 の理論基盤として以下を整理しました。

- **Pregel BSP モデルと Super-step**：並列実行・バリア同期・トランザクション性を保証する実行基盤。
- **State・Channel・Reducer**：部分更新を `f(left, right)` で安全にマージする仕組みと、`add_messages` / `Overwrite` の使い分け。
- **Node・Edge・Compilation**：責務分離、冪等性要件、そして構造検証を行う `compile()`。
- **Thinking in the Graph API**：離散化・4 分類・Raw State 原則・エラー回復設計の 5 ステップ。

次回の **Part 1 実践編** では、この理論をそのままコードへ落とし込み、`MiniCPM-V-4.7-35B-A3B` を使って `StateGraph` と `operator.add` Reducer による最小ステートフルパイプラインを構築・実測します。

---

### 参考リンク
- [LangGraph Graph API Overview（公式ドキュメント）](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Thinking in the Graph API（公式ドキュメント）](https://docs.langchain.com/oss/python/langgraph/thinking-in-graph-api)
- [LangGraph ソースコード（StateGraph）](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)
