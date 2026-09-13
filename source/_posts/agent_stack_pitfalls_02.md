---
title: 実践トラブルシュート記録 Part 2：LangGraph / DeepAgents 実践で踏んだ落とし穴（StoreBackend API 変更・オフロード閾値・Interrupt / Time Travel の罠）と設計総括
date: 2026-08-07 14:06:31
categories:
  - AI
tags:
  - Python
  - LLM
  - AI Agent
  - LangChain
  - LangGraph
  - DeepAgents
  - トラブルシューティング
  - 実践記録
lang: ja
description: LangGraphとDeepAgents 0.7.41の実践ラボで踏んだStoreBackend破壊的変更、CompositeBackendのプレフィックス剥離、Large Tool Result Offloading閾値とMoEレイテンシ急増、Interrupt/HITLの冪等性、Time Travelのチェックポイント選択の5つの罠を徹底解説。
---

前回の Part 1（`agent_stack_pitfalls_01`）では、環境構築や OpenAI 互換プロキシ、MoE モデル選定、レート制限、Structured Output といった基盤レイヤーの 5 つの落とし穴を整理しました。

本記事（Part 2）では、さらに一歩踏み込んで **LangGraph と DeepAgents 0.7.41 のフレームワーク内部挙動**に起因する 5 つの実践的な罠を取り上げます。いずれも「公式の古いスニペットや直感通りに書くと確実にハマる」一方で、一度内部の仕組みを理解すればエージェント設計の解像度が劇的に上がるポイントばかりです。

Part 1 と同じく、各トピックを **問題 / 発生条件 / 試行錯誤 / 根本原因 / まとめ・再発防止策** の 5 段構成で記録します。

<!-- more -->

---

## 落とし穴 1：`deepagents 0.7.41` の破壊的変更による `StoreBackend` の `TypeError`

### 問題

DeepAgents Lab 2（`lab2_composite_backend`）で、ローカルディスク（`FilesystemBackend`）と長期メモリ（`StoreBackend`）を組み合わせたハイブリッドストレージを構築しようとしたところ、`StoreBackend(store=store)` のインスタンス化時点で即座に `TypeError` が発生しました。

```text
TypeError: StoreBackend.__init__() missing 1 required keyword-only argument: 'namespace'
```

### 発生条件

- `deepagents==0.7.41` を使用している
- 以前のバージョンや旧ブログ記事の感覚で、`StoreBackend(store=InMemoryStore())` のように `StoreBackend` 側へ直接 `store` インスタンスを渡し、`namespace` 引数を省略した場合

### 試行錯誤

1. 最初は単純に `StoreBackend(store=store, namespace=("memories",))` と渡せば通るかと思い試したが、`StoreBackend.__init__` のシグネチャを確認するとそもそも `store` 引数を受け取る設計から変わっており、`namespace` も `ToolRuntime` を受け取るファクトリ関数（`Callable[[ToolRuntime], tuple[str, ...]]`）を想定していた。
2. `inspect.signature(StoreBackend.__init__)` および `create_deep_agent` の引数定義を確認したところ、**`store` インスタンス自体は `create_deep_agent(..., store=store)` に渡し、`StoreBackend` 側はランタイムから `store` を解決しつつ `namespace` 関数だけを受け取る**アーキテクチャへ刷新されていた（参考：[langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)）。
3. `StoreBackend(namespace=lambda rt: ("memories",))` と `create_deep_agent(..., backend=..., store=store)` の組み合わせに書き換えたところ、エラーが解消し、Lab 2 が **7.23 秒**で正常完了した。

### 根本原因

DeepAgents の進化に伴い、ストレージの実体（`BaseStore`）のライフサイクル管理は LangGraph のグラフ実行基盤（`create_deep_agent` が内部でコンパイルする StateGraph）に一元化されました。

これにより、マルチテナントやユーザー単位の分離（例：`rt.config["configurable"]["user_id"]` に応じた動的名前空間）を安全に実現できるよう、`StoreBackend` は「どの名前空間に読み書きするか」を決める **キーワード専用引数 `namespace`** を必須とし、実際の `store` 参照は実行時の `ToolRuntime` 経由で自動取得する設計に変更されたのです。

### まとめ・再発防止策

`deepagents 0.7.x` 以降で `StoreBackend` を組む際は、以下の **「`store` はエージェントへ、`namespace` は `StoreBackend` へ」** という分離パターンをテンプレート化してください。

```python
import os
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend

# LLM クライアント（キーは環境変数から取得）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    timeout=45.0,
    max_retries=2,
)

# 1. Store 実体はエージェント生成時に渡すために用意する
store = InMemoryStore()

# 2. StoreBackend には store ではなく、キーワード専用引数 namespace を渡す
#    動的テナント分離したい場合は lambda rt: (rt.config["configurable"]["user_id"], "memories") のように書ける
backend = CompositeBackend(
    default=FilesystemBackend(root_dir="./workspace", virtual_mode=True),
    routes={
        "/memories/": StoreBackend(namespace=lambda rt: ("memories",)),
    },
)

# 3. create_deep_agent に backend と store の両方を渡す
agent = create_deep_agent(
    model=llm,
    backend=backend,
    store=store,
)
```

---

## 落とし穴 2：`CompositeBackend` のプレフィックス剥離挙動（`/memories/pref.txt` → `/pref.txt`）

### 問題

落とし穴 1 を乗り越えて `CompositeBackend` で `/notes.txt`（ローカルディスク）と `/memories/pref.txt`（`InMemoryStore`）の両方へファイルを書き込んだあと、検証コードで `store.get(("memories",), "/memories/pref.txt")` を呼んだところ、**`None` が返ってきてテストが落ちました**。

しかしエージェントの最終応答では以下のように書き込み成功を報告しています。

```text
完了しました。
- /notes.txt に「DeepAgents CompositeBackend OK」を書き込みました。
- /memories/pref.txt に「lang=ja」を書き込みました。
```

### 発生条件

- `CompositeBackend(default=..., routes={"/memories/": StoreBackend(...)})` のようにプレフィックスルーティングを構成している
- エージェントが `write_file("/memories/pref.txt", "lang=ja")` を実行した直後に、テストコードやバックエンド監視側で `store.search(...)` や `store.get(...)` を使ってキーを直接参照する場合

### 試行錯誤

1. 「`write_file` が実際には呼ばれず、LLM が嘘をついただけか？」と疑ったが、`tools_used` を確認すると `["write_file", "write_file"]` が確実に 2 回実行されており、ディスク側の `disk_exists` も `True`（`"DeepAgents CompositeBackend OK"`）だった。
2. そこで `store.search(("memories",))` で名前空間内の全アイテムをダンプしてみたところ、実際に保存されていたキー一覧（`store_keys`）は次の通りだった。

```json
"store_keys": [
  "/pref.txt"
]
```

`/memories/pref.txt` ではなく、**ルートプレフィックス `/memories` が剥離された `/pref.txt`** として保存されていたのです。

### 根本原因

`CompositeBackend` は UNIX のマウントポイント（`mount`）と同じ思想で設計されています。`routes={"/memories/": sub_backend}` と定義した場合、`/memories/` 配下へのアクセスは、マウントプレフィックスを取り除いた相対的なルートパス（`/pref.txt`）に変換されて配下の `StoreBackend` に委譲されます。

これにより、`StoreBackend` 側は自分が `/memories/` にマウントされているのか `/user_data/` にマウントされているのかを意識せずに済む（疎結合になる）わけですが、**テストコードや運用スクリプトから `store` を直接覗くときにはプレフィックス剥離を知っていないと必ず空振り**します。

{% mermaid %}
flowchart LR
    Agent["DeepAgent<br/>write_file 呼び出し"] --> CB["CompositeBackend<br/>ルーティング層"]

    CB -- "パス: /notes.txt<br/>(routes に不一致)" --> FS["default: FilesystemBackend<br/>実ファイル: ./workspace/notes.txt"]
    CB -- "パス: /memories/pref.txt<br/>('/memories/' に一致 → プレフィックス剥離)" --> SB["routes['/memories/']: StoreBackend<br/>namespace=('memories',)<br/>保存キー: '/pref.txt'"]

    style Agent fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style CB fill:#fef3c7,stroke:#d97706,color:#0f172a
    style FS fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style SB fill:#ede9fe,stroke:#7c3aed,color:#0f172a
{% endmermaid %}

### まとめ・再発防止策

- `CompositeBackend` 経由の書き込みを検証する際は、**`store.get(ns, "/memories/pref.txt")` ではなく `store.get(ns, "/pref.txt")`（または `store.search(ns)` での一覧確認）**を使う。
- 複数ルートを同じ `InMemoryStore` に相乗りさせる場合は、キー衝突を防ぐために `StoreBackend(namespace=lambda rt: ("memories",))` のように**ルートごとに異なる `namespace` を割り当てる**のが安全です。

```python
# Store 側の実キーを確認するデバッグ用スニペット
items = store.search(("memories",))
store_keys = [item.key for item in items]
print("Store に保存された実キー一覧:", store_keys)  # -> ['/pref.txt']

# 特定のキーを取得する場合もプレフィックス剥離後のキーを指定する
pref_item = store.get(("memories",), "/pref.txt")
assert pref_item is not None
```

---

## 落とし穴 3：Large Tool Result Offloading が 600 行で発火せず、1,400 行（約 120KB）で MoE レイテンシが急増した問題

### 問題

DeepAgents Lab 3（`lab3_context_offload`）では、`FilesystemMiddleware` による **Large Tool Result Offloading（巨大なツール出力をコンテキストからファイルシステムへ自動退避する機能）** を実証しようとしました。

ところが、最初に用意した **600 行のダミーログ**ではオフロードが一切発火せず、ログ全文がそのまま LLM のコンテキストに流し込まれました。そこで発火条件を超えるよう **1,400 行（約 120KB）のログ**へ増量したところ、今度は無事にオフロードが発火したものの、ラボ全体の実行時間が **63.33 秒**（他ラボの 10〜30 倍）にまで跳ね上がりました。

### 発生条件

- `FilesystemMiddleware` のデフォルト設定（オフロード閾値：**20,000 トークン**）のまま巨大ログを返すツールを実行する
- ログ行数が少ない（600 行程度）と 20,000 トークン未満に収まり、オフロードされずにプロンプトを圧迫する
- 逆に 1,400 行（約 120KB）まで増やすとオフロードは発火するが、その前後処理やファイル探索ステップの往復、さらにコンテキストに残るプレビュー＋推論負荷により、MoE モデル（`MiniCPM-V-4.7-35B-A3B`）でも応答レイテンシが大きく伸びる

### 試行錯誤

1. 最初は「600 行のログなら十分大きいだろう」と考え `dump_large_log()` を実行したが、State の `files` 辞書に `/large_tool_results/...` が作成されず、ツール出力がそのまま `ToolMessage` に入っていた。
2. `FilesystemMiddleware` の実装を確認したところ、オフロード判定は「行数」ではなく **トークン数換算（デフォルト 20,000 トークン、文字数換算で約 80,000 文字前後）** で行われていた。
3. ログを 1,400 行（約 120KB）に増やして再実行したところ、ついにオフロードが発火！ State に `/large_tool_results/chatcmpl-tool-954917c8055113ae` が生成され、`ToolMessage` には以下の退避案内だけが残った。

```text
Tool result too large, the result of this tool call chatcmpl-tool-954917c8055113ae
was saved in the filesystem at this path: /large_tool_results/chatcmpl-tool-954917c8055113ae

You can read the result from the filesystem ...
```

4. その後、エージェントは自律的に `grep` ツールを呼び出し、1,400 行の中から唯一のエラー行 `2026-07-20T10:59:59Z ERROR billing-api CRITICAL_TIMEOUT db_pool_exhausted active=150/150` を正確に抽出した（`tools_used: ["dump_large_log", "grep"]`）。
5. ただし全体の所要時間が **63.33 秒**かかったため、タイムアウトを短くしすぎていた場合（例：25 秒）は途中で失敗していたことが判明。

### 根本原因

- **閾値ギャップ**：デフォルトの 20,000 トークンという閾値は、128K〜200K コンテキストを持つ超大型商用モデルを前提にした値です。セルフホストの 35B MoE モデル（プリフィル計算量や KV キャッシュ帯域に制約がある）にとっては、**10,000〜19,000 トークンの「オフロードされない巨大テキスト」が最もレイテンシを悪化させる危険地帯**になります。
- **オフロード後の複数ステップ往復**：オフロードが発火すると、`dump_large_log` →（退避通知）→ `grep` / `read_file` → 最終回答、と最低でも 3 回の LLM 推論が直列に走るうえ、巨大ファイルの生成・シリアライズ・検索コストも加算されます。

### まとめ・再発防止策

セルフホスト MoE モデルで DeepAgents を運用する場合は、デフォルトの 20,000 トークンを鵜呑みにせず、**モデルの快適なプリフィル上限に合わせてオフロード閾値を低め（例：4,000〜8,000 トークン）にチューニング**するか、ツール側で最初からページネーション・フィルタリング引数を提供するのがベストプラクティスです。また、ログ解析系エージェントでは `ChatOpenAI(timeout=90.0)` のように十分なタイムアウトを確保します。

```python
from langchain_core.tools import tool

# ベストプラクティス: ツール側で最初からフィルタリングできるようにし、
# それでも巨大な場合だけ DeepAgents のオフロードに委ねる
@tool
def query_service_logs(service: str, level: str = "ERROR", limit: int = 50) -> str:
    """指定サービスのログをレベルで絞り込んで最大 limit 行だけ返す。"""
    # 実運用では DB / Loki / CloudWatch 等にクエリを投げる
    return (
        f"2026-07-20T10:59:59Z {level} {service} "
        "CRITICAL_TIMEOUT db_pool_exhausted active=150/150"
    )
```

---

## 落とし穴 4：`HumanInTheLoopMiddleware` / `interrupt()` の必須条件と再実行時の冪等性

### 問題

LangChain Lab 6（`lab6_guardrails_hitl`）、LangGraph Lab 5（`lab5_interrupt_timetravel`）、DeepAgents Lab 6（`lab6_permissions_hitl`）の 3 つのラボで人間承認（Human-in-the-Loop）を実装した際、以下の 3 つの罠に連続して遭遇しました。

1. `checkpointer` を付けずに `interrupt()` や `HumanInTheLoopMiddleware` を使うと、中断状態を保存できず実行時エラーになる。
2. `config={"configurable": {"thread_id": "..."}}` を渡さない、あるいは再開時に別の `thread_id` を渡すと中断箇所から再開できない。
3. **`Command(resume=...)` で再開したとき、中断したノードの「途中から」ではなく「ノードの先頭から」再実行される**ため、副作用のある処理（DB 書き込み・メール送信・カウンタ加算など）を `interrupt()` より前に書いていると 2 回実行されてしまう。

### 発生条件

- `HumanInTheLoopMiddleware(interrupt_on={...})` や LangGraph の `interrupt(...)` を利用している
- `checkpointer=MemorySaver()`（本番は Postgres / SQLite）または `thread_id` の設定漏れ
- 1 つのノード関数内で `send_notification()` → `interrupt()` → `execute_deploy()` のように、`interrupt()` の手前に非冪等な副作用を配置している
- `HumanInTheLoopMiddleware` への再開ペイロード形式（`{"decisions": [{"type": "approve"}]}` や `{"decisions": [{"type": "reject", "message": "..."}]}`）を誤った場合

### 試行錯誤

1. まず LangGraph Lab 5 で `interrupt()` を呼ぶノードの先頭にデバッグログを仕込んだところ、初回実行で 1 回、さらに `Command(resume=True)` での再開時にもう 1 回、**合計 2 回ノード先頭が通過する**ことを実測で確認した。
2. DeepAgents Lab 6（`delete_production_table` の拒否フロー）では、中断時のペイロード構造を確認したうえで `Command(resume={"decisions": [{"type": "reject", "message": "本番データベースの削除は禁止されています。"}]})` を渡したところ、ツールを実行せずにエージェントが安全に拒否応答（`"本番データベースの削除は禁止されています。temp_logs テーブルの削除は実行できません。"`）を返した（所要時間 **1.93 秒**）。
3. LangChain Lab 6（`deploy_release` の承認フロー）でも同様に `Command(resume={"decisions": [{"type": "approve"}]})` を渡すことで、`v2.4.0` の `production` デプロイが承認後にのみ 1 回だけ実行されることを確認した。

### 根本原因

LangGraph の `interrupt()` は Python のスレッドやコルーチンをメモリ上で凍結する仕組みではありません。**「チェックポイントに現在の State を保存してグラフ実行を一度終了し、`Command(resume=value)` が届いたらそのノードを先頭から再実行する。2 回目の実行時は `interrupt()` が例外的に中断せず、代わりに `value` を戻り値として返す」** というリプレイモデル（Replay Model）で動作します。

だからこそ、`checkpointer` と `thread_id` が絶対必須であり、かつ **`interrupt()` より手前のコードは必ず冪等（副作用なし）である必要がある**のです。

### まとめ・再発防止策

- `interrupt()` を使うノードでは、**`interrupt()` の前に外部副作用を絶対に書かない**（副作用は必ず `interrupt()` の後、または別ノードに分離する）。
- `HumanInTheLoopMiddleware` を使う場合は、`interrupt_on` の対象ツールに対して `{"decisions": [{"type": "approve" | "reject" | "edit", ...}]}` のスキーマを厳守する。

```python
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


def review_node(state: dict) -> dict:
    """承認ゲート用ノード。interrupt() より前に副作用を置かないのが鉄則。"""
    # OK: 純粋な読み取りやペイロード組み立てのみ行う
    proposal = state["proposal"]

    # 初回はここで中断し、再開時はノード先頭から再実行されて resume 値が返る
    approved = interrupt({
        "question": "このプランを承認しますか？",
        "proposal": proposal,
    })

    # OK: 承認判定後の状態更新や副作用は interrupt() の後に配置する
    if approved:
        return {"status": "approved", "result": f"EXECUTED: {proposal}"}
    return {"status": "rejected", "result": "CANCELLED"}


# 再開時は必ず初回と同じ thread_id を指定する
config = {"configurable": {"thread_id": "deploy-req-001"}}
# 1回目: 中断（__interrupt__ が返る）
# paused = graph.invoke({"proposal": "Plan: DBインデックス追加"}, config=config)
# 2回目: 承認して再開
# resumed = graph.invoke(Command(resume=True), config=config)
```

---

## 落とし穴 5：LangGraph Time Travel で最新チェックポイントではなく `h.next == ("review",)` を選ぶ理由

### 問題

LangGraph Lab 5（`lab5_interrupt_timetravel`）で、一度 `EXECUTED: Plan: DBインデックス追加` まで完了した実行履歴から過去へ巻き戻し（Time Travel）、プランを `Plan: 読み取りレプリカ増設（フォーク版）` に差し替えて別ルートを再実行しようとしました。

このとき、直感的に `list(graph.get_state_history(config))[0]`（最新のチェックポイント）を `update_state()` して `invoke(None, fork_cfg)` を呼んでも、**グラフはすでに `END` に到達しているため何も再実行されず、古い結果のまま終了してしまいます**。

### 発生条件

- `get_state_history(config)` の戻り値（新しい順＝降順で並ぶ）の先頭要素をそのまま Time Travel の分岐点として選んでしまった場合
- あるいは「`review` ノードを実行し直したい」のに、`review` 実行**後**のチェックポイントを選んでしまった場合

### 試行錯誤

1. Lab 5 の全チェックポイント履歴（`total_checkpoints: 5`）をステップ番号と `h.next`（次に実行予定のノードタプル）付きでダンプしてみた。
   - 最新（Step 4）：`next=()` → すでに完了済み（ここをフォークしても `invoke(None)` は何もしない）
   - 途中（Step 2）：`next=("review",)` → **`draft` が終わり、まさにこれから `review` ノードへ入ろうとしている直前の状態**
2. そこで `target = next(h for h in history if h.next == ("review",))` で分岐点を特定し、`fork_cfg = graph.update_state(target.config, {"proposal": "Plan: 読み取りレプリカ増設（フォーク版）"})` で新しい分岐チェックポイントを作成。
3. 続けて `graph.invoke(None, fork_cfg)` を実行すると、見事に `review` ノードの `interrupt()` で再度停止し、`Command(resume=True)` を渡すことで `fork_result: "EXECUTED: Plan: 読み取りレプリカ増設（フォーク版）"` を得ることができた（全体所要時間 **0.01 秒**、LLM 不要の純粋グラフ制御）。

### 根本原因

LangGraph のチェックポイントは「あるスーパーステップが完了し、**次に実行すべきノード集合 `next` が予約されているスナップショット**」です。

- `update_state(checkpoint_config, values)` は、そのチェックポイントの State を書き換えた**新しい分岐チェックポイント（`fork_cfg`）**を生成します（元の履歴は破壊されない）。
- その直後に `graph.invoke(None, fork_cfg)` を呼ぶと、LangGraph は `fork_cfg` に記録されている **`next` ノードから実行を再開**します。
- したがって、最新チェックポイント（`next == ()`）をフォークしても走るノードが存在せず、`review` ノードを新しい入力でやり直したいなら **`h.next == ("review",)` のチェックポイントを選ぶ**必要があるのです。

{% mermaid %}
flowchart TD
    S0["Step 0: START<br/>next=('draft',)"] --> S1["Step 1: draft 完了<br/>proposal='DBインデックス追加'<br/>next=('review',)"]
    S1 --> S2["Step 2: review 中断<br/>interrupt() 待機"]
    S2 -- "Command(resume=True)" --> S3["Step 3: review 完了<br/>result='EXECUTED: DBインデックス追加'<br/>next=()"]

    S1 -. "get_state_history() で<br/>h.next == ('review',) を選択<br/>→ update_state() でフォーク" .-> F1["Fork Checkpoint<br/>proposal='読み取りレプリカ増設（フォーク版）'<br/>next=('review',)"]
    F1 -- "invoke(None, fork_cfg)<br/>→ Command(resume=True)" --> F2["Fork 完了<br/>result='EXECUTED: 読み取りレプリカ増設（フォーク版）'<br/>next=()"]

    style S1 fill:#fef3c7,stroke:#d97706,color:#0f172a
    style S3 fill:#e2e8f0,stroke:#64748b,color:#0f172a
    style F1 fill:#ede9fe,stroke:#7c3aed,color:#0f172a
    style F2 fill:#dcfce7,stroke:#16a34a,color:#0f172a
{% endmermaid %}

### まとめ・再発防止策

Time Travel を実装する際は、インデックス番号（`history[1]` など）をハードコードせず、**「どのノードの直前からやり直したいか」を `h.next` で宣言的に検索する**のが安全です。

```python
from langgraph.types import Command

# 1. 指定 thread_id のチェックポイント履歴を新しい順に取得
history = list(graph.get_state_history(config))
print(f"総チェックポイント数: {len(history)}")  # Lab 5 実測: 5

# 2. 最新 (next=()) ではなく、review ノード実行直前のスナップショットを特定する
target_snapshot = next(
    h for h in history if h.next == ("review",)
)

# 3. 過去のチェックポイントから State を書き換えて新しい分岐を作成
fork_cfg = graph.update_state(
    target_snapshot.config,
    {"proposal": "Plan: 読み取りレプリカ増設（フォーク版）"},
)

# 4. None 入力でフォーク地点 (next=('review',)) から再実行 → 再度 interrupt で停止
graph.invoke(None, fork_cfg)

# 5. フォーク版を承認して完了させる
fork_out = graph.invoke(Command(resume=True), fork_cfg)
print(fork_out["result"])
# -> EXECUTED: Plan: 読み取りレプリカ増設（フォーク版）
```

---

## 全 21 ラボ実践を通じた設計総括

Part 1・Part 2 を通じて、LangChain v1.5.2 / LangGraph / DeepAgents 0.7.41 の 3 層スタックで実際に踏んだ 10 個の落とし穴をすべて解剖してきました。最後に、本番グレードの AI エージェントを設計するうえでの 5 つの指針として締めくくります。

1. **モデルと API 境界を最初に固める**：スモークテスト（基本応答＋Tool Calling）、`timeout`/`max_retries`、`InMemoryRateLimiter`、そして互換エンドポイント向けの `ToolStrategy` は全プロジェクト共通のスターターキットにする。
2. **ライブラリの破壊的変更は `inspect.signature` で即確認する**：特に急速に進化する `deepagents` では、`StoreBackend(namespace=...)` のように最新のシグネチャを直接確認する習慣が最短ルートになる。
3. **仮想パスと実ストレージキーの境界を意識する**：`CompositeBackend` のプレフィックス剥離のように、抽象化レイヤーが変換する境界を把握しておけば、テストや運用監視で迷わない。
4. **コンテキストサイズと推論レイテンシは常にセットで設計する**：巨大出力のオフロード閾値はデフォルト値（20,000 トークン）を盲信せず、採用モデルの特性に合わせて調整する。
5. **状態永続化は「リプレイモデル」で考える**：LangGraph の `interrupt()` も Time Travel も、すべて「チェックポイント＋`next` ノード＋冪等なノード再実行」という一貫した原理の上に成り立っている。この原理さえ押さえれば、承認フローもやり直し（フォーク）も驚くほどシンプルに組める。

これから LangChain / LangGraph / DeepAgents の 3 層スタックに挑戦される方が、本記録によって同じ落とし穴を華麗に回避できれば幸いです。
