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
