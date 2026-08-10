---
title: "DeepAgents Part 2 実践編：CompositeBackend によるローカルディスク×StoreBackend 経路分割と namespace 設計の実証"
date: 2026-07-07 22:24:49
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, CompositeBackend, StoreBackend, FilesystemBackend, Python]
lang: ja
description: "CompositeBackend で FilesystemBackend（ローカルディスク）と StoreBackend（LangGraph BaseStore）を統合し、1 回のエージェント実行で経路分割される様子を実測検証。StoreBackend のキーワード専用 namespace 必須仕様と /memories/ プレフィックス除去の挙動を詳しく解析します。"
---

# DeepAgents Part 2 実践編：`CompositeBackend` によるローカルディスク×`StoreBackend` 経路分割と `namespace` 設計の実証

前回の **Part 2 理論編** では、DeepAgents の仮想ファイルシステムを支える `BackendProtocol` と、`StateBackend`・`FilesystemBackend`・`StoreBackend`・`CompositeBackend` の役割分担およびプレフィックスルーティングの仕組みを解説しました。

本記事（Part 2 実践編）では、OpenAI 互換 MoE モデル **`MiniCPM-V-4.7-35B-A3B`** を用いて、**`default=FilesystemBackend(virtual_mode=True)`** と **`routes={"/memories/": StoreBackend(namespace=...)}`** を組み合わせた `CompositeBackend` を実装・検証します。また、実装時に実際に遭遇した **`StoreBackend` の `namespace` キーワード専用引数エラー** と、実行後の `store.search()` で **`/memories/` プレフィックスが剥がれて `"/pref.txt"` として保存される現象** を実測データとともに解説します。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、`StoreBackend` の現行 API と `CompositeBackend` のルーティング仕様について次のように整理しました。

> **新 API（現行標準）**: Backend インスタンスを直接渡し、ランタイム情報（`get_store()`, `get_runtime()`）は内部で自動解決する。`StoreBackend` には **キーワード専用引数 `namespace`** として `NamespaceFactory = Callable[[Runtime], tuple[str, ...]]` を渡す。

> エージェントが `/memories/pref.txt` へ書き込むと、`CompositeBackend` はマッチしたルートプレフィックスを取り除いた **`/pref.txt`** を子バックエンド（`StoreBackend` や `FilesystemBackend`）へ渡します。一方、エージェントが `ls("/")`・`glob`・`grep` を実行した際には、全マウント先の結果を集約し、元のプレフィックス（`/memories/pref.txt`）を復元して返します。

今回の実践ラボ（Lab 2）では、1 回のユーザー指示で次の 2 ファイル書き込みを行わせ、物理的な保存先が完全に分離されることを実測確認します。

1. `/notes.txt` に `"DeepAgents CompositeBackend OK"` を書き込む → デフォルト経路の **`FilesystemBackend`**（一時ローカルディレクトリ）に実ファイルとして作成されること。
2. `/memories/pref.txt` に `"lang=ja"` を書き込む → マウント経路の **`StoreBackend`**（`InMemoryStore` の `("user", "demo")` 名前空間）へ格納されること。

---

## 2. 実験アーキテクチャと実装コード（Lab 2）

今回の経路分割アーキテクチャと、書き込み先・検証方法の対応図を以下に示します。

{% mermaid %}
flowchart LR
    U["ユーザー指示<br/>1) /notes.txt へ書き込み<br/>2) /memories/pref.txt へ書き込み"] --> AGENT["create_deep_agent<br/>(MiniCPM-V-4.7-35B-A3B)<br/>store=InMemoryStore()"]
    AGENT -->|"write_file #1<br/>/notes.txt"| COMP["CompositeBackend"]
    AGENT -->|"write_file #2<br/>/memories/pref.txt"| COMP

    COMP -->|"デフォルト経路一致"| FS["FilesystemBackend<br/>(root_dir=tmp_dir, virtual_mode=True)"]
    COMP -->|"ルート '/memories/' 一致<br/>プレフィックス除去 → '/pref.txt'"| SB["StoreBackend<br/>(namespace=lambda rt: ('user', 'demo'))"]

    FS --> DISK["ローカルディスク<br/>tmp_dir/notes.txt<br/>(os.path.exists で検証)"]
    SB --> STORE["InMemoryStore<br/>namespace=('user', 'demo')<br/>key='/pref.txt' (store.search で検証)"]

    style AGENT fill:#ebf8ff,stroke:#2b6cb0,stroke-width:2px,color:#1a365d
    style COMP fill:#faf5ff,stroke:#6b46c1,stroke-width:2px,color:#44337a
    style FS fill:#fefcbf,stroke:#b7791f,stroke-width:1.5px,color:#744210
    style SB fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style DISK fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
    style STORE fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

### 実装コード（Lab 2：修正後の正常動作版）

API キー・URL は `.env` から `os.getenv("BASE_URL")` / `os.getenv("API_KEY")` で読み込みます。一時ディレクトリは `try ... finally` で確実に後片付けします。

```python
import json
import os
import shutil
import tempfile
import time
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StoreBackend,
)

# Pydantic 互換性警告を抑止し、環境変数をロード（APIキーのハードコードは厳禁）
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
    max_tokens=512,
    timeout=40,
    max_retries=2,
)


def run_lab2():
    t0 = time.perf_counter()
    # 検証用の隔離された一時ディレクトリを作成
    tmp_dir = tempfile.mkdtemp(prefix="da_lab2_")
    try:
        # LangGraph の長期ストアを準備（本番では PostgresStore / RedisStore 等を利用）
        store = InMemoryStore()

        # CompositeBackend でパスごとの保存先を振り分ける
        # - デフォルト: ローカルディスク（必ず virtual_mode=True でパス脱出を防止）
        # - /memories/: LangGraph BaseStore（キーワード専用引数 namespace が必須）
        composite = CompositeBackend(
            default=FilesystemBackend(root_dir=tmp_dir, virtual_mode=True),
            routes={
                "/memories/": StoreBackend(
                    namespace=lambda rt: ("user", "demo"),
                )
            },
        )

        # store インスタンスは StoreBackend ではなく create_deep_agent(..., store=store) に渡す
        agent = create_deep_agent(
            model=llm,
            backend=composite,
            store=store,
            system_prompt="ファイルツールを使って作業し、完了したら日本語で簡潔に報告してください。",
        )

        # 1 回のリクエストで 2 つの異なるマウント先への書き込みを指示
        res = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "/notes.txt に 'DeepAgents CompositeBackend OK' と書き込み、"
                            "次に /memories/pref.txt に 'lang=ja' と書き込んでください。"
                        ),
                    }
                ]
            }
        )

        # 1. ローカルディスク側 (/notes.txt) の実在と内容を検証
        notes_path = os.path.join(tmp_dir, "notes.txt")
        disk_exists = os.path.exists(notes_path)
        disk_content = (
            open(notes_path, encoding="utf-8").read() if disk_exists else None
        )

        # 2. LangGraph Store 側 (/memories/pref.txt) の保存キーを直接検索して検証
        store_items = [item.key for item in store.search(("user", "demo"))]

        return {
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "disk_exists": disk_exists,
            "disk_content": disk_content,
            "store_keys": store_items,
            "tools_used": [
                tc["name"]
                for m in res["messages"]
                for tc in getattr(m, "tool_calls", [])
            ],
            "final": res["messages"][-1].content,
        }
    finally:
        # テスト用ディレクトリを確実にクリーンアップ
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run_lab2(), indent=2, ensure_ascii=False))
```

---

## 3. 実測出力の解析：`store_keys: ["/pref.txt"]` の意味

実際に実行して得られた生出力は以下の通りです（所要時間 **7.23 秒**）。

```json
{
  "lab2_composite_backend": {
    "status": "ok",
    "elapsed_s": 7.23,
    "output": {
      "disk_exists": true,
      "disk_content": "DeepAgents CompositeBackend OK",
      "store_keys": [
        "/pref.txt"
      ],
      "tools_used": [
        "write_file",
        "write_file"
      ],
      "final": "完了しました。\n\n- /notes.txt に「DeepAgents CompositeBackend OK」を書き込みました。\n- /memories/pref.txt に「lang=ja」を書き込みました。"
    }
  }
}
```

### 挙動の裏付け（なぜこの結果になったのか）

1. **`tools_used: ["write_file", "write_file"]` による透明な操作**
   モデル側から見ると、`/notes.txt` も `/memories/pref.txt` も同じ仮想ファイルシステム上のファイルに過ぎません。モデルは裏側がローカルディスクなのか `BaseStore` なのかを一切意識せず、共通の `write_file` ツールを 2 回呼び出すだけで異種ストレージへの書き込みを完了しています。

2. **ローカルディスク側の検証（`disk_exists: true`）**
   `tmp_dir/notes.txt` が実ファイルとして生成され、内容も `"DeepAgents CompositeBackend OK"` と完全一致しました。一方で `tmp_dir/memories/` ディレクトリは作成されておらず、`/memories/` 配下の書き込みがローカルディスクに漏れていないことも確認できます。

3. **Store 側のキーが `"/memories/pref.txt"` ではなく `"/pref.txt"` になる理由**
   `store.search(("user", "demo"))` が返したキー一覧は **`["/pref.txt"]`** でした。これは、`CompositeBackend` がルート `"/memories/"` に一致した時点で **プレフィックス `"/memories"` を剥がしてから子バックエンド（`StoreBackend`）へ `write("/pref.txt", "lang=ja")` を委譲する** ためです。
   - エージェントが次回 `read_file("/memories/pref.txt")` を呼ぶと、再び `CompositeBackend` が `"/memories"` を剥がして `StoreBackend` から `"/pref.txt"` を読み出すため、エージェント視点では一貫して `/memories/pref.txt` として見えます。
   - 反対に、**管理画面やバッチ処理など外部の Python コードから `store.get(("user", "demo"), ...)` で直接読み出す場合は、プレフィックスなしの `"/pref.txt"` をキーとして指定する必要がある** ことを押さえておく必要があります。

{% mermaid %}
sequenceDiagram
    autonumber
    participant Py as アプリケーション検証コード
    participant Agent as DeepAgent
    participant Comp as CompositeBackend
    participant Store as InMemoryStore<br/>ns=("user", "demo")

    Agent->>Comp: write_file("/memories/pref.txt", "lang=ja")
    Comp->>Store: put(ns=("user", "demo"), key="/pref.txt", val="lang=ja")
    Store-->>Comp: OK
    Comp-->>Agent: "/memories/pref.txt" 書き込み成功
    Py->>Store: store.search(("user", "demo"))
    Store-->>Py: [Item(key="/pref.txt", ...)]
    Note over Py,Store: 外部から Store を直接参照すると<br/>マウント接頭辞 "/memories" は含まれない！
{% endmermaid %}

---

## 4. 注意点・ハマりどころ（Pay Attention）

Lab 2 の構築・検証を通じて実際に踏んだエラーと、本番設計で必ず押さえるべき 4 つの落とし穴をまとめます。

### ① `StoreBackend(store=store)` は NG！`namespace` はキーワード専用で必須
最初の試行では、直感的に `routes={"/memories/": StoreBackend(store=store)}` と書いてしまいましたが、DeepAgents v0.7 では **`StoreBackend` の `namespace` 引数はキーワード専用（keyword-only）かつ実質必須** であり、`store` の供給は LangGraph ランタイム側（`create_deep_agent(..., store=store)`）で行う設計になっています。

```python
# ❌ NG: namespace 未指定・旧来の store 直渡しパターン
# routes={"/memories/": StoreBackend(store=store)}

# ✅ OK: namespace ファクトリをキーワード引数で渡し、store は create_deep_agent 側へ渡す
composite = CompositeBackend(
    default=FilesystemBackend(root_dir=tmp_dir, virtual_mode=True),
    routes={"/memories/": StoreBackend(namespace=lambda rt: ("user", "demo"))},
)
agent = create_deep_agent(model=llm, backend=composite, store=store)
```

さらにマルチユーザーの本番環境では、今回のような固定タプル `("user", "demo")` ではなく、LangGraph の `Runtime` から動的にテナント ID やユーザー ID を取り出す形にしてください。

```python
# 本番マルチテナント向けの namespace ファクトリ例
routes={
    "/memories/": StoreBackend(
        namespace=lambda rt: (rt.context.tenant_id, rt.context.user_id, "memories")
    ),
}
```

### ② `CompositeBackend` のプレフィックス除去と事前シード（Pre-seeding）の罠
外部スクリプトで `store.put(("user", "demo"), "/memories/pref.txt", ...)` のように事前データ（シード）を投入してしまうと、エージェントが `read_file("/memories/pref.txt")` を呼んだ際に `StoreBackend` はキー `"/pref.txt"` を探すため **ファイルが見つかりません（File not found）**。`StoreBackend` へ外部から事前シードする場合は、マウントプレフィックスを除いた相対ルートパス（`"/pref.txt"`）をキーに揃える必要があります。

### ③ 本番では `default=FilesystemBackend(...)` を避け、`default=StateBackend()` にする
今回の Lab 2 では `/notes.txt` のディスク出力を最小構成で実証するためにあえて `default=FilesystemBackend(...)` としましたが、**本番アプリケーションでは `default=StateBackend()` にし、実ディスクは `routes={"/workspace/": FilesystemBackend(...)}` のように明示マウントするのが鉄則** です。
理由は、DeepAgents が内部的に生成する `/large_tool_results/` や `/conversation_history/` がすべて `default` バックエンドへ書き込まれるため、ローカルディスクを `default` にするとプロジェクトフォルダが内部アーティファクトで汚れてしまうからです。

### ④ `FilesystemBackend` の `virtual_mode=True` を絶対に忘れない
`FilesystemBackend(root_dir=tmp_dir)` と `virtual_mode` を省略するとデフォルト `False` になり、`root_dir` 外のファイルへのアクセス制限が効きません。ローカルディスクをマウントする際は必ず `virtual_mode=True` をセットで記述する習慣を徹底してください。

---

## 5. まとめ

本記事（Part 2 実践編）では、`CompositeBackend` を用いたローカルディスク × `StoreBackend` の経路分割を実測検証しました。

- 1 回の指示（2 回の `write_file`・所要時間 7.23 秒）で、`/notes.txt` はローカルディスクへ、`/memories/pref.txt` は `InMemoryStore` の `("user", "demo")` 名前空間へ正確に振り分けられた。
- `StoreBackend` はキーワード専用引数 `namespace` の明示と `create_deep_agent(..., store=store)` の組み合わせが必須であり、Store 内部のキーはマウントプレフィックスが除去された `"/pref.txt"` として格納される。

次回の **Part 3 理論編** では、DeepAgents の真骨頂である **Context Engineering（大容量ツール出力の自動オフロード・自動要約・プロンプトキャッシュ・Overflow リカバリ）** の内部メカニズムを徹底解説します。
