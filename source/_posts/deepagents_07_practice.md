---
title: DeepAgents Part 7 実践編：updates ストリームで可視化する内部 Middleware ノード遷移とマルチターン記憶検証
date: 2026-08-02 20:27:04
categories:
  - AI
tags:
  - DeepAgents
  - LangGraph
  - Streaming
  - Checkpointer
  - Middleware
  - Python
  - AI Agent
lang: ja
description: DeepAgents の stream_mode=["updates"] を使って内部の Middleware ノード遷移（PatchToolCallsMiddleware.before_agent → model → _ParentNotifierMiddleware.after_agent）を実測し、Checkpointer と thread_id によるマルチターン記憶復元を検証する実践レポート。
---

[Part 7 理論編](/2026/08/01/deepagents_07_theory/)では、本番運用における Streaming・Checkpointer 永続化・Rubric 評価・標準プロトコル連携（ACP / AG-UI / A2A）のアーキテクチャを整理しました。

シリーズ最終回となる本実践編（Lab 7）では、最も基本でありながら本番デバッグや UI 構築で差がつく以下の2点を実機で検証します。

1. `agent.stream(..., stream_mode=["updates"])` を回したとき、**DeepAgents の内部で実際にどの Middleware ノードがどの順番で通過しているのか**を実測する。
2. Turn 1 で `stream()` 経由で記憶させた情報（コードネーム `Alpha-9`）が、Checkpointer と同一 `thread_id` を介して Turn 2 の `invoke()` で正確に復元されることを実証する。

---

## 1. 理論編のポイント振り返り

本ラボで実機確認する理論編の核心部分は以下の2点です。

> `stream_mode=["updates"]` でストリームを購読すると、表面上の `model` や `tools` だけでなく、DeepAgents ハーネスが内部に組み込んだ **`PatchToolCallsMiddleware.before_agent`** や **`_ParentNotifierMiddleware.after_agent`** といったライフサイクル Middleware ノードの遷移までそのまま観測できる。

> Checkpointer はステップごとにグラフ状態をスナップショット保存するため、同一の `thread_id` を渡すだけでマルチターンの会話記憶が自動的に復元されるだけでなく、ワーカークラッシュ時の途中再開（Durable Execution）やタイムトラベルによる巻き戻しデバッグも可能になる。

---

## 2. 実験フローと内部ノード遷移の設計

今回はツール呼び出しを伴わないシンプルな記憶タスク（Turn 1: `stream` で記憶 → Turn 2: `invoke` で想起）を実行し、DeepAgents の最小パス（ツール未使用時の最短ノード遷移）を観察します。

{% mermaid %}
flowchart LR
    subgraph Turn1["Turn 1: agent.stream(stream_mode=['updates'])"]
        U1["User: 'コードネーム Alpha-9 を記憶してください。'"] --> N1["PatchToolCallsMiddleware.before_agent<br/>(未完了 tool_calls の整合性チェック・修復)"]
        N1 --> N2["model<br/>(LLM 推論・記憶完了を返答)"]
        N2 --> N3["_ParentNotifierMiddleware.after_agent<br/>(実行終了フック・親通知処理)"]
    end

    N3 -->|"状態スナップショット保存"| CP[("InMemorySaver<br/>thread_id='da-prod-07'")]

    subgraph Turn2["Turn 2: agent.invoke()"]
        CP -->|"会話履歴を自動復元"| U2["User: '記憶したコードネームは何ですか？'"]
        U2 --> M2["model<br/>(履歴から 'Alpha-9' を想起)"]
        M2 --> ANS["AI: '記憶したコードネームは「Alpha-9」です。'"]
    end

    style N1 fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style N2 fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style N3 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style CP fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

---

## 3. 実行コード（Lab 7：`updates` ストリーム観測 ＋ マルチターン記憶検証）

接続設定は環境変数（`BASE_URL`, `API_KEY`）から取得します。Turn 1 で `stream_mode=["updates"]` を使って通過した全ノード名を収集し、Turn 2 で同じ `thread_id` を渡して記憶を問い合わせます。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

# .env からエンドポイントと API キーを取得（ハードコード禁止）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"

def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """検証用の OpenAI 互換チャットモデルを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=40,
        max_retries=2,
    )

def run_lab7_streaming_production():
    llm = make_llm()

    # マルチターン記憶を保持する Checkpointer（本番では PostgresSaver 等を利用）
    checkpointer = InMemorySaver()

    # 標準ハーネス構成で DeepAgent を生成
    agent = create_deep_agent(
        model=llm,
        checkpointer=checkpointer,
        system_prompt="日本語で簡潔に1文で回答してください。",
    )

    # Turn 1 と Turn 2 で共有する thread_id
    cfg = {"configurable": {"thread_id": "da-prod-07"}}

    # Turn 1: updates モードでストリーム実行し、通過した全ノード名を記録する
    # stream_mode にリストを渡しているため、各チャンクは (mode, chunk) タプルで返る
    stream_nodes = []
    for mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "コードネーム Alpha-9 を記憶してください。"}]},
        config=cfg,
        stream_mode=["updates"],
    ):
        if isinstance(chunk, dict):
            # updates の辞書キーが、そのステップで完了したグラフノード名
            stream_nodes.extend(list(chunk.keys()))

    # Turn 2: 同一 thread_id で invoke() し、Checkpointer からの記憶復元を検証
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "記憶したコードネームは何ですか？"}]},
        config=cfg,
    )

    print("stream_nodes:", stream_nodes)
    print("turn2_reply:", r2["messages"][-1].content)

if __name__ == "__main__":
    run_lab7_streaming_production()
```

---

## 4. 実測出力と内部 Middleware ノード遷移の解剖

上記スクリプトの実機実行結果（`deepagents_labs.json`、所要時間 `1.43s`）は以下の通りです。

```json
{
  "lab7_streaming_production": {
    "status": "ok",
    "elapsed_s": 1.43,
    "output": {
      "stream_nodes": [
        "PatchToolCallsMiddleware.before_agent",
        "model",
        "_ParentNotifierMiddleware.after_agent"
      ],
      "turn2_reply": "記憶したコードネームは「Alpha-9」です。"
    }
  }
}
```

### なぜこの3ノードが、この順序で現れたのか？

`stream_nodes` に記録された3つのノード名は、DeepAgents のランタイム設計（[langchain-ai/deepagents](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py)）の内部構造を如実に物語っています。

1. **`PatchToolCallsMiddleware.before_agent`（エージェント開始前フック）**
   グラフの最初に必ず通過する防御用 Middleware ノードです。もし前のターンがネットワーク断・強制キャンセル・HITL 中断などで途中終了し、履歴の中に「`AIMessage` の `tool_calls` はあるのに、対応する `ToolMessage` が欠落している（dangling tool call）」状態が残っていた場合、OpenAI 互換 API は 400 Bad Request で失敗します。このノードは LLM を呼ぶ前に履歴を走査して自動修復する役割を担います。
2. **`model`（LLM 推論ノード）**
   メインの推論ステップです。今回は「コードネーム Alpha-9 を記憶してください。」という指示に対して外部ツールを呼ぶ必要がなかったため、`tool_calls` を含まない通常の応答を生成し、`tools` ノードへ分岐せずに次の終了フェーズへ進みました。
3. **`_ParentNotifierMiddleware.after_agent`（エージェント終了後フック）**
   エージェントの実行サイクル完了時に通過する内部 Middleware ノードです。DeepAgents ではメインエージェントとサブエージェントが共通のハーネス構造を共有しており、終了時の状態整理や親エージェントへの通知・ライフサイクル連携を担うフックがグラフ末尾に配置されています。

さらに Turn 2 では、Turn 1 を `.stream()` で実行したにもかかわらず、`InMemorySaver` に `thread_id="da-prod-07"` のチェックポイントが確実に永続化されていたため、Turn 2 の `.invoke()` は Turn 1 の履歴を引き継いで **「記憶したコードネームは「Alpha-9」です。」** と即座に正答しました。

---

## 5. 本番 UI 向け：内部ノードのフィルタリングとマルチモード購読

実測で分かった通り、`stream_mode=["updates"]` には `PatchToolCallsMiddleware.before_agent` や `_ParentNotifierMiddleware.after_agent` といった内部 Middleware ノード名も含まれます。これをそのままエンドユーザー向け UI に表示するとノイズになるため、本番フロントエンドへ中継する際は**「表示対象ノードのホワイトリスト化」**を行うのが定石です。

{% mermaid %}
sequenceDiagram
    participant UI as フロントエンド UI
    participant API as バックエンド配信層
    participant Agent as DeepAgent Stream

    Agent-->>API: ("updates", {"PatchToolCallsMiddleware.before_agent": ...})
    Note over API: 内部 Middleware ノードはデバッグログのみ記録し UI には非表示
    Agent-->>API: ("messages", (AIMessageChunk("記憶しました"), meta))
    API-->>UI: テキストトークンを逐次描画
    Agent-->>API: ("updates", {"model": ...})
    API-->>UI: ステップ完了インジケータ更新
    Agent-->>API: ("updates", {"_ParentNotifierMiddleware.after_agent": ...})
    Note over API: 内部終了フックをフィルタ除外し、完了イベントを送信
    API-->>UI: RUN_FINISHED
{% endmermaid %}

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ユーザー向け UI に露出する意味のあるノード名のみを定義
VISIBLE_NODES = {"model", "tools"}

def stream_for_production_ui(agent, user_text: str, thread_id: str) -> None:
    """内部 Middleware ノードを除外しつつ、トークンと進捗を同時配信する例。"""
    config = {"configurable": {"thread_id": thread_id}}

    for mode, data in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates" and isinstance(data, dict):
            # 割り込みは最優先で捕捉する
            if "__interrupt__" in data:
                print("\n[HITL 承認待ち]", data["__interrupt__"][0].value)
                continue

            # 内部 Middleware ノード（PatchToolCallsMiddleware 等）を除外
            for node_name in data.keys():
                if node_name in VISIBLE_NODES:
                    print(f"\n[step完了] {node_name}")

        elif mode == "messages":
            token, _metadata = data
            if token.content:
                # トークンをリアルタイム出力
                print(token.content, end="", flush=True)

    print()
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

本番環境で Streaming・Checkpointer・外部プロトコルを運用する際によく踏む落とし穴をまとめます。

1. **`stream_mode` を文字列で渡すかリストで渡すかでアンパック形状が変わる**
   `stream_mode="updates"`（文字列）だと `for chunk in agent.stream(...):` になりますが、`stream_mode=["updates"]`（リスト）だと要素が1つでも `for mode, chunk in agent.stream(...):` の2要素タプルになります。さらに `subgraphs=True` を足すと `(namespace, mode, chunk)` の3要素タプルになるため、形状の取り違えによる `ValueError: too many values to unpack` に注意してください。
2. **内部 Middleware ノード名にビジネスロジックをハードコードしない**
   `_ParentNotifierMiddleware.after_agent` のように先頭にアンダースコアが付くノードは内部実装の詳細であり、バージョンアップで名称が変わる可能性があります。UI 制御や集計では `VISIBLE_NODES = {"model", "tools"}` のように安定した主要ノードをホワイトリスト参照するのが安全です。
3. **`InMemorySaver` はプロセス再起動で全記憶が消える**
   ラボ検証で使った `InMemorySaver` はメモリ保持専用です。本番で複数ワーカー運用や再起動耐性を持たせるには、PostgreSQL バックエンドの Checkpointer、または LangSmith Agent Server のマネージド永続化を使用してください。
4. **サブエージェント多用時の `recursion_limit` 不足**
   DeepAgents は自律的に計画・調査・サブエージェント委譲を繰り返すため、デフォルトのステップ上限に達して途中終了することがあります。本番呼び出しでは必要に応じて `config={"configurable": {"thread_id": ...}, "recursion_limit": 100}`（複雑なマルチサブエージェント案件ではさらに大きく）を明示設定しましょう。
5. **A2A 連携時の `contextId` フォーマット制約**
   Agent Server の A2A エンドポイントは `contextId` をそのまま LangGraph の `thread_id` として扱うため、**`contextId` は UUID 形式が必須**です。初回メッセージでは `contextId` を省略してサーバーに採番させ、2ターン目以降はその値をそのまま引き回してください（`"session-42"` のような独自文字列は `-32602 Invalid thread ID` で拒否されます）。

---

## 7. シリーズ総括

Part 1 から Part 7 までの全14記事（理論編×7・実践編×7）を通じて、DeepAgents の全貌をコードと実測ログで検証してきました。

- **Part 1**：`create_deep_agent` の基本構造と `write_todos` 計画ループ
- **Part 2**：仮想ファイルシステムと `CompositeBackend`（`FilesystemBackend` + `StoreBackend`）
- **Part 3**：コンテキストエンジニアリング（大容量ツール結果の自動オフロードと `grep` 検索）
- **Part 4**：Skills（`SKILL.md` Progressive Disclosure）と Memory（`AGENTS.md`）
- **Part 5**：Subagents（`task` ツールによる専門エージェント委譲とコンテキスト分離）
- **Part 6**：Permissions（`FilesystemPermission`）・Human-in-the-Loop（`interrupt_on`）・Sandboxes
- **Part 7**：本番運用設計（Streaming・Checkpointer・`RubricMiddleware`・ACP / AG-UI / A2A プロトコル連携）

DeepAgents は単なるプロンプトテンプレートではなく、**「コンテキスト管理・永続化・権限と人間介入・隔離実行・標準プロトコル」**を統合したエージェント実行ハーネスです。ぜひ本シリーズの検証コードをベースに、自身のプロダクト要件に合わせた本番 DeepAgent を構築してみてください。
