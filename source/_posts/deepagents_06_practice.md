---
title: DeepAgents Part 6 実践編：危険ツール delete_production_table の割込停止と reject フィードバックによる安全制御
date: 2026-07-31 14:20:33
categories:
  - AI
tags:
  - DeepAgents
  - LangGraph
  - Human-in-the-Loop
  - interrupt_on
  - Python
  - AI Agent
lang: ja
description: DeepAgents の interrupt_on と Checkpointer を使い、本番テーブル削除ツール delete_production_table の実行直前でグラフを割り込み停止させ、Command(resume=...) による reject フィードバックで安全に阻止する実践検証レポート。
---

[Part 6 理論編](/2026/07/31/deepagents_06_theory/)では、DeepAgents の3層防御（Permissions・Human-in-the-Loop・Sandboxes）の設計思想と内部メカニズムを整理しました。本実践編では、その中核となる **Human-in-the-Loop（`interrupt_on`）** を実際に動かし、危険なツール `delete_production_table` の呼び出しが本当に実行直前で停止すること、そして `reject` フィードバックによってツール未実行のまま安全に回答が返ることを実機ログで検証します。

---

## 1. 理論編のポイント振り返り

まず、今回の実験で検証する理論編の2つの重要ポイントを確認しておきます。

> `interrupt_on` を設定して実行を中断・再開する場合、エージェントの状態（メッセージ履歴・保留中のツール呼び出し・割り込みメタデータ）を保持する **Checkpointer** と、中断前と再開時で同一の **`thread_id`** を指定することが絶対条件となる。

> `reject` を返した場合、対象ツール関数は一度も実行されないままスキップされ、指定した `message`（省略時はデフォルトの拒否メッセージ）がエージェントへフィードバックされる。これにより、モデルは拒否理由を踏まえてユーザーへ安全に回答を返すことができる。

これらが実際のランタイム（`deepagents` + LangGraph `InMemorySaver` + `Command(resume=...)`）でどのようなペイロード構造と状態遷移を描くのかを確かめます。

---

## 2. 実験アーキテクチャと検証フロー

今回のラボ（Lab 6）では、本番データベースのテーブルを削除する危険なツール `delete_production_table(table: str)` を定義し、`interrupt_on={"delete_production_table": True}` を設定した DeepAgent に削除を指示します。

{% mermaid %}
flowchart LR
    T1["Turn 1: ユーザー指示<br/>'temp_logs テーブルを削除して'"] --> A1["agent.invoke(..., thread_id='da-hitl-06')"]
    A1 --> M1["LLM (MiniCPM-V-4.7-35B-A3B)<br/>tool_call: delete_production_table(table='temp_logs')"]
    M1 --> HITL["HumanInTheLoopMiddleware<br/>interrupt_on に一致"]
    HITL -->|"状態保存"| CP[("InMemorySaver<br/>thread_id='da-hitl-06'")]
    HITL -->|"__interrupt__ 返却"| R1["r1: interrupted = True<br/>(ツール未実行で停止)"]
    R1 -->|"人間が拒否決定"| CMD["Command(resume=<br/>{'decisions': [{'type': 'reject', 'message': '本番DBの削除は禁止されています。'}]})"]
    CMD --> A2["agent.invoke(CMD, thread_id='da-hitl-06')"]
    A2 -->|"拒否理由を ToolMessage 化"| M2["LLM 最終推論"]
    M2 --> R2["r2: 安全な拒否回答を返却"]

    style HITL fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style CP fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    style R1 fill:#ffebee,stroke:#c62828,color:#b71c1c
    style R2 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

---

## 3. 実行コード（Lab 6：危険ツールの割り込み停止と `reject` 制御）

API キーやエンドポイントは環境変数（`BASE_URL`, `API_KEY`）から読み込みます。コード内のコメントで各ステップの役割を明示しています。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from deepagents import create_deep_agent

# .env から接続情報を読み込み（API キーのハードコードは厳禁）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"

def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント経由で検証用モデルを初期化する。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=40,
        max_retries=2,
    )

def run_lab6_hitl():
    llm = make_llm()

    # 本番環境で絶対に無断実行させてはならない危険ツールを定義
    @tool
    def delete_production_table(table: str) -> str:
        """本番テーブルを削除する危険なツール。"""
        # もし誤って実行された場合は、この文字列が返る
        return f"DELETED table {table}"

    # HITL の一時停止・再開には Checkpointer の指定が必須
    checkpointer = InMemorySaver()

    # interrupt_on で対象ツールに承認ゲートを設定
    agent = create_deep_agent(
        model=llm,
        tools=[delete_production_table],
        interrupt_on={"delete_production_table": True},
        checkpointer=checkpointer,
        system_prompt="指示されたツールを実行してください。",
    )

    # 中断前と再開時で同一の thread_id を指定するための config
    cfg = {"configurable": {"thread_id": "da-hitl-06"}}

    # Step 1: 危険ツールの実行を指示（ここで割り込みが発生し、ツール本体は呼ばれない）
    r1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "temp_logs テーブルを delete_production_table で削除してください。",
                }
            ]
        },
        config=cfg,
    )

    # __interrupt__ キーの有無で割り込み停止を判定
    interrupted = "__interrupt__" in r1
    print("interrupted:", interrupted)
    print("interrupt_preview:", str(r1.get("__interrupt__", ""))[:260])

    # Step 2: 明確な拒否理由を添えて Command(resume=...) で再開する
    r2 = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": "本番DBの削除は禁止されています。",
                    }
                ]
            }
        ),
        config=cfg,  # 必ず同じ thread_id の config を使う
    )

    # 拒否フィードバックを受けた LLM の最終応答を表示
    print("after_reject:", r2["messages"][-1].content)

if __name__ == "__main__":
    run_lab6_hitl()
```

---

## 4. 実測出力とペイロードの徹底解剖

上記のコードを実際に実行した際の出力（`deepagents_labs.json` の実測値、所要時間 `1.93s`）は以下の通りです。

```json
{
  "lab6_permissions_hitl": {
    "status": "ok",
    "elapsed_s": 1.93,
    "output": {
      "interrupted": true,
      "interrupt_preview": "[Interrupt(value={'action_requests': [{'name': 'delete_production_table', 'args': {'table': 'temp_logs'}, 'description': \"Tool execution requires approval\\n\\nTool: delete_production_table\\nArgs: {'table': 'temp_logs'}\"}], 'review_configs': [{'action_name': 'de",
      "after_reject": "本番データベースの削除は禁止されています。temp_logs テーブルの削除は実行できません。"
    }
  }
}
```

### なぜこの挙動になったのか？（内部動作の解析）

1. **`interrupted: true` と `Interrupt` オブジェクトの構造**
   LLM はユーザー指示に従い `delete_production_table(table="temp_logs")` の `tool_calls` を生成しました。しかし、実行グラフが `tools` ノードへ進む直前に `HumanInTheLoopMiddleware` が呼び出しを捕捉し、LangGraph の `interrupt()` を発火させました。
   返却された `r1["__interrupt__"][0].value` には、承認 UI をそのまま描画できる構造化データが含まれています。
   - `action_requests[0]["name"]`: `'delete_production_table'`
   - `action_requests[0]["args"]`: `{'table': 'temp_logs'}`
   - `action_requests[0]["description"]`: `"Tool execution requires approval\n\nTool: delete_production_table\nArgs: {'table': 'temp_logs'}"`
   - `review_configs`: 対象アクション名と許可された決定種別（今回は `True` 指定のため `approve` / `edit` / `reject` / `respond` が許可）

2. **ツール本体が未実行であることの証明**
   もし `delete_production_table` が実行されていれば、`"DELETED table temp_logs"` という結果がモデルに渡り、モデルも「削除しました」と報告したはずです。しかし `Command(resume={"decisions": [{"type": "reject", "message": "本番DBの削除は禁止されています。"}]})` を渡したことで、`HumanInTheLoopMiddleware` がツール実行をスキップし、拒否メッセージを合成 `ToolMessage` としてスレッド履歴に注入しました。

3. **LLM が拒否理由を正確に反映して終了した理由**
   拒否 `ToolMessage` を受け取った LLM は、ツールが実行されなかった事実と「本番DBの削除は禁止されています。」というポリシー理由をコンテキストとして認識し、無意味な再試行（リトライループ）に陥ることなく、**「本番データベースの削除は禁止されています。temp_logs テーブルの削除は実行できません。」**と明確に回答してグラフを正常終了しました。

---

## 5. 応用：ストリーミング UI と組み合わせた実運用パターン

本番の Web アプリケーションや CLI では、`.invoke()` ではなく `.stream(stream_mode=["messages", "updates"])` を使い、LLM の思考・出力をリアルタイム表示しつつ `updates` チャンク内の `__interrupt__` を捕捉するループ構造にします。

{% mermaid %}
stateDiagram-v2
    [*] --> Streaming: stream_input = ユーザー発話
    Streaming --> CheckInterrupt: ストリーム完了
    CheckInterrupt --> [*]: pending_interrupt が None<br/>(正常完了)
    CheckInterrupt --> ReviewUI: __interrupt__ を検出
    ReviewUI --> ResumeStream: 人間が approve / edit / reject を選択<br/>stream_input = Command(resume=...)
    ResumeStream --> Streaming: 同一 thread_id で再ストリーム
{% endmermaid %}

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command

load_dotenv()

def run_with_streaming_hitl(agent, user_prompt: str, thread_id: str) -> str:
    """トークンを逐次表示しつつ、__interrupt__ 検出時に人間の判断で再開するループ。"""
    config = {"configurable": {"thread_id": thread_id}}
    stream_input = {"messages": [{"role": "user", "content": user_prompt}]}

    while True:
        pending_interrupt = None

        # messages（トークン表示）と updates（割り込み検知）を同時購読
        for mode, data in agent.stream(
            stream_input,
            stream_mode=["messages", "updates"],
            config=config,
        ):
            if mode == "messages":
                token, _metadata = data
                if token.content:
                    print(token.content, end="", flush=True)
            elif mode == "updates" and isinstance(data, dict):
                if "__interrupt__" in data:
                    pending_interrupt = data["__interrupt__"][0]

        # 割り込みが発生しなければエージェント実行完了
        if pending_interrupt is None:
            print()
            break

        # 承認待ちアクション一覧を表示し、同じ順序で decisions を構築
        payload = pending_interrupt.value
        action_requests = payload["action_requests"]
        decisions = []
        for req in action_requests:
            print(f"\n[承認要求] Tool={req['name']} Args={req['args']}")
            # 実運用ではここで UI や標準入力から判断を取得する
            decisions.append(
                {
                    "type": "reject",
                    "message": f"{req['name']} の実行は運用ポリシーにより却下されました。再試行せずユーザーに報告してください。",
                }
            )

        # 次のループでは Command(resume=...) を入力として同一スレッドを再開
        stream_input = Command(resume={"decisions": decisions})
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

実務で `interrupt_on` や `permissions` を導入する際、特につまずきやすいポイントをまとめます。

1. **Checkpointer 未指定だと割り込み再開ができない**
   `interrupt_on` や `FilesystemPermission(mode="interrupt")` を設定したにもかかわらず `checkpointer` を渡し忘れると、一時停止状態を保存できずエラーになります。ローカル検証は `InMemorySaver()`、本番環境では PostgreSQL 等の永続 Checkpointer（または LangSmith Agent Server の自動 Checkpointer）を必ず接続してください。
2. **再開時に異なる `thread_id` を渡してしまうミス**
   `Command(resume=...)` は「その `thread_id` の最新チェックポイントに保存された保留中の割り込み」を復元します。リクエストごとに新しい UUID を生成してしまうと再開対象が見つからず失敗します。フロントエンドとバックエンド間で `thread_id` を確実に引き回す設計が必須です。
3. **`decisions` の件数・順序不一致**
   LLM が1ターンで複数の要承認ツールを同時呼び出しした場合、`action_requests` には複数要素が入ります。`decisions` の長さが `action_requests` と一致しない、あるいは順序がずれていると意図しないツールが承認／拒否されたりバリデーションエラーになります。
4. **`reject` の `message` が曖昧だと LLM が再試行ループに入る**
   単に `{"type": "reject"}` だけ返すよりも、`"本番DBの削除は禁止されています。再試行せず、ユーザーに代替案を提示してください。"` のように**「なぜ却下されたか」「次にどう振る舞うべきか」**を具体的に書くと、モデルが即座に安全側へ収束します。
5. **副作用ツールの拒否に `respond` を使ってはいけない**
   `respond` は `ask_user` のように人間がツール結果を代理返却するための Decision です。削除ツール等の拒否に `respond` を使うと、モデルが「ツールが成功してその文字列を返した」と誤認する恐れがあるため、拒否には必ず `reject` を使用してください。
6. **Sandbox 併用時の `FilesystemPermission` スコープエラー**
   `CompositeBackend` のデフォルトに Sandbox を指定している場合、`paths=["/**"]` のようなルールを渡すと `NotImplementedError` になります。Sandbox 上の操作は `execute` 経由で迂回可能なため、パス権限は `/memories/**` 等の非 Sandbox ルートに限定し、Sandbox 側の危険操作は `interrupt_on={"execute": True}` や Auth Proxy で制御してください。

---

## 7. まとめ

本実践編では、`delete_production_table` に対して `interrupt_on` を適用し、以下を実機で確認しました。

- Checkpointer（`InMemorySaver`）と `thread_id` を組み合わせることで、危険ツール実行直前に `__interrupt__` が確実に発火する。
- `action_requests` と `review_configs` により、呼び出し予定のツール名・引数を構造化データとして検査できる。
- `Command(resume={"decisions": [{"type": "reject", "message": ...}]})` を返すことで、ツール本体を一度も実行せずに拒否理由をモデルへ伝達し、安全に回答を完結させられる。

次回（Part 7）では、いよいよシリーズの総仕上げとして、**本番運用設計（Streaming・Checkpointer 永続化・Rubric 自己評価・ACP / AG-UI / A2A プロトコル連携）**を理論・実践の両面から掘り下げます。
