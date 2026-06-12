---
title: "LangChain v1 Part 6 実践編：メール PII 自動マスキングと本番デプロイ承認フロー（HITL）の実装検証"
date: 2026-05-19 07:32:32
categories: [AI]
tags: [LangChain, LLM, AI Agent, Guardrails, PII, Human-in-the-Loop, Python]
lang: ja
description: "LangChain v1 の PIIMiddleware と HumanInTheLoopMiddleware を組み合わせ、メールアドレスの自動マスキングと本番環境デプロイ承認フローを実装・実行検証します。実測ログと挙動の内部解説、本番導入時のハマりどころを詳述します。"
---

本稿（LangChain v1 シリーズ Part 6 実践編）では、前回の理論編で解説した **Context Engineering・Guardrails（`PIIMiddleware`）・Human-in-the-Loop（`HumanInTheLoopMiddleware`）** を 1 つの本番想定シナリオへ統合し、実際の LLM 推論と Checkpointer を用いて挙動を検証します。

今回の検証テーマは **「担当者のメールアドレスを含む本番デプロイ指示を受け取り、(1) 入力段階での PII 自動マスキング と (2) 危険ツール実行前の一時停止・人間承認後の再開 を同時に成立させる」** パイプラインです。

---

## 1. 理論編の振り返りと今回の検証ポイント

理論編では、ガードレールと承認フローの内部動作について以下の 3 つの原則を確認しました。

> Model Context への変更（`wrap_model_call` による `request.override(...)`）は**一時的（Transient）**であり単一の LLM 呼び出しのみに作用するのに対し、`before_model` / `after_model` などのライフサイクルフックや `PIIMiddleware` によるステート更新は**永続的（Persistent）**に `AgentState`（チェックポイント）へ書き込まれます。

> `PIIMiddleware` は `strategy="redact"` で `[REDACTED_{PII_TYPE}]` へ置換し、`apply_to_input=True` によってユーザー入力が LLM やチェックポイントへ渡る前に機密情報を無害化します。決定論的ガードレールであるため低レイテンシかつ 100% の再現性で動作します。

> `HumanInTheLoopMiddleware` はツール実行前に `interrupt()` を発行してグラフ実行を一時停止し、LangGraph の永続化層（Checkpointer）へ状態を保存します。再開時は同じ `thread_id` と `Command(resume=...)` を渡すことで、`approve`・`edit`・`reject`・`respond` の4種類の意思決定を安全に反映できます。

本実践ラボ（`lab6_guardrails_hitl`）では、これら 3 点が実際にどう連携して動作するかを以下の観点で実測検証します。

1. ユーザー入力の `alice@example.com` が `AgentState["messages"][0]` 上で永続的に `[REDACTED_EMAIL]` へ書き換わっているか
2. `deploy_release` ツール呼び出しが LLM から提案された瞬間に実行が停止し、`__interrupt__` ペイロードが返却されるか
3. 同一の `thread_id` に対して `Command(resume={"decisions": [{"type": "approve"}]})` を渡したとき、正しくデプロイツールが実行されて最終応答へ到達するか

---

## 2. アーキテクチャと処理フロー

今回の実装における Middleware 積層順と、中断・再開のフローを可視化します。

{% mermaid %}
flowchart LR
    In["ユーザー入力<br/>担当者 alice@example.com ...<br/>v2.4.0 を production へデプロイ"]
    M1["Layer 1: PIIMiddleware<br/>email / strategy='redact'<br/>apply_to_input=True"]
    LLM1["LLM 推論 (1回目)<br/>deploy_release ツールを提案"]
    M2["Layer 2: HumanInTheLoopMiddleware<br/>interrupt_on={'deploy_release': True}"]
    CP[("InMemorySaver<br/>thread_id='thread-hitl-06'")]
    Pause["一時停止 (__interrupt__)<br/>action_requests & review_configs"]
    Human["人間承認<br/>Command(resume=approve)"]
    Exec["deploy_release 実行<br/>DEPLOYED v2.4.0 to production"]
    LLM2["LLM 推論 (2回目)<br/>完了報告を生成"]

    In --> M1 --> LLM1 --> M2 --> CP --> Pause
    Pause -.-> Human --> CP --> Exec --> LLM2

    style M1 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style M2 fill:#ffebee,stroke:#c62828,color:#b71c1c
    style CP fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style Pause fill:#fff8e1,stroke:#f57f17,color:#e65100
    style Exec fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
{% endmermaid %}

---

## 3. 実装コード（`lab6_guardrails_hitl`）

以下は実際に検証を実行した完全な Python コードです。エンドポイント URL と API キーは環境変数（`BASE_URL`, `API_KEY`）から読み込みます。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# .env から接続設定を読み込む（API キーのハードコード厳禁）
load_dotenv()

def make_llm(model: str = "MiniCPM-V-4.7-35B-A3B", temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント向けの ChatOpenAI インスタンスを生成する。"""
    return ChatOpenAI(
        model=model,
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=temperature,
        max_tokens=512,
        timeout=30,
        max_retries=1,
    )

def run_lab6_guardrails_hitl() -> dict:
    llm = make_llm()

    # 1. 副作用を伴う危険操作ツールを定義
    @tool
    def deploy_release(version: str, env: str) -> str:
        """指定バージョンを指定環境にデプロイする（危険操作）。"""
        return f"DEPLOYED {version} to {env} successfully."

    # 2. HITL の一時停止・再開には Checkpointer が必須
    checkpointer = InMemorySaver()

    # 3. PIIMiddleware と HumanInTheLoopMiddleware を積層してエージェントを構築
    agent = create_agent(
        model=llm,
        tools=[deploy_release],
        middleware=[
            # 前段: 入力中のメールアドレスを LLM 送信前に [REDACTED_EMAIL] へ置換
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            # 後段: deploy_release 実行前に必ず人間の承認を要求（batched モード）
            HumanInTheLoopMiddleware(interrupt_on={"deploy_release": True}),
        ],
        checkpointer=checkpointer,
        system_prompt="指示に従いツールを実行してください。",
    )

    # 4. スレッド ID を指定して初回呼び出し（PII を含むデプロイ依頼）
    cfg = {"configurable": {"thread_id": "thread-hitl-06"}}
    r1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "担当者 alice@example.com の承認済み案件として v2.4.0 を production にデプロイしてください。",
                }
            ]
        },
        config=cfg,
    )

    # 中断フラグ・割り込みペイロード・マスキング後のユーザーメッセージを取得
    interrupted = "__interrupt__" in r1
    interrupt_payload = str(r1.get("__interrupt__", ""))[:300]
    redacted_user_msg = r1["messages"][0].content

    # 5. 人間が内容を確認し、同一 thread_id に対して approve で実行を再開
    r2 = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=cfg,
    )

    return {
        "redacted_input": redacted_user_msg,
        "interrupted": interrupted,
        "interrupt_preview": interrupt_payload,
        "resumed_final": r2["messages"][-1].content,
    }

if __name__ == "__main__":
    import json
    result = run_lab6_guardrails_hitl()
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

## 4. 実測出力と詳細分析

上記コードを `MiniCPM-V-4.7-35B-A3B` に対して実行した際の実測結果（`langchain_labs.json` より抜粋、総実行時間 `39.01s`）は以下の通りです。

```json
{
  "status": "ok",
  "elapsed_s": 39.01,
  "output": {
    "redacted_input": "担当者 [REDACTED_EMAIL] の承認済み案件として v2.4.0 を production にデプロイしてください。",
    "interrupted": true,
    "interrupt_preview": "[Interrupt(value={'action_requests': [{'name': 'deploy_release', 'args': {'version': 'v2.4.0', 'env': 'production'}, 'description': \"Tool execution requires approval\\n\\nTool: deploy_release\\nArgs: {'version': 'v2.4.0', 'env': 'production'}\"}], 'review_configs': [{'action_name': 'deploy_release', 'al",
    "resumed_final": "v2.4.0 の production へのデプロイが完了しました。"
  }
}
```

### なぜこの挙動になったのか？（内部メカニズムの解説）

#### 1. `redacted_input` が状態レベルで書き換わっている理由
`r1["messages"][0].content` を確認すると、元の `alice@example.com` はすでに消去され、`[REDACTED_EMAIL]` に置換されています。これは、`PIIMiddleware` が `before_model` フックとして動作し、**LLM へリクエストを送る前に `AgentState["messages"]` を永続的（Persistent）に書き換えている**ためです。その結果、以下の 2 つのセキュリティ保証が同時に達成されます。

- 外部 LLM プロバイダへメールアドレスが送信されない
- `InMemorySaver`（本番では PostgreSQL 等の Checkpointer）に保存される会話スナップショットにも生メールが残らない

#### 2. `__interrupt__` の構造と中断タイミング
`interrupted: true` が示す通り、LLM が `deploy_release(version="v2.4.0", env="production")` を提案した直後、ツール関数本体が実行される前に `HumanInTheLoopMiddleware` の `after_model` フックが介入して `interrupt()` を発行しています。ペイロードには次の 2 要素が含まれます。

- `action_requests`: 実行予定のツール名（`deploy_release`）、引数（`{'version': 'v2.4.0', 'env': 'production'}`）、人間向けの説明文
- `review_configs`: 対象ツールに対して許可された意思決定リスト（`interrupt_on={"deploy_release": True}` のため `approve`, `edit`, `reject`, `respond` すべてが許可）

#### 3. `Command(resume=...)` による再開と最終応答
2 回目の `agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg)` では、新しいユーザー入力を渡していません。同じ `thread_id: "thread-hitl-06"` をキーに Checkpointer から中断直前のグラフ状態が復元され、承認された `deploy_release` が実行されます。ツールが返した `"DEPLOYED v2.4.0 to production successfully."` を受け取った LLM が 2 回目の推論を行い、`"v2.4.0 の production へのデプロイが完了しました。"` という最終応答を生成して完了します。

### 状態とデータ変化のタイムライン

{% mermaid %}
stateDiagram-v2
    [*] --> InputReceived: "alice@example.com を含む入力"
    InputReceived --> Sanitized: "PIIMiddleware (before_model)<br/>[REDACTED_EMAIL] へ永続置換"
    Sanitized --> ToolProposed: "LLM 推論 (1回目)<br/>deploy_release(v2.4.0, production)"
    ToolProposed --> Interrupted: "HITL (after_model)<br/>Checkpointer 保存 & __interrupt__"
    Interrupted --> ToolExecuted: "Command(resume=approve)<br/>同一 thread_id で復元・実行"
    ToolExecuted --> Completed: "LLM 推論 (2回目)<br/>デプロイ完了報告を生成"
    Completed --> [*]
{% endmermaid %}

---

## 5. 応用：却下（`reject`）や引数修正（`edit`）で再開する場合

実務の承認 UI では、`approve` だけでなく `edit`（デプロイ先を `staging` に変更）や `reject`（メンテナンス時間帯外のため却下）の分岐を実装します。

```python
from langgraph.types import Command

# ケース A: 本番ではなく staging 環境へ引数を書き換えて実行する（edit）
r_edit = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "deploy_release",
                        "args": {"version": "v2.4.0", "env": "staging"},
                    },
                }
            ]
        }
    ),
    config=cfg,
)

# ケース B: 変更凍結期間のためデプロイを却下し、理由をエージェントへ伝える（reject）
r_reject = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": "現在はリリース凍結期間のため production デプロイは禁止です。再試行せずユーザーへ報告してください。",
                }
            ]
        }
    ),
    config=cfg,
)
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

本番環境へ `PIIMiddleware` と `HumanInTheLoopMiddleware` を導入する際、特に踏みやすい落とし穴を 5 つ整理します。

1. **Checkpointer と `thread_id` の欠落による実行時エラー**
   - `HumanInTheLoopMiddleware` は Checkpointer なしでは中断状態を保存できず動作しません。開発時は `InMemorySaver()` で検証できますが、本番のマルチプロセス／コンテナ環境では **必ず `AsyncPostgresSaver` や `MongoDBSaver` などの永続 Checkpointer** を使用し、リクエストごとに一意な `thread_id` を渡してください。
2. **Middleware の登録順序（Ordering）**
   - 本ラボのように `PIIMiddleware` を `HumanInTheLoopMiddleware` より前に配置することで、承認 UI に表示されるコンテキストやチェックポイントからも PII を排除できます。また、`interrupt_mode="per_call"` を使う場合、`HumanInTheLoopMiddleware` は **`ToolRetryMiddleware` や `ToolErrorMiddleware` より前**に配置する必要があります（後ろに置くと、ツール失敗のリトライごとに人間へ再承認を求めてしまいます）。
3. **条件付き割り込み（`when` 述語）の決定性**
   - `when` 述語は中断時だけでなく**再開時にも再評価**されます。現在時刻やランダム値、外部 API の変動する状態に依存する述語を書くと、再開時に `False` を返してしまい、人間の `reject` が無視されてツールが実行される重大バグにつながります。`when` は必ず引数（`request.tool_call["args"]`）のみに基づく純粋関数にしてください。
4. **`edit` 時の大幅な引数変更によるモデル再試行**
   - `edit` でモデルが意図した内容から大きく逸脱する引数を渡すと、モデルが「意図した操作が完了していない」と判断して同じツールを再度呼び出すことがあります。`edit` は安全側の微調整（パスや環境名の修正など）に留め、方針転換が必要な場合は `reject` + 明確な指示メッセージを使うのが安全です。
5. **ストリーミング時の PII カスタム検出器の長さ上限**
   - `apply_to_output=True` でストリーミング出力を保護する場合、LangChain のストリーム変換器は末尾 **128 文字** をバッファリングしてチャンク境界をまたぐ PII を検知します。128 文字を超える長いパターンをカスタム正規表現で定義すると、先頭部分が先にクライアントへ送出されてしまうため、カスタムパターンは 128 文字以内に収まる設計にしてください。

---

## 7. まとめ

本稿では、`PIIMiddleware` と `HumanInTheLoopMiddleware` を組み合わせた実践コードを実行し、以下を実測レベルで確認しました。

- `alice@example.com` が LLM 呼び出し前かつ状態保存前に `[REDACTED_EMAIL]` へ永続置換されること
- `deploy_release` の呼び出し直前で確実に `__interrupt__` が発行され、ツール実行がブロックされること
- 同一 `thread_id` と `Command(resume={"decisions": [{"type": "approve"}]})` で安全に実行が再開され、最終応答まで完了すること

次回の **Part 7 理論編** では、単一エージェントの限界を超える **Multi-Agent パターン比較（Subagents / Handoffs / Router / Skills）** と、外部ツール連携の標準規格 **MCP（Model Context Protocol）統合アーキテクチャ** を解説します。
