---
title: DeepAgents ソースコード解析 Part 6：SubAgentMiddleware・Permissions（HITL）・PatchToolCallsMiddleware の内部実装
date: 2026-09-29 03:44:33
categories: [AI]
tags: [Python, LLM, AI Agent, DeepAgents, Multi-Agent, ソースコード解析]
lang: ja
description: DeepAgents 連載の最終回。SubAgentMiddleware の同期委譲・実験的 fork モード（キャッシュプレフィックス共有と再帰委譲ブロック）、AsyncSubAgentMiddleware の Agent Protocol 経由バックグラウンド実行、FilesystemPermission と HumanInTheLoopMiddleware を繋ぐ _fs_interrupt、および PatchToolCallsMiddleware の宙ぶらりんツール修復機構を徹底解説する。
---

# DeepAgents ソースコード解析 Part 6：`SubAgentMiddleware`・Permissions（HITL）・`PatchToolCallsMiddleware` の内部実装

DeepAgents ソースコード解析の最終回となる Part 6 では、マルチエージェント協調と実行時の安全性・堅牢性を支える 4 つのコンポーネントをまとめて解剖します。

1. [`middleware/subagents.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py)：同期サブエージェント（`task` ツール）の状態分離・差分書き戻し・`mode="fork"` のキャッシュ共有設計
2. [`middleware/async_subagents.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/async_subagents.py)：リモート Agent Protocol サーバー上で非同期実行する `AsyncSubAgentMiddleware` のライフサイクル管理
3. [`middleware/permissions.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/permissions.py) ＆ [`middleware/_fs_interrupt.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_fs_interrupt.py)：`FilesystemPermission` の `interrupt` ルールを `HumanInTheLoopMiddleware` の動的 `when` 述語へ変換する接着層
4. [`middleware/patch_tool_calls.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/patch_tool_calls.py)：中断・キャンセル・壊れた引数で残された「宙ぶらりん `tool_call`」をエージェント起動前に修復するガード

---

## 1. `SubAgentMiddleware`：同期委譲と `mode="fork"` の精密な設計

[`SubAgentMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L1031) は、親エージェントへ `task(description, subagent_type)` ツールを提供し、専門サブエージェント（宣言的 `SubAgent` または事前コンパイル済み `CompiledSubAgent`）へタスクを委譲します。

{% mermaid %}
flowchart TD
    P["親 Agent が task(description, subagent_type) を呼び出し"] --> CHK{"_deepagents_forked_context<br/>が True か?"}
    CHK -- "Yes (再帰委譲)" --> REF["_FORK_RECURSION_REFUSAL を返却<br/>(孫サブエージェント生成をブロック)"]
    CHK -- "No" --> MODE{"subagent の mode?"}

    MODE -- "isolated (既定)" --> ISO["親 State から _EXCLUDED_STATE_KEYS と<br/>private_state_keys を除去<br/>messages = [HumanMessage(description)]"]
    MODE -- "fork (実験的)" --> FORK["_fork_messages():<br/>親の要約後有効履歴 + _FORK_TASK_PREAMBLE +<br/>独自指示 + description を構築<br/>_deepagents_forked_context = True をセット"]

    ISO --> RUN["_run_subagent() / _arun_subagent()<br/>stream_mode=['values', 'checkpoints'] で実行<br/>checkpoint_ns を捕捉"]
    FORK --> RUN

    RUN --> OUT["_return_command_with_state_update()<br/>1) 未変更キー (_is_unchanged) を除去<br/>2) structured_response or 最後の非空 AIMessage を抽出<br/>3) artifact={'checkpoint_ns': ns} 付き ToolMessage を返却"]

    style P fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style REF fill:#ffebee,stroke:#c62828,color:#b71c1c
    style FORK fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    style OUT fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

### 1.1 並列 `task` 実行での競合を防ぐ状態フィルタと `_is_unchanged`

親エージェントは 1 ターンに複数の `task` ツール呼び出しを並列発行できます。もし各サブエージェントが「親から受け取ったカスタム State フィールド」をそのまま返却すると、リデューサを持たないフィールドは LangGraph の `InvalidUpdateError` で衝突し、加算リデューサを持つフィールドは値が二重加算されてしまいます。

[`_return_command_with_state_update`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L836-L881) はこれを防ぐため、**`_EXCLUDED_STATE_KEYS`・`private_state_keys`・そして親からの入力値と変わらないキー（`_is_unchanged`）をすべて除外した差分だけ**を `Command(update=...)` で親へ書き戻します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py
state_update = {
    k: v
    for k, v in result.items()
    if k not in _EXCLUDED_STATE_KEYS
    and k not in private_state_keys
    and not _is_unchanged(k, v, inputs)
}

structured = result.get("structured_response")
if structured is not None:
    # response_format 指定時は構造化出力を JSON シリアライズして ToolMessage 本文にする
    content = structured.model_dump_json() if hasattr(structured, "model_dump_json") else json.dumps(structured)
elif refusal := _refusal_in(result["messages"][-1] if result["messages"] else None):
    content = _refusal_notice(refusal)
else:
    # Anthropic 等が最終ツール呼び出し後に空の end_turn AIMessage を出すことがあるため、
    # 末尾から遡って「最初の非空 AIMessage.text」を親への報告として採用する
    content = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            text = msg.text.rstrip() if msg.text else ""
            if text:
                content = text
                break

# クライアント UI がサブエージェントの実行ログを辿れるよう checkpoint_ns を artifact に載せる（モデルには非表示）
artifact = {"checkpoint_ns": checkpoint_ns} if checkpoint_ns else None
return Command(
    update={
        **state_update,
        "messages": [ToolMessage(content, tool_call_id=tool_call_id, artifact=artifact)],
    }
)
```

また `TaskToolSchema` の `_reject_unknown_keys` バリデータでは、LLM が誤って `description` の代わりに `prompt` などの未定義キーへ指示を書いてしまう事故を検知し、明示的な `ValidationError` をツールエラーとしてモデルへ返却する工夫も入っています。

### 1.2 `mode="fork"`：プロンプトキャッシュをヒットさせるための巧妙な仕掛け

`mode="fork"` のサブエージェントは、親の会話履歴を引き継いで作業を継続します。ここで最も面白いのが、**親の Anthropic Prompt Cache プレフィックスを最大限に再利用するための 3 つの設計**です。

1. **システムプロンプトをバイト単位で同一に保つ**：Part 1 の `graph.py` にて、fork サブエージェント自身の `system_prompt` は親のシステムプロンプトに混ぜず、プライベートキー `_FORK_INSTRUCTIONS_KEY` に退避し、タスク投入時の末尾 `HumanMessage` 側へ付与します。
2. **ツール定義配列の順序も一致させる（`_ForkTaskToolMiddleware`）**：fork 側から `task` ツールを削るとツール定義リストが親と変わりキャッシュがミスします。そこで **あえて親とまったく同じ位置・同じ説明文の `task` ツールを fork にも持たせ**、実際に呼び出されたときに `_deepagents_forked_context=True` を検知して `_FORK_RECURSION_REFUSAL` で実行時拒否します。
3. **親の要約イベントを適用済みの履歴を渡す（`_fork_messages`）**：末尾の未完了 `AIMessage(tool_calls=...)` をポップしたうえで `_DeepAgentsSummarizationMiddleware._apply_event_to_messages()` を通し、親が既に圧縮した古い履歴をそのまま引き継ぎます。

---

## 2. `AsyncSubAgentMiddleware`：Agent Protocol による非同期バックグラウンド実行

同期の `SubAgentMiddleware` が `task` 呼び出しをブロックするのに対し、[`AsyncSubAgentMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/async_subagents.py) は LangGraph SDK（`get_client` / `get_sync_client`）経由でリモート（またはローカル ASGI）の Agent Protocol サーバーへバックグラウンド実行を投入し、即座に `task_id`（= リモート `thread_id`）を返します。

公開されるツールは次の 5 つです。

- `start_async_task`：リモートスレッドを作成してバックグラウンド Run を起動（`AsyncSubAgentState.async_tasks` に追跡登録）
- `check_async_task`：ステータス（`pending` / `running` / `waiting` / `success` / `error` / `cancelled`）と結果を確認
- `update_async_task`：実行中のタスクへ追加指示を送信（`update_strategy="interrupt"` で現 Run を中断して再開、または `"enqueue"` でキュー積み）
- `cancel_async_task`：現行 Run および `superseded_run_ids` を一括キャンセル
- `list_async_tasks`：状態フィルタ付きで全バックグラウンドタスクを一覧

同一デプロイメント内（`url` 省略時）であれば、`_trace_headers()` によって LangSmith の分散トレースヘッダ（`langsmith-trace` と `ls_nest_under_parent` baggage）が自動伝播され、さらに `notify_parent=True` 時は子タスク完了時に親スレッドへ通知メッセージ（`source="async_subagent"`）を送り、アイドル状態の親エージェントを自動起床させる仕組みまで備えています。

---

## 3. Permissions と HITL を繋ぐ `_fs_interrupt.py`

[`middleware/permissions.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/permissions.py) 自体は後方互換の再エクスポートモジュールで、実体の `FilesystemPermission` は `filesystem.py` に、`HumanInTheLoopMiddleware` との連携ロジックは [`middleware/_fs_interrupt.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_fs_interrupt.py) に切り出されています。

`FilesystemMiddleware` 自体は HITL を一切知らず、`deny` ルールの拒否と検索結果からの除外だけを担当します。一方、`create_deep_agent` が呼び出す `_build_interrupt_on_from_permissions()` は、`mode="interrupt"` ルールをツールごとの **`InterruptOnConfig(when=...)` 述語** へコンパイルします。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_fs_interrupt.py
# exact ツール (read_file / write_file / edit_file): 対象パスと symlink 解決後パスの両方を評価
def _make_exact_when_predicate(rules, operation, path_arg_name, backend=None):
    def when(req: ToolCallRequest) -> bool:
        raw_path = req.tool_call.get("args", {}).get(path_arg_name)
        if not isinstance(raw_path, str):
            return False
        try:
            normalized = validate_path(raw_path)
        except ValueError:
            return False
        # 要求パスと symlink 解決先パスの双方をチェック
        decisions = {_check_fs_permission(rules, operation, path) for path in _candidate_paths(backend, normalized)}
        # 先行する deny ルールがある場合は HITL 承認を出さず、ツール側の permission denied に任せる
        return "interrupt" in decisions and "deny" not in decisions
    return when
```

一方、`ls` / `glob` / `grep` / `delete` のような **bulk スコープ** のツールでは、`_make_bulk_when_predicate()` が以下の抜け穴をすべて塞いでいます。

- `path=None`（パス未指定の一括検索）や `path="."`（`validate_path` 後に `"/"` へ正規化）は全ツリーに到達し得るため無条件で interrupt を発火。
- `glob(pattern="/secrets/**", path="/workspace")` のように、`path` は安全でも絶対 `pattern` や `..` を含む相対 `pattern` で別ツリーへ飛ぶ呼び出しを `_bulk_pattern_fires()` で捕捉して interrupt を発火。

---

## 4. `PatchToolCallsMiddleware`：宙ぶらりん `tool_call` の自動修復

エージェント実行がプロセス強制終了・タイムアウト・ユーザー中断・JSON 引数破損（`invalid_tool_calls`）などで途中停止すると、チェックポイントの `messages` に **「`AIMessage` に `tool_call` があるのに、対応する `ToolMessage` が存在しない（dangling tool call）」** 状態が残ります。このまま次回推論を再開すると、OpenAI や Anthropic の API はメッセージ構造違反（400 Bad Request）で即座に失敗します。

[`PatchToolCallsMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/patch_tool_calls.py) はわずか 52 行のシンプルな実装ながら、Base Stack に常駐してこの致命的エラーを未然に防ぎます。

{% mermaid %}
flowchart LR
    A["before_agent(state)"] --> B["全 ToolMessage から<br/>answered_ids を収集"]
    B --> C{"全 AIMessage の<br/>tool_calls + invalid_tool_calls に<br/>未回答 ID があるか?"}
    C -- "なし (正常)" --> D["None を返却 (No-op)"]
    C -- "あり (宙ぶらりん検知)" --> E["未回答 tool_call ごとに<br/>status='error' の合成 ToolMessage を<br/>該当 AIMessage 直後へ挿入"]
    E --> F["RemoveMessage(REMOVE_ALL_MESSAGES)<br/>+ patched_messages で安全に置換"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style D fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style E fill:#fff3e0,stroke:#ef6c00,color:#e65100
{% endmermaid %}

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/patch_tool_calls.py
for msg in messages:
    patched_messages.append(msg)
    if not isinstance(msg, AIMessage):
        continue
    # 通常の tool_calls だけでなく、JSON パース失敗等で生じた invalid_tool_calls も対象にする
    for tool_call in (*msg.tool_calls, *msg.invalid_tool_calls):
        tool_call_id = tool_call["id"]
        if tool_call_id is None or tool_call_id in answered_ids:
            continue
        name = tool_call["name"] or "unknown"
        if tool_call.get("type") == "invalid_tool_call":
            content = f"Tool call {name} with id {tool_call_id} could not be executed - arguments were malformed or truncated."
        else:
            content = (
                f"Tool call {name} with id {tool_call_id} did not complete - no result was recorded. "
                "It may have been cancelled or interrupted."
            )
        patched_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id, status="error"))
```

---

## 連載の総括

全 6 回にわたって DeepAgents のソースコードをコア機能ごとに読み解いてきました。振り返ると、DeepAgents の設計哲学は次の 3 本柱に集約されます。

1. **Context Engineering の多層化**：`DeltaChannel`、Large Tool Result Offloading、SHA-256 Blob CAS、非破壊イベント要約、末尾緊急クリップ、そして Skills の Progressive Disclosure まで、すべてが「コンテキストを汚さず、かつ失わない」ために噛み合っています。
2. **プロンプトキャッシュとプロバイダ制約への実戦的最適化**：Middleware スタックの配置順序、`MemoryMiddleware` の 2 段目キャッシュブレークポイント、fork サブエージェントのプレフィックス完全一致、Claude の bound reasoning 除去など、本番運用でコストと安定性を左右する泥臭い問題がフレームワーク層に封じ込められています。
3. **Fail-Closed な安全境界**：`FilesystemBackend` の `virtual_mode` + `O_NOFOLLOW`、`FilesystemPermission` の symlink 解決・再帰削除チェック・HITL `when` 述語、`SkillsMiddleware` の実行時ツールゲート、そして `PatchToolCallsMiddleware` による履歴自己修復まで、自律エージェントを安心して長時間走らせるためのガードレールが張り巡らされています。

ぜひ公式リポジトリ [`langchain-ai/deepagents`](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents/deepagents) のコードを手元で追いながら、独自のカスタム Middleware や Backend 設計の参考にしてみてください。
