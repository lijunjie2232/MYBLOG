---
title: "LangChain v1 ソースコード解析 Part 5：Built-in Middleware 解析② — HumanInTheLoopMiddleware・PIIMiddleware・ModelFallbackMiddleware / ModelRetryMiddleware"
date: 2026-09-04 17:17:38
categories: [AI]
tags: [Python, LLM, AI Agent, LangChain, Middleware, HITL, Security, ソースコード解析]
lang: ja
description: "LangChain v1 の安全性・可用性ミドルウェア 4 種（HumanInTheLoopMiddleware, PIIMiddleware, ModelFallbackMiddleware, ModelRetryMiddleware）のソースコードを精読し、batched/per_call 中断、ストリーミング PII スライディングバッファ、プロバイダ固有キャッシュ除去、GraphBubbleUp 例外透過の設計を解説する。"
---

Part 4 ではコンテキスト圧縮系の組み込みミドルウェアを解析しました。Part 5 では、本番運用のエージェントに欠かせない **「人間の承認（HITL）」「個人情報保護（PII）」「障害耐性（Fallback / Retry）」** を担う 4 つの Built-in Middleware のソースコードを横断解析します。

- [`agents/middleware/human_in_the_loop.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py)
- [`agents/middleware/pii.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/pii.py)
- [`agents/middleware/model_fallback.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_fallback.py)
- [`agents/middleware/model_retry.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_retry.py)

---

## 1. `HumanInTheLoopMiddleware`：2 つの割り込みモードと編集通知設計

[`HumanInTheLoopMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py#L402-L1049) は、危険なツール呼び出しの実行前に LangGraph の `interrupt()` を発火させ、人間から `approve` / `edit` / `reject` / `respond` の 4 種類の決定（`Decision`）を受け取ります。

最大のアーキテクチャ上の特徴は、**`interrupt_mode` による 2 系統の実装**（`batched` と `per_call`）です。

{% mermaid %}
flowchart TD
    subgraph BATCH["interrupt_mode='batched' (ターン単位)"]
        B1["after_model()<br/>全対象ツールを HITLRequest に集約"] --> B2["interrupt(hitl_request)<br/>decisions[] を一括受領"]
        B2 --> B3["edit 内容を PrivateStateAttr<br/>hitl_edited_tool_calls に退避<br/>(元の AIMessage は改変しない)"]
        B3 --> B4["wrap_tool_call()<br/>実行時に _apply_edit() で差し替え<br/>+ _EDIT_NOTICE を結果先頭に付与"]
    end

    subgraph PER["interrupt_mode='per_call' (ツール呼出単位)"]
        P1["wrap_tool_call()<br/>_decision_schema() で動的 Pydantic 型生成"] --> P2["interrupt(ToolApprovalRequest,<br/>response_schema=schema)"]
        P2 --> P3["approve: そのまま実行<br/>edit: _apply_edit + 通知付与<br/>reject/respond: 合成 ToolMessage 返却"]
    end

    style BATCH fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style PER fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

### 1.1 `per_call` モードの動的 `response_schema` 生成

`per_call` モードでは、ツールごとに許可された決定と引数スキーマを厳密に検証するため、`_decision_schema()`（[`human_in_the_loop.py#L222-L266`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py#L222-L266)）が **実行時に Pydantic モデルと `Discriminator("type")` 付き Union を動的構築** して `interrupt(value, response_schema=response_schema)` へ渡します。

```python
def _edit_decision(name: str, tool: BaseTool | None) -> object:
    # ツール名を Literal[name] に固定し、per_call 編集での他ツールへのすり替えを構造的に禁止
    edited_action = create_model(
        "EditedAction",
        name=(cast("TypeForm[Any]", Literal[name]), ...),
        args=(cast("TypeForm[Any]", _edit_args(tool)), ...),
    )
    decision = create_model(
        "EditDecision",
        type=(cast("TypeForm[Any]", Literal["edit"]), ...),
        edited_action=(edited_action, ...),
    )
    # WrapValidator(_as_sent) により検証だけ行い、生の dict 構造をそのまま下流へ通す
    return Annotated[decision, WrapValidator(_as_sent)]
```

### 1.2 レビュアー編集時に「元の `AIMessage` を書き換えない」理由

`batched` モードで人間が `edit` を選んだ際、コードは元の `AIMessage.tool_calls` を書き換えず、内部状態 `_HumanInTheLoopState.hitl_edited_tool_calls`（`PrivateStateAttr`）に退避し、`wrap_tool_call()` で実行時置換したうえで `_EDIT_NOTICE`（[`#L56-L60`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py#L56-L60)）を `ToolMessage` の先頭に付加します。これにより「LLM 自身が生成した履歴の改ざん」を避けつつ、「人間の判断で引数が変更されたため元の呼び出しを再発行するな」とモデルに正確に伝えられます。

---

## 2. `PIIMiddleware`：ステート層 + ストリーミング層の二段防御

[`PIIMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/pii.py#L513-L889) は、メールアドレス・クレジットカード（Luhn チェック付き）・IP・MAC・URL・カスタム正規表現を検出し、`block` / `redact` / `mask` / `hash` の 4 戦略を適用します。

ソースコードを読んで最も驚かされるのが、**State レベルのフック（`before_model` / `after_model`）だけでなく、`StreamTransformer`（`_PIIStreamTransformer`）によるインフライト・ストリーミング検閲** を実装している点です（[`pii.py#L50-L511`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/pii.py#L50-L511)）。

### 2.1 チャンク境界を跨ぐ PII を防ぐ 128 文字スライディングバッファ

トークン単位で届くストリーミング応答では、`user@ex` と `ample.com` が別々のデルタに分断されます。`_PIIStreamTransformer`（`before_builtins = True` により標準ストリーム変換より先に実行）は、`(run_id, content_block_index)` ごとに末尾 128 文字（`_DEFAULT_STREAM_LOOKBACK = 128`）を保留するスライディングウィンドウを持ちます。

```python
def _mutate_string_field_delta(self, delta: dict[str, Any], payload: dict[str, Any], run_id: str, field: str) -> None:
    text = delta.get(field)
    if not isinstance(text, str) or not text:
        return
    key = (run_id, payload["index"])
    held = self._buffers.get(key, "")
    combined = held + text

    # 保留分 + 新着デルタを結合した全体に対して PII 検出を実行（境界またぎを捕捉）
    # strategy="block" の場合、apply_strategy() が即座に PIIDetectionError を送出してストリームを遮断
    matches = self._rule.detector(combined)
    if matches:
        combined = apply_strategy(combined, matches, self._rule.strategy)

    # 末尾 lookback(128) 文字は次デルタとの境界判定のためバッファに残し、安全な先頭部分のみ送出
    emit_end = max(0, len(combined) - self._lookback)
    self._buffers[key] = combined[emit_end:]
    delta[field] = combined[:emit_end]
```

この仕組みは通常の `text-delta` だけでなく、推論モデルの思考過程である `reasoning-delta`、`tool-call-delta`、`tool-output-delta`、さらには `values` チャネルのステートスナップショットにまで網羅的に適用されます。

---

## 3. `ModelFallbackMiddleware` と `ModelRetryMiddleware` の協調と防衛設計

可用性を担う 2 つのラッパー型ミドルウェア（`wrap_model_call` / `awrap_model_call`）には、LangGraph 上で安全に動作するための共通ルールと独自の工夫があります。

{% mermaid %}
sequenceDiagram
    participant R as "ModelRetryMiddleware (外側)"
    participant F as "ModelFallbackMiddleware (内側)"
    participant M as "_execute_model_sync"

    R->>F: "handler(request) [attempt 0]"
    F->>M: "Primary Model 呼び出し"
    M--xF: "APIError (503)"
    Note over F: "_sanitize_request_for_fallback()<br/>非対応キャッシュヘッダ/マーカー除去"
    F->>M: "Fallback Model 1 呼び出し"
    alt "Fallback 成功"
        M-->>F: "ModelResponse"
        F-->>R: "ModelResponse"
    else "全モデル失敗 (非 GraphBubbleUp 例外)"
        F--xR: "last_exception 再送出"
        Note over R: "should_retry_exception() 判定<br/>指数バックオフ + ±25% jitter 待機"
        R->>F: "handler(request) [attempt 1]"
    end
{% endmermaid %}

### 3.1 `except GraphBubbleUp: raise` の鉄則

[`ModelFallbackMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_fallback.py#L411-L412) と [`ModelRetryMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_retry.py#L244-L245) の両方で、**`except Exception:` の直前に必ず `except GraphBubbleUp: raise` を置いています**。LangGraph の `interrupt()` などのグラフ制御例外は `GraphBubbleUp` を継承しているため、これを通常のモデル障害としてキャッチしてしまうと人間の承認待ちやグラフ中断が握りつぶされてしまうためです。

### 3.2 `ModelFallbackMiddleware`：クロスプロバイダ・キャッシュサニタイズ

外側にプロンプトキャッシュ用ミドルウェアが配置されていると、プライマリモデル（例：Anthropic）向けに付与された `cache_control` マーカーが `request.messages` や `request.tools` に残ったまま、フォールバック先（例：OpenAI）へ送られて 400 エラーを引き起こします。

`model_fallback.py` の `_sanitize_request_for_fallback()`（[`#L113-L166`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_fallback.py#L113-L166)）は、フォールバック先モデルの `_llm_type` を検査し、非対応プロバイダ向けのキャッシュ設定だけを自動的に剥ぎ取ります。

```python
# _llm_type ベースでプロバイダ互換性を判定（将来のモデル ID にも追従可能）
_ANTHROPIC_LLM_TYPES = frozenset({
    "anthropic-chat",
    "anthropic-bedrock-chat",
    "anthropic-chat-vertexai",
})

# フォールバックループ内でのサニタイズ（try ブロックの外で実行し、サニタイズ自体のバグを隠蔽しない）
for fallback_model in self.models:
    fallback_request = _sanitize_request_for_fallback(request, fallback_model)
    try:
        return handler(fallback_request.override(model=fallback_model))
    except GraphBubbleUp:
        raise
    except Exception as e:
        last_exception = e
        continue
```

- Anthropic 非対応モデルへフォールバックする場合：`model_settings`・`system_message`・`messages`・`tools`（および `extras` / `metadata` 内）から `cache_control` を再帰除去。
- Fireworks 非対応の場合：`extra_headers` から `x-session-affinity` を除去。
- OpenAI / Azure OpenAI / Fireworks 以外の場合：`prompt_cache_key` を除去。

### 3.3 `ModelRetryMiddleware`：例外フィルタと失敗時フォールバック応答

[`ModelRetryMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/model_retry.py#L221-L327) は、共有ヘルパー `_retry.py` の `should_retry_exception()` と `calculate_delay()`（指数バックオフ + ±25% ジッター）を利用します。

```python
for attempt in range(self.max_retries + 1):
    try:
        return handler(request)
    except GraphBubbleUp:
        raise
    except Exception as exc:
        attempts_made = attempt + 1
        # retry_on に合致しない例外（例: 認証エラー等）は on_failure を通さず即座に再送出
        if not should_retry_exception(exc, self.retry_on):
            raise
        if attempt < self.max_retries:
            delay = calculate_delay(
                attempt,
                backoff_factor=self.backoff_factor,
                initial_delay=self.initial_delay,
                max_delay=self.max_delay,
                jitter=self.jitter,
            )
            if delay > 0:
                time.sleep(delay)  # 非同期版 awrap_model_call では await asyncio.sleep(delay)
        else:
            # リトライ枯渇時: on_failure="error" なら再送出、"continue" / callable なら AIMessage を返して継続
            return self._handle_failure(exc, attempts_made)
```

---

## まとめ

今回解析した 4 つの Built-in Middleware からは、LangChain v1 の「本番運用を前提とした設計哲学」が色濃く読み取れます。

- **`HumanInTheLoopMiddleware`**：`per_call` での動的 Pydantic スキーマ検証と、履歴を汚さない編集通知設計。
- **`PIIMiddleware`**：State フックと 128 文字 lookback 付き `StreamTransformer` の二段構えで、ストリーミング経由の情報漏洩を完全に塞ぐ。
- **`ModelFallbackMiddleware` / `ModelRetryMiddleware`**：`GraphBubbleUp` の透過と、プロバイダを跨ぐ際のキャッシュ設定サニタイズ。

最終回となる次回 Part 6 では、プロバイダ動的ロードの要である [`chat_models/base.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/chat_models/base.py) の `init_chat_model` と、FastMCP 統合を担う [`mcp/adapter.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/adapter.py)・[`mcp/tools.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/mcp/tools.py) を解析します。
