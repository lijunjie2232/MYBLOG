---
title: "LangChain v1 ソースコード解析 Part 4：Built-in Middleware 解析① — SummarizationMiddleware と ContextEditingMiddleware"
date: 2026-08-20 07:25:17
categories: [AI]
tags: [Python, LLM, AI Agent, LangChain, Middleware, Context Engineering, ソースコード解析]
lang: ja
description: "LangChain v1 のコンテキスト制御ミドルウェア SummarizationMiddleware と ContextEditingMiddleware のソースコードを精読し、破壊的状態置換 vs 非破壊リクエスト編集の設計対比、二分探索カットオフ、AI/Tool ペア保護、オーバーフロー自動回復を解説する。"
---

長時間稼働する AI エージェントにとって最大の課題は、会話履歴とツール出力の肥大化による **コンテキストウィンドウ圧迫** です。LangChain v1 には、この問題に対してアプローチの異なる 2 つの組み込みミドルウェアが用意されています。

- [`SummarizationMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py)：古い履歴を LLM で要約し、グラフ状態（`state["messages"]`）を **破壊的に置換** する。
- [`ContextEditingMiddleware`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/context_editing.py)：古いツール出力を `[cleared]` に置き換え、モデル呼び出し時のみ **非破壊的にリクエストを書き換える**。

Part 4 では、この 2 つの実装をソースコードレベルで比較・解剖します。

---

## 1. フック選択の決定的な違い：`before_model` vs `wrap_model_call`

両ミドルウェアの最も本質的な違いは、**どのフックを使い、どこを書き換えるか** にあります。

{% mermaid %}
flowchart LR
    subgraph SUM["SummarizationMiddleware (before_model)"]
        S1["state['messages'] 検査<br/>TriggerClause (AND/OR)"] --> S2["二分探索 + AI/Toolペア保護<br/>cutoff_index 決定"]
        S2 --> S3["要約生成 + Overflow時中抜き再試行<br/>_create_summary"]
        S3 --> S4["RemoveMessage(REMOVE_ALL_MESSAGES)<br/>+ 要約HumanMessage + 保持履歴"]
    end

    subgraph CTX["ContextEditingMiddleware (wrap_model_call)"]
        C1["deepcopy(request.messages)<br/>※ Graph State は変更しない"] --> C2["ClearToolUsesEdit.apply<br/>古い ToolMessage を [cleared] 化"]
        C2 --> C3["request.override(messages=edited)<br/>→ handler(req) へ転送"]
    end

    style SUM fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style CTX fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

| 比較項目 | `SummarizationMiddleware` | `ContextEditingMiddleware` |
| :--- | :--- | :--- |
| 実装フック | `before_model` / `abefore_model` | `wrap_model_call` / `awrap_model_call` |
| Graph State への影響 | `REMOVE_ALL_MESSAGES` で永続置換（チェックポイントにも反映） | `deepcopy` したコピーのみ編集（State 原本は無傷） |
| 追加 LLM コスト | 要約用モデルの呼び出しが発生（`TAG_NOSTREAM` 付与） | ゼロ（ローカル文字列置換のみ） |
| 主な適用場面 | 長期マルチターン会話全体の記憶圧縮 | 巨大 JSON/ログを返すツール呼び出しの即時削減 |

---

## 2. `SummarizationMiddleware` の内部アルゴリズム

### 2.1 AND / OR トリガー正規化とプロバイダ報告トークンの併用

`SummarizationMiddleware`（[`summarization.py#L263-L1040`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L263-L1040)）は、`trigger` に単一タプル `("tokens", 3000)`、AND 条件辞書 `{"tokens": 4000, "messages": 10}`、さらにそれらの OR リストを受け付けます。`_normalize_trigger()`（[`#L600-L674`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L600-L674)）がすべてを「句のリスト（DNF：選言標準形）」へ正規化し、`bool` が `int` のサブクラスであることによる誤設定（`{"messages": True}`）まで明示的に拒否します。

さらに巧妙なのが、トークン閾値の判定ロジック `_should_summarize()`（[`#L676-L738`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L676-L738)）です。

```python
def _should_summarize_based_on_reported_tokens(
    self, messages: list[AnyMessage], threshold: float
) -> bool:
    """直近の AIMessage に記録されたプロバイダ実測トークン数で閾値超過を判定する。"""
    last_ai_message = next(
        (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
        None,
    )
    if (
        isinstance(last_ai_message, AIMessage)
        and last_ai_message.usage_metadata is not None
        and (reported_tokens := last_ai_message.usage_metadata.get("total_tokens", -1))
        and reported_tokens >= threshold
        # モデル切替（別プロバイダ）によるトークン計算ズレを防ぐためプロバイダ一致を確認
        and (message_provider := last_ai_message.response_metadata.get("model_provider"))
        and _provider_matches(
            message_provider,
            self.model._get_ls_params().get("ls_provider"),
        )
    ):
        return True
    return False
```

近似トークンカウンタ（Anthropic モデルでは実験値 `chars_per_token=3.3` に自動チューニング：[`#L252-L260`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L252-L260)）と、直近 `AIMessage.usage_metadata["total_tokens"]` の **両方** をチェックし、どちらか一方が閾値を超えたら要約を発火します。しかも `_LS_PROVIDER_ALIASES`（`amazon_bedrock` ↔ `bedrock_converse` 等）でプロバイダ表記揺れまで吸収しています。

### 2.2 二分探索カットオフと AI/Tool ペア分断の防止

要約対象と保持対象の境界 `cutoff_index` の決定は、**トークン予算ベースの二分探索**（[`_find_token_based_cutoff`, `#L752-L799`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L752-L799)）と、**AI/Tool ペア保護**（[`_find_safe_cutoff_point`, `#L878-L911`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L878-L911)）の 2 段階で実装されています。

```python
# 二分探索で「messages[mid:] が target_token_count 以下になる最小のインデックス」を O(log N) で探索
left, right = 0, len(messages)
cutoff_candidate = len(messages)
max_iterations = len(messages).bit_length() + 1
for _ in range(max_iterations):
    if left >= right:
        break
    mid = (left + right) // 2
    # サブリストに対するスケーリング歪みを避けるため use_usage_metadata_scaling=False のカウンタを使用
    if self._partial_token_counter(messages[mid:]) <= target_token_count:
        cutoff_candidate = mid
        right = mid
    else:
        left = mid + 1
```

もし算出された `cutoff_candidate` が `ToolMessage` の位置に落ちてしまうと、対応する `AIMessage(tool_calls=...)` が要約側へ消え、孤立した `ToolMessage` が保持側に残って LLM API エラーを引き起こします。`_find_safe_cutoff_point()` はこれを防ぐため、次のように境界を自動調整します。

```python
@staticmethod
def _find_safe_cutoff_point(messages: list[AnyMessage], cutoff_index: int) -> int:
    if cutoff_index >= len(messages) or not isinstance(messages[cutoff_index], ToolMessage):
        return cutoff_index

    # cutoff_index 以降に連続する ToolMessage の tool_call_id をすべて収集
    tool_call_ids: set[str] = set()
    idx = cutoff_index
    while idx < len(messages) and isinstance(messages[idx], ToolMessage):
        if messages[idx].tool_call_id:
            tool_call_ids.add(messages[idx].tool_call_id)
        idx += 1

    # 後方検索して、該当 tool_call_id を持つ親 AIMessage を見つけたらそこまでカット位置を戻す
    for i in range(cutoff_index - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
            if tool_call_ids & ai_tool_call_ids:
                return i

    # 親 AIMessage が見つからない異常系では ToolMessage 群の後ろへ進めて孤立を防ぐ
    return idx
```

### 2.3 要約モデル自体の `ContextOverflowError` 回復機構

要約処理そのものが要約モデルのコンテキスト長を超過するケースに対しても、`_create_summary()` / `_overflow_retry_prompt()`（[`#L938-L1020`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py#L938-L1020)）が自動回復を実装しています。

- 通常時は `self._summary_model`（`with_retry(retry_if_exception_type=default_retry_on)` で最大 3 回リトライ設定済み）に `get_buffer_string(messages, format="xml")` で整形したプロンプトを送信。
- もし `ContextOverflowError` が発生した場合、**トークン予算を元の 50%（かつ要約モデル上限の 40% 以下）に縮小し、先頭 1/4（セッションの初期目的）と末尾 3/4（直近の作業状況）を残して中間を中抜き** したプロンプトで 1 回だけ自動再試行します。
- 要約呼び出しには `"tags": [TAG_NOSTREAM]` と `internal_call_metadata()` が付与され、ユーザー向けストリーミング出力に要約テキストが漏れ出さないよう隔離されています。

---

## 3. `ContextEditingMiddleware` と `ClearToolUsesEdit` の内部実装

[`agents/middleware/context_editing.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/context_editing.py) は、Anthropic の `clear_tool_uses_20250919` に着想を得た、**モデル非依存のツール結果プルーニング機構** です。

`ContextEdit` プロトコル（[`#L44-L55`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/context_editing.py#L44-L55)）により編集戦略をプラガブルに差し替え可能であり、標準実装として `ClearToolUsesEdit`（[`#L57-L185`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/context_editing.py#L57-L185)）が提供されています。

{% mermaid %}
flowchart TD
    A["count_tokens(messages) <= trigger ?"] -- "Yes" --> RET["何もしない"]
    A -- "No (閾値超過)" --> B["全 ToolMessage を抽出し<br/>末尾 keep 件を保護して除外"]
    B --> C["古い ToolMessage から順にループ"]
    C --> D{"すでに cleared 済み<br/>または exclude_tools 該当?"}
    D -- "Yes" --> C
    D -- "No" --> E["ToolMessage を model_copy()<br/>content='[cleared]', artifact=None<br/>response_metadata['context_editing'] 付与"]
    E --> F{"clear_tool_inputs == True ?"}
    F -- "Yes" --> G["対応 AIMessage の tool_call['args'] = {} に置換"]
    F -- "No" --> H{"clear_at_least > 0 かつ<br/>回収トークン >= clear_at_least ?"}
    G --> H
    H -- "Yes" --> DONE["早期終了 (必要最小限の削減)"]
    H -- "No" --> C

    style RET fill:#eceff1,stroke:#455a64,color:#263238
    style E fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style G fill:#fce4ec,stroke:#c2185b,color:#880e4f
    style DONE fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

### 3.1 コアコード精読：`ClearToolUsesEdit.apply()`

```python
for idx, tool_message in candidates:
    # 冪等性チェック: すでに編集済みの ToolMessage はスキップ
    if tool_message.response_metadata.get("context_editing", {}).get("cleared"):
        continue

    # 対応する親 AIMessage と tool_call を後方検索（ツール名解決と入力クリアに必要）
    ai_message = next((m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)), None)
    if ai_message is None:
        continue
    tool_call = next(
        (call for call in ai_message.tool_calls if call.get("id") == tool_message.tool_call_id),
        None,
    )
    if tool_call is None or (tool_message.name or tool_call["name"]) in excluded_tools:
        continue

    # ToolMessage 本体を置換（artifact も None にしてメモリ/トークン解放）
    messages[idx] = tool_message.model_copy(
        update={
            "artifact": None,
            "content": self.placeholder,  # デフォルト: "[cleared]"
            "response_metadata": {
                **tool_message.response_metadata,
                "context_editing": {"cleared": True, "strategy": "clear_tool_uses"},
            },
        }
    )

    # オプション: 呼び出し側の巨大引数（例: write_file の全文）も空辞書 {} に削減可能
    if self.clear_tool_inputs:
        messages[messages.index(ai_message)] = self._build_cleared_tool_input_message(
            ai_message, tool_message.tool_call_id
        )

    # clear_at_least が指定されていれば、目標削減量に達した時点で早期打ち切り
    if self.clear_at_least > 0:
        new_token_count = count_tokens(messages)
        if max(0, tokens - new_token_count) >= self.clear_at_least:
            break
```

実務で嬉しいのは `clear_tool_inputs=True` と `clear_at_least` の組み合わせです。ファイル編集ツールなどでは、`ToolMessage`（出力）よりも `AIMessage.tool_calls[i]["args"]`（入力引数）の方が数万トークンを占めるケースが多いため、入力側もピンポイントで `{}` に消去できます。また、`ContextEditingMiddleware.wrap_model_call()`（[`#L250-L300`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/context_editing.py#L250-L300)）では `deepcopy(list(request.messages))` に対して編集を適用し `request.override(messages=edited_messages)` で下流へ渡すため、LangGraph のチェックポイント履歴（UI 表示や監査ログ用）には元のツール結果がそのまま保存されます。

---

## まとめ

`SummarizationMiddleware` と `ContextEditingMiddleware` は、併用することで最も高い効果を発揮します。

1. まず `ContextEditingMiddleware` で古い巨大なツール入出力を **コストゼロ・非破壊** に `[cleared]` 化してトークン消費を抑える。
2. それでも会話全体がコンテキスト上限（例：`("fraction", 0.8)`）に近づいたら、`SummarizationMiddleware` が **二分探索 + AI/Tool ペア保護** で安全に履歴を要約圧縮し、オーバーフロー時すら中抜きリトライで回復する。

次回 Part 5 では、Built-in Middleware 解析の第 2 弾として、`HumanInTheLoopMiddleware`、`PIIMiddleware`、`ModelFallbackMiddleware`、`ModelRetryMiddleware` の 4 つをコードレベルで解剖します。
