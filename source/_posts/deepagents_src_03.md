---
title: DeepAgents ソースコード解析 Part 3：FilesystemMiddleware と Large Tool Result Offloading・Truncation 機構
date: 2026-09-23 09:57:20
categories: [AI]
tags: [Python, LLM, AI Agent, DeepAgents, Context Engineering, ソースコード解析]
lang: ja
description: DeepAgents の中核ミドルウェア FilesystemMiddleware（middleware/filesystem.py）を徹底解析。wrap_tool_call / wrap_model_call による巨大ツール結果・巨大 HumanMessage の自動退避（Offloading）、並列ファイル変更ガード、symlink 解決を含む細粒度 FilesystemPermission、マルチモーダルメディアの並び替えと自動リカバリ機構を解説する。
---

# DeepAgents ソースコード解析 Part 3：`FilesystemMiddleware` と Large Tool Result Offloading・Truncation 機構

長期実行エージェントが最も頻繁に直面する破綻原因は「**1 回のツール呼び出し（巨大なログ出力、長大なファイル読み込み、大量の grep 結果）がコンテキストウィンドウを一瞬で食い潰すこと**」です。

DeepAgents の [`FilesystemMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py) は、単なるファイル操作ツール群（`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, `execute`）の提供者にとどまらず、以下の 4 つの防御機構を `wrap_tool_call` と `wrap_model_call` の両輪で実装しています。

1. **Large Tool Result Offloading**：閾値を超えた `ToolMessage` をバックエンドの `/large_tool_results/{tool_call_id}` へ自動退避し、先頭＋末尾プレビュー付きスタブに置換
2. **Oversized HumanMessage Eviction**：巨大なユーザー入力をバックエンドへ退避し、State を壊さずモデル向けリクエストのみを縮小
3. **並列ファイル変更ガードと `FilesystemPermission`**：同一ターン内の競合編集を阻止し、symlink 解決後パスまで含めて `allow` / `deny` / `interrupt` を判定
4. **マルチモーダル・バイナリ最適化**：動画フレーム抽出、合成 `HumanMessage` の並び替え、および `ModelInvalidRequestError` 時の自動フォールバック

---

## 1. `wrap_tool_call` と `wrap_model_call` の全体パイプライン

[`FilesystemMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L1903) はツール実行の出口（`wrap_tool_call`）と LLM 呼び出しの入口（`wrap_model_call`）の両方をインターセプトします。

{% mermaid %}
flowchart TD
    subgraph TOOL["wrap_tool_call (ツール実行フック)"]
        direction TB
        T1["ToolCallRequest 到着"] --> T2{"_parallel_file_mutation_error()<br/>同一ターンで同じ file_path への<br/>複数 write/edit/delete か?"}
        T2 -- "重複あり" --> T3["エラー ToolMessage を即返却"]
        T2 -- "問題なし" --> T4["with _tool_blob_store(...):<br/>handler(request) 実行"]
        T4 --> T5{"read_file かつ<br/>バイナリ含む?"}
        T5 -- "Yes" --> T6["_offload_tool_result()<br/>SHA-256 Content-Addressed Blob 化"]
        T5 -- "No" --> T7
        T6 --> T7{"テキスト長 > 4 * limit<br/>かつ除外対象外ツール?"}
        T7 -- "超過" --> T8["_intercept_large_tool_result()<br/>/large_tool_results/{id} へ退避<br/>Head+Tail プレビュー付きスタブ生成"]
        T7 -- "閾値内" --> T9["そのまま返却"]
    end

    subgraph MODEL["wrap_model_call (LLM 呼び出しフック)"]
        direction TB
        M1["ModelRequest 到着"] --> M2["_filter_unsupported_tools_and_apply_prompt()<br/>execute/delete 可否判定 & ルート説明付与"]
        M2 --> M3["_move_media_results_after_tool_results()<br/>合成メディア HumanMessage を ToolMessage 群の後ろへ"]
        M3 --> M4["_evict_and_truncate_messages()<br/>巨大 HumanMessage を /conversation_history/ へ退避"]
        M4 --> M5["_offload_history_messages & _hydrate_messages<br/>State は Blob 参照化 / リクエスト時のみ再水和"]
        M5 --> M6["handler(request) で LLM 呼び出し"]
        M6 -- "ModelInvalidRequestError" --> M7["_replace_rejected_file_content()<br/>不正マルチモーダルを置換して 1 回リトライ"]
    end

    style TOOL fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style MODEL fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

---

## 2. Large Tool Result Offloading と Head+Tail プレビュー

### 2.1 `wrap_tool_call` と `_process_large_message`

ツール実行が完了すると、[`wrap_tool_call`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L3941-L3970) は返り値が単一の `ToolMessage` であっても複数メッセージを含む `Command` であっても透過的に検査します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    # 同一 AIMessage 内で同じファイルに対して複数の write_file / edit_file / delete が並列発行された場合、
    # 後続の呼び出しを拒否してレースコンディションによる破壊を防ぐ
    if error := _parallel_file_mutation_error(request):
        return error

    # ツール実行中のみ ContextVar 経由で Blob ストアを公開（グラフ State には入れない）
    with _tool_blob_store(self.backend if self._offload_binary_content else None, self._blobs_prefix):
        tool_result = handler(request)

    # read_file が返した画像・PDF 等のバイナリブロックを SHA-256 Blob に退避
    if self._offload_binary_content and request.tool_call["name"] == "read_file":
        tool_result = _offload_tool_result(tool_result, self.backend, self._blobs_prefix, {})

    # 退避無効、または read_file 等の除外対象ツールならそのまま返す
    if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
        return tool_result

    # 閾値超過のテキストをバックエンドへ書き出し、プレビュースタブへ置換
    return self._intercept_large_tool_result(tool_result)
```

サイズ判定を行う [`_process_large_message`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L3618-L3662) では、**マルチモーダルブロック（画像や音声の base64）をサイズ計測から除外**し、テキストブロックのみを抽出して `NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict`（`NUM_CHARS_PER_TOKEN = 4`）と比較します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py
def _process_large_message(
    self,
    message: ToolMessage,
    resolved_backend: BackendProtocol,
) -> tuple[ToolMessage, bool]:
    if not self._tool_token_limit_before_evict:
        return message, False

    # 非テキストブロック（画像等）は除外してテキスト長のみで判定する
    content_str = _extract_text_from_message(message)

    # 4文字/トークンの保守的見積もりで閾値以下なら退避しない
    if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
        return message, False

    # {large_tool_results_prefix}/{sanitized_tool_call_id} へ全文保存し、置換 ToolMessage を作成
    processed_message = _offload_tool_message_content(
        message,
        content_str,
        resolved_backend,
        self._large_tool_results_prefix,
    )
    if processed_message is None:
        # バックエンド書き込みに失敗した場合はデータ損失を防ぐため元メッセージを保持
        return message, False
    return processed_message, True
```

### 2.2 偽マーカー誤認を防ぐ `ContentPreview` 設計

実際のプレビュー生成は [`_message_eviction.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_message_eviction.py#L64-L157) の `_create_content_preview()` が担います。特筆すべきは、**「先頭 5 行＋末尾 5 行の行省略（`lines_omitted`）」と「1 行 1,000 文字超の行内クリッピング（`lines_clipped`）」を独立したブールフラグとして `ContentPreview` に記録する設計**です。

- ミニファイされた JSON や `.jsonl` のように「行数は 3 行しかないが 1 行が 10 万文字ある」ケースでは、`lines_omitted=False, lines_clipped=True` となり、行内クリップが行われた旨だけを正確に注記します。
- コンテンツ自体に `... [N lines truncated] ...` という文字列が含まれていても、生成側のフラグで判定するため誤認しません。
- 置換後の `ToolMessage` は `_build_evicted_content()` によって **非テキストブロック（画像等）をそのまま温存** し、テキストブロックだけを退避パスと `read_file(offset=..., limit=...)` の案内文に差し替えます。

### 2.3 巨大 `HumanMessage` の非破壊退避と `DeltaChannel` の協調

ユーザーが巨大なログやドキュメントをプロンプトに貼り付けた場合、[`_evict_and_truncate_messages`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L3768-L3797) が `wrap_model_call` 内で発動します。

ここでの巧妙な設計は、**LangGraph State 上の `HumanMessage` 本文は破壊せず、`additional_kwargs["lc_evicted_to"] = file_path` のタグだけを付与したコピーを `Command(update={"messages": [tagged]})` で返す**点です。

- 同じ `id` を持つメッセージを `Command` で流すことで、Part 1 で見た `DeltaChannel` のリデューサが `REMOVE_ALL_MESSAGES` センチネルなしにインプレース更新します。
- モデルへ送信する直前のリクエスト配列上でのみ `_build_truncated_human_message()` を適用して縮小するため、UI 側の会話履歴表示には元のユーザー入力がそのまま残ります。

---

## 3. `read_file` / `grep` における細かな Truncation と UX 工夫

`FilesystemMiddleware` は退避だけでなく、各ファイルツール固有の切り詰めロジックも細かく作り込んでいます。

- **`grep` のマッチ中心クリッピング（[`_clip_grep_line`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L874-L887)）**：1 行が `GREP_LINE_CHAR_LIMIT`（2,000 文字）を超えるミニファイ済みコード等でヒットした場合、行頭から切るのではなく **マッチ位置 `pattern` を中央に据えた 2,000 文字ウィンドウ** を切り出し、前後に `[N chars clipped]` を付与します。
- **`grep` の正規表現ヒント**：`BackendProtocol.grep` はリテラル検索であるため、ヒット 0 件かつパーミッションによる伏せ字（redaction）でもない場合に限り、`regex_literal_hint(pattern)` で「正規表現ではなくリテラル検索である」旨のヒントを付与します。
- **`read_file` の `no_lines_requested` 区別**：`limit=0` で呼ばれた際に「ファイルが空である」と誤認させないよう、`EMPTY_CONTENT_WARNING` と `NO_LINES_REQUESTED_WARNING` を厳密に分離し、後続の `write_file` による誤上書きを防止しています。

---

## 4. `FilesystemPermission`：symlink 解決と再帰削除の安全判定

[`FilesystemPermission`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L375-L495) は、`operations`（`read` / `write`）、`paths`（`wcmatch` の `BRACE | GLOBSTAR` パターン）、`mode`（`allow` / `deny` / `interrupt`）の 3 要素でアクセス制御を行います。

{% mermaid %}
flowchart LR
    P["リクエストパス<br/>例: /work/link_to_secret"] --> C1{"要求パス自体が<br/>deny ルールに一致?"}
    C1 -- "Yes" --> D1["即座に permission denied"]
    C1 -- "No" --> R["_resolve_real_path(backend, path)<br/>CompositeBackend ルート追跡 +<br/>FilesystemBackend._real_path()"]
    R -- "解決失敗 (循環symlink/脱出)" --> D2["cannot resolve エラー (fail-closed)"]
    R -- "解決成功: /secrets/key.pem" --> C2{"実パスが<br/>deny ルールに一致?"}
    C2 -- "Yes" --> D3["permission denied (resolves to ...)"]
    C2 -- "No" --> OK["許可 (allow)<br/>※ interrupt は HITL 層で事前承認"]

    style D1 fill:#ffebee,stroke:#c62828,color:#b71c1c
    style D2 fill:#ffebee,stroke:#c62828,color:#b71c1c
    style D3 fill:#ffebee,stroke:#c62828,color:#b71c1c
    style OK fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

さらに再帰削除ツール `delete` に対しては、[`_find_delete_deny_patterns`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L678-L737) が極めて慎重な判定を行います。

- **単一ファイルと確定できた場合（`has_descendants=False`）**：`write_file` / `edit_file` と同じく宣言順のファーストマッチで判定。
- **ディレクトリまたは子孫を持ち得る場合（`has_descendants=True`）**：上位ディレクトリに対する `allow` ルールがあっても、配下に 1 つでも `deny` ルールが重なる（`_wildcard_delete_overlap` / `_paths_overlap`）なら削除を拒否します。これにより「`/work` は allow、`/work/secrets/**` は deny」という設定で `delete("/work")` が秘密ディレクトリごと吹き飛ばす抜け穴を完全に塞いでいます。

---

## 5. マルチモーダル対応とプロバイダ制約の吸収

`read_file` が動画ファイル（`.mp4` 等）を読み込むと、[`_handle_video_read`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L282-L356) は `0.5 fps` でフレームをサンプリングし、テキストの `ToolMessage` に加えて **フレーム画像を載せた合成 `HumanMessage`** を `Command` で返します。

しかし、多くの LLM プロバイダ API は「`AIMessage(tool_calls=...)` の直後にすべての `ToolMessage` が連続すること」を要求し、途中に `HumanMessage` が挟まると 400 エラーを返します。これを解決するため、`wrap_model_call` 内で [`_move_media_results_after_tool_results`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py#L238-L266) が全 `ToolMessage` バッチの末尾へ合成メディアメッセージを自動移動します。さらに、それでもプロバイダが `ModelInvalidRequestError` を返した場合は、`_replace_rejected_file_content()` が直近ターンのマルチモーダルブロックだけをテキストエラーに置換して 1 回リトライすることで会話全体の中断を防ぎます。

---

## まとめ

`FilesystemMiddleware` は、ファイルツールを提供するだけでなく、コンテキスト肥大化の第一防衛線（プロアクティブ退避）、権限・並列変更の安全境界、マルチモーダル API 互換レイヤを兼ね備えています。

次回 Part 4 では、プロアクティブ退避だけでは収まりきらない長大な会話履歴をどう圧縮・リカバリするか——**コンテキスト圧縮パイプライン**（`summarization.py`, `_blob_offload.py`, `_message_eviction.py`, `_overflow_clip.py`）の 4 モジュール連携を徹底解剖します。
