---
title: DeepAgents ソースコード解析 Part 4：コンテキスト圧縮パイプライン（summarization / blob_offload / message_eviction / overflow_clip）
date: 2026-09-28 05:00:25
categories: [AI]
tags: [Python, LLM, AI Agent, DeepAgents, Context Engineering, ソースコード解析]
lang: ja
description: DeepAgents の多層コンテキスト圧縮パイプラインを完全解析。SummarizationMiddleware のイベントベース非破壊要約・事前ツール引数切り詰め・ContextOverflowError リアクティブリカバリ、_blob_offload の SHA-256 Content-Addressed バイナリ退避と再水和、_overflow_clip の末尾 ToolMessage 緊急クリップまでソースコードから解説する。
---

# DeepAgents ソースコード解析 Part 4：コンテキスト圧縮パイプライン（`summarization`, `_blob_offload`, `_message_eviction`, `_overflow_clip`）

DeepAgents が数百ステップにわたる長期タスクを「コンテキスト長超過（`ContextOverflowError`）」で墜落させずに完走できる理由は、単一の要約プロンプトではなく、**4 つの内部モジュールが連携する多層コンテキスト圧縮パイプライン**にあります。

Part 4 では、以下の 4 ファイルの役割分担とデータフローをソースコードレベルで徹底解析します。

1. [`middleware/summarization.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py)：事前引数切り詰め・履歴退避・イベントベース非破壊要約・オーバーフロー時フォールバック
2. [`middleware/_blob_offload.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_blob_offload.py)：SHA-256 Content-Addressed なバイナリ（画像等）退避とオンデマンド再水和
3. [`middleware/_message_eviction.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_message_eviction.py)：プロアクティブ／リアクティブ共通の Head+Tail プレビュー生成と `ToolMessage` 退避
4. [`middleware/_overflow_clip.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_overflow_clip.py)：`ContextOverflowError` 発生時に残存末尾の巨大 `ToolMessage` バッチを緊急圧縮する最終防衛線

---

## 1. コンテキスト圧縮パイプラインの全体像

1 回の LLM 呼び出し（`SummarizationMiddleware.wrap_model_call`）の中で、メッセージ列は「軽量な前処理 → 閾値判定 → 要約＋履歴退避 → 予算チェック＆緊急クリップ」という段階的なパイプラインを通過します。

{% mermaid %}
flowchart TD
    IN["ModelRequest (全履歴 messages)"] --> S0["_get_effective_messages()<br/>過去の _summarization_event を適用し<br/>[summary_msg, *cutoff以降の生メッセージ] を復元"]
    S0 --> S1["_count_tokens(messages, system, tools)<br/>ツールスキーマ込みで 1 回だけ計測"]
    S1 --> S2["Step 1: _truncate_args()<br/>古い AIMessage.tool_calls の巨大引数を軽量カット"]
    S2 --> S3{"Step 2: should_summarize?<br/>trigger 到達 or _over_budget?"}

    S3 -- "No" --> CALL1["handler(request) で通常呼び出し"]
    CALL1 -- "成功" --> RET1["ModelResponse を返却"]
    CALL1 -- "ContextOverflowError 発生!" --> S4

    S3 -- "Yes" --> S4["Step 3: _determine_cutoff_index()<br/>ツール呼び出しペアを壊さない安全位置で分割"]
    S4 --> S5["_offload_inline_media()<br/>data: URL を /conversation_history/media/ へ退避"]
    S5 --> S6["並列実行 (async時は asyncio.gather)<br/>1) _offload_to_backend (/conversation_history/{sid}.md)<br/>2) _create_summary (LLM 要約生成)"]
    S6 --> S7["Claude 等の bound reasoning ブロックを<br/>preserved_messages から除去 (_drop_bound_reasoning)"]
    S7 --> S8["_call_with_budget()<br/>予算超過または再 Overflow 時に<br/>_clip_overflow_tail() で末尾 ToolMessage を緊急退避"]
    S8 --> RET2["ExtendedModelResponse<br/>_summarization_event & 置換 ToolMessage を State 更新"]

    style IN fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style S2 fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style S6 fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    style S8 fill:#ffebee,stroke:#c62828,color:#b71c1c
    style RET2 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
{% endmermaid %}

---

## 2. `SummarizationMiddleware`：非破壊イベント方式と 3 段階の賢い圧縮

### 2.1 なぜ State から古いメッセージを削除せず `_summarization_event` を使うのか

LangChain 標準の `SummarizationMiddleware` は古いメッセージを `RemoveMessage` で State から消去しますが、DeepAgents 版（[`_DeepAgentsSummarizationMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py#L621)）は **生の `messages` 配列を State に丸ごと残したまま、プライベート状態 `_summarization_event` だけを更新する非破壊方式** を採用しています。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py
class SummarizationEvent(TypedDict):
    cutoff_index: int                      # 要約された境界インデックス
    cutoff_message_id: NotRequired[str | None]  # 前方削除が起きても境界がずれないよう message.id でも追跡
    summary_message: HumanMessage          # 要約本文 + 退避ファイルパス案内を含む HumanMessage
    file_path: str | None                  # /conversation_history/{session_id}.md のパス
    created_at_index: NotRequired[int]     # 要約生成時の messages 長（bound reasoning 除去範囲の判定用）
```

この設計には 3 つの大きな利点があります。

1. **監査・デバッグ・UI 表示の完全性**：チェックポイントには生の会話履歴が一切欠損なく残る。
2. **プロバイダ固有の思考ブロック（Thinking）制約への対処**：Anthropic Claude や Bedrock Claude は `thinking` / `redacted_thinking` ブロックを直前の会話履歴に暗号的にバインドしているため、過去の会話が要約で書き換わると API がリクエストを拒否します。`created_at_index` と `_bound_reasoning_types()` を組み合わせ、**要約前から持ち越された Claude の reasoning ブロックだけをモデル向けリクエストから安全に除去**します（Gemini や OpenAI モデルの思考ブロックは温存）。
3. **モデルプロファイル連動の自動デフォルト（`compute_summarization_defaults`）**：モデルに `profile["max_input_tokens"]` が定義されていれば `trigger=("fraction", 0.85)`, `keep=("fraction", 0.10)` の比率ベース、未定義なら `170,000 tokens` / `6 messages` の固定値ベースへ自動切替します。

### 2.2 `awrap_model_call` のコア実装

[`awrap_model_call`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py#L1772-L1912) の中核ロジックを見ると、非同期実行時には **履歴ファイルの書き出しと LLM 要約呼び出しを `asyncio.gather` で並列実行** してレイテンシを最小化していることがわかります。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py
# 前回の _summarization_event を反映した「現在の有効メッセージ列」を再構築
effective_messages = self._get_effective_messages(request)

# ツールスキーマ変換を含むトークン計測は高コストなため、1 回だけ計測して使い回す
total_tokens = self._count_tokens(effective_messages, request.system_message, request.tools)

# Step 1: 要約より軽量な「古い write_file / edit_file 引数の事前切り詰め」を試行
truncated_messages, truncate_modified = self._truncate_args(effective_messages, total_tokens)
if truncate_modified:
    total_tokens = self._count_tokens(truncated_messages, request.system_message, request.tools)

should_summarize = self._should_summarize(truncated_messages, total_tokens) or self._over_budget(request, total_tokens)

overflow_error: Exception | None = None
if not should_summarize:
    try:
        return await handler(request.override(messages=truncated_messages))
    except Exception as exc:
        # プロバイダ固有の Context Overflow（400/413/422 + メッセージ判定）以外はそのまま再送出
        if not _is_context_overflow(exc):
            raise
        overflow_error = exc  # オーバーフロー時のみ要約フォールバックへ落下させる

# Step 3: 履歴退避 & 要約生成
cutoff_index = self._determine_cutoff_index(truncated_messages)
messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

# インライン data: URL 画像等を先にメディアフォルダへアップロードし、XML タグ参照に置換
offloaded_media_messages, failed_media = await self._aoffload_inline_media(backend, messages_to_summarize)
session_id = self._get_session_id(request.state)

# 履歴 Markdown のバックエンド追記と、LLM による要約生成は独立なので並列実行！
file_path, summary = await asyncio.gather(
    self._aoffload_to_backend(backend, offloaded_media_messages, session_id),
    self._acreate_summary(offloaded_media_messages),
)
```

さらに `DEEPAGENTS_DEFAULT_SUMMARY_PROMPT` には `<media_reference_information>` ブロックが差し込まれており、`<image url="/conversation_history/media/{hash}.png" />` のような退避メディア参照タグを要約文に保持するよう要約モデルへ指示します。これにより、要約後のエージェントも必要になれば `read_file` で過去の画像を再取得できます。

---

## 3. `_blob_offload.py`：SHA-256 Content-Addressed 退避とオンデマンド再水和

画像や PDF を `read_file` やユーザー入力で扱うと、base64 文字列が LangGraph チェックポイントに毎回保存され、DB 容量が爆発します。[`_blob_offload.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_blob_offload.py) は、この問題を **Content-Addressed Storage（CAS）パターン** で解決しています。

{% mermaid %}
sequenceDiagram
    participant State as "LangGraph State / Checkpoint"
    participant MW as "FilesystemMiddleware<br/>(_blob_offload.py)"
    participant Cache as "_blob_payloads<br/>(UntrackedValue 実行内キャッシュ)"
    participant Backend as "Backend<br/>({artifacts_root}/blobs/<sha256>)"
    participant LLM as "LLM Provider"

    Note over MW,Backend: 1. 退避フェーズ (wrap_tool_call / wrap_model_call)
    MW->>MW: base64 デコード & SHA-256 ダイジェスト計算
    MW->>Backend: upload_files([(blobs/<sha256>, raw_bytes)])
    MW->>Cache: cache[digest] = base64_payload
    MW->>State: base64 を削除し {"deepagents_blob": "<sha256>"} スタブを保存

    Note over MW,LLM: 2. 再水和フェーズ (wrap_model_call 直前)
    MW->>Cache: ダイジェストが実行内キャッシュにあるか確認
    alt キャッシュミス（別ターン / チェックポイント再開）
        MW->>Backend: download_files([blobs/<sha256>])
        MW->>MW: sha256(downloaded) == digest を改ざん検証!
    end
    MW->>LLM: base64 を復元した一時メッセージ配列のみを送信
{% endmermaid %}

ここで極めてセキュリティ意識が高いのが [`_accept_downloads`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_blob_offload.py#L199-L206) の実装です。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_blob_offload.py
def _accept_downloads(
    missing: list[str],
    responses: Sequence[FileDownloadResponse],
    payloads: dict[str, str],
    cache: dict[str, str],
) -> None:
    for digest, response in zip(missing, responses, strict=False):
        # Blob はエージェント自身が書き込み可能なファイルシステム上に置かれているため、
        # 再水和前に必ず SHA-256 を再計算して改ざん・破損がないか検証する！
        if response.error is not None or response.content is None or hashlib.sha256(response.content).hexdigest() != digest:
            continue
        payload = base64.b64encode(response.content).decode("ascii")
        cache[digest] = payload
        payloads[digest] = payload
```

Blob ファイルが削除・改ざんされていた場合でも例外でクラッシュせず、`[Binary content from an earlier read_file call is no longer available. Re-read the file if you still need it.]` という案内テキストへグレースフルに縮退します。また `_blob_payloads` チャネルは LangGraph の `UntrackedValue` として定義されているため、チェックポイントには永続化されず、同一ラン内での高速キャッシュとしてのみ機能します。

---

## 4. `_overflow_clip.py`：要約しても収まらない場合の緊急末尾クリップ

要約（Step 3）を実行しても、**直前のターンで並列実行された巨大な `ToolMessage` 群（`preserved_messages` の末尾）自体がコンテキスト上限を超えている**ケースがあります。これを救済するのが [`_overflow_clip.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_overflow_clip.py) です。

`_call_with_budget()` から呼び出される `_clip_overflow_tail()` は、末尾に連続する `ToolMessage` バッチの合計トークン数が閾値（`keep` 予算、またはフォールバック `5,000 tokens`）以上の場合に発動し、**ツール種別に応じて 2 通りの最適処理**を行います。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/_overflow_clip.py
def _clip_one_tail_message(
    msg: ToolMessage,
    tc_index: dict[str, dict[str, Any]],
    backend: BackendProtocol,
    large_tool_results_prefix: str,
) -> ToolMessage | None:
    # 直前の AIMessage.tool_calls から tool_call_id に対応する元の呼び出しを引く
    original_path = _read_file_original_path(msg, tc_index)
    if original_path is not None:
        # パス A: read_file の結果なら、元ファイルが既にバックエンドにあるため新規書き込み不要！
        # 先頭 4,000 文字にスライスし、original_path への offset/limit 再読込ガイドを付ける
        return _slice_read_file_tm(msg, original_path)
    # パス B: それ以外のツール結果（execute / grep 等）は共通ヘルパで
    # /large_tool_results/{tool_call_id} へ退避し、Head+Tail プレビュースタブへ置換
    return _offload_tool_message_content(
        msg, _extract_text_from_message(msg), backend, large_tool_results_prefix
    )
```

そして重要なのは、クリップされた置換 `ToolMessage` 群が **元のメッセージ ID を保持したまま `Command(update={"messages": new_tail})` で State に書き戻される**点です。これにより、次のターン以降で同じ巨大な末尾ツール結果が再びオーバーフローを引き起こす無限ループを断ち切ります。

---

## まとめ

DeepAgents のコンテキスト圧縮パイプラインは、「1. 軽量な古いツール引数カット → 2. バイナリの CAS 退避 → 3. 非破壊イベントによる履歴退避＆並列要約 → 4. 末尾 `ToolMessage` の緊急クリップ」という 4 段構えで、情報損失を最小限に抑えながらコンテキスト上限への耐性を極限まで高めています。

次回 Part 5 では、エージェントの能力とドメイン知識をコンテキストを圧迫せずに拡張する **`SkillsMiddleware` と `MemoryMiddleware` の動的プロンプト注入と Progressive Disclosure** を解析します。
