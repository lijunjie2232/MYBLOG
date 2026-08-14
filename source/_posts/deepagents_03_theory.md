---
title: "DeepAgents Part 3 理論編：Context Engineering（Large Tool Result Offloading・Summarization・Prompt Caching・Overflow Clipping）"
date: 2026-07-16 12:00:36
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, Context Engineering, Summarization, Prompt Caching, Architecture]
lang: ja
description: "長時間稼働するエージェントをコンテキスト破綻から守る DeepAgents の Context Engineering を徹底解説。20,000 トークン閾値での Large Tool Result Offloading、85% 閾値の非破壊 Summarization、ContextOverflowError リカバリ、そしてキャッシュ効率を最大化する Prompt Caching 設計を詳解します。"
---

# DeepAgents Part 3 理論編：Context Engineering（Large Tool Result Offloading・Summarization・Prompt Caching・Overflow Clipping）

複雑な調査・ログ解析・コード修正をエージェントに自律実行させると、ほぼ確実に直面する壁が **コンテキストウィンドウの枯渇と劣化（Context Rot）** です。巨大な API レスポンスや数千行のログが履歴にそのまま積み上がると、トークン課金が跳ね上がるだけでなく、モデルの注意（Attention）がノイズに奪われて指示違反やハルシネーションが急増します。

DeepAgents はこの問題に対し、開発者が手動で履歴をスライスしなくても自律的にコンテキストを健全に保つ **多層 Context Engineering 機構** を標準搭載しています。本記事（Part 3 理論編）では、以下の 4 つのコアメカニズムを体系的に解説します。

1. **Large Tool Result / Input Offloading**（仮想ファイルシステムへの自動退避）
2. **Summarization & Filesystem Preservation**（85% 閾値での非破壊圧縮と履歴保全）
3. **Overflow Clipping / Recovery**（`ContextOverflowError` 発生時の即時フォールバック）
4. **Prompt Caching**（スタック順序と TTL 最適化による推論高速化・コスト削減）

---

## 1. DeepAgents における 5 種類のコンテキスト分類

DeepAgents の公式設計（[Context Engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering)）では、エージェントが扱う情報をライフサイクルとスコープによって次の 5 層に整理しています。

| コンテキスト種別 | 構成要素 | スコープ・特徴 |
| :--- | :--- | :--- |
| **1. Input Context** | `system_prompt`・`memory`（`AGENTS.md`）・`skills`（`SKILL.md`）・Tool prompts | 起動時にシステムプロンプトへ組み立てられる静的基盤。 |
| **2. Runtime Context** | `context_schema` + `invoke(..., context=...)` | 実行ごとの不変設定（ユーザー ID・API キー・DB 接続）。モデルには自動露出せず、`ToolRuntime` 経由でツール・Middleware が参照。サブエージェントへも自動伝播。 |
| **3. Custom Graph State** | `state_schema`（`DeepAgentState` サブクラス） | チェックポイントに永続化される可変状態。`DeltaChannel` リデューサを維持し、会話が伸びてもチェックポイント増分を線形に抑える。 |
| **4. Context Compression** | **Offloading** ＋ **Summarization** | コンテキスト上限に近づいた際に自動で発火する 2 段構えの圧縮機構。 |
| **5. Isolation & Long-term Memory** | Subagents（`task` ツール）＋ `CompositeBackend`（`/memories/`） | 重い作業のコンテキスト隔離と、スレッド横断の長期記憶。 |

このうち、単一エージェントの長時間稼働を直接支える心臓部が **4. Context Compression（Offloading ＋ Summarization）** です。

---

## 2. Large Tool Result & Input Offloading（自動オフロード機構）

`FilesystemMiddleware`（[`deepagents/middleware/filesystem.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py)）は、すべてのツール入出力を監視し、**デフォルト閾値 20,000 トークン（ASCII テキストで約 80KB 相当）** を超える巨大なペイロードを自動的に仮想ファイルシステムへ退避（Offload）します。

{% mermaid %}
flowchart TD
    TC["ツール実行完了<br/>(例: dump_large_log)"] --> CHK{"ツール出力が<br/>閾値 (既定 20,000 tokens ≒ 80KB)<br/>を超過したか？"}
    CHK -->|"No (閾値以下)"| INLINE["そのまま ToolMessage として<br/>会話履歴にインライン格納"]
    CHK -->|"Yes (閾値超過)"| SAVE["デフォルト Backend に保存<br/>/large_tool_results/&lt;tool_call_id&gt;"]
    SAVE --> STUB["ToolMessage を軽量ポインタに置換<br/>・保存パス通知<br/>・先頭 10 行のプレビューのみ記載"]
    STUB --> MODEL["LLM がパスを認識し<br/>grep / read_file(offset, limit) で<br/>必要箇所だけをピンポイント取得"]

    style CHK fill:#fefcbf,stroke:#b7791f,stroke-width:2px,color:#744210
    style INLINE fill:#edf2f7,stroke:#4a5568,stroke-width:1.5px,color:#2d3748
    style SAVE fill:#e6fffa,stroke:#2c7a7b,stroke-width:1.5px,color:#234e52
    style STUB fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1.5px,color:#1a365d
    style MODEL fill:#f0fff4,stroke:#2f855a,stroke-width:1.5px,color:#22543d
{% endmermaid %}

### 入力側（Tool Call Inputs）と出力側（Tool Call Results）の 2 方向オフロード

1. **Tool Call Results > 20,000 トークン（出力オフロード）**
   巨大なログや検索結果が返った瞬間、生のテキストを `ToolMessage` に入れる代わりにデフォルト Backend の `/large_tool_results/<tool_call_id>` へ保存します。`ToolMessage` には **保存先パスと先頭 10 行のプレビュー** だけが残され、モデルは以降 `grep` や `read_file(offset=..., limit=...)` を使って必要な行だけをオンデマンドに引き出します。
2. **Tool Call Inputs > 20,000 トークン（入力オフロード）**
   エージェントが `write_file` や `edit_file` で大きなソースコードを書き込んだ場合、その引数自体が過去の `AIMessage.tool_calls` に丸ごと残り続けます。すでにファイルシステムに永続化済みの内容を履歴に持ち続けるのは冗長なため、セッション全体のコンテキストが **モデル上限の 85%** を超えたタイミングで、古いファイル書き込みツール呼び出しの引数をディスク上のファイルポインタへ切り詰めます。
3. **Binary Content Offloading（`deepagents>=0.7.21`、オプトイン）**
   画像や PDF などの Base64 ペイロードがチェックポイントを肥大化させる問題に対しては、`FilesystemMiddleware(backend=..., offload_binary_content=True)` を有効化することで、バイナリを `/blobs/<sha256>`（コンテンツアドレス指定）へ自動退避できます。同一メディアの再読込時は同じ Blob を再利用し、モデル送信直前に復元します（注：Blob 退避先は `StateBackend` ではなく外部ディスクや Sandbox にルーティングする必要があります）。

---

## 3. Summarization（自動要約）と Overflow Clipping / Recovery

オフロードだけでは対応しきれない「数十ターン以上に及ぶ長い対話履歴」を圧縮するのが、Bare Stack に標準装備されている **`SummarizationMiddleware`**（[`deepagents/middleware/summarization.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py)）です。

### 発動条件と 2 段構えの保全戦略

`SummarizationMiddleware` は以下のルールで自動起動します。

- **発動トリガー（`trigger`）**: モデルのプロファイル（`max_input_tokens`）の **85%** に到達した時点。
  - **重要（フォールバック値）**: 自前ホストの OpenAI 互換モデルなどでモデルプロファイルが未登録の場合、フォールバックとして **170,000 トークン到達 / 直近 6 メッセージ保持** が適用されます。
- **保持ウィンドウ（`keep`）**: コンテキスト上限の **10%** 分の直近メッセージは要約せずそのまま保持し、直前の作業文脈を壊さないようにします。
- **非破壊変換（Non-destructive）**: 要約は **LLM へ送信する直前のリクエスト配列のみを書き換え、永続化された `state["messages"]` 自体は削除・改変しません**。これにより、後から LangSmith での評価・リプレイ・監査を行う際の完全性が保たれます。
- **Dual Approach（要約＋原文退避）**:
  1. **In-context summary**: LLM が「セッションの目的・作成済み成果物・次のステップ」を構造化要約し、`lc_source="summarization"` のメタデータ付き `HumanMessage` としてモデルへ渡す（ストリーミング時は `metadata.get("lc_source") == "summarization"` で UI 表示からフィルタリング可能）。
  2. **Filesystem preservation**: 圧縮前のオリジナル会話ログをテキスト化して `/conversation_history/` に保存する。要約で抜けた細部が必要になれば、エージェント自身が `grep` / `read_file` で原文を再検索できる。

### Overflow Clipping & 即時リカバリ

トークン推定の誤差や急激な単発入力によって、事前チェックをすり抜けて LLM プロバイダから **`ContextOverflowError`** が返された場合でも、DeepAgents は即座にクラッシュしません。`SummarizationMiddleware` が例外を捕捉し、**その場で緊急要約（＋必要に応じたオーバーフロー分のクリッピング）を実行して「要約＋直近保持メッセージ」で自動的にリトライ** します。

{% mermaid %}
sequenceDiagram
    autonumber
    participant Loop as Agent Loop
    participant Summ as SummarizationMiddleware
    participant FS as Default Backend<br/>(/conversation_history/)
    participant LLM as Primary / Summarizer LLM

    Loop->>Summ: 次ターンのメッセージ履歴を送信
    alt トークン数 >= 85% (または未登録時 170k)
        Summ->>FS: 原文の会話履歴をテキスト退避
        Summ->>LLM: 要約生成 (summarizer= で安価モデル指定可)
        LLM-->>Summ: 構造化サマリー (目的 / 成果物 / 次手順)
        Summ->>LLM: [サマリー(lc_source=summarization) + 直近10%] で本推論
    else 85% 未満 → 通常送信
        Summ->>LLM: そのまま本推論を実行
        opt ContextOverflowError が発生した場合
            LLM-->>Summ: ContextOverflowError
            Note over Summ: 即時フォールバック要約 & リトライ
            Summ->>FS: 原文退避
            Summ->>LLM: 緊急要約 + 直近メッセージで再送
        end
    end
    LLM-->>Loop: 推論結果を返却 (state["messages"] 原文は非破壊)
{% endmermaid %}

### 実務で効く 3 つのカスタマイズ手法

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import SummarizationMiddleware
from deepagents.middleware.summarization import create_summarization_tool_middleware

load_dotenv()

primary_llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

backend = StateBackend()

# 1. trigger / keep を明示調整（プロファイル未登録の 32K/64K モデルでは必須級）
# 2. summarizer= に安価・高速モデルを指定して要約コストを分離（deepagents>=0.7.26）
custom_summarization = SummarizationMiddleware(
    model=primary_llm,
    backend=backend,
    trigger=("tokens", 24_000),  # 例: 32K コンテキストモデルなら 24k で早めに圧縮
    keep=("messages", 8),        # 直近 8 メッセージをそのまま保持
    # summarizer="openai:gpt-4.1-mini",  # 任意で軽量モデルを要約専任に指定可能
)

# 3. エージェント自身がタスク区切りで自律圧縮できる compact_conversation ツールを追加
agent = create_deep_agent(
    model=primary_llm,
    backend=backend,
    middleware=[
        custom_summarization,  # .name 一致によりデフォルト SummarizationMiddleware を置換
        create_summarization_tool_middleware(primary_llm, backend),
    ],
)
```

---

## 4. Prompt Caching：スタック順序が生むコスト・レイテンシ最適化

長いエージェントループでは、毎ターン同じ「ベースシステムプロンプト」「ツールスキーマ」「スキル一覧」が繰り返し LLM に送られます。DeepAgents は **Anthropic モデル** および **AWS Bedrock（Claude / Nova）モデル** を検出すると、`AnthropicPromptCachingMiddleware` / `BedrockPromptCachingMiddleware` を自動有効化し、静的プレフィックスをプロンプトキャッシュに乗せます（非対応モデルでは `unsupported_model_behavior="ignore"` により自動的に no-op になります）。

Part 1 理論編で確認した Middleware スタック順序がここで効いてきます。

- `PatchToolCallsMiddleware`（4 番目）と `SkillsMiddleware`（8 番目）は **Prompt Caching（9 番目）の前** に配置され、安定したプレフィックスをキャッシュ対象に含める。
- 一方、会話中に更新されうる `MemoryMiddleware`（10 番目、`AGENTS.md`）は **Prompt Caching の直後** に配置し、メモリ追記によるキャッシュ無効化（Cache Busting）を防ぐ。
- ターン間隔が長いエージェントでは、デフォルトの TTL（`5m`）を以下のように `1h` へ上書きすることでヒット率をさらに高められます。

```python
from deepagents import create_deep_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

# デフォルトの 5m TTL を 1h に延長（.name 一致でデフォルトキャッシュ層を置換）
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[AnthropicPromptCachingMiddleware(ttl="1h")],
)
```

---

## 5. Runtime Context・Custom State・ストリーミング時の要約フィルタリング

コンテキスト圧縮と並んで実務設計で押さえておくべきなのが、「モデルに見せない実行時設定（Runtime Context）」「グラフ状態に永続化する独自フィールド（Custom State）」、そして「ストリーミング UI で要約トークンを隠す方法」の 3 点です。

### 5.1 Runtime Context vs Custom State の使い分け

| 比較項目 | Runtime Context（`context_schema`） | Custom State（`state_schema`） |
| :--- | :--- | :--- |
| **可変性** | 実行中は不変（Read-only） | ツールや Middleware から動的に更新可能 |
| **チェックポイント保存** | 保存されない（呼び出し時に毎回渡す） | LangGraph チェックポイントに永続化される |
| **典型的な用途** | `user_id`, `tenant_id`, API キー, DB コネクション, Feature Flags | 調査済み URL リスト, リトライ回数, 監査カウンタ |
| **サブエージェント伝播** | 全サブエージェントへ自動伝播 | 宣言的 `SubAgent` は継承（`CompiledSubAgent` 等は手動統一） |

```python
from dataclasses import dataclass
from deepagents import DeepAgentState, create_deep_agent
from langchain.tools import ToolRuntime, tool


@dataclass
class AppContext:
    """実行ごとに外部から注入する不変のランタイムコンテキスト。"""
    tenant_id: str
    user_id: str


class AuditState(DeepAgentState):
    """DeepAgentState を継承し、DeltaChannel リデューサを維持したまま独自項目を追加。"""
    inspected_services: list[str]


@tool
def fetch_tenant_metrics(service: str, runtime: ToolRuntime[AppContext]) -> str:
    """テナント固有のメトリクスを取得する（APIキーやテナントIDをプロンプトに出さない）。"""
    tenant_id = runtime.context.tenant_id
    return f"[{tenant_id}] metrics for {service}: error_rate=0.02%"
```

`state_schema` を独自定義する際は、必ず **`DeepAgentState` を継承** してください。これにより `messages` チャンネルの組み込み `DeltaChannel` リデューサが保持され、会話が長くなってもチェックポイント容量の増加が線形に抑えられます。

### 5.2 ストリーミング時の要約トークン（`lc_source="summarization"`）フィルタリング

`agent.stream(..., stream_mode="messages")` でフロントエンドへトークンを逐次送信している場合、`SummarizationMiddleware` が裏側で生成した内部サマリーのトークンもストリームに流れてきます。ユーザー向けチャット画面に内部要約がそのまま表示されるのを防ぐには、`metadata.get("lc_source")` を検査してスキップします。

```python
# ストリーミング出力から SummarizationMiddleware の内部生成トークンを除外する例
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "長時間の調査を続けてください"}]},
    stream_mode="messages",
):
    if metadata.get("lc_source") == "summarization":
        # 必要に応じて UI 側に「会話履歴を圧縮中...」インジケータだけを表示する
        continue
    print(token.content, end="", flush=True)
```

---

## 6. まとめと次回の実践編に向けて

本記事（Part 3 理論編）では、DeepAgents の Context Engineering を支える 4 本柱を解説しました。

- **Offloading**: 20,000 トークン（約 80KB）超のツール結果を `/large_tool_results/<tool_call_id>` に自動退避し、`ToolMessage` をパス＋先頭 10 行プレビューへ軽量化する。
- **Summarization**: 85% 閾値（未登録時は 170,000 トークン）で非破壊要約を実行しつつ、原文を `/conversation_history/` に保全する。
- **Overflow Recovery**: 予期せぬ `ContextOverflowError` 時も即時フォールバック要約でリトライし、ループの停止を防ぐ。
- **Prompt Caching**: `MemoryMiddleware` をキャッシュ層の後ろに置くスタック設計により、キャッシュヒット率を最大化する。

次回の **Part 3 実践編** では、大量ログを返すシミュレータツール `dump_large_log` を実際に走らせ、**「600 行（約 36KB）ではオフロードされずインライン保持されたケース」** と **「1,400 行（約 120KB）で実際に `/large_tool_results/` へオフロードされ、エージェントが自律的に `grep` で `ERROR` 行を特定したケース」** を実測データで対比検証します。
