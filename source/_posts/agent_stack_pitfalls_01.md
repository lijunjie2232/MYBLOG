---
title: 実践トラブルシュート記録 Part 1：環境構築・OpenAI 互換 MoE モデル接続・レート制限・Structured Output で踏んだ落とし穴と解決総括
date: 2026-08-03 12:37:50
categories:
  - AI
tags:
  - Python
  - LLM
  - AI Agent
  - LangChain
  - LangGraph
  - DeepAgents
  - トラブルシューティング
  - 実践記録
lang: ja
description: LangChain v1.5.2・LangGraph・DeepAgents 0.7.41 の全21ラボ実践で実際に遭遇した環境構築・OpenAI互換プロキシ・MoEモデル選定・429レート制限・Structured Outputの5つの落とし穴と根本解決を体系的に記録する。
---

AI エージェント開発のチュートリアルや公式ドキュメントは「正常系」を美しく見せてくれますが、いざローカルの Windows 環境からセルフホストの OpenAI 互換 API プロキシ経由で **LangChain v1.5.2 / LangGraph / DeepAgents 0.7.41** を組み合わせて全 21 ラボ（各フレームワーク 7 ラボずつ）を完走しようとすると、ドキュメントの行間に潜む現実の罠が次々と牙を剥きます。

本記事（Part 1）では、インフラ・接続・モデル選定・レート制限・構造化出力といった**基盤レイヤーで実際に踏み抜いた 5 つの落とし穴**を、以下の統一フォーマットで赤裸々に記録します。

- **問題**：何が起きたか（実際のエラーメッセージ・ログ・計測値）
- **発生条件**：どのような環境・コード・実行パターンで再現するか
- **試行錯誤**：解決に至るまでに試したアプローチと失敗
- **根本原因**：フレームワークやプロトコル仕様のどこに原因があるのか
- **まとめ・再発防止策**：今後同じ罠を踏まないためのテンプレートと運用ルール

<!-- more -->

---

## 全体像：基盤レイヤーのトラブルシュート決定木

まずは、今回の 21 ラボ実践で遭遇した基盤系トラブルの切り分けフローを俯瞰しておきます。ローカル実行から API 疎通、モデル応答、並列実行、構造化出力まで、どの段階で詰まっているかを即座に特定するための地図です。

{% mermaid %}
flowchart TD
    Start["ラボスクリプト実行開始"] --> EnvCheck{"PowerShell 構文エラー<br/>または ModuleNotFoundError?"}
    EnvCheck -- "Yes" --> Fix1["#1 PowerShell 5.1 は ';'<br/>venv の python.exe を明示指定"]
    EnvCheck -- "No" --> ApiCall["OpenAI 互換 API へリクエスト送信"]

    ApiCall --> HttpStatus{"HTTP ステータス / タイムアウト"}
    HttpStatus -- "400 not a chat model" --> Fix3A["#3 チャット対応モデルへ切替<br/>(d3-flash を除外)"]
    HttpStatus -- "25s Timeout / 9-17s 低速" --> Fix3B["#3 高速 MoE 選定 +<br/>timeout / max_retries を明示"]
    HttpStatus -- "429 Rate limit exceeded" --> Fix4["#4 20 RPM 制限を考慮し<br/>ペーシング + 指数バックオフ"]
    HttpStatus -- "200 OK" --> ParseCheck{"レスポンス解析・警告確認"}

    ParseCheck -- "PydanticSerializationUnexpectedValue" --> Fix2["#2 プロキシの metadata 型不一致<br/>警告の無害性を確認しフィルタ管理"]
    ParseCheck -- "Structured Output 失敗" --> Fix5["#5 ProviderStrategy から<br/>ToolStrategy へフォールバック"]
    ParseCheck -- "正常完了" --> Done["ラボ検証成功"]

    style Start fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style Done fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style Fix1 fill:#fef3c7,stroke:#d97706,color:#0f172a
    style Fix2 fill:#fef3c7,stroke:#d97706,color:#0f172a
    style Fix3A fill:#fee2e2,stroke:#dc2626,color:#0f172a
    style Fix3B fill:#fee2e2,stroke:#dc2626,color:#0f172a
    style Fix4 fill:#ede9fe,stroke:#7c3aed,color:#0f172a
    style Fix5 fill:#fce7f3,stroke:#db2777,color:#0f172a
{% endmermaid %}

---

## 落とし穴 1：Windows PowerShell 5.1 の `&&` 構文エラーと新規 venv の `ModuleNotFoundError`

### 問題

検証用の仮想環境（`.venv`）を新規作成して最初のラボスクリプトを実行しようとした瞬間、以下の 2 種類のエラーに連続して足止めされました。

```text
At line:1 char:24
+ .\.venv\Scripts\activate && python run_lab1.py
+                        ~~
The token '&&' is not a valid statement separator in this version.
```

さらに、セミコロン `;` でつなぎ直したり、別シェルから `python` を直接叩いたりすると、今度はパッケージが入っているはずなのに以下が発生します。

```text
Traceback (most recent call last):
  File "run_lab1.py", line 3, in <module>
    from langchain.agents import create_agent
ModuleNotFoundError: No module named 'langchain'
```

### 発生条件

- OS：Windows（デフォルトの **Windows PowerShell 5.1**、`pwsh` 7 系ではない）
- 新規に `python -m venv .venv` で仮想環境を作成し、自動実行ツールや単発コマンドセッションから `activate` と `python` を連続実行しようとした場合
- もしくは、システムのグローバル Python と `.venv` 内の Python が混在しており、`pip install` 先と実行時の `python` バイナリがズレている場合

### 試行錯誤

1. Linux / macOS のクセで `cd lab && .\.venv\Scripts\activate && python run_lab1.py` と書き、PowerShell 5.1 のパーサーエラーに弾かれた。
2. `;` でつないで `.\.venv\Scripts\activate; python run_lab1.py` にしたが、実行ポリシー（ExecutionPolicy）やサブシェル環境の影響で `activate.ps1` が静かに失敗・スキップされ、グローバル Python が呼ばれて `ModuleNotFoundError: No module named 'langchain'` が出た。
3. 仮想環境をアクティベートするのをやめ、**`.venv\Scripts\python.exe` をフルパス（または相対パス）で直接呼び出す**方式に切り替えたところ、100% 確実に意図した環境で実行できるようになった。

### 根本原因

- **PowerShell 5.1 の仕様**：`&&` や `||` のパイプラインチェーン演算子が導入されたのは PowerShell 7.0 以降であり、Windows 標準搭載の 5.1 では構文エラーになります。
- **シェルセッション非永続性と `activate` の脆さ**：コマンド実行のたびに新しいシェルセッションが立ち上がる環境（CI、タスクランナー、エージェントのコマンド実行ツールなど）では、`activate` スクリプトによる `PATH` 書き換えはセッションをまたぐと消えます。また、PowerShell の `ExecutionPolicy` が `Restricted` のままだと `Activate.ps1` 自体がブロックされます。

### まとめ・再発防止策

Windows PowerShell 5.1 環境で Python プロジェクトを回す際は、**「`activate` に頼らない」「依存コマンドの連結は `$?` または直接パス呼び出しで制御する」**のが鉄則です。

```powershell
# NG: PowerShell 5.1 では && が構文エラーになる
# .\.venv\Scripts\activate && python run_langchain_labs.py

# OK: venv 内の python.exe を直接指定し、インストールも python -m pip に統一する
.\.venv\Scripts\python.exe -m pip install -U langchain==1.5.2 langgraph deepagents==0.7.41
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe run_langchain_labs.py
}
```

さらにスクリプト側でも、起動時に実行パスと主要パッケージのバージョンを 1 行出力しておくと、環境の取り違えを 1 秒で見抜けます。

```python
import sys
import langchain
import langgraph
import deepagents

# 実行中のインタープリタとライブラリバージョンを冒頭で必ず確認する
print(f"Python: {sys.executable}")
print(
    f"langchain={langchain.__version__}, "
    f"deepagents={deepagents.__version__}"
)
```

---

## 落とし穴 2：セルフホスト OpenAI 互換 API プロキシが `metadata` を `list` で返すことによる `PydanticSerializationUnexpectedValue` 警告

### 問題

LangChain の `ChatOpenAI` 経由でセルフホストの OpenAI 互換 API プロキシを呼び出すと、エージェントの推論自体は成功しているにもかかわらず、標準エラー出力に大量の Pydantic シリアライズ警告が流れ続けました。

```text
UserWarning: Pydantic serializer warnings:
  PydanticSerializationUnexpectedValue(
    Expected `dict[str, str]` - serialized value may not be as expected [field_name='metadata', input_value=[], input_type=list]
  )
```

ラボ出力を JSON 形式で収集・検証している最中に、この警告がログを埋め尽くし、「本当にスキーマ破損が起きていないか？」と調査に時間を奪われました。

### 発生条件

- セルフホストの OpenAI 互換ゲートウェイ（一部の vLLM / LiteLLM / 独自リバースプロキシ実装）を利用している
- `openai` Python SDK（Pydantic v2 ベース）経由でレスポンスオブジェクトを `model_dump()` するタイミング（LangChain の `langchain-openai` が内部でレスポンスを辞書化する際など）
- プロキシ側が OpenAI 公式仕様では `dict[str, str] | None` であるはずの `metadata` フィールドを、空のリスト `[]` として返却している

### 試行錯誤

1. 最初は自分の `ChatOpenAI` 初期化コードで `metadata=[]` を渡してしまったのかと疑い、呼び出し側の引数を総点検したが該当箇所はなし。
2. `httpx` のログレベルを上げて生の HTTP レスポンス JSON を確認したところ、プロキシから返る JSON のトップレベルに `"metadata": []` が含まれていた。
3. LangChain 側のバグではなく、OpenAI SDK の Pydantic モデル定義（`Optional[Dict[str, str]]`）とプロキシのレスポンス型の不一致であることが確定。

### 根本原因

OpenAI の公式 Chat Completions API 仕様では、`metadata` はキーと値のペア（JSON Object / Python `dict[str, str]`）または `null` です。しかし一部の OpenAI 互換プロキシ実装（特に PHP や Go の一部シリアライザ、あるいは動的型付けの変換層）では、空の連想配列を JSON の `[]` としてシリアライズしてしまうことがあります。

Pydantic v2 はデシリアライズ時に寛容に値を受け入れる一方で、LangChain 内部で `response.model_dump()` を呼んだ際に「定義型 `dict[str, str]` と実値 `list` が一致しない」と検知し、`PydanticSerializationUnexpectedValue` の `UserWarning` を送出します。

### まとめ・再発防止策

プロキシ側の改修権限がある場合は `"metadata": {}` または `null` を返すように修正するのがベストです。プロキシ側を変更できない利用者側では、**「無害な既知の型不一致であること」を検証したうえで、対象警告だけをピンポイントに抑制**します。

```python
import warnings

# セルフホスト OpenAI 互換プロキシが metadata=[] を返すことによる
# 既知の Pydantic v2 シリアライズ警告のみをピンポイントで抑制する
warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*metadata.*",
    category=UserWarning,
)
```

> [!WARNING]
> `warnings.filterwarnings("ignore", category=UserWarning)` のように広範囲に潰すと、LangChain や DeepAgents の非推奨 API 警告（DeprecationWarning / UserWarning）まで見逃してしまいます。必ず `message` 正規表現で `metadata` 起因のものに限定してください。

---

## 落とし穴 3：モデル選定の罠（400 Error / 25s Timeout / 低速応答）と `timeout`/`max_retries` 未設定によるハング

### 問題

全 21 ラボ（ツール呼び出し、並列 `Send`、Subagent 委譲、HITL 再開など多数回の LLM 往復が発生）を回すにあたり、利用可能なセルフホストエンドポイント上の複数モデルをベンチマークしたところ、モデルによって致命的な挙動差が出ました。

| モデル名 | テスト結果 | 基本応答レイテンシ | Tool Calling レイテンシ | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| `d3-flash` | `400 BadRequestError: not a chat model` | — | — | **利用不可**（チャット API 非対応） |
| `DeepSeek-V4-Flash` | `APITimeoutError`（25s で打ち切り） | > 25.0s | Timeout | **利用不可**（高負荷・ハング要因） |
| `DeepSeek-V4.1-Flash` | 成功するが極めて低速 | 9.0s 〜 17.0s | 15s 以上 | **開発サイクルに不向き** |
| `MiniCPM-V-4.7-35B-A3B` | 高速・安定（Active 3B の MoE モデル） | **0.64s** | **1.21s** | **採用**（全 21 ラボ完走） |

さらに、`ChatOpenAI` に `timeout` や `max_retries` を明示せずデフォルトのまま実行した際、遅いモデルや一時的なプロキシ詰まりに当たると、スクリプトが数分間無応答のままハングし、どこで止まっているのか分からない状態に陥りました。

### 発生条件

- 複数のモデルがぶら下がっている OpenAI 互換プロキシ環境で、モデル名だけを見て十分な疎通・ツール呼び出しテストをせずにエージェント本編へ突入した場合
- `ChatOpenAI(model=..., base_url=..., api_key=...)` に `timeout` と `max_retries` を指定していない場合（デフォルトの `openai` SDK タイムアウトは 600 秒＝10 分）

### 試行錯誤

1. まず軽量そうな名前だった `d3-flash` を指定 → 即座に `400 Error: not a chat model` で失敗（Completion 専用または別用途エンドポイントだった）。
2. 次に `DeepSeek-V4-Flash` を試す → 25 秒待ってもレスポンスが返らずタイムアウト。
3. `DeepSeek-V4.1-Flash` に切り替えたところ応答は返ったものの、1 ターンに 9〜17 秒かかり、ツール呼び出しを 3〜4 回繰り返すマルチステップラボ（例：LangChain Lab 7 のマルチエージェントや DeepAgents Lab 5 の Subagent）では 1 ラボだけで 1 分超を要する見込みになった。
4. そこで全候補モデルに対して **「基本応答」と「Tool Calling 応答」の 2 項目スモークテスト**を実行したところ、MoE（Mixture of Experts）アーキテクチャの **`MiniCPM-V-4.7-35B-A3B`** が **基本応答 0.64 秒 / Tool Calling 1.21 秒** という圧倒的な速度とツール呼び出し精度を示した。
5. 実際、採用後の全ラボ計測でも、LangChain Lab 5（Middleware）が **0.39 秒**（`wrap_model_call: 381.5ms`）、LangGraph Lab 7（Functional API）が **0.57 秒**、DeepAgents Lab 1（Harness Basics）が **2.70 秒** と、快適な反復開発が可能になった。

### 根本原因

- **エージェント開発は LLM 呼び出し回数の掛け算**：単発チャットなら 10 秒待てても、ReAct ループ＋Subagent＋Structured Output の組み合わせでは 1 タスクあたり 4〜10 回の推論が直列に走ります。MoE モデル（総パラメータ 35B / 推論時アクティブ 3B）は、知識量と推論速度のバランスがエージェントの反復ツール呼び出しに極めてよく噛み合います。
- **デフォルトタイムアウト 600 秒の罠**：`ChatOpenAI` は `timeout` を省略すると `openai` クライアントのデフォルト（10 分）が適用されるため、バックエンドが詰まったときに即座に失敗せず、開発者の時間を無為に溶かします。

### まとめ・再発防止策

エージェント開発を始める前に、必ず **10 秒で終わるモデル選定スモークテスト**を回し、`ChatOpenAI` のファクトリ関数で `timeout` と `max_retries` を強制設定します。

```python
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


def create_chat_model(
    model_name: str = "MiniCPM-V-4.7-35B-A3B",
    temperature: float = 0.0,
    timeout: float = 45.0,
    max_retries: int = 2,
) -> ChatOpenAI:
    """タイムアウトとリトライを明示した安全な ChatOpenAI ファクトリ。"""
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    if not base_url or not api_key:
        raise EnvironmentError("環境変数 BASE_URL と API_KEY を設定してください。")

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        # デフォルトの600秒ハングを防ぎ、異常を早期検知する
        timeout=timeout,
        # 一時的なネットワーク揺らぎのみ自動リトライ
        max_retries=max_retries,
    )


@tool
def ping_tool(target: str) -> str:
    """疎通確認用のダミーツール。"""
    return f"pong:{target}"


def smoke_test_model(model_name: str) -> None:
    """基本応答と Tool Calling の両方を計測するスモークテスト。"""
    llm = create_chat_model(model_name=model_name, timeout=25.0, max_retries=0)

    # 1. 基本応答の計測
    t0 = time.perf_counter()
    resp = llm.invoke("1単語で OK とだけ返してください。")
    t_basic = time.perf_counter() - t0

    # 2. Tool Calling 応答の計測
    llm_with_tools = llm.bind_tools([ping_tool])
    t1 = time.perf_counter()
    tool_resp = llm_with_tools.invoke("target='api' で ping_tool を呼んでください。")
    t_tool = time.perf_counter() - t1

    print(
        f"[{model_name}] basic={t_basic:.2f}s, tool_call={t_tool:.2f}s, "
        f"calls={len(tool_resp.tool_calls)}"
    )
```

---

## 落とし穴 4：並列ラボ実行時に踏んだ `429 Rate limit exceeded: maximum 20 requests per minute`

### 問題

高速な `MiniCPM-V-4.7-35B-A3B` を手に入れたことで調子に乗り、7 つのラボスクリプトや LangGraph の `Send` による並列ファンアウト、さらに複数の検証プロセスを同時に走らせたところ、即座に HTTP 429 で叩き落とされました。

```text
openai.RateLimitError: Error code: 429 - {
  'error': {
    'message': 'Rate limit exceeded: maximum 20 requests per minute',
    'type': 'rate_limit_error'
  }
}
```

特に LangChain Lab 6（HITL で複数ターン＋検証、実測 `39.01s`）や Lab 7（Supervisor + Subagent の入れ子推論、実測 `19.42s`）を直列・並列で連続実行すると、1 ラボあたり 3〜6 リクエストを消費するため、わずか 30 秒足らずで **20 RPM（Requests Per Minute）** の上限に到達してしまいます。

### 発生条件

- API プロキシ側に **20 requests / minute（平均 3 秒に 1 リクエスト）** のレート制限が課されている
- 複数のラボランナーを並列起動したり、LangGraph の `Send` で 3 並列ワーカーを一気に投げた直後に次のラボを待機なしで開始した場合
- しかも `MiniCPM-V-4.7-35B-A3B` が 0.64s〜1.21s と**速すぎる**ため、低速モデル時代には顕在化しなかった RPM 制限を容易に突き抜ける

### 試行錯誤

1. 最初は `ChatOpenAI(max_retries=2)` だけで乗り切れると考えたが、SDK の短い自動リトライ間隔では 60 秒スライディングウィンドウが回復せず、結局 `RateLimitError` でラボ全体が落ちた。
2. ラボ間に固定で `time.sleep(2)` を入れてみたが、1 ラボ内で 5 回 LLM を呼ぶケース（例：並列ツール呼び出しや Subagent）では依然としてバーストが発生した。
3. 最終的に、**① ラボランナーの直列実行＋ラボ間ペーシング（クールダウン）** と、**② `RateLimitError` 専用の指数バックオフ＋ジッター（Exponential Backoff with Jitter）ラッパー** の 2 段構えにしたことで、全 21 ラボの完走率が 100% になった。

### 根本原因

モデルが高速化すると、スループットのボトルネックが「推論時間」から「API ゲートウェイの RPM 制限」へ移動します。特にエージェントは 1 回の `agent.invoke()` の裏で複数回の HTTP リクエスト（プランニング → ツール呼び出し → ツール結果の統合 → Subagent 推論）を発行するため、**ユーザー目線の「1 実行」がゲートウェイ目線では「5 リクエスト」**としてカウントされます。

以下のフローのように、事前ペーシングと事後バックオフの両方を組み合わせるのが堅牢です。

{% mermaid %}
flowchart LR
    Req["ラボ / エージェント実行要求"] --> Pace["事前ペーシング<br/>(トークンバケット / 最小間隔 3.2s)"]
    Pace --> Call["LLM API 呼び出し"]
    Call --> Check{"結果判定"}
    Check -- "200 OK" --> Next["次のステップ / ラボへ"]
    Check -- "429 Rate Limit" --> RetryCheck{"リトライ上限<br/>未満?"}
    RetryCheck -- "Yes" --> Backoff["指数バックオフ待機<br/>min(60s, 4s * 2^attempt + jitter)"]
    Backoff --> Call
    RetryCheck -- "No" --> Fail["エラー送出・ログ記録"]

    style Req fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style Pace fill:#fef3c7,stroke:#d97706,color:#0f172a
    style Backoff fill:#ede9fe,stroke:#7c3aed,color:#0f172a
    style Next fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style Fail fill:#fee2e2,stroke:#dc2626,color:#0f172a
{% endmermaid %}

### まとめ・再発防止策

テストスイートやバッチ処理でエージェントを連続実行する際は、以下のようなリトライ＋ペーシングユーティリティを共通モジュール化しておきます。LangChain の `InMemoryRateLimiter` を `ChatOpenAI(rate_limiter=...)` に渡すアプローチも非常に有効です。

```python
import random
import time
from typing import Callable, TypeVar
from langchain_core.rate_limiters import InMemoryRateLimiter
from openai import RateLimitError

T = TypeVar("T")

# 20 RPM 制限に対し、安全マージンを取って 18 RPM (0.3 req/sec) に制限する RateLimiter
safe_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.3,   # 約 3.3 秒に 1 リクエスト
    check_every_n_seconds=0.1, # ポーリング間隔
    max_bucket_size=3,         # 瞬間バーストを最大 3 リクエストまでに抑制
)


def run_with_rate_limit_backoff(
    fn: Callable[[], T],
    max_attempts: int = 5,
    base_delay_s: float = 4.0,
    max_delay_s: float = 60.0,
) -> T:
    """429 RateLimitError 発生時に指数バックオフ＋ジッターで再試行するラッパー。"""
    for attempt in range(max_attempts):
        try:
            return fn()
        except RateLimitError as exc:
            if attempt == max_attempts - 1:
                raise
            # 指数的に待機時間を延ばし、ランダムなジッターで再衝突を防ぐ
            jitter = random.uniform(0.5, 2.0)
            wait_s = min(max_delay_s, base_delay_s * (2 ** attempt) + jitter)
            print(
                f"[429 Retry {attempt + 1}/{max_attempts}] "
                f"{wait_s:.1f}s 待機します... ({exc})"
            )
            time.sleep(wait_s)
    raise RuntimeError("到達不能コード")
```

---

## 落とし穴 5：OpenAI 互換カスタムエンドポイントでの Structured Output（`ProviderStrategy` vs `ToolStrategy`）

### 問題

LangChain v1.5.2 の `create_agent(..., response_format=IncidentReport)` を使い、障害レポートを Pydantic スキーマ（`service`, `severity`, `root_cause`, `action_items`）で厳密に構造化抽出しようとした際、OpenAI 互換カスタムエンドポイントでは**プロバイダネイティブの `json_schema` 構造化出力（`ProviderStrategy`）が不安定、あるいは未対応で失敗する**という落とし穴に直面しました。

### 発生条件

- `ChatOpenAI` で公式 OpenAI API ではなく、セルフホストの OpenAI 互換エンドポイント（`base_url` 指定）に接続している
- LangChain v1.5.2 の `create_agent()` に `response_format=PydanticModel` をそのまま渡すか、`ProviderStrategy` を明示指定した
- エンドポイント側のサービングエンジンが OpenAI の `response_format: {"type": "json_schema", "strict": true}`（Structured Outputs API）を完全実装していない、またはモデル固有の制約でスキーマ違反や 400 エラーを返す

### 試行錯誤

1. まず `ProviderStrategy`（ネイティブ JSON Schema 制約）の挙動を確認したところ、OpenAI 互換プロキシ越しではバックエンド実装に強く依存し、一部スキーマ（`Literal` やネスト配列など）でプロキシがパラメータを無視したり、プレーンテキスト混じりの出力になってパースに失敗するケースがあった。
2. LangChain v1 の `langchain.agents.structured_output` モジュールを確認したところ、ネイティブ制約に頼る `ProviderStrategy` とは別に、**モデルの Tool Calling 機能を使って構造化オブジェクトを生成させる `ToolStrategy`** が用意されていることが分かった。
3. Lab 3（`lab3_structured_streaming`）で `response_format=ToolStrategy(IncidentReport)` を明示したところ、Tool Calling に強い `MiniCPM-V-4.7-35B-A3B` で一発で完璧な構造化レスポンスが得られた（実行時間 **7.70 秒**、後続のストリーミング検証も **27 チャンク**で完走）。

実際の Lab 3 で取得できた `structured_response` は以下の通りです。

```json
{
  "service": "payment-gateway",
  "severity": "P1",
  "root_cause": "DBコネクションプールの枯渇",
  "action_items": [
    "プール上限を50から150へ引き上げる",
    "スロークエリの調査を実施する"
  ]
}
```

### 根本原因

LangChain v1 の `create_agent` における Structured Output には 2 つの戦略が存在します（参考：[langchain-ai/langchain](https://github.com/langchain-ai/langchain)）。

| 戦略クラス | 内部メカニズム | 強み | OpenAI 互換プロキシでの落とし穴 |
| :--- | :--- | :--- | :--- |
| **`ProviderStrategy`** | OpenAI 等の `response_format: {type: "json_schema"}` を直接送信 | 公式 API ではトークンレベルでスキーマ厳守 | 互換プロキシや非対応モデルで `400 Error` やサイレント無視が起きやすい |
| **`ToolStrategy`** | Pydantic モデルを人工的なツール定義としてバインドし、Tool Call 引数を検証 | **Tool Calling 対応モデルならどの互換 API でも動く**。検証エラー時の自動リトライも効く | 1 ステップぶんツール呼び出し形式を経由する |

OpenAI 互換エンドポイントでは「JSON Schema 厳格モード（Structured Outputs）」の実装成熟度にバラつきがある一方、「Tool Calling（Function Calling）」はほぼすべてのモダンなオープンウェイトモデル／サービングエンジンで最優先にサポートされています。

### まとめ・再発防止策

セルフホストやサードパーティの OpenAI 互換エンドポイントを使う場合、**最初から `ToolStrategy` を明示指定する**のが最も移植性と安定性の高い選択です。

```python
import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class IncidentReport(BaseModel):
    """障害インシデントの構造化レポートスキーマ。"""
    service: str = Field(description="障害が発生したサービス名")
    severity: Literal["P1", "P2", "P3"] = Field(description="重要度")
    root_cause: str = Field(description="推定される根本原因")
    action_items: list[str] = Field(description="実施すべき対応アクション一覧")


# OpenAI 互換エンドポイントへの接続（キーは環境変数から取得）
llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
    timeout=45.0,
    max_retries=2,
)

# 互換プロキシでの json_schema 非対応を回避するため ToolStrategy を明示
agent = create_agent(
    model=llm,
    tools=[],
    response_format=ToolStrategy(IncidentReport),
)

result = agent.invoke({
    "messages": [
        (
            "user",
            "payment-gateway で DB コネクションプール枯渇による P1 障害が発生。"
            "プール上限を50から150へ引き上げ、スロークエリ調査を行う報告書を作成して。",
        )
    ]
})

# result["structured_response"] に検証済みの IncidentReport インスタンスが格納される
report: IncidentReport = result["structured_response"]
print(report.model_dump_json(indent=2, ensure_ascii=False))
```

---

## Part 1 まとめ：基盤レイヤーのチェックリスト

今回の 5 つの落とし穴を振り返ると、どれも「コードロジックの複雑さ」ではなく、**実行環境・プロキシ仕様・モデル特性・レート制限・プロトコル互換性**という周辺境界で発生していました。新しい AI エージェント検証環境を立ち上げる際は、以下の 5 項目を初日のチェックリストにしておくと無駄なデバッグ時間を大幅に削れます。

1. **Windows PowerShell 5.1**：`&&` を使わず、`.\.venv\Scripts\python.exe` を直接呼ぶ。
2. **プロキシの型差異**：`PydanticSerializationUnexpectedValue` のような警告は生レスポンスを確認したうえでピンポイントに制御する。
3. **モデル選定とタイムアウト**：本編前に「基本応答 + Tool Calling」の 2 項目スモークテストを回し、`ChatOpenAI` に必ず `timeout` と `max_retries` を設定する。
4. **RPM レート制限**：高速 MoE モデルほど 20 RPM 制限に当たりやすいため、`InMemoryRateLimiter`＋指数バックオフを標準装備する。
5. **Structured Output**：OpenAI 互換カスタムエンドポイントでは `ProviderStrategy` に賭けず、`ToolStrategy` を明示する。

次回の **Part 2** では、いよいよフレームワーク内部の深部——**DeepAgents 0.7.41 の `StoreBackend` 破壊的変更、`CompositeBackend` のプレフィックス剥離、Large Tool Result Offloading 閾値と 63 秒レイテンシ急増、そして LangGraph の `interrupt()` 冪等性と Time Travel のチェックポイント選択**——で踏んだ 5 つの落とし穴を徹底解剖します。
