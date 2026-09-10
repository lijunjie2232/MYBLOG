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
