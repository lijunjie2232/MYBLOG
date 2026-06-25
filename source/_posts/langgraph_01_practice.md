---
title: "LangGraph Part 1 実践編：StateGraph と Reducer で構築する最小ステートフルパイプライン（MiniCPM-V-4.7-35B-A3B 実践）"
date: 2026-05-30 15:09:48
categories: [AI]
tags: [LangGraph, StateGraph, Reducer, MiniCPM, Python, 実践]
lang: ja
description: "LangGraph Part 1 理論編の概念を実践へ落とし込み、MiniCPM-V-4.7-35B-A3B を用いて StateGraph・TypedDict・operator.add Reducer による最小ステートフルパイプラインを構築・検証する。"
---

# LangGraph Part 1 実践編：StateGraph と Reducer で構築する最小ステートフルパイプライン

本記事では、[LangGraph Part 1 理論編](/2026/05/27/langgraph_01_theory/) で学んだ **Pregel BSP の Super-step**、**Channel と Reducer**、**Raw State 原則** を、実際の実行可能な Python コードへ落とし込みます。

LLM バックエンドには OpenAI 互換エンドポイント経由で **`MiniCPM-V-4.7-35B-A3B`** を使用し、「入力トピックの正規化 → LLM による 1 文要約」を行う最小ステートフルパイプラインを構築して、実測ログから内部挙動を検証します。

---

## 1. 理論編の振り返りと実践アーキテクチャ

Part 1 理論編では、State 更新の基本契約について次のように整理しました。

> **State 更新の基本契約**：ノードは State 全体をインプレースで書き換えるのではなく、自身が変更したいキーだけを含む**部分更新辞書（Partial Update）**を返却します。LangGraph は各チャネルに紐づく **Reducer** を介して、既存値（`left`）と新しい更新値（`right`）を安全にマージします。

また、State の設計方針として次の原則も確認しました。

> **Raw State の原則**：State には整形済みのプロンプト文字列ではなく**生データ（Raw Data）**のみを格納し、プロンプトの組み立ては各ノード内でオンデマンドに行います。

今回の実践ラボでは、この 2 つの原則を検証するために 3 つのチャネルを持つ `PipelineState` を設計します。

- `topic: str`（デフォルト Reducer＝上書き）：生のトピック文字列を保持し、`normalize` ノードで前後空白を除去した値へ上書きする。
- `logs: Annotated[list[str], operator.add]`（追記 Reducer）：各ノードの実行証跡を自動的にリスト連結して蓄積する。
- `summary: str`（デフォルト Reducer＝上書き）：`summarize` ノードが `MiniCPM-V-4.7-35B-A3B` を呼び出して生成した 1 文要約を格納する。

{% mermaid %}
flowchart LR
    IN(["入力辞書<br/>topic: '  Pregel BSP モデル  '<br/>logs: [] / summary: ''"]) --> N1["normalize ノード<br/>空白除去 & ログ追記"]
    N1 -->|"topic 上書き<br/>logs += ['normalized:...']"| N2["summarize ノード<br/>MiniCPM-V-4.7-35B-A3B 呼び出し"]
    N2 -->|"summary 上書き<br/>logs += ['summarized']"| OUT(["最終 State<br/>全チャネルの統合結果"])

    style IN fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    style N1 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100
    style N2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style OUT fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
{% endmermaid %}

---

## 2. 環境準備と LLM クライアント初期化

API キーやエンドポイント URL はソースコードへハードコードせず、必ず環境変数（`BASE_URL`、`API_KEY`）から読み込みます。`.env` ファイルを用意した上で、`python-dotenv` と `langchain-openai` を組み合わせて `ChatOpenAI` インスタンスを生成します。

```python
import operator
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# .env ファイルから環境変数（BASE_URL, API_KEY）をロードする
load_dotenv()

# 接続先エンドポイントと API キーは必ず環境変数経由で取得する
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"

def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """OpenAI 互換エンドポイント経由で MiniCPM-V-4.7-35B-A3B を初期化するヘルパー。"""
    return ChatOpenAI(
        model=MODEL_NAME,        # 使用するモデル識別子
        base_url=BASE_URL,       # 環境変数から取得した OpenAI 互換 API ベース URL
        api_key=API_KEY,         # 環境変数から取得した認証キー
        temperature=temperature, # 再現性を高めるためデフォルトは 0.0（決定的出力）
        max_tokens=256,          # 1 文要約に十分なトークン上限を設定
        timeout=30,              # ネットワークハングを防ぐためのタイムアウト（秒）
        max_retries=1,           # クライアント層での最小リトライ回数
    )
```

---

## 3. StateGraph と Reducer パイプラインの実装

続いて、`PipelineState` の定義、2 つのノード関数（`normalize_node` / `summarize_node`）、そして `StateGraph` の配線とコンパイルを実装します。各ノードが**変更したいキーのみを部分更新辞書として返却している点**に注目してください。

```python
def run_lab1() -> dict:
    # 決定的な要約を得るため temperature=0.0 で LLM を生成
    llm = make_llm(temperature=0.0)

    # 1. State スキーマの定義（TypedDict を採用してオーバーヘッドを最小化）
    class PipelineState(TypedDict):
        # デフォルト Reducer（Last-Write-Wins）：新しい値で上書きされる
        topic: str
        # operator.add Reducer：ノードが返す list[str] が既存リストへ自動連結される
        logs: Annotated[list[str], operator.add]
        # デフォルト Reducer：summarize_node の生成結果で上書きされる
        summary: str

    # 2. 前処理ノード：入力トピックの前後空白を除去し、処理ログを追記する
    def normalize_node(state: PipelineState) -> dict:
        cleaned = state["topic"].strip()
        # summary キーは返さない（変更しないチャネルはそのまま維持される）
        return {
            "topic": cleaned,
            "logs": [f"normalized:{cleaned}"],
        }

    # 3. LLM 推論ノード：Raw State の topic からオンデマンドでプロンプトを組み立てる
    def summarize_node(state: PipelineState) -> dict:
        prompt = f"「{state['topic']}」について日本語で1文で説明してください。"
        resp = llm.invoke(prompt)
        # topic キーは返さない（normalize_node の結果が保持される）
        return {
            "summary": resp.content,
            "logs": ["summarized"],
        }

    # 4. StateGraph の構築と compile() による構造検証
    graph = (
        StateGraph(PipelineState)           # スキーマを紐づけてビルダーを初期化
        .add_node("normalize", normalize_node)  # 前処理ノードを登録
        .add_node("summarize", summarize_node)  # LLM 要約ノードを登録
        .add_edge(START, "normalize")       # エントリポイント START -> normalize
        .add_edge("normalize", "summarize") # 固定エッジ normalize -> summarize
        .add_edge("summarize", END)         # 終端エッジ summarize -> END
        .compile()                          # 孤立ノード等を検証し CompiledStateGraph を生成
    )

    # 5. 初期 State を投入して同期実行（invoke）
    result = graph.invoke({
        "topic": "  Pregel BSP モデル  ",   # 前後に意図的な空白を含む生データ
        "logs": [],                         # 空リストで初期化
        "summary": "",                      # 空文字で初期化
    })
    return result

if __name__ == "__main__":
    import json
    output = run_lab1()
    print(json.dumps(output, ensure_ascii=False, indent=2))
```

---

## 4. 実測実行結果と内部挙動の徹底解剖

上記のコードを `MiniCPM-V-4.7-35B-A3B` 環境で実際に実行した結果（実行時間 `1.29 秒`）は以下の通りです。

```json
{
  "topic": "Pregel BSP モデル",
  "logs": [
    "normalized:Pregel BSP モデル",
    "summarized"
  ],
  "summary": "Pregel BSP（Bulk Synchronous Parallel）モデルは、グラフ処理を「メッセージ送信」「計算」「同期」の反復サイクルで実行する分散処理パラダイムです。"
}
```

### なぜこの出力になったのか？ Super-step と Reducer の時系列トレース

Part 1 理論編で解説した通り、LangGraph は各ノードを独立した Super-step で実行し、Super-step 境界のバリア同期で各チャネルの Reducer を適用します。

{% mermaid %}
sequenceDiagram
    participant Caller as 呼び出し元 (invoke)
    participant Runtime as Pregel ランタイム
    participant N1 as normalize (Super-step 1)
    participant N2 as summarize (Super-step 2)

    Caller->>Runtime: 初期入力 {topic: "  Pregel BSP モデル  ", logs: [], summary: ""}
    Runtime->>N1: State スナップショットを渡す
    N1-->>Runtime: 部分更新 {topic: "Pregel BSP モデル", logs: ["normalized:Pregel BSP モデル"]}
    Note over Runtime: Reducer 適用<br/>topic: 上書き -> "Pregel BSP モデル"<br/>logs: [] + ["normalized:..."]<br/>summary: 未更新 -> "" のまま保持
    Runtime->>N2: 更新後 State スナップショットを渡す
    N2-->>Runtime: 部分更新 {summary: "Pregel BSP（Bulk Synchronous...）", logs: ["summarized"]}
    Note over Runtime: Reducer 適用<br/>topic: 未更新 -> "Pregel BSP モデル" 保持<br/>logs: ["normalized:..."] + ["summarized"]<br/>summary: 上書き -> LLM回答
    Runtime-->>Caller: 最終 State 辞書を返却
{% endmermaid %}

ここで重要な観察ポイントが 3 つあります。

1. **上書き（Last-Write-Wins）と未指定キーの保持**：`normalize_node` は `summary` を返していませんが、`summary` は消えずに空文字のまま保持されます。同様に `summarize_node` は `topic` を返していませんが、正規化済みの `"Pregel BSP モデル"` がそのまま最終結果に残ります。
2. **`operator.add` Reducer による自動集約**：両ノードとも `logs` に要素 1 つだけのリストを返していますが、ランタイムが `left + right` を自動評価するため、最終結果には `["normalized:Pregel BSP モデル", "summarized"]` が順序通りに蓄積されました。
3. **Raw State 原則の効果**：`PipelineState` にはプロンプトテンプレート文字列を一切持たせず、`topic` という生データだけを保持しました。そのため、仮に「英語で 3 箇条書きにする」などプロンプト仕様が変わっても、`PipelineState` のスキーマや `normalize_node` を一切変更する必要がありません。

---

## 5. 応用パターン：`Overwrite` による Reducer チャネルのリセット

もしパイプラインの途中に「監査ログや一時エラーリストを一度クリアしてから再処理する」ステップを追加したい場合、単に `{"logs": []}` を返しても `operator.add` により既存ログが残ってしまいます。その場合は以下のように `Overwrite` を使います。

```python
from langgraph.types import Overwrite

def reset_logs_node(state: dict) -> dict:
    # 誤り: return {"logs": []}  -> existing_logs + [] となり消えない
    # 正解: Overwrite でラップすると operator.add をバイパスして直接置換される
    return {"logs": Overwrite(["logs_reset_complete"])}
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

Part 1 の基本実装で初学者が最も踏みやすい落とし穴を 4 点整理します。

1. **ノード内で `state["logs"].append(...)` のようにインプレース変更（Mutation）してはいけない**
   LangGraph はノードの返却辞書を Reducer に通して新しいスナップショットを構築します。引数で渡された `state` のリストや辞書を直接書き換えると、並列ブランチ間での競合状態（Race Condition）やチェックポイントの不整合を引き起こします。必ず新しいリストや辞書を `return` してください。
2. **集約型 Reducer フィールドに空リスト `[]` を返してもリセットされない**
   `Annotated[list[str], operator.add]` のチャネルに対して `{"logs": []}` を返すと `left + []` が評価され、過去データがそのまま残ります。リセットしたい場合は必ず `Overwrite([])` を使用してください。
3. **初期入力でキーを省略したときの `KeyError` に注意**
   `TypedDict` は Python の実行時には単なる `dict` です。`graph.invoke({"topic": "AI"})` のように `logs` を渡さずに起動した場合、デフォルト Reducer なしのキーや参照先で `state["summary"]` を直接読むと `KeyError` になることがあります。ノード内で未初期化の可能性があるキーを読む際は `state.get("summary", "")` を使うか、`dataclass` でデフォルト値を定義するのが安全です。
4. **`compile()` を呼び忘れると `invoke()` できない**
   `StateGraph` インスタンス自体は「設計図（ビルダー）」であり実行機能を持ちません。必ず `.compile()` を呼んで `CompiledStateGraph` を得てから `.invoke()` / `.stream()` を実行してください。また、どのエッジからも到達できない孤立ノードがある場合も `compile()` 時または実行時に構造的な問題として表面化します。

---

## 7. まとめ

本記事では、`StateGraph`・`TypedDict`・`operator.add` Reducer・`MiniCPM-V-4.7-35B-A3B` を組み合わせて最小のステートフルパイプラインを構築し、Super-step ごとに部分更新と Reducer がどう評価されるかを実測データで検証しました。

次回の **Part 2 理論編** では、線形パイプラインから一歩進み、**Conditional Edges・循環ループ（Cycles）・`Send` による動的並列化（Map-Reduce）・`Command` による状態更新とルーティングの統合**を掘り下げます。
