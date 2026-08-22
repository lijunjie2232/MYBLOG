---
title: "DeepAgents Part 4 実践編：release-check スキルのオンデマンド読込と AGENTS.md チーム規約の自動適用検証"
date: 2026-07-29 13:57:10
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, AIAgent, Skills, Memory, Python]
lang: ja
description: "DeepAgents で release-check スキル（SKILL.md）とチーム規約（AGENTS.md）を構成し、Progressive Disclosure によるオンデマンド read_file と常時ロード規約の自動適用を実測ログで検証します。"
---

# はじめに

本記事は **DeepAgents Part 4 実践編** です。[Part 4 理論編](/2026/07/28/deepagents_04_theory/) で解説した **Skills（Progressive Disclosure）** と **Long-Term Memory（`AGENTS.md`）** の動作メカニズムを、実際に動作する Python コードと実測ログを用いて検証します。

理論編では、両者の役割分担について次のように整理しました。

> Skills は Progressive Disclosure（段階的開示）を採用しており、起動時には各スキルの name と description だけをシステムプロンプトへ注入し、タスクに一致した瞬間だけ read_file で SKILL.md 本文をオンデマンドに読み込む。

> Memory（AGENTS.md）は起動時に MemoryMiddleware によって全文がシステムプロンプトへ常時ロードされるため、「毎ターン必ず守るべきチーム規約・出力フォーマット・ユーザー嗜好」に最適であり、タスク特化型の手順書である Skills と補完関係にある。

今回はこの 2 つの原則が同時に機能することを、以下の 2 点を観測することで実証します。

1. ユーザーが「リリース前チェックの 3 項目」を尋ねた瞬間、エージェントが `read_file` を自律的に呼び出して `/skills/release-check/SKILL.md` をオンデマンド読込するか？
2. 一方で `AGENTS.md` は `read_file` を呼ばずとも起動時に自動適用され、回答末尾に `[Checked by DeepAgent]` が付与されるか？

---

## 1. 検証アーキテクチャとディレクトリ構成

検証では、`tempfile.mkdtemp()` で一時ワークスペースを作成し、`FilesystemBackend(root_dir=tmp_dir, virtual_mode=True)` で仮想ルート `/` にサンドボックス化します。

```text
<tmp_dir>/                    # 仮想ルート "/" としてマウント
├── AGENTS.md                 # MemoryMiddleware が起動時に全文ロード
└── skills/                   # SkillsMiddleware が走査する親コンテナ ("/skills/")
    └── release-check/        # スキルディレクトリ（name と一致）
        └── SKILL.md          # 起動時は Frontmatter のみ、必要時に read_file で本文読込
```

全体のデータフローは以下の通りです。

{% mermaid %}
flowchart LR
    subgraph Disk["一時ワークスペース (FilesystemBackend virtual_mode=True)"]
        MFile["/AGENTS.md\n(末尾署名ルール)"]
        SFile["/skills/release-check/SKILL.md\n(Frontmatter + 3項目チェックリスト)"]
    end

    subgraph Agent["create_deep_agent (MiniCPM-V-4.7-35B-A3B)"]
        MM["MemoryMiddleware\n起動時に全文注入"]
        SM["SkillsMiddleware\n起動時に name/description のみ注入"]
        LLM["LLM 推論ループ"]
        RF["read_file ツール"]
    end

    MFile -- "起動時常時ロード" --> MM --> LLM
    SFile -- "Level 1: Metadata のみ" --> SM --> LLM
    LLM -- "タスク一致を検出" --> RF
    RF -- "Level 2: 本文をオンデマンド読込" --> SFile
    SFile -- "3項目を返却" --> LLM

    style MM fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style SM fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RF fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
{% endmermaid %}

---

## 2. 実装コード：`release-check` スキルと `AGENTS.md` の検証

OpenAI 互換エンドポイント（`BASE_URL` / `API_KEY`）を環境変数から読み込み、`MiniCPM-V-4.7-35B-A3B` モデルで検証を実行する完全なコードです。

```python
import os
import shutil
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# .env から接続先 URL と API キーを読み込む（ハードコード禁止）
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "MiniCPM-V-4.7-35B-A3B"


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """再現性確保のため temperature=0.0 で OpenAI 互換 LLM クライアントを生成する。"""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=512,
        timeout=40,
        max_retries=2,
    )


def run_skills_and_memory_lab() -> dict:
    """Skills のオンデマンド読込と Memory (AGENTS.md) の自動適用を同時検証する。"""
    llm = make_llm()
    tmp_dir = tempfile.mkdtemp(prefix="da_lab4_")
    try:
        # 1. スキルディレクトリ（/skills/release-check/SKILL.md）を作成
        #    name フィールドは親ディレクトリ名 "release-check" と一致させる
        skill_dir = os.path.join(tmp_dir, "skills", "release-check")
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(
                "---\n"
                "name: release-check\n"
                "description: リリース前チェックリストを確認するときに使用するスキル\n"
                "---\n\n"
                "# Release Check\n"
                "1. 全テスト通過を確認\n"
                "2. マイグレーション手順を確認\n"
                "3. ロールバック手順を明記\n"
            )

        # 2. チーム規約メモリ（/AGENTS.md）を作成
        #    毎ターン必ず守るべき出力フォーマット制約を記述する
        with open(os.path.join(tmp_dir, "AGENTS.md"), "w", encoding="utf-8") as f:
            f.write(
                "# Team Memory\n"
                "- 回答の末尾に必ず `[Checked by DeepAgent]` を付けること。\n"
            )

        # 3. FilesystemBackend(virtual_mode=True) でサンドボックス化し、
        #    skills には親コンテナ "/skills/"、memory には "/AGENTS.md" を指定
        agent = create_deep_agent(
            model=llm,
            backend=FilesystemBackend(root_dir=tmp_dir, virtual_mode=True),
            skills=["/skills/"],
            memory=["/AGENTS.md"],
            system_prompt="スキルとメモリに従って日本語で簡潔に回答してください。",
        )

        # 4. リリース前チェックの 3 項目を質問して挙動を観測
        res = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "リリース前チェックの3項目を教えてください。"}
                ]
            }
        )

        # 実行中に呼び出された全ツール名と最終回答を抽出
        tools_used = [
            tc["name"]
            for m in res["messages"]
            for tc in getattr(m, "tool_calls", [])
        ]
        return {
            "tools_used": tools_used,
            "final": res["messages"][-1].content,
        }
    finally:
        # 検証用の一時ディレクトリをクリーンアップ
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    output = run_skills_and_memory_lab()
    print("Tools used:", output["tools_used"])
    print("Final answer:\n", output["final"])
```

---

## 3. 実測結果（Execution Output）

上記スクリプトを実際に実行した結果（`deepagents_labs.json` の `lab4_skills_memory` 記録）は以下の通りです。

```json
{
  "lab4_skills_memory": {
    "status": "ok",
    "elapsed_s": 3.46,
    "output": {
      "tools_used": [
        "read_file"
      ],
      "final": "リリース前チェックの3項目は以下の通りです：\n\n1. 全テスト通過を確認\n2. マイグレーション手順を確認\n3. ロールバック手順を明記\n\n[Checked by DeepAgent]"
    }
  }
}
```

所要時間はわずか **3.46 秒**。期待通りの 3 項目が正確に回答され、さらに末尾には `[Checked by DeepAgent]` が確実に付与されています。

---

## 4. なぜこの挙動になったのか？（内部トレース解説）

注目すべきは `tools_used` の配列が `["read_file"]` の **1 回だけ** である点です。ここには Skills と Memory の設計上の違いが鮮明に表れています。

1. **`AGENTS.md` に対して `read_file` が呼ばれていない理由**:
   `memory=["/AGENTS.md"]` を指定したことで、`MemoryMiddleware` がエージェント起動時（最初の LLM 呼び出し前）に `/AGENTS.md` をバックエンドから読み込み、システムプロンプトへ全文を埋め込みました。そのため LLM はツールを呼ばずに最初から「末尾に `[Checked by DeepAgent]` を付ける」規約を認識しています。
2. **`SKILL.md` に対して `read_file` が 1 回呼ばれた理由**:
   `skills=["/skills/"]` を指定したことで、`SkillsMiddleware` は起動時に `release-check` の `name` と `description`（`リリース前チェックリストを確認するときに使用するスキル`）だけをシステムプロンプトへ注入しました。
   ユーザーから「リリース前チェックの 3 項目」と聞かれた LLM は、その時点では 3 項目の中身を知りません。しかし `description` との一致を判断し、自律的に `read_file` で `/skills/release-check/SKILL.md` の本文を取得（Level 2 Progressive Disclosure）した上で回答を組み立てました。

このやり取りを時系列シーケンス図で追うと以下のようになります。

{% mermaid %}
sequenceDiagram
    participant U as User
    participant MW as Memory & Skills Middleware
    participant LLM as DeepAgent LLM
    participant FS as FilesystemBackend

    Note over MW,FS: エージェント起動フェーズ (before_agent)
    MW->>FS: /AGENTS.md を全文読込
    MW->>FS: /skills/*/SKILL.md の Frontmatter のみ走査
    MW->>LLM: System Prompt 注入<br/>(AGENTS.md 全文 + release-check の name/description)

    U->>LLM: "リリース前チェックの3項目を教えてください。"
    Note over LLM: description に一致!<br/>本文は未読込なので read_file を選択
    LLM->>FS: tool_call: read_file("/skills/release-check/SKILL.md")
    FS-->>LLM: ToolMessage: "# Release Check\n1. 全テスト通過を確認\n2. マイグレーション手順を確認\n3. ロールバック手順を明記"

    Note over LLM: SKILL.md の 3 項目を整理し、<br/>AGENTS.md の末尾署名規約を適用
    LLM-->>U: "1. 全テスト通過を確認\n2. マイグレーション手順を確認\n3. ロールバック手順を明記\n\n[Checked by DeepAgent]"
{% endmermaid %}

---

## 5. 発展パターン：本番環境向け `StoreBackend` 永続化と書き込み保護

ローカル検証では `FilesystemBackend` を使いましたが、本番のマルチユーザー環境では `CompositeBackend` + `StoreBackend` を使い、チーム規約や承認済みスキルを **Read-Only** で保護するのが定石です。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data

load_dotenv()

llm = ChatOpenAI(
    model="MiniCPM-V-4.7-35B-A3B",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    temperature=0.0,
)

# 本番デプロイ時は LangGraph Platform Store や PostgresStore を使用
store = InMemoryStore()

# 注意: CompositeBackend はルートプレフィックスを剥がして StoreBackend へ渡すため、
# シード時のキーは "/memories/AGENTS.md" ではなく "/AGENTS.md" とする
store.put(
    ("team-alpha",),
    "/AGENTS.md",
    create_file_data("# Team Memory\n- 回答の末尾に必ず `[Checked by DeepAgent]` を付けること。\n"),
)
store.put(
    ("team-alpha",),
    "/release-check/SKILL.md",
    create_file_data(
        "---\nname: release-check\ndescription: リリース前チェックリストを確認するときに使用するスキル\n---\n\n"
        "1. 全テスト通過を確認\n2. マイグレーション手順を確認\n3. ロールバック手順を明記\n"
    ),
)

agent = create_deep_agent(
    model=llm,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda _rt: ("team-alpha",)),
            "/skills/": StoreBackend(namespace=lambda _rt: ("team-alpha",)),
        },
    ),
    memory=["/memories/AGENTS.md"],
    skills=["/skills/"],
    # 共有スキルとチーム規約への不正な書き換え（プロンプトインジェクション対策）を禁止
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/memories/**", "/skills/**"],
            mode="deny",
        )
    ],
    store=store,
)
```

---

## 6. 注意点・ハマりどころ（Pay Attention）

実運用で Skills と Memory を導入する際、特につまずきやすいポイントをまとめます。

1. **`skills` には「スキル本体」ではなく「親コンテナディレクトリ」を渡す**:
   `skills=["/skills/release-check/"]` のように `SKILL.md` が直接入っているディレクトリを指定しても、**エラーにならず静かに無視（0 件検出）**されます。必ずその 1 つ上の親ディレクトリ `skills=["/skills/"]` を指定してください。
2. **`name` とディレクトリ名の不一致・曖昧な `description`**:
   `SKILL.md` の `name` が親ディレクトリ名と異なると仕様違反でスキップされます。また `description: "リリース関連"` のように曖昧だと LLM が `read_file` を発動せず自前の一般論で回答してしまいます。発動条件となる具体的なキーワード（例:「リリース前チェックリストを確認するときに使用」）を明記しましょう。
3. **`CompositeBackend` + `StoreBackend` のキープレフィックス除去の罠**:
   エージェントからは `/memories/AGENTS.md` や `/skills/release-check/SKILL.md` として見えても、`CompositeBackend` がルートプレフィックスを剥がして転送するため、`store.put()` で初期データを投入する際のキーは `/AGENTS.md` や `/release-check/SKILL.md` にする必要があります。
4. **スレッド単位のスキルキャッシュと 10 MB 上限**:
   スキル一覧はスレッド開始時に 1 度だけロードされるため、実行済みスレッドの途中で `SKILL.md` を書き換えても自動反映されません（新スレッド作成またはメタデータのリセットが必要）。また 10 MB を超える `SKILL.md` は発見フェーズで無言スキップされます。
5. **カスタムサブエージェントは `skills` を継承しない**:
   親エージェントに `skills=["/skills/"]` を設定しても、カスタム定義したサブエージェントからは見えません（`general-purpose` サブエージェントのみ自動継承）。必要な場合はサブエージェント定義側にも個別に `skills` を設定してください。

---

## まとめ

本記事では、`release-check` スキルと `AGENTS.md` チーム規約を組み合わせた実測検証を通じて、DeepAgents の 2 種類のコンテキスト管理が意図通りに協調動作することを確認しました。

- **Skills**: `tools_used: ["read_file"]` が示す通り、必要なタスクが来た瞬間だけ `SKILL.md` 本文をオンデマンド読込。
- **Memory**: `read_file` なしで起動時から常時適用され、末尾の `[Checked by DeepAgent]` 規約を確実に遵守。

次回の **Part 5 理論編** では、単一コンテキストの限界を突破する **Subagents（コンテキスト隔離とタスク委譲）** の 5 つのアーキテクチャパターンを解説します。
