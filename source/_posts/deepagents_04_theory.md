---
title: "DeepAgents Part 4 理論編：Skills（SKILL.md Progressive Disclosure）と Long-Term Memory（AGENTS.md / MemoryMiddleware）の設計"
date: 2026-07-28 13:46:08
categories: [AI]
tags: [DeepAgents, LangChain, LangGraph, AIAgent, Skills, Memory, ContextEngineering]
lang: ja
description: "DeepAgents における Skills（SKILL.md の 3 階層 Progressive Disclosure）と Long-Term Memory（AGENTS.md / MemoryMiddleware）のアーキテクチャ、スコープ設計、権限管理、使い分け基準を理論面から徹底解説します。"
---

# はじめに

AI エージェントが単純なチャットボットから「複雑な業務ワークフローを自律的に完遂するハーネス」へと進化するにつれ、**ドメイン知識と永続コンテキストをどう管理するか**が最大の設計課題になります。

あらゆる業務手順・コーディング規約・リファレンスドキュメントをすべて単一の `system_prompt` に詰め込むアプローチは、すぐに以下の 3 つの壁に突き当たります。

1. **コンテキスト膨張（Context Bloat）とコスト増大**: 現在のタスクと無関係な手順書まで毎ターン全トークン消費する。
2. **注意力の分散（Lost in the Middle）**: 指示が長大になるほど、LLM が重要なルールを見落としやすくなる。
3. **セッション横断の学習不能**: ユーザーが伝えた好みやチーム固有の規約が、次のスレッド（会話）に引き継がれない。

LangChain の **DeepAgents** は、この課題を **Skills（手続き的記憶 / オンデマンド読み込み）** と **Long-Term Memory（宣言的記憶 / 起動時常時ロード）** という 2 つの補完的な機構で解決しています。本記事（Part 4 理論編）では、両者の内部アーキテクチャと設計原則を深掘りします。

---

## 1. Skills アーキテクチャと Progressive Disclosure（段階的開示）

DeepAgents の Skills は、[Agent Skills 仕様](https://agentskills.io/specification)に準拠した「再利用可能なドメイン知識パッケージ」です。1 つのスキルは専用のディレクトリとして構成され、必ず `SKILL.md`（YAML Frontmatter + Markdown 手順書）をルートに含みます。

> Skills は Progressive Disclosure（段階的開示）を採用しており、起動時には各スキルの name と description だけをシステムプロンプトへ注入し、タスクに一致した瞬間だけ read_file で SKILL.md 本文をオンデマンドに読み込む。

### 1.1 3 階層の読み込みモデル

`create_deep_agent(..., skills=["/skills/"])` を指定すると、内部のミドルウェアスタックに [`SkillsMiddleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/skills.py) が組み込まれます。Skills は以下の **3 階層（Level 1 → 2 → 3）** で段階的にコンテキストへ展開されます。

| 階層 | 読み込まれる内容 | 読み込みタイミング | 担当コンポーネント |
| :--- | :--- | :--- | :--- |
| **Level 1: Metadata** | `SKILL.md` の Frontmatter（`name`, `description`） | エージェント起動時（全スキル対象） | `SkillsMiddleware` |
| **Level 2: Instructions** | `SKILL.md` の Markdown 本文全体 | ユーザー依頼が `description` に一致し、エージェントが `read_file` を呼んだ瞬間 | LLM + `FilesystemMiddleware` |
| **Level 3: Resources** | `scripts/`, `references/`, `assets/` 配下の補助ファイル | `SKILL.md` の手順内で必要と指示された瞬間のみ | LLM（`read_file` / Sandbox 実行） |

この段階的開示のフローを Mermaid 図で可視化します。

{% mermaid %}
flowchart TD
    Start["エージェント起動 (before_agent)"] --> Scan["SkillsMiddleware が skills パスを走査"]
    Scan --> Parse["各ディレクトリの SKILL.md Frontmatter を解析"]
    Parse --> L1["Level 1: name と description のみを System Prompt へ注入"]

    L1 --> UserReq["ユーザーからのタスク依頼"]
    UserReq --> Match{"LLM がタスクと description の一致を判定"}

    Match -- "一致しない" --> Normal["通常応答 / 他ツール実行\n(SKILL.md 本文は未読込・トークン節約)"]
    Match -- "一致する" --> L2["Level 2: read_file で SKILL.md 本文をオンデマンド読込"]

    L2 --> Follow["手順書に従ってワークフローを実行"]
    Follow --> NeedRes{"補助リソース参照が必要?"}
    NeedRes -- "不要" --> Done["最終回答を生成"]
    NeedRes -- "必要" --> L3["Level 3: references/ や scripts/ を読込・実行"]
    L3 --> Done

    style L1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style L2 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style L3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Normal fill:#f5f5f5,stroke:#757575,stroke-dasharray: 5 5
{% endmermaid %}

### 1.2 `SKILL.md` のディレクトリ構造と Frontmatter 仕様

スキルディレクトリは以下のように整理します。

```text
skills/
└── release-check/
    ├── SKILL.md          # 必須: YAML Frontmatter + 手順書（推奨 500 行 / 5,000 tokens 以下）
    ├── scripts/          # 任意: 決定論的に実行する検証スクリプト等（Sandbox 環境で実行）
    ├── references/       # 任意: 詳細な仕様書・エラーコード一覧など（1 階層以内で参照）
    └── assets/           # 任意: テンプレート・スキーマ・画像等の静的リソース
```

Frontmatter で定義できるフィールドは以下の通りです。

| フィールド | 必須 | 制約・役割 |
| :--- | :--- | :--- |
| `name` | Yes | 小文字英数字とハイフン（1〜64 文字）。**親ディレクトリ名と完全一致**させる必要がある。 |
| `description` | Yes | 最大 1,024 文字。エージェントが起動時に唯一見る判断材料のため、「何をするか」と「いつ使うか（キーワード）」を具体的に書く。 |
| `license` | No | ライセンス名または同梱ライセンスファイルへの参照。 |
| `compatibility` | No | 必要な実行環境・ネットワーク要件（最大 500 文字）。 |
| `metadata` | No | 任意の Key-Value。DeepAgents 独自の `include_tools` もここに記述する。 |
| `allowed-tools` | No | 事前承認ツールのスペース区切りリスト（実験的機能）。 |

### 1.3 Layered Override（Last One Wins）と Skill Tools

DeepAgents の Skills は、エンタープライズ運用を想定した高度な拡張機能を備えています。

1. **複数ソースのレイヤリング（Last One Wins）**:
   `skills=["/skills/org-base/", "/skills/project-override/"]` のように複数パスを指定した場合、同じ `name` を持つスキルは**後ろに指定したソースが優先**されます。共通スキルをプロジェクト固有ルールで上書きする設計が可能です。
2. **Skill Tools（`metadata.include_tools`、`deepagents>=0.7.22`）**:
   スキル固有のツール群をエージェント起動時には隠蔽（Gated）しておき、`SKILL.md` を読んだ後だけ動的にツールスキーマを解放できます。`include_tools` は**スペース区切りの文字列**（YAML リストは不可）で指定します。
   さらに `SkillsMiddleware(tools=resolve_skill_tools)` にリゾルバ関数を渡せば、MCP サーバー由来のツール群を `runtime.context.user_id` に応じて動的に解決できます。
3. **Pinning Skills（`pinned_skills`）**:
   UI のスラッシュコマンド（例: `/release-check ...`）のように、呼び出し時点で使うスキルが確定している場合、入力ステートに `"pinned_skills": ["release-check"]` を渡すことで、LLM の選択を待たずに最初から `SKILL.md` 本文をコンテキストへ固定できます。

```python
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import SkillsMiddleware
from langchain_core.tools import tool

# スキル専用のツールを定義（起動時は隠蔽され、SKILL.md 読込後にのみ解放される）
@tool
def create_release_ticket(version: str) -> str:
    """リリースチケットを起票してチケット ID を返す。"""
    return f"REL-{version}-001"

backend = FilesystemBackend(root_dir="./workspace", virtual_mode=True)

# SkillsMiddleware を明示的に構成して Skill Tools を紐付ける例
agent = create_deep_agent(
    model="openai:gpt-4o",
    backend=backend,
    middleware=[
        SkillsMiddleware(
            backend=backend,
            sources=["/skills/"],          # スキルディレクトリの親コンテナを指定
            tools=[create_release_ticket], # SKILL.md の metadata.include_tools と対応
        ),
    ],
)
```

---

## 2. Long-Term Memory（`AGENTS.md` / `MemoryMiddleware`）の設計

DeepAgents における Memory は「ファイルシステムを第一級の記憶媒体とする（Filesystem-backed Memory）」という一貫した思想に基づいています。

- **Short-Term Memory（短期記憶）**: 単一スレッド内の会話履歴・スクラッチファイル。LangGraph の Checkpointer と `StateBackend` が自動管理します。
- **Long-Term Memory（長期記憶）**: スレッドを跨いで永続化される知識・規約・ユーザー嗜好。`create_deep_agent(..., memory=["/AGENTS.md"])` のようにパスを指定すると、内部の `MemoryMiddleware` が起動時にファイルを読み込み、システムプロンプトへ注入します。

> Memory（AGENTS.md）は起動時に MemoryMiddleware によって全文がシステムプロンプトへ常時ロードされるため、「毎ターン必ず守るべきチーム規約・出力フォーマット・ユーザー嗜好」に最適であり、タスク特化型の手順書である Skills と補完関係にある。

### 2.1 3 つのメモリスコープと `CompositeBackend` ルーティング

実運用では、記憶の「公開範囲（Scope）」を `CompositeBackend` と `StoreBackend` の `namespace` 関数で厳密に分離します。

1. **Agent-scoped Memory**: `namespace=lambda rt: (rt.server_info.assistant_id,)`。全ユーザー共通のエージェントの人格・学習済みナレッジを保持。
2. **User-scoped Memory**: `namespace=lambda rt: (rt.server_info.user.identity,)`。ユーザーごとの出力嗜好・業務コンテキストを完全に隔離（User A の記憶が User B へ漏洩しない）。
3. **Organization-level Memory**: `namespace=lambda rt: (rt.context.org_id,)`。コンプライアンス規定や社内ポリシーなど全社共通ルールを配置。プロンプトインジェクション防止のため、原則 **Read-only** に設定する。

これらを 1 つのエージェント内で同時に組み合わせるアーキテクチャは次の通りです。

{% mermaid %}
flowchart LR
    Agent["DeepAgent\n(MemoryMiddleware + SkillsMiddleware)"] --> Router["CompositeBackend\n(パスプレフィックスで振分)"]

    Router -- "default (作業用)" --> State["StateBackend\n(スレッド内短期記憶)"]
    Router -- "/memories/ (プレフィックス除去)" --> UserStore["StoreBackend\nnamespace: (user_id,)\nRead-Write"]
    Router -- "/policies/ (プレフィックス除去)" --> OrgStore["StoreBackend\nnamespace: (org_id,)\nRead-Only (deny write)"]
    Router -- "/skills/ (プレフィックス除去)" --> SkillStore["StoreBackend\nnamespace: (assistant_id,)\nProgressive Disclosure"]

    UserStore --> DB[("LangGraph Store\n(PostgreSQL / InMemoryStore)")]
    OrgStore --> DB
    SkillStore --> DB

    style Agent fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style Router fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style State fill:#e3f2fd,stroke:#1565c0
    style UserStore fill:#e8f5e9,stroke:#2e7d32
    style OrgStore fill:#ffebee,stroke:#c62828
    style SkillStore fill:#e0f2f1,stroke:#00695c
{% endmermaid %}

### 2.2 更新戦略：Hot Path vs Background Consolidation（Sleep Time Compute）

長期記憶の更新タイミングには 2 つのパターンがあります。

| 更新戦略 | 仕組み | メリット | デメリット |
| :--- | :--- | :--- | :--- |
| **Hot Path（会話中更新 / デフォルト）** | ユーザーとの会話中に、エージェント自身が `edit_file` で `/memories/AGENTS.md` を更新する。 | 即座に記憶へ反映され、ユーザーにも透明性が高い。 | 応答レイテンシが増加し、エージェントが本業と記憶管理を同時処理する必要がある。 |
| **Background Consolidation（非同期統合）** | 別デプロイの Consolidation Agent を Cron（例: 6 時間毎）で起動し、過去スレッド群から重要事実を抽出・統合する。 | ユーザー向けレイテンシゼロ。複数スレッド横断での矛盾解消・重複削除が可能。 | 反映は次回会話以降。2 つ目のエージェントと Cron の運用が必要。 |

また、会話の「事実（Semantic Memory）」ではなく「過去にどうデバッグしたかという一連の経緯（**Episodic Memory**）」が必要な場合は、Checkpointer に保存された過去スレッドを `client.threads.search(metadata={"user_id": ...})` で検索するツールを持たせることで実現できます。

---

## 3. Skills vs Memory vs Tools の使い分け基準

設計時に迷いやすい 3 要素の役割を整理します。

| 比較軸 | Skills (`SKILL.md`) | Memory (`AGENTS.md`) | Tools (`@tool`) |
| :--- | :--- | :--- | :--- |
| **主な目的** | タスク特化の手順・ノウハウの段階的開示 | 常に有効であるべき規約・嗜好・コンテキスト | 外部 API 呼出や決定論的なプログラム処理 |
| **ロード方式** | Level 1 のみ起動時、本文はオンデマンド | 起動時に全文をシステムプロンプトへ常時ロード | 毎ターンツールスキーマとしてバインド（遅延ロード除く） |
| **複数指定時の挙動** | 同名スキルは**後勝ち上書き（Last Wins）** | 複数ファイルの内容が**すべて結合（Combined）** | エージェント作成時に登録された集合 |
| **適したユースケース** | リリース手順、SQL 監査手順、大規模リファレンス | 「末尾に署名」「日本語で簡潔に」「社内コンプラ規約」 | DB 照会、外部 API 実行、ファイル操作 |

> **設計の指針**: 「どの質問が来ても毎回守ってほしいこと」は **Memory** に、「特定の業務を依頼されたときだけ詳しく参照してほしい手順」は **Skills** に分離することで、プロンプトのトークン効率と指示遵守率を同時に最大化できます。

---

## 4. セキュリティ・権限管理とサブエージェントへの継承

本番環境で Skills / Memory を運用する際は、以下の 2 つの境界管理が不可欠です。

### 4.1 `FilesystemPermission` による書き込み制御

共有の組織ポリシーや承認済みスキルにエージェントが勝手に書き込めてしまうと、**共有ステート経由のプロンプトインジェクション**の温床になります。DeepAgents では `FilesystemPermission` でパス単位の書き込み拒否（`mode="deny"`）や人間承認（`mode="interrupt"`）を宣言的に強制できます。

```python
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 承認済みスキルと組織ポリシーを Read-Only に保護し、ユーザー記憶のみ書き込み許可する構成
agent = create_deep_agent(
    model="openai:gpt-4o",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda rt: ("user", rt.context.user_id)),
            "/policies/": StoreBackend(namespace=lambda rt: ("org", rt.context.org_id)),
            "/skills/": StoreBackend(namespace=lambda rt: ("curated-skills",)),
        },
    ),
    memory=["/memories/preferences.md", "/policies/compliance.md"],
    skills=["/skills/"],
    permissions=[
        # 組織ポリシーと承認済みスキルへの書き込みを完全に拒否
        FilesystemPermission(operations=["write"], paths=["/policies/**", "/skills/**"], mode="deny"),
    ],
    store=store,
)
```

### 4.2 サブエージェントへの Skills 継承ルール

Part 5 で詳述する Subagents と組み合わせる際、Skills の継承には明確な非対称性があります。

- **General-purpose サブエージェント**: 親エージェントの `skills` を**自動継承**する。
- **カスタムサブエージェント**: 親エージェントの `skills` を**継承しない**。必要な場合はサブエージェント定義辞書に明示的に `"skills": ["/skills/specialist/"]` を渡す必要があり、スキル状態は親子間で完全に隔離される。

---

## まとめ

本記事では、DeepAgents のコンテキスト管理の中核を担う **Skills** と **Long-Term Memory** の理論設計を整理しました。

- **Skills (`SKILL.md`)** は `SkillsMiddleware` による 3 階層 Progressive Disclosure によって、多数の業務手順を持ちながらも初期コンテキストを軽量に保つ。
- **Memory (`AGENTS.md`)** は `MemoryMiddleware` によって起動時に常時ロードされ、`CompositeBackend` + `StoreBackend` の `namespace` 設計により Agent / User / Organization スコープを安全に分離できる。
- 両者を `FilesystemPermission` で保護することで、学習能力とセキュリティを両立した本番エージェント基盤を構築できる。

次回の **Part 4 実践編** では、実際に `release-check` スキルと `AGENTS.md` を作成し、オンデマンド `read_file` の発火とチーム規約の自動適用を実測ログ付きで検証します。
