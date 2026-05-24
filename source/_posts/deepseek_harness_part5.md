---
title: DeepSeek Harness 技術解析 Part 5：単一ツール呼び出しを超えて — PTC Runtime・Subagents・Agent Teams

date: 2026-05-11 22:51:09
categories: [AI]
tags: [Deep Learning, TypeScript, Node.js, LLM, AI Agent, DeepSeek, Multi-Agent, PTC, 機械学習, AI, 人工知能, 生成AI]
lang: ja
description: DeepSeek Harness の高度な実行拡張を解説します。モデルが JS コードを書いてホストツールを並列・合成呼び出しする PTC Runtime（Programmatic Tool Calling）、ローカル子エージェントと外部エージェント製品を統一扱う Subagent Seam、そして DAG ベースの共有タスクボードで協調する Agent Teams の設計理論を紐解きます。
---

## はじめに

これまでの Part 1〜4 で、単一エージェントが安全かつ堅牢にツールを呼び出す基盤を見てきました。しかし、実務の複雑なタスク（大量ファイルの集計、複数ステップの調査と並列実装など）では、従来の「LLM が JSON でツールを 1 回ずつ呼び、巨大な生出力をそのままコンテキストに流し込む」方式には限界があります。

Part 5 では、DeepSeek Harness がこの限界を突破するために備えている **3 つの拡張実行モデル** を解説します。

1. **PTC Runtime（Programmatic Tool Calling）**：コード実行によるツール合成
2. **Subagents**：ローカル子ループと外部エージェントを統一する委譲 Seam
3. **Agent Teams**：共有タスク DAG と Inbox によるマルチエージェント協調

---

## 1. PTC Runtime（Programmatic Tool Calling）

### 1.1 なぜ「コードでツールを呼ぶ」のか

通常の Tool Calling では、たとえば「20 個のファイルを読み、特定条件でフィルタし、合計値だけ知りたい」場合でも、20 個分のファイル全文が LLM のコンテキストウィンドウに流れ込み、トークン消費と推論ターン数が爆発します。

**PTC（Programmatic Tool Calling）** は、モデルに **サンドボックス化された JS/Node プログラムを書かせ、その中で既存のホストツールを `tools.<name>(args)` として非同期呼び出しさせる** 仕組みです（参照：[docs/subsystems/ptc-runtime.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/ptc-runtime.md) / [`packages/ptc-runtime/ptc-runtime/src/types.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/ptc-runtime/ptc-runtime/src/types.ts)）。

{% mermaid %}
graph TB
    Model["LLM 応答<br/>予約ツール run_code(program) を呼び出し"] --> Pipeline1["ToolExecutionPipeline<br/>run_code 自体を検証・承認"]
    Pipeline1 --> Runtime["ctx.ptcRuntime (dsh-ptc-runtime-node)<br/>サンドボックス化された Node.js 実行環境"]
    Runtime -->|"グローバル tools.* 呼び出し<br/>ロスレス JSON 境界"| Pipeline2["ToolExecutionPipeline<br/>サブ呼び出しごとに pre/guards/approval/post 適用<br/>tool/ptc-dispatch ログ記録"]
    Pipeline2 --> HostTools["既存ホストツール<br/>fs / bash / search / subagent 等"]
    HostTools -->|"型付き JSON 結果"| Runtime
    Runtime -->|"最終完了値 value と標準出力 output のみ"| ModelResult["tool/result<br/>中間データをコンテキストに溢れさせず返却"]

    style Model fill:#e1f5ff
    style Runtime fill:#fff9c4
    style Pipeline2 fill:#fce4ec
    style ModelResult fill:#c8e6c9
{% endmermaid %}

### 1.2 PTC Seam の 4 つの設計原則

1. **既存ツールパイプラインをバイパスしない**
   - `run_code` 内部から呼ばれた各ツール（サブ呼び出し）も、親トークンを携えて **通常の `ToolExecutionPipeline` を必ず通過** します。権限承認・単調ガード・サンドボックス・`tool/ptc-dispatch` 監査ログがそのまま効き、拒否された場合はプログラム内で型付き Promise 拒否として返ります。
2. **ロスレス JSON 境界とプロトタイプ汚染防御**
   - ホスト関数は `PtcBindingNamespace` としてプログラムのグローバルオブジェクトに露出されますが、バインディング名は own property として定義され `__proto__` 経由の走査を遮断します。引数・戻り値・最終完了値（`PtcRunResult.value`）はすべてロスレス JSON 境界を通過します。
3. **エラーは例外ではなく結果フィールドで返す**
   - `PtcRunResult` では、プログラムの実行エラーやタイムアウト・強制中断（`AbortSignal` による無限ループ停止を含む）も `run()` の reject ではなく **`result.error` フィールド** として解決されます。失敗をモデルへ構造化報告する責務を呼び出し側に委ねる設計です。
4. **状態保持（`stateOwner`）と構文形式（`programFormat`）の直交設計**
   - `function-body` / `repl` / `script` / `scoped-repl` という構文評価形式と、`stateOwner`（セッション生存中のみメモリ上で変数を引き継ぐオプション）を独立に設計しています。永続セッション形式には揮発オブジェクトを一切持ち込まないため、Part 2 のイベントソーシング原則とも矛盾しません。

---

## 2. Subagents：ローカル子ループと外部エージェントの統一 Seam

タスクを別コンテキストに委譲する **Subagent**（参照：[docs/subsystems/subagent.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.md)）は、`ctx.subagents` という 1 つのプロバイダレジストリ背後に、性質の異なるバックエンドを統一的に収めています。

- **インプロセス生成（`dsh-subagent-spawn-in-process`）**：新しい空のコンテキストで子 `Agent` を起動。
- **インプロセス分岐（`dsh-subagent-fork-in-process`）**：Part 2 の `buildForkSeed()` を使い、親の会話履歴を継承した子 `Agent` を分岐起動。
- **外部エージェント製品／別プロセス SDK**：DSH SDK 別プロセスや外部エージェント（Claude Agent SDK / Codex 等）を同じ `SubagentActivation` 契約で駆動。

### 2.1 宣言的能力チェック（`SubagentCapabilities`）と配送モード

各バックエンドは起動前に `SubagentCapabilities`（`agentOptions`, `agentPreset`, `outputSchema`, `depthLimit`）を宣言し、マネージャが非対応オプションを起動前に即座に拒否します。

また、`SubagentActivationSpec.delivery` によって結果の届け方を 2 通りに分けています。

- **`delivery: 'parent'`（モデル向けツール用）**：子の完了結果を親エージェントの Inbox へ自動通知し、親の次ステップを起動する。
- **`delivery: 'caller'`（Workflow / PTC 向け）**：親の会話へメッセージを注入せず、呼び出し元が `activation.result` の Promise を直接 `await` して構造化結果（`outputSchema` 検証済みの `structured`）を受け取る。

さらに、階層的な名前空間（ルートからのスラッシュ区切り名）、`SubagentCapacity` による同時実行数上限、および `maxDepth` による再帰委譲の深さ制限がすべてランタイムで一元管理されます。

---

## 3. Agent Teams：共有タスク DAG と Inbox 協調

複数のサブエージェントが並列で 1 つの大きなプロジェクトを進める場合、単なる親子委譲だけでは「誰がどのタスクを担当しているか」「どのタスクが他の完了を待っているか」「どのファイルを同時に書き換えようとしているか」が混乱します。

**Agent Teams**（[`packages/team/agent-team`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/team/agent-team/src/task-board.ts) / [docs/subsystems/agent-team.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/agent-team.md)）は、ルートセッション（Lead）のイベントログを権威あるジャーナルとして **共有タスク DAG（有向非巡回グラフ）** を提供します。

### 3.1 `TeamTaskBoard` の設計：CAS・循環検出・アドバイザリ書き込み範囲

[`packages/team/agent-team/src/task-board.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/team/agent-team/src/task-board.ts#L56-L83) の `create()` を見ると、マルチエージェント協調でも Part 2 のイベントソーシングとトランザクション設計が徹底されていることが分かります。

```ts
// packages/team/agent-team/src/task-board.ts (L56-L83 抜粋)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/team/agent-team/src/task-board.ts#L56-L83

  async create(caller: Agent, root: Agent, request: CreateTeamTaskRequest, signal?: AbortSignal): Promise<TeamTaskView> {
    // 1. Lead セッション(root.id)単位の直列トランザクションで並列更新の競合を防ぐ
    return this.journal.transact(root.id, async () => {
      signal?.throwIfAborted()
      const state = this.state(root)

      // 2. 削除済み(tombstone)を除くアクティブタスク数が上限を超えていないか検証
      const active = state.tasks.filter(task => task.status !== 'deleted').length
      if (active >= this.maxTasks) {
        throw new TeamError(`Team task limit ${this.maxTasks} reached`, 'TEAM_TASK_LIMIT')
      }

      // 3. ブランド付きタスク ID（task-1, task-2, ...）を採番し、初期スナップショットを構築
      const id = brandString<TeamTaskId>(`task-${state.nextTaskNumber}`)
      const task: TeamTaskSnapshot = {
        id,
        revision: 1, // CAS（楽観ロック）用の初期リビジョン
        subject: requiredText(request.subject, 'subject', 200),
        description: requiredText(request.description, 'description', 16_384),
        status: 'pending',
        blockedBy: this.dependencies(request.blockedBy ?? [], state), // ブロッカー解決
        writeScopes: this.writeScopes(request.writeScopes ?? []),    // アドバイザリ書き込み範囲の正規化
      }

      // 4. DAG 整合性検証（存在しない ID・重複・循環依存があれば即座にエラー送出）
      this.assertTaskGraph(state, task)
      // トランザクション待機中に caller が既に終了していないか（ツリー所属）を再確認
      this.ctx.subagents.rootOf(caller)

      // 5. Lead の永続セッションログへイベント追記＋フラッシュし、ready/警告を付与したビューを返す
      await this.journal.appendAndFlush(root, task)
      return projectTaskView(state, task)
    })
  }
```

- **直列トランザクション＋楽観ロック（`revision`）**：すべてのタスク変更は Lead セッション単位の `journal.transact()` で直列化され、更新ごとに `revision` がインクリメント（Compare-and-Set）されます。
- **DAG の循環検出（`assertTaskGraph`）**：`blockedBy` に削除済みタスクや循環依存があると即座に `TEAM_TASK_DEPENDENCY_CYCLE` で拒否され、`TeamTaskView.ready` が「全ブロッカー完了済みか」を自動算出します。
- **ロックではなくアドバイザリな `writeScopes`**：ファイルパス重複をハードロックしてデッドロックを起こすのではなく、重複を `writeScopeWarnings` として可視化してエージェント自身に調整させます。
- **タスク変更とエージェント起床の分離**：タスクボードの更新だけではアイドル中のエージェントを勝手に起こしません。誰かに動いてほしいときは、通常のサブエージェント間メッセージ（Inbox）を明示的に送るというシンプルな責務分離を採っています。

---

## まとめと次回予告

Part 5 では、DeepSeek Harness が単一ツール呼び出しの制約をどう乗り越えているかを見てきました。

- **PTC Runtime**：既存ツールパイプラインとサンドボックスを維持したまま、モデルに JS コードでツールを合成・フィルタリングさせることでトークンと往復数を大幅削減。
- **Subagents**：ローカル生成・Fork・外部製品を 1 つの `SubagentActivation` Seam に統合し、`parent` / `caller` の配送モードを使い分け。
- **Agent Teams**：Lead セッションログ上の直列トランザクション＋DAG＋`SessionProjection`（`agentTeam`）により、リプレイ可能で透明なマルチエージェント協調を実現。

最終回となる **Part 6** では、エージェント自身がランタイムを検査・自己拡張する **Extensions（`ctx.cordisInspect` / `ctx.dynamicCordisRunner`）** と、1 つのプラグインツリーを Web・Desktop・SDK へ投影する **Typert RPC・Slot UI アーキテクチャ** を解説します。
