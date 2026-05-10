---
title: DeepSeek Harness 技術解析 Part 4：Capability Seam（能力の継ぎ目）— 実行世界の差し替え・OS サンドボックス・人間承認

date: 2026-05-02 10:38:44
categories: [AI]
tags: [Deep Learning, TypeScript, Node.js, LLM, AI Agent, DeepSeek, Sandbox, Security, SSH, 機械学習, AI, 人工知能, 生成AI]
lang: ja
description: DeepSeek Harness の設計パターン「Capability Seam（Service Definition / Provider / Consumer）」を解説します。ファイルシステムとサブプロセスを差し替えるだけで Bash・PTY・LSP がそのまま SSH リモートへ移行する仕組み、OS レベルのプロセス隔離（bwrap / Landlock / Seatbelt / Windows ACL）、そして Fail-Closed な人間承認（ctx.approval）をソースコードとフロー図で紐解きます。
---

## はじめに

[Part 3](/2026/04/23/deepseek_harness_part3/) では、Cordis・`dsh-scope`・レイヤー合成によって「特権的コアなし」でプラグインを束ねる仕組みを解説しました。

Part 4 では、そのプラグイン基盤の上で **「エージェントが外界（OS・ファイル・シェル・人間）とどう安全に接するか」** を支える **Capability Seam（能力の継ぎ目）** を掘り下げます。

---

## 1. Capability Seam の基本理論：3 役割の分離

エージェントに「ファイルを読む」「コマンドを実行する」「サンドボックスで隔離する」といった能力を与えるとき、ツール実装の中に `node:fs` や `child_process` を直接書いてしまうと、後で「リモート SSH 先で実行したい」「テストでモックしたい」「OS ごとに隔離方式を変えたい」となった瞬間に全ツールをフォークする羽目になります。

DeepSeek Harness はすべての外界アクセスを **3 つの役割からなる Capability Seam** として設計します（参照：[docs/architecture.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md#capability-seams) / [docs/capability-seams.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/capability-seams.md)）。

1. **Service Definition（サービス定義）**：型・インターフェース・エラー契約・`ctx.<key>` 宣言だけを所有（例：`dsh-fs`, `dsh-subprocess`, `dsh-sandbox`, `dsh-user-approval`）。
2. **Service Provider（プロバイダ実装）**：具体的な OS やトランスポートへの実装を提供（例：`dsh-fs-local` vs `dsh-fs-ssh`、`dsh-sandbox-local` vs `dsh-sandbox-ssh`）。
3. **Consumer（利用者）**：主にモデル向けツールや上位サービス（例：`dsh-tool-fs`, `dsh-tool-bash`, `dsh-tool-lsp`, `dsh-tool-terminal`）。Consumer は **必ず Service Definition だけに依存し、具象 Provider を一切 `import` しない**。

さらにパッケージ境界では **Explicit > Implicit（明示的な解決）** の原則を貫き、`run()` の中で暗黙に `?? default` するのではなく、必ず **`resolve(request) -> Spec`** の 1 段を置いてデフォルト値・パス正規化・権限境界を確定させます。

---

## 2. 実行世界の差し替え：ローカルから SSH リモートへ

Capability Seam の威力が最も分かりやすく現れるのが **SSH リモート実行**（参照：[docs/subsystems/ssh.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/ssh.md)）です。

{% mermaid %}
graph TB
    Tools["Consumer ツール群<br/>tool-fs / tool-bash / tool-terminal / tool-lsp / ptc-runtime"]
    Seams["Service Definition 層<br/>ctx.fs / ctx.subprocess / ctx.sandbox"]

    Tools --> Seams

    Seams -->|"ローカル構成"| Local["Local Providers<br/>dsh-fs-local + dsh-subprocess-local + dsh-sandbox-local"]
    Seams -->|"SSH 構成"| Ssh["SSH Providers<br/>dsh-fs-ssh + dsh-subprocess-ssh + dsh-sandbox-ssh"]
    Ssh --> Conn["SshConnection (dsh-ssh)<br/>SHA-256 検証済みリモートヘルパー<br/>管理 RPC と独立転送 Unix ソケット"]

    style Tools fill:#e1f5ff
    style Seams fill:#fff9c4
    style Local fill:#e8f5e9
    style Ssh fill:#fce4ec
{% endmermaid %}

- **SSH 専用ツールは 1 つも増やさない**：`dsh-ssh` ファミリーは既存の `ctx.fs`・`ctx.subprocess`・`ctx.sandbox` をそのまま実装します。Harness 本体・LLM 通信・セッションログは手元のホストに置いたまま、ファイルシステムとプロセスの「実行世界」だけを丸ごとリモート POSIX マシンへ差し替えることで、Bash・PTY・LSP・PTC がツール側の変更ゼロでリモート動作します。
- **検証と非再接続（Non-Reconnecting）設計**：`SshConnection` は接続前にリモートヘルパーの `helperHash`（SHA-256）を検証し、一致しなければ起動を拒否します。また通信断時には **曖昧な副作用を自動リプレイしない** ため再接続せず、進行中の操作を明示的に無効化してリモート側もハートビートリース失効で自動クリーンアップします。

---

## 3. OS レベルのプロセス隔離：`ctx.sandbox`

シェルコマンドを実行するエージェントにとって、OS レベルのファイル書き込み制限は最後の防波堤です。DeepSeek Harness のプロセスサンドボックス（[`packages/sandbox/sandbox/src/index.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/sandbox/sandbox/src/index.ts) / [docs/subsystems/sandbox.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/sandbox.md)）は、**呼び出し単位（Per-Call）** でファイル効果ポリシーを `argv` にラップする仕組みです。

### 3.1 モードと強制完全性（Enforcement）の分離

```ts
// packages/sandbox/sandbox/src/index.ts (L38-L89 抜粋)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/sandbox/sandbox/src/index.ts#L38-L89

// ファイル書き込みの許可レベル（ネットワークやプロセス可視性はこの語彙の対象外）
// - read-only: 書き込み禁止（POSIX シェルに必要な /dev/null 等の最小シンクのみ許可）
// - workspace-write: ワークスペース配下＋明示追加ルート＋一時領域への書き込みを許可
// - danger-full-access: 隔離をバイパス（ctx.sandbox を経由せず元 argv を直接起動）
export type SandboxMode = 'read-only' | 'workspace-write' | 'danger-full-access'

// サンドボックスプロバイダへ実際に渡せる「隔離あり」モードだけに型を狭める
export type ConfinedSandboxMode = Exclude<SandboxMode, 'danger-full-access'>

// プラットフォームの強制完全性：古い Landlock ABI や Windows ACL の制約下では 'partial' を明示申告する
export type SandboxEnforcement = 'full' | 'partial'

// 1 回の能力呼び出しごとに解決される完全な実行ポリシー
export interface SandboxExecutionPolicy {
  mode: SandboxMode
  workspaceRoot: string                        // セッションの不変 cwd から導出される絶対ルート
  additionalWritableRoots?: readonly string[]  // workspace-write 時に追加許可する物理ディレクトリ群
  sessionId?: SessionId                        // Windows ACL 等のバックエンドがセッション別 SID/一時領域を識別する鍵
}

// プロバイダ(ctx.sandbox)が受け取る確定ポリシー：プロバイダに状態を持たせず毎回の呼び出しで渡す
export interface SandboxPolicy extends SandboxExecutionPolicy {
  mode: ConfinedSandboxMode
}
```

設計上の重要ポイントが 3 つあります。

1. **プロバイダに状態を持たせず、毎回の呼び出しで `SandboxPolicy` を渡す**
   - 同じプロセス内で「Bash は `read-only` で走らせつつ、別セッションのツールは `workspace-write`、人間が承認した 1 回限りの昇格リトライは広い権限で再実行する」といった並列動作が状態競合なしで可能です。
2. **プラットフォームごとの実装を `dsh-sandbox-local` に隠蔽**
   - Linux：`bwrap` / Landlock（`native/system` のネイティブアドオンを利用）
   - macOS：Seatbelt
   - Windows：ACL Restricted Token（セッションごとに独立した一時ディレクトリと SID を割り当て）
3. **`full` / `partial` の誠実な申告と失敗分類の分離**
   - 古い Landlock ABI や Windows ACL の制約などで完全隔離できない場合は `enforcement: 'partial'` を正直に報告します。
   - さらに `ConfinedArgv` は **`runnerFailureRules`（サンドボックス基盤自体の起動失敗）** と **`denialSignatures`（サンドボックスが正常に動作してコマンドの書き込みをブロックした証拠）** を直交に分類します。これにより、基盤障害を通常のコマンド失敗と誤認しません。

---

## 4. 人間承認（`ctx.approval`）と Fail-Closed 設計

エージェントが危険な操作（サンドボックス昇格や外部変更など）を行おうとするとき、ユーザーに確認を求めるのが `dsh-user-approval`（[`packages/interaction/user-approval/src/index.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/interaction/user-approval/src/index.ts) / [docs/subsystems/approval.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/approval.md)）です。

### 4.1 閉じた結果型と `ask` / `never` ポリシー

```ts
// packages/interaction/user-approval/src/index.ts (L57-L69, L90-L110 抜粋)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/interaction/user-approval/src/index.ts#L57-L110

// 承認結果の閉じた 4 値：'allowed-once' 以外はすべて Fail-Closed で拒否扱いになる
const OUTCOMES: readonly ApprovalOutcome[] = ['allowed-once', 'rejected', 'cancelled', 'unavailable']

// セッション単位の承認方針：'never' は対話回答者へ渡す前に決定論的に 'rejected' を返す
export type ApprovalPolicy = 'ask' | 'never'

// セッションログから現在の承認ポリシーとターン開閉状態を導出する同期 Projection 定義
const approvalStateDefinition = {
  key: 'approvalState',
  stateVersion: 1,
  // 永続キャッシュ読み込み時に検証するための Zod スキーマ
  stateSchema: zod.object({
    policy: zod.enum(['ask', 'never']).nullable(),
    turnOpen: zod.boolean(),
  }),
  init: (): ApprovalState => ({ policy: null, turnOpen: false }),
  apply(state: ApprovalState, event: SessionEvent): ApprovalState {
    switch (event.type) {
      case 'approval/policy':
        // 値が変わらないときは同一参照 state を返し、不要な下流通知を抑止する（Object.is 最適化）
        return state.policy === event.data.policy ? state : { ...state, policy: event.data.policy }
      case 'turn/start':
        // 承認要求が「開いたターン内」であることを AgentRegistry に依存せず自前で追跡する
        return state.turnOpen ? state : { ...state, turnOpen: true }
      case 'turn/end':
        return state.turnOpen ? { ...state, turnOpen: false } : state
      default:
        // 無関係なイベントでは元の参照をそのまま返す
        return state
    }
  },
} satisfies ProjectionDefinition<'approvalState', ApprovalState>
```

ここには DeepSeek Harness らしい堅牢なセキュリティ設計が詰まっています。

- **徹底した Fail-Closed**：結果 `ApprovalOutcome` は `'allowed-once' | 'rejected' | 'cancelled' | 'unavailable'` の 4 値に閉じており、回答者（UI 等）が不在・例外送出・タイムアウトした場合はすべて `'unavailable'` となり、**呼び出し側は `'allowed-once'` 以外を必ず拒否** します。
- **`never` ポリシーの強制順序**：CI や無人実行（Headless）向けの `'never'` ポリシーは、`approval/request` waterfall を発火する **前** にサービス内部で判定されます。そのため、後から `prepend: true` で登録されたプラグインであっても `'never'` を回避できません。
- **監査ログとモデル認識の分離**：`ctx.approval.request()` は必ずオープンなターン内で実行され、`approval/asked` と `approval/decided` のペアをログ専用イベントとして記録します（モデル履歴には入れません）。一方、現在の `ask` / `never` 方針自体は Part 1 で触れた `<system-reminder>` ランタイムコンテキストに反映されるため、モデルも「承認プロンプトが無効だから昇格を要求しても無駄だ」と正しく把握できます。

---

## まとめと次回予告

Part 4 では、DeepSeek Harness が外界アクセスと安全性をどう両立しているかを解説しました。

- **Capability Seam（Definition / Provider / Consumer）** により、ツールを書き換えずにローカルと SSH リモートの実行世界を丸ごと切り替え可能。
- **`ctx.sandbox`** は Per-Call のポリシー指定、`full`/`partial` の明示、基盤障害と正当なブロックの分離で信頼性を担保。
- **`ctx.approval`** は Fail-Closed な 4 値結果・`never` の事前強制・`SessionProjection` による状態追跡で、対話 UI と無人実行の両方を安全に支える。

次回の **Part 5** では、従来の「1 ツール呼び出し＝1 JSON 往復」の枠を超える **PTC Runtime（Programmatic Tool Calling）**、統一プロバイダ越しの **Subagents**、そして共有タスク DAG で協調する **Agent Teams** を解説します。
