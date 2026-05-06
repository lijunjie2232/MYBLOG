---
title: DeepSeek Harness 技術解析 Part 1：全体像・プロジェクト構成と LLM エージェントフローの核心理論

date: 2026-04-03 21:42:18
categories: [AI]
tags: [Deep Learning, Python, Node.js, TypeScript, LLM, AI Agent, DeepSeek, Cordis, 機械学習, AI, 人工知能, 深層学習, 生成AI]
lang: ja
description: DeepSeek AI が公開したオープンソースのエージェントハーネス「DeepSeek Harness (dsh)」の全体像、リポジトリ構成・依存パッケージ、そして中核となる LLM エージェント実行フロー（Turn/Step 状態遷移、KV キャッシュ最適化、ツール実行パイプライン、異常系リカバリ）をソースコードとフロー図で徹底解説します。
---

## はじめに：シリーズ構成と本記事の位置づけ

[DeepSeek Harness (`dsh`)](https://github.com/deepseek-ai/deepseek-harness) は、DeepSeek AI が開発するオープンソースの **AI エージェントハーネス（Agent Harness）** です。単なるチャット UI や簡易的な ReAct ループではなく、**「すべてがプラグイン（Everything is a Plugin）」** という設計思想と、**イベントソーシング（Event Sourcing）** に基づく堅牢な実行基盤を備えています。

本シリーズでは、DeepSeek Harness の内部理論を以下の全 6 回に分けて解剖します。

1. **Part 1（本記事）**：全体像・基本理論・プロジェクト構成と **LLM エージェントフローの核心**
2. **Part 2**：イベントソース型セッション設計（`Model-Visible ⟺ Logged`、Projection、世代マイグレーション）
3. **Part 3**：「Everything is a Plugin」を支える **Cordis**・スコープ付きライフタイム・レイヤー合成
4. **Part 4**：Capability Seam（能力の継ぎ目）— 実行環境の差し替え、OS サンドボックス、人間承認
5. **Part 5**：単一ツール呼び出しを超えて — PTC Runtime（コード実行型ツール呼び出し）、サブエージェント、Agent Teams
6. **Part 6**：ランタイム自己拡張（モデルによるプラグイン生成）とマルチサーフェス展開（Web / Desktop / SDK）

第 1 回となる本記事では、まず予備知識となる **基本理論とアプリケーションの振る舞い・特徴**、**リポジトリ構造と主要パッケージ** を押さえたうえで、心臓部である **LLM エージェントの実行フロー** を実際の TypeScript コードとフロー図で追っていきます。

---

## 1. DeepSeek Harness とは何か：基本理論とアプリの特徴

### 1.1 「エージェントハーネス」の基本理論

LLM を自律エージェントとして動かすとき、モデル本体（推論エンジン）の周りを包む制御層を **ハーネス（Harness）** と呼びます。多くの簡易エージェントフレームワークは「プロンプト文字列を組み立てて LLM API を叩き、返ってきた JSON で関数を実行する `while` ループ」をハードコードしています。

これに対し、DeepSeek Harness は次の **3 つの核心理論** を土台にしています。

1. **Everything is a Plugin（特権的コアを持たないプラグイン構成）**
   - プラグインフレームワーク [Cordis](https://github.com/cordiverse/cordis) を基盤とし、LLM アダプタ、ツールレジストリ、セッションログ、さらには **エージェントループ本体（`dsh-agent-loop`）すら 1 つの差し替え可能なプラグイン** として実装されています。
   - 登録されたフック・ツール・プロンプト要素はすべて「可逆的なエフェクト（`ctx.effect()`）」として管理され、アンロード時に自動で巻き戻されます。
2. **Model-Visible ⟺ Logged（モデルに見えるものは必ずログに記録される）**
   - LLM に送られるメッセージ履歴はメモリ上の配列を直接ミューテートするのではなく、**追記専用（Append-Only）の `SessionEvent` ログから毎回導出（`deriveMessages()`）** されます。
   - これにより、セッションの再開（Resume）、途中分岐（Fork）、決定論的リプレイ、UI 描画がすべて単一の事実源（Single Source of Truth）から保証されます。
3. **Capability Seam（能力の継ぎ目：Definition / Provider / Consumer の分離）**
   - ファイルシステム（`ctx.fs`）やサブプロセス（`ctx.subprocess`）、サンドボックス（`ctx.sandbox`）といった能力を「インターフェース定義・実装プロバイダ・利用者（ツール）」の 3 役割に分離します。
   - そのため、プロバイダをローカルから SSH やリモート環境に差し替えるだけで、Bash・PTY・LSP ツールが一切の書き換えなしに別環境へ移行します。

### 1.2 アプリケーションとしての振る舞いと主な特徴

エンドユーザーや開発者から見た `dsh` は、**1 つのプラグインツリーを複数の実行形態（Profile）に投影できる** アプリケーションです。

| プロファイル / 形態 | 起動コマンド例 | 主な振る舞い・用途 |
|---|---|---|
| **Web UI (`web`)** | `npx @deepseek-ai/dsh web` / `pnpm dsh web` | ローカル HTTP サーバー（既定 `127.0.0.1:3080`）を起動し、ブラウザ上でチャット・ファイル閲覧・承認 UI・プラグイン管理を提供 |
| **Desktop (Electron)** | `pnpm run dev:desktop` | 署名済みバンドルランタイムを内蔵し、Node IPC 経由で認証済み Host と Web アセットを接続するデスクトップアプリ |
| **Headless (`headless`)** | `pnpm dsh --profile headless "task"` | サーバーを立てずに 1 タスクを自律実行して終了する CLI モード |
| **SDK (`sdk` / `sdk-minimal`)** | TypeScript SDK / Python SDK から起動 | JSON-RPC サーバーとして動作し、外部プログラムからエージェントを制御可能 |
| **ACP (`acp`)** | `dsh --profile acp` | Agent Client Protocol 準拠の自動化専用サーバー |

主な機能面の特徴は以下の通りです。

- **安全機構と人間協調（Human-in-the-Loop）**：OS レベルのプロセス隔離（Linux の `bwrap` / Landlock、macOS の Seatbelt）と、ツール実行前の承認プロンプト・権限プリセットを統合。
- **並列ツール実行と排他制御**：読み取り系などの並列安全なツールは有界プールで並列実行し、破壊的操作は排他バリアとして直列化。
- **コンテキスト圧縮（Compaction）**：コンテキスト長の上限に近づいた際やオーバーフロー時に、ツール結果の枝刈り（Pruning）と要約を自動実行し、KV キャッシュへの影響を最小化。
- **高度な実行拡張**：サンドボックス化された Node.js 上でモデル自身が JS コードを書いてツールを呼び出す **PTC（Programmatic Tool Calling）**、**Subagent / Agent Teams**、さらにモデル自身がプラグインを定義・ロードする **ランタイム自己拡張** までサポート。

---

## 2. リポジトリ構造と主要パッケージ・依存ライブラリ

### 2.1 モノレポ全体構造

DeepSeek Harness は `pnpm` ワークスペースによる大規模 TypeScript モノレポ（ESM 専用、Node `^24.22.3 || >=26.13.1`）として構成されています（参照：[pnpm-workspace.yaml](https://github.com/deepseek-ai/deepseek-harness/blob/master/pnpm-workspace.yaml) / [AGENTS.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/AGENTS.md)）。

```text
deepseek-harness/
├── vendor/          # ベンダリングされた Cordis / Cosmokit / Schemastery
├── packages/        # 50 以上の機能グループに分かれた @deepseek-ai/dsh-* パッケージ群
│   ├── core/        # 中核 API スパイン（session, system-prompt, tools, agent, agent-loop, scope）
│   ├── llm/         # LLM 抽象サービスとプロバイダアダプタ（DeepSeek / pi-ai / retry / replay）
│   ├── bundle/      # プロファイル構成用パッチレイヤー（base, web-app, headless, sdk-app 等）
│   ├── fs/ shell/ subprocess/ sandbox/ terminal/ lsp/ ssh/  # 実行環境・ツール群
│   ├── compaction/ context/ guard/ spill/                   # コンテキスト制御・ループ衛生
│   ├── ptc-runtime/ subagent/ team/ workflow/ jobs/         # 高度な実行・マルチエージェント
│   ├── extensions/ mcp/ skill/                              # 拡張・外部ツール連携
│   ├── session/ session-query/ storage/ attachment/         # 永続化・検索・Projection
│   ├── interaction/ credentials/ settings/ account/         # 人間協調・認証・設定
│   ├── api/ typert/ host/ client/ sdk/ acp/ boot/           # 通信・RPC・UI・起動層
│   └── ...
├── apps/            # 製品アセンブリ（cli, web, desktop 等）
├── native/          # @deepseek-ai/node-addon-system（Landlock 等のネイティブアドオン）
├── python/          # Python SDK とランタイムホイール
├── docs/            # アーキテクチャ・サブシステム設計書（英・中バイリンガル）
├── scripts/         # ビルド・型チェック・ドキュメント同期・アーキテクチャ検証ゲート
└── website/         # VitePress ドキュメントサイト
```

### 2.2 コアパッケージ（`packages/core/` + `packages/llm/`）

LLM エージェントフローを直接担うのは以下の 7 パッケージです（参照：[docs/architecture.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)）。

| パッケージ | 責務 | `ctx` キー |
|---|---|---|
| `@deepseek-ai/dsh-session` | 追記専用 `SessionEvent` ログとインメモリストア、`deriveMessages()` | `ctx.sessions` |
| `@deepseek-ai/dsh-system-prompt` | プロンプトセクションとツールスキーマの組み立て | `ctx.systemPrompt` |
| `@deepseek-ai/dsh-tools` | スコープ付きツールレジストリとガード付き実行パイプライン | `ctx.tools` |
| `@deepseek-ai/dsh-agent` | `Agent` インターフェース、ライブレジストリ、`agent/*` イベント定義 | `ctx.agents` |
| `@deepseek-ai/dsh-agent-loop` | デフォルトの `Agent` ドライバ実装（`ReactLoopAgent`） | `ctx.agentLoop` |
| `@deepseek-ai/dsh-scope` | エージェント単位のスコープ付き登録プリミティブ | ライブラリ（キーなし） |
| `@deepseek-ai/dsh-llm` | メッセージ・ストリーム語彙と LLM アダプタ継ぎ目 | `ctx.llm` |

### 2.3 ベースバンドル（`dsh-base`）と主要外部ライブラリ

`web` / `headless` / `sdk` / `acp` プロファイルはすべて共通の第一層として [`packages/bundle/base/package.json`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/bundle/base/package.json)（`@deepseek-ai/dsh-base`）を読み込みます。ここには以下のパッケージ群が束ねられています。

- **モデル・リトライ層**：`dsh-llm`、`dsh-llm-deepseek-api-key`、`dsh-llm-deepseek-account`、`dsh-llm-pi-ai`、`dsh-llm-retry`
- **コンテキスト・圧縮層**：`dsh-compaction-basic`、`dsh-compaction-tool-result-pruner`、`dsh-compaction-image-offload`、`dsh-token-meter`
- **実行・安全層**：`dsh-fs-local`、`dsh-fs-sandbox`、`dsh-subprocess-local`、`dsh-bash-sandbox`、`dsh-sandbox-local`、`dsh-permission-presets`
- **ツール・実行器**：`dsh-tool-executor-native`、`dsh-tool-executor-ptc`、`dsh-ptc-runtime-node`、`dsh-tool-fs`、`dsh-tool-bash`、`dsh-tool-jobs`、`dsh-tool-agent-team`、`dsh-subagent`
- **永続化・検索**：`dsh-session-persistence-jsonl`、`dsh-session-projection`、`dsh-session-query-sqlite`

また、外部依存（[package.json](https://github.com/deepseek-ai/deepseek-harness/blob/master/package.json) / [pnpm-workspace.yaml](https://github.com/deepseek-ai/deepseek-harness/blob/master/pnpm-workspace.yaml)）には特徴的な選定が見られます。

- **Cordis / Cosmokit / Schemastery（`vendor/`）**：DI・イベントバス・スキーマ検証の中核をピン留めベンダリング。
- **`@earendil-works/pi-ai`**：OpenAI 互換・Anthropic・Gemini 等の複数 LLM API プロバイダを統一扱うオプションバックエンド。
- **`node-pty`**：Windows の ConPTY を含むクロスプラットフォーム永続ターミナル。
- **`koffi`**：Windows での JSONL 書き込み時の原子的ファイル置換（`MoveFileExW`）に利用。
- **`tsdown` / `vitest` / `oxlint`**：高速バンドル・テスト・Lint 基盤。

---

## 3. LLM エージェントフローの核心理論：Turn / Step / Inbox

ここからが本記事の本題である **LLM エージェントフロー** です。
DeepSeek Harness のデフォルトドライバ `ReactLoopAgent`（[`packages/core/agent-loop/src/agent.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/src/agent.ts)）は、実行単位を **Turn（ターン）** と **Step（ステップ）** の 2 階層に明確に分離しています。

- **Step（ステップ）**：**1 回の LLM リクエスト** ＋ その応答に含まれる **ツール呼び出しバッチの実行**。
- **Turn（ターン）**：**0 回以上の Step** を包むユーザー応答の論理単位。最初の入力を受け取る前に開き（`turn/start`）、ツール実行による継続要求も割り込み入力もなくなった時点で閉じる（`turn/end`）。

### 3.1 3 種類の Inbox 入力モード

エージェントへの入力はすべて単一の永続化可能な **Inbox** を経由し、用途に応じて 3 つのメソッドに分かれます。

```ts
// packages/core/agent-loop/src/agent.ts (L168-L187)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/src/agent.ts#L168-L187

  send(message: UserMessage, target: InboxTarget, wakeup: boolean): void {
    // すでに中断(abort)された進行中ターンに起床入力を混ぜると処理が迷子になるため、
    // その場合は必ず「次のターン(next-turn)」へ振り替える。
    // splice 監視フックからの再入キャンセルで分類が変わらないよう、挿入前に確定させておく。
    const wakingAfterAbort = wakeup && this.phase.kind !== 'idle' && this.phase.abort.signal.aborted
    const resolvedTarget = wakingAfterAbort ? 'next-turn' : target

    // Inbox の末尾(Infinity)に 0 件削除・1 件挿入としてメッセージを追記し、
    // 永続イベント agent/inbox/spliced と通知 agent/inbox/inserted を発行する
    this.inbox.splice(resolvedTarget, Infinity, 0, [message])

    // wakeup=true のときだけドライバを起動（または再起動をラッチ予約）する
    if (wakeup) this.wakeDriver(wakingAfterAbort)
  }

  followup(input: UserMessage): void {
    // 通常のユーザー発話：現ターン完了後の「次のターン」に積み、アイドル中なら起床させる
    this.send(input, 'next-turn', true)
  }

  steer(input: UserMessage): void {
    // 実行中の軌道修正（ステアリング）：進行中ターンの「次のステップ」に割り込み、必要なら起床させる
    this.send(input, 'next-step', true)
  }

  inject(input: UserMessage): void {
    // 受動的な文脈注入：次のステップに相乗りさせるだけで、単独ではドライバを起こさない(wakeup=false)
    this.send(input, 'next-step', false)
  }
```

- `followup`：通常のユーザー発言。現在のターンが終わった後に新ターンとして処理。
- `steer`：エージェントがツール実行中などにユーザーが軌道修正（ステアリング）する入力。次のステップの直前で回収される。
- `inject`：システム通知やプラグインからの補足コンテキスト。単体では LLM を叩かず、別の起床トリガーが来たときに一緒にモデルへ渡される。

### 3.2 Turn / Step ライフサイクル全体フロー

以下の Mermaid 図は、入力受付から Turn/Step 実行、エラーリカバリ、終了チェックポイントまでの全体フローです（参照：[docs/agent-lifecycle.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/agent-lifecycle.md)）。

{% mermaid %}
graph TB
    UserInput["ユーザー入力 / プラグイン入力<br/>followup / steer / inject"] --> Inbox["Inbox へ追加<br/>agent/inbox/spliced & inserted"]
    Inbox --> Wake["ドライバ起床 wakeDriver()<br/>agent/status: running"]
    Wake --> TurnStart["セッション記録: turn/start"]
    TurnStart --> Claim["Inbox から入力を回収 claim()<br/>先頭は next-turn、以降は next-step"]
    Claim --> ResolveTools["ステップ有効ツールの解決<br/>ctx.tools.resolve()"]
    ResolveTools --> Assemble["プロンプト組み立て<br/>ctx.systemPrompt.assemble()"]
    Assemble --> PreStep["waterfall: agent/pre-step<br/>圧縮・書き換え・拒否の判定"]

    PreStep -->|"reject または初回空バッチ"| TurnEnd["セッション記録: turn/end<br/>agent/status: idle"]
    PreStep -->|"enter(messages)"| StepStart["セッション記録: step/start"]

    StepStart --> PrepareReq["prepareRequest()<br/>agent/request waterfall & llm.prepareCall()"]
    PrepareReq --> Admit["セッションへ確定記録<br/>system/message, user/message<br/>request/header, request/context"]
    Admit --> Derive["ログから履歴を導出 & deepFreeze<br/>session.deriveMessages()"]
    Derive --> Stream["LLM ストリーミング実行<br/>llm/stream waterfall → agent/assistant-stream"]

    Stream -->|"エラー / 中断"| Attempt["assistant/attempt を記録"]
    Attempt --> ReqErr["waterfall: agent/request-error<br/>リトライ・コンテキスト圧縮の機会"]
    ReqErr -->|"action.kind === retry"| PrepareReq
    ReqErr -->|"未処理エラー"| Recovery["未応答ツールの補償記録<br/>ToolCallRecovery"]

    Stream -->|"成功"| AsstMsg["assistant/message を記録<br/>コンパクト時刻付きストリームを内包"]
    AsstMsg --> CheckTools{"tool-call ブロックあり？"}
    CheckTools -->|"なし / max-tokens"| StepEnd["セッション記録: step/end<br/>toolSet.dispose()"]
    CheckTools -->|"あり"| ExecTools["ツール実行パイプライン<br/>ctx.tools.execute()"]
    ExecTools --> StepEnd

    Recovery --> StepEnd
    StepEnd --> CheckContinue{"ツール継続あり または<br/>next-step 入力あり？"}
    CheckContinue -->|"あり"| Claim
    CheckContinue -->|"なし"| TurnStopping["serial: agent/turn-stopping<br/>終了直前チェックポイント（Goal 等）"]
    TurnStopping --> CheckAfterStop{"フックが next-step を追加？"}
    CheckAfterStop -->|"あり"| Claim
    CheckAfterStop -->|"なし"| TurnEnd

    style UserInput fill:#e1f5ff
    style PreStep fill:#fff9c4
    style Stream fill:#e8f5e9
    style ExecTools fill:#fce4ec
    style ReqErr fill:#ffebee
    style TurnEnd fill:#c8e6c9
{% endmermaid %}

### 3.3 コアコード解説：`preStep()` と `turn()` の協調

注目すべき設計ポイントが 3 つあります。

1. **「確定前のキャンセル」ではログを汚さない**
   `preStep()` や `prepareRequest()`（`llm.prepareCall()`）の非同期処理中にキャンセルされた場合、`system/message` も `user/message` もまだセッションログに書かれていません。そのため履歴が中途半端に破損しません。
2. **ツール集合（`toolSet`）の寿命を 1 ステップに固定**
   `preStep()` で解決した `ResolvedToolSet` は、そのステップのプロンプト宣言とツール実行の両方で共有され、`step/end` の `finally` 節で必ず `decision.toolSet.dispose()` されます。これにより、ステップ途中でツール定義が変化しても整合性が保たれます。
3. **`agent/turn-stopping` による非侵入的な継続制御**
   モデルがツールを呼ばずに終了しようとした瞬間、シリアルイベント `agent/turn-stopping` が発火します。例えば「ゴール達成まで自動継続するプラグイン（`dsh-goal`）」は、ループ本体を書き換えるのではなく、このイベントを購読して `agent.steer()` で次ステップを差し込むだけで実現されています。

実際の `turn()` の骨格を見てみましょう（[`packages/core/agent-loop/src/agent.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/src/agent.ts#L337-L397)）。

```ts
// packages/core/agent-loop/src/agent.ts (L340-L397 抜粋)
      while (true) {
        signal.throwIfAborted()
        const step = phase.step + 1

        // 1. Inbox 回収 → ツール集合解決 → プロンプト組み立て → agent/pre-step waterfall を実行
        const decision = await this.preStep(target, { turn, step })
        if (decision.kind === 'reject') {
          // ポリシーやフックがステップ開始を拒否した場合、ステップを消費せずターンを blocked 終了
          turnEnds = { kind: 'blocked' }
          return false
        }
        try {
          // すでに終了理由が確定しており、追加メッセージも空ならループを抜ける
          if (turnEnds && decision.messages.length === 0) break
          // ターン初回で入力が空に書き換えられた場合も、LLM を叩かずに正常終了する
          if (phase.step === 0 && decision.messages.length === 0) {
            turnEnds = { kind: 'completed' }
            return false
          }

          signal.throwIfAborted()
          // 2. ステップ境界を開き、このステップ中に発行された tool/call を監視する復旧器を接続
          this.session.append('step/start', { turn, step })
          phase.step = step
          const toolRecovery = new ToolCallRecovery()
          const stopRecovery = this.ctx.on('session/event', (session, event) => {
            if (session === this.session) toolRecovery.observe(event)
          })

          try {
            // 3. LLM リクエスト送信〜ストリーミング〜ツールバッチ実行を行う
            const stepEnd = await this.step(decision)
            // max-tokens は sticky（粘着）：一度トークン上限に達したら、
            // 後続ステップが通常完了してもターン終了理由を completed へ降格させない
            if (turnEnds === null || turnEnds.kind !== 'max-tokens') turnEnds = stepEnd
          } catch (error: unknown) {
            // 4. 異常終了時：未応答のまま残った tool/call に対し合成の tool/result を補完記録する
            for (const event of toolRecovery.results()) {
              this.session.append('tool/result', event.data, { surfaceOp: 'append', ... })
            }
            throw error
          } finally {
            // 復旧リスナーを解除し、必ず対になる step/end をログへ確定する
            stopRecovery()
            this.session.append('step/end', { turn, step })
          }
        } finally {
          // 5. このステップ専用に解決したツール集合(toolSet)を解放し、定義の寿命を 1 ステップに限定
          decision.toolSet.dispose()
        }

        signal.throwIfAborted()
        // 6. 自然終了直前で next-step 入力が空なら、終了前チェックポイント(agent/turn-stopping)を発火
        //    Goal プラグイン等はここで steer() を差し込んでターンを継続できる
        if (turnEnds && this.inbox.nextStep.length === 0) {
          await this.dispatch.serial('agent/turn-stopping', { turn, signal })
          signal.throwIfAborted()
        }
        // チェックポイント後も追加の next-step 入力がなければターン完了
        if (turnEnds && this.inbox.nextStep.length === 0) break
        // ツール継続または割り込み入力がある場合は、次ステップから next-step キューを読む
        target = 'next-step'
      }
```

---

## 4. 二重チャネル設計と KV キャッシュ最適化

### 4.1 永続イベント（`session/event`）とライブイベント（`agent/*`）の分離

DeepSeek Harness はイベントを役割ごとに明確に分離しています。

- **`session/event`（永続チャネル）**：`turn/*`、`step/*`、`system/message`、`user/message`、`assistant/message`、`assistant/attempt`、`tool/call`、`tool/result`、`request/header` など。リプレイや履歴導出に必要な「確定した事実」だけを記録します。
- **`agent/*`（ライブ制御チャネル）**：`agent/pre-step`、`agent/request`、`agent/request-error`、`agent/turn-stopping`（プラグインが介入するフック）、および `agent/assistant-stream`（UI 向けのリアルタイムなチャンク配信）。

LLM からのトークンストリームを 1 チャンクごとに永続ログへ書くと I/O が肥大化するため、ストリーミング中はプロセスローカルの `agent/assistant-stream` で UI に流し、**完了（Settlement）した瞬間に 1 件の `assistant/message`（または失敗時の `assistant/attempt`）としてコンパクトな時刻付きストリームごと永続化** します。

### 4.2 KV キャッシュを壊さないプロンプト更新戦略

LLM API（特に DeepSeek API の Context Caching）では、リクエスト先頭からの **プレフィックス一致（KV キャッシュヒット）** がレイテンシとコストを劇的に左右します。もしステップごとにシステムプロンプトの先頭を書き換えてしまうと、過去の全履歴のキャッシュが毎回無効化されてしまいます。

DeepSeek Harness はこの問題に対し、極めて合理的なアプローチを採用しています（参照：[packages/core/agent-loop/README.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/README.md#model-experience)）。

1. **システムプロンプトの変更は「追記」で表現する**
   - 初期システム指示を先頭に配置した後、途中でプロンプトセクションが変化しても先頭を書き換えません。`assembly.systemUpdates()` が差分を通常の `system/message` として **末尾へ追記** します。
   - 先頭のシステムノードを再構築・統合するのは、**コンテキスト圧縮（Compaction）でいずれにせよプレフィックスが切り替わる直後のステップだけ** です。
2. **動的コンテキストは `<system-reminder>` で構造化差分管理する**
   - カレントディレクトリや時刻などの実行時コンテキスト（Runtime Context）は、`RuntimeContextProjection` が前回の確定スナップショットと構造比較し、**変化があったときだけ** `<system-reminder>` タグ付きの `user/message` を末尾に追記します。変化がなければ何も出力しません。
3. **ツール一覧の変化も `developer/message` で追記通知する**
   - ステップ間で利用可能なツールが追加・削除された場合、`buildRequest()` が `request/header` の差分を検出し、`developer/message`（`tool-addition` / `tool-removal`）を履歴末尾へ追記します。
4. **リクエストの不変性（`deepFreeze`）とキャッシュ再利用**
   - `buildRequest()` は `session.deriveMessages()` で導出したメッセージオブジェクトを `WeakSet<Message>`（`this.frozenMessages`）で追跡しながら一度だけ `deepFreeze()` し、リクエスト全体を `Object.freeze()` してアダプタへ渡します。これにより、ミドルウェアによる意図しない履歴破壊を防ぎつつ、毎ステップの再凍結コストも抑えています。

```ts
// packages/core/agent-loop/src/agent.ts (L716-L733 抜粋)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/src/agent.ts#L716-L733

    // 1. リクエストヘッダ（モデル設定＋ツールスキーマ）を再帰凍結し、アダプタ側の破壊的変更を防ぐ
    deepFreeze(header)

    // 2. 追記専用セッションログからモデル向けのメッセージ列を導出する（Model-Visible ⟺ Logged）
    const boundaryMessages = session.deriveMessages()
    for (const message of boundaryMessages) {
      // すでにこのエージェントで凍結済みのメッセージ参照は WeakSet でスキップし、毎ステップの再凍結コストを削減
      if (this.frozenMessages.has(message)) continue
      deepFreeze(message)
      this.frozenMessages.add(message)
    }
    // 配列コンテナ自体も凍結する
    Object.freeze(boundaryMessages)

    // 3. 最終的な LLM リクエストを構築して凍結し、「agent-loop 由来」の内部ブランドを付与
    //    ※ signal（AbortSignal）だけはライブなキャンセル通知のため凍結せずそのまま渡す
    const request = markAgentLoopRequest(Object.freeze({
      ...header.config,
      messages: boundaryMessages,
      toolHistory: session.toolHistory(), // アダプタ固有のツール宣言履歴
      ...header.tools !== undefined ? { tools: header.tools } : {},
      sessionId: this.session.id,
      signal,
    }))
    return request
```

---

## 5. ツール実行パイプライン（Guarded Tool Execution Pipeline）

LLM が応答に 1 つ以上の `tool-call` ブロックを含めたとき、`ReactLoopAgent` は生の呼び出しバッチをそのまま `ctx.tools.execute()` に委譲します。ツール実行器（`dsh-tool-executor-native` や `dsh-tool-executor-ptc`）は、共通の `ToolExecutionPipeline`（[`packages/core/tools/src/pipeline.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/tools/src/pipeline.ts)）を通してすべての呼び出しを処理します。

### 5.1 並列実行と排他バリア（Fail-Closed スケジューリング）

モデルが 1 ステップで複数のツールを同時に呼び出した場合、パイプラインは各呼び出しの `executionMode` を判定します。

```ts
// packages/core/tools/src/pipeline.ts (L106-L115)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/tools/src/pipeline.ts#L106-L115

  executionMode(exec: ToolExecutionInput): ToolExecutionMode {
    // 呼び出し元のエージェントスコープ・ステップ有効ツール集合(toolSet)からツール定義を解決
    const tool = this.resolveExecution(exec.name, exec.agent, exec.parent !== undefined, exec.toolSet)
    // ツール未発見、または並列安全判定フック(isConcurrencySafe)が未定義なら安全側(exclusive)へ倒す
    if (!tool?.isConcurrencySafe) return { kind: 'exclusive' }
    try {
      // 引数の中身（例：read-only フラグの有無など）に応じて動的に並列安全性を判定
      const concurrencySafe: unknown = tool.isConcurrencySafe(exec.arguments)
      // 厳密に boolean の true のみ並列許可（truthy な別値は拒否）
      return concurrencySafe === true ? { kind: 'parallel' } : { kind: 'exclusive' }
    } catch {
      // 判定関数が例外を投げた場合も Fail-Closed で排他バリア扱いとする
      return { kind: 'exclusive' }
    }
  }
```

- **Fail-Closed 設計**：`tool.isConcurrencySafe(args)` が厳密に `true` を返したときだけ `parallel`（有界ローリングプール：既定 `maxParallelToolCalls = 10`）で並列実行されます。未定義・不明ツール・例外発生時はすべて安全側に倒し `exclusive`（排他バリア）として直列実行します。
- **モデル順序の保証**：並列実行で完了順序が前後しても、`tool/result` のセッション記録は必ずモデルが呼び出した順序（model-order）で確定されます。

### 5.2 ガード付きツール実行パイプラインのフロー

各ツール呼び出しは、以下の多段パイプラインを通過します（参照：[docs/tool-execution-pipeline.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/tool-execution-pipeline.md)）。

{% mermaid %}
graph TB
    ModelCall["モデル応答の tool-call ブロック"] --> LogCall["セッション記録: tool/call<br/>実行前に必ずログ化"]
    LogCall --> PreExec["waterfall: tools/pre-execute<br/>フック・権限・サンドボックス判定"]

    PreExec -->|"allow"| Guards["単調ガード Monotonic Guards<br/>順序変更不可の拒否/棄権チェック"]
    PreExec -->|"ask"| Approval["人間承認 ctx.approval<br/>ワンショット確認プロンプト"]
    PreExec -->|"deny"| Denied["拒否: ツール本体をスキップ"]

    Approval -->|"allowed-once"| Guards
    Approval -->|"rejected / cancelled / unavailable"| Denied

    Guards -->|"allow"| Around["waterfall: tools/execute<br/>タイムアウト・リトライ・メトリクス"]
    Guards -->|"deny"| Denied

    Around --> Body["ツール本体 execute() 実行<br/>fs/* ゲート・ツール固有イベント発行"]
    Body --> Project["ToolDefinition.projectContent<br/>実行結果テキスト・画像の整形"]
    Denied --> Project

    Project --> PostExec["waterfall: tools/post-execute<br/>結果の受理・ブロック・置換・文脈追加"]
    PostExec --> Finalize["ToolDefinition.finalizeContent<br/>最終コンテンツ不変条件の適用"]
    Finalize --> Notify["同期通知: tools/result<br/>凍結された確定結果を配信"]
    Notify --> LogResult["セッション記録: tool/result<br/>モデルへ返す単一の確定結果"]
    LogResult --> DeferCtx["バッチ完了後: additionalContexts を<br/>next-step へ FIFO 注入"]

    style ModelCall fill:#e1f5ff
    style PreExec fill:#fff9c4
    style Approval fill:#ffebee
    style Body fill:#e8f5e9
    style LogResult fill:#c8e6c9
{% endmermaid %}

このパイプラインの巧みな点は **「ポリシーをツール本体から完全に切り離しつつ、セキュリティ不変条件を守れる」** ことにあります。

- 汎用フックや権限プリセットは `tools/pre-execute` / `tools/post-execute` の waterfall リスナーとして外から挿し込めます。
- 一方で「他のプラグインに並び替えられては困る所有者の安全ルール」は、waterfall ではなく **単調ガード（Monotonic Guards）** に登録され、`allow` 後に必ず実行されて `deny`（拒否）を覆せません。
- 引数は実行前に `snapshotJsonValue()` ＋ `deepFreeze()` でロスレス JSON 境界を通過させるため、プロトタイプ汚染や後から引数が改変される事故を構造的に防ぎます。

---

## 6. 失敗・リトライ・キャンセル時の不変条件

実用エージェントで最も壊れやすいのが「API エラー・コンテキスト超過・ユーザーによる途中キャンセル」です。DeepSeek Harness は以下の 3 つの不変条件でログの一貫性を守っています。

### 6.1 `assistant/message` と `assistant/attempt` の分離

- **成功時**：`assistant/message` を記録（履歴導出の対象）。
- **失敗・リトライ・ストリームエラー時**：`assistant/attempt` を記録。観測・デバッグ・テレメトリ用にストリームは残りますが、**`deriveMessages()` のモデル履歴には含まれない** ため、次のリトライ時に壊れた応答が混入しません。
- **ユーザーによる途中キャンセル時**：すでにユーザー画面へ配信済みのテキストブロックがある場合は `interrupted: true` を付けた `assistant/message` として確定します。これにより、次のターンで「先ほど途中まで出力した内容」をモデルも正しく認識できます。

### 6.2 ステップ内リトライ（`agent/request-error`）

LLM リクエストがエラー（レート制限やコンテキスト長超過など）で失敗すると、ドライバは `assistant/attempt` を記録した後、waterfall イベント `agent/request-error` を発火します（[`agent.ts` L538-L554](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/src/agent.ts#L538-L554)）。

- 例えば `dsh-compaction-basic` は、コンテキスト超過エラーを受け取ると **その場（開いたステップ内）でツール結果の枝刈りや会話要約を実行** し、`{ kind: 'retry' }` を返します。
- ループは `continue` で同じステップのまま `prepareRequest()` へ戻り、プロンプト再組み立てやユーザーメッセージの重複登録を一切行わずに、圧縮後の履歴でリクエストを再送します。

### 6.3 ツール呼び出し・結果のペアリング保証（`ToolCallRecovery`）

LLM API の多くは「`tool_call` があるのに、対応する `tool_result` が欠落している履歴」をバリデーションエラーとして拒否します。
そこで `turn()` はステップ開始時に `ToolCallRecovery` を `session/event` に接続しておき、ステップが例外やキャンセルで異常終了した際、**未応答のまま残ったすべての `tool/call` に対して合成エラー結果を自動追記** します。

- キャンセルによりディスパッチ前に打ち切られた呼び出し → `ABORTED_BEFORE_DISPATCH`
- `tool/call` は記録されたが結果が確定する前に失敗 → `TOOL_OUTCOME_UNKNOWN`
- 呼び出し開始前に失敗 → `TOOL_NOT_STARTED`

これにより、失敗したターンの直後にユーザーが会話を続けても、セッション履歴の整合性が絶対に崩れません。

---

## 7. まとめと次回予告

本記事（Part 1）では、DeepSeek Harness (`dsh`) の全体像と、中核となる **LLM エージェントフロー** の設計理論をソースコードベースで解説しました。要点を整理すると次の 4 点に集約されます。

1. **特権的コアを持たないプラグイン構成** と **イベントソーシング** により、エージェントループ自体も差し替え・拡張可能。
2. **Turn / Step の 2 階層制御** と **3 種の Inbox（`followup` / `steer` / `inject`）** で、自律ツールループと人間の割り込みを自然に両立。
3. **追記型の `system/message` / `<system-reminder>`** により、KV キャッシュのプレフィックスヒット率を高く維持。
4. **Fail-Closed な並列/排他ツールパイプライン** と **`assistant/attempt`・`ToolCallRecovery`** により、異常系でもモデル履歴が破綻しない。

次回の **Part 2** では、このエージェントフローを裏側で支える **イベントソース型セッション（`dsh-session` / `dsh-session-projection` / 永続化と世代マイグレーション）** の内部構造を詳しく掘り下げます。
