---
title: DeepSeek Harness 技術解析 Part 2：イベントソース型セッション設計（Model-Visible ⟺ Logged・Projection・永続化と Fork）

date: 2026-04-12 14:17:53
categories: [AI]
tags: [Deep Learning, TypeScript, Node.js, LLM, AI Agent, DeepSeek, Event Sourcing, CQRS, 機械学習, AI, 人工知能, 生成AI]
lang: ja
description: DeepSeek Harness の状態管理の要であるイベントソース型セッション（dsh-session）を徹底解説します。追記専用ログからモデル履歴を導出する Surface 層、CQRS 型 SessionProjection、単一ライター所有権に基づく永続化・クラッシュ復旧、そして非破壊な世代マイグレーションと Fork の仕組みをソースコードとフロー図で紐解きます。
---

## はじめに

[Part 1](/2026/04/03/deepseek_harness_part1/) では、DeepSeek Harness (`dsh`) の全体像と LLM エージェントフロー（Turn / Step / Inbox / ツールパイプライン）を解説しました。

Part 2 である本記事では、そのエージェントフローの裏側を支える **イベントソース型セッション（Event-Sourced Session）** の設計理論に踏み込みます。エージェント開発で頻発する「UI とモデル履歴のズレ」「コンテキスト圧縮後のログ消失」「クラッシュ時の不整合」「途中分岐（Fork）の難しさ」を、DeepSeek Harness がどのように解決しているかを見ていきましょう。

---

## 1. 基本理論：なぜ「Model-Visible ⟺ Logged」なのか

多くのエージェント実装では、会話履歴を `messages: Message[]` のような可変配列として持ち、コンテキスト圧縮時に古い要素を `splice()` で消したり、システムプロンプトを直接書き換えたりします。しかし、この方式には致命的な弱点があります。

- **UI 表示とモデル入力が衝突する**：コンテキスト圧縮で古いメッセージを消すと、人間のチャット履歴まで消えてしまう。
- **リプレイやデバッグが不可能**：失敗したリトライや途中キャンセルの痕跡が失われる。
- **途中分岐（Fork）や再開（Resume）で状態が壊れる**：配列以外の付随状態（未完了ツール、選択中のモデル、Todo 状態など）を別管理すると必ず不整合が起きる。

DeepSeek Harness はこれらを解決するために、**イベントソーシング（Event Sourcing）** と **CQRS（Command Query Responsibility Segregation）** を徹底しています。

> **核心不変条件：`Model-Visible ⟺ Logged`**
> モデルリクエストに到達するすべての情報は、追記専用（Append-Only）の `SessionEvent` ログから再構成可能でなければならない。新しいモデル可視入力を追加するには、必ず対応するセッションイベントを定義する。

---

## 2. `SessionEventMap` と Surface 層（人間向け履歴 vs モデル向け履歴）

セッションログの語彙は [`packages/core/session/src/types.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/types.ts) の `SessionEventMap` で定義され、TypeScript の宣言マージ（Declaration Merging）によりプラグインから拡張可能です（例：`compaction/*` や `hook/*`）。

### 2.1 全イベントと Surface イベントの区別

ログに記録されるイベントはすべて連続したシーケンス番号（`seq`）を持つロスレス JSON ですが、**すべてが LLM に送られるわけではありません**。モデルへの入力メッセージを生成するイベント（**Surface Event**）は [`packages/core/session/src/surface.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/surface.ts#L50-L56) の 5 種類だけです。

```ts
// packages/core/session/src/surface.ts (L50-L56)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/surface.ts#L50-L56

// モデルへの入力メッセージ（Surface）を生成できる 5 種類のイベントのみをホワイトリスト定義。
// turn/*, step/*, tool/call, assistant/attempt などはここに入らず、履歴導出から除外される。
const SURFACE_EVENT_TYPES = new Set<string>([
  'system/message',    // 初期システム指示および追記型のセクション更新
  'developer/message', // ツール追加・削除などの開発者向けインクリメンタル通知
  'user/message',      // 人間のプロンプト・inject された文脈・<system-reminder>
  'assistant/message', // 成功確定（または途中中断）したモデル応答
  'tool/result',       // ツール実行パイプラインが確定した単一の返却結果
])
```

- **Surface イベント**（上記 5 種）：`surfaceOp: 'append'`（末尾追加）または `surfaceOp: { op: 'replace', ... }`（コンテキスト圧縮などによる範囲置換）のいずれかを持ちます。
- **非 Surface イベント**：`turn/*`、`step/*`、`tool/call`、`assistant/attempt`、`request/header`、`request/context`、`session/end-seed` など。ライフサイクルや監査・リプレイ用に残りますが、`deriveMessages()` には直接入りません。

### 2.2 「人間のチャット履歴」と「モデルが見る履歴」の分離

この `surfaceOp` の区別により、1 本の追記専用ログから 2 つの異なるビューを矛盾なく取り出せます（[`surface.ts` L79-L106](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/surface.ts#L79-L106)）。

- **人間向けトランスクリプト（UI）**：`isAppendSurfaceEvent()`（`surfaceOp === 'append'`）のみを採用。コンテキスト圧縮で `replace` が起きても、ユーザーがすでに見た過去の会話は UI から消えません。
- **モデル向け履歴（`deriveMessages()`）**：`replace` された範囲を要約ノードでシャドウイング（影に隠す）した最新の Surface ビューを導出します。さらにプラグインが既存メッセージの内容を純粋変換したい場合（例：画像のオフロード）は、`SessionMessageProjection` を登録して元のログを書き換えずに変換結果を適用します。

{% mermaid %}
graph TB
    Log["追記専用 SessionEvent ログ<br/>seq: 0, 1, 2, ... (Single Source of Truth)"]

    Log --> SurfaceFold["Surface 畳み込み foldSurface()<br/>append / replace と MessageProjection を適用"]
    Log --> HumanView["人間向けトランスクリプト<br/>isAppendSurfaceEvent() のみ抽出<br/>圧縮後も過去の会話が消えない"]
    Log --> ProjReg["ctx.sessionProjections<br/>ドメインごとの同期 Pure Fold"]

    SurfaceFold --> Derive["deriveEventMessage() / deriveMessages()<br/>モデルへ送る不変 Message[]"]
    ProjReg --> HostState["Host 状態読み取り<br/>handle.stateOf(session)"]
    ProjReg --> ClientSnap["Client スナップショット<br/>snapshot(session) & 差分 Push"]

    style Log fill:#e1f5ff
    style Derive fill:#e8f5e9
    style HumanView fill:#fff9c4
    style ClientSnap fill:#fce4ec
{% endmermaid %}

実際の 1 イベント→メッセージ導出関数 `deriveEventMessage()` は驚くほど簡潔です（[`packages/core/session/src/surface.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/surface.ts#L120-L152)）。

```ts
// packages/core/session/src/surface.ts (L120-L152)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/surface.ts#L120-L152

export function deriveEventMessage(
  event: SessionEvent,
  projectedMessages?: ReadonlyMap<SessionSeq, Message>,
): Message | null {
  // 1. 画像オフロード等のプラグイン由来 MessageProjection がこの seq を置換済みなら、
  //    元のログを書き換えずに変換後メッセージを優先採用する
  const projected = projectedMessages?.get(event.seq)
  if (projected !== undefined) return projected

  // 2. イベント種別ごとに LLM ワイヤー用の Message へ射影する
  switch (event.type) {
    case 'user/message': {
      // ユーザー入力・inject 文脈は生成元が決めたフォーマットをそのまま渡す
      return event.data
    }
    case 'system/message':
    case 'developer/message':
    case 'assistant/message': {
      // 空ノード（例：初期予約された空システムノードや、usage だけ持つ max-tokens 応答）は
      // Surface 上の位置だけ保持し、LLM API へ空メッセージを送らないよう null を返す
      if (event.data.message.content.length === 0) return null
      return event.data.message
    }
    case 'tool/result': {
      // tool/result の内部 error/meta フィールドは除外し、モデル向け message 本体のみ渡す
      return event.data.message
    }
    default:
      // 非 Surface イベント（境界・attempt・ログ専用レコード）はモデル履歴に含めない
      // 宣言マージで拡張可能なユニオンのため、ここでは意図的に assertNever を使わない
      return null
  }
}
```

---

## 3. CQRS 型 `ctx.sessionProjections`：ドメイン状態の統一畳み込み

セッションログから導出したいのは LLM メッセージだけではありません。たとえば：

- 現在の **Inbox 状態**（`AgentLoop` が所有）
- 選択中の **モデル設定（`modelSelection`）**
- **Todo リスト**、**作業ディレクトリ**、**Goal 進捗**、**セッションタイトル**

もし各プラグインがバラバラに `session/event` を購読して独自キャッシュを持つと、クライアントに送るスナップショットの時刻（`seq`）がズレたり、リプレイ時に状態が再現できなくなったりします。

そこで `dsh-session-projection`（[`docs/subsystems/session-projection.md`](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session-projection.md)）が **「フレームワークが駆動し、ドメインは純粋計算だけを持つ」** という Projection Seam を提供します。

### 3.1 `ProjectionDefinition` の 4 原則

各ドメインプラグインは `ProjectionDefinition<K, S>` だけを登録します。

1. **完全同期（Synchronous）**：`init(header, inheritedEventCount)` と `apply(state, event)` は必ず同期関数。非同期を禁止することで、全ドメインの状態を同一 `asOfSeq` の一貫した断面（Consistency Cut）で取得できます。
2. **参照等価によるゼロコスト最適化**：関心のないイベントでは `apply()` が **同じ `state` 参照をそのまま返す** 契約。フレームワークは `Object.is(prev, next)` で変化なしを即判定し、下流の `view()` 計算やクライアント配信を一切行いません。
3. **Host 状態と Client ビューの分離**：クライアントへ送る必要があるユニットだけが `wire: { viewSchema, view(state) }` を定義します。クライアント側でイベントを畳み込む必要はなく、Zod 検証済みの完成値だけを受け取ります。
4. **バージョン付き永続キャッシュ**：`state` はプレーン JSON＋`stateVersion` を持つため、チェックポイント保存＋差分リプレイ（suffix replay）が安全に機能します。

---

## 4. 永続化 Seam・単一ライター所有権・クラッシュ復旧

メモリ上の `Session` をディスクへ永続化する責務は、`dsh-session-persistence`（Service Definition）と `dsh-session-persistence-jsonl`（Provider）に分離されています（参照：[docs/subsystems/persistence.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.md)）。

### 4.1 `SessionHandle` と単一ライター所有権

永続化ログへのアクセスはすべて `create()` / `open(id, 'read' | 'write')` が返す **`SessionHandle`**（`read` / `append` / `flush` / `close`）を経由します。

- **単一ライター（Single-Writer Ownership）**：同一セッションに対して 2 つ目の `open(id, 'write')` を行うと即座に `SessionAlreadyOwnedError` で拒否されます。これにより、ライブ実行中のターンと別プロセス／修復処理が競合する事故を防ぎます。
- **Write-Behind バッファと Flush チェックポイント**：`session/event` は同期イベントですが、永続化バックエンドは有界のバッチ窓（write-behind window）で非同期にまとめて書き込み、プロデューサをブロックしません。耐久性を保証したい境界では `session/flush` が待機とドレインを完了させます。
- **`SessionHeader` の分離**：フォーマットバージョン・作成日時・初期 `cwd`・`isSeeded` フラグはイベントログの外側の不変ヘッダとして保持され、モデル履歴には混ざりません。

### 4.2 クラッシュ復旧は「切り詰めず、閉じる」

長時間タスクの途中でプロセスが強制終了すると、ログ末尾は `turn/start` が開いたまま `turn/end` が欠落した状態になります。

DeepSeek Harness の永続化層は **途中でクラッシュしたターンを切り捨て（truncate）しません**。破損した物理末尾フレーム（未完了の Zstandard ブロックなど）だけを除いた有効プレフィックスを返し、**修復はリーダー（`agent-loop` の Resume）が行います**。

- 書き込み権限を持つ Resume 処理が、未応答ツールエラー＋開いた `step/end`＋合成の `turn/end { reason: { kind: 'interrupted' } }`（`interruptedTurnClosers`）を通常のバッチとして追記します。
- 読み取り専用の観測者（`session-query` 等）はディスクへ一切書き戻さず、メモリ上でのみ同じクローズ補完を適用します。

### 4.3 非破壊な隣接世代マイグレーション（`vN -> vN+1`）

長期運用されるセッションログのフォーマット進化に対応するため、JSONL バックエンドは **歴史ファイルを絶対に上書き・削除しない** 設計を取っています（参照：[docs/architecture.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md#session-log)）。

- v0 は `session.jsonl[.zstd]`、v1 以降は `session.vN.jsonl[.zstd]` として保存されます。
- `read` で古い世代を開いた場合はメモリ上で `vN -> vN+1` の隣接マイグレーション連鎖を適用するだけでディスクには書きません。
- `write` で開いた場合のみ、最新世代の新ファイル（例：`session.v4.jsonl.zstd`）を検証付きで排他発行し、旧世代ファイルはそのまま残します。

---

## 5. Turn 境界での Fork（セッション分岐）

イベントソーシングの最大の恩恵の 1 つが **Fork（会話の途中分岐）** です。[`packages/core/session/src/fork.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/fork.ts#L21-L30) を見ると、Fork のシード構築がわずか 10 行で実装されていることが分かります。

```ts
// packages/core/session/src/fork.ts (L21-L30)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/fork.ts#L21-L30

export function buildForkSeed(events: readonly SessionEvent[], boundary: SessionSeqType): SessionEvent[] {
  // 1. 親ログの先頭(seq=0)から指定境界(boundary)までを包含スライス（元イベントオブジェクトを共有）
  const prefix = events.slice(0, boundary + 1)

  // 2. 継承プレフィックスの直後に「ここまでが親からの継承」を示すマーカーを挿入
  //    タイムスタンプは境界イベントと揃え、seq は連続番号(boundary + 1)を採番する
  prefix.push({
    type: 'session/end-seed', seq: SessionSeq(boundary + 1),
    time: events[boundary]!.time,
    data: { inherited: true },
  })

  // 3. 分岐点が未完了のターン／ステップ途中だった場合、
  //    未応答ツールの合成結果・step/end・turn/end({ kind: 'forked' }) を末尾へ自動補完して返す
  return prefix.concat(openTurnClosers(prefix, { kind: 'forked' }))
}
```

1. 親ログの `0..boundary` をそのまま切り出す（`inheritedEventCount` として記録）。
2. 境界直後に `{ type: 'session/end-seed', data: { inherited: true } }` マーカーを挿入し、どこまでが親からの継承かを明示。
3. もし分岐点が未完了のターン／ステップ途中であれば、`openTurnClosers(prefix, { kind: 'forked' })` で合成の終了イベントを追記して整合性を保ちます。

---

## まとめと次回予告

Part 2 では、DeepSeek Harness の記憶と状態を支えるイベントソーシング設計を解説しました。

- **Surface 層** により、1 本の追記専用ログから「人間向け履歴」と「圧縮・変換済みモデル履歴」を矛盾なく両立。
- **`ctx.sessionProjections`** により、全ドメインの派生状態を同期 Pure Fold＋`Object.is` 差分で統一管理。
- **`SessionHandle`・非破壊マイグレーション・`buildForkSeed()`** により、クラッシュ耐性・長期互換性・途中分岐をシンプルに実現。

次回の **Part 3** では、これらすべてのコンポーネントを「特権的コアなし」で繋ぎ合わせるプラグイン基盤 **Cordis**、エージェント単位の **Scoped Lifetime（`dsh-scope`）**、そして **Profiles / Bundles / Patch Layers** による宣言的構成を解説します。
