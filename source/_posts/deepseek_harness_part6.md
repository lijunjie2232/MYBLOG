---
title: DeepSeek Harness 技術解析 Part 6：ランタイム自己拡張とマルチサーフェス展開（Extensions・Typert RPC・Web Client）

date: 2026-05-22 16:24:37
categories: [AI]
tags: [Deep Learning, TypeScript, Node.js, React, Electron, LLM, AI Agent, DeepSeek, RPC, Cordis, 機械学習, AI, 人工知能, 生成AI]
lang: ja
description: DeepSeek Harness シリーズ最終回。エージェント自身が稼働中の Cordis 環境を検査し、バージョン付き Host/Client プラグインを動的に生成・マウントする「ランタイム自己拡張（Extensions）」と、型グラフ生成ベース RPC「Typert」およびブラウザ側 Cordis＋Slot 構造によるマルチサーフェス展開（Web / Desktop / SDK）の設計を解説します。
---

## はじめに

最終回となる Part 6 では、DeepSeek Harness の中でも特に先進的な 2 つの領域を解説します。

1. **ランタイム自己拡張（Extensions）**：エージェント自身が稼働中のハーネスを検査し、新しい Host/Client プラグインを書いてその場でロード・アンロードする仕組み。
2. **マルチサーフェス展開（Typert RPC & Web Client）**：これまで見てきた Host 側のプラグインツリーを、どうやって型安全かつ疎結合に **Web UI・Electron Desktop・TypeScript/Python SDK** へ投影しているのか。

---

## 1. ランタイム自己拡張：エージェントが自身のプラグインを書く

Part 3 で見たように、DeepSeek Harness ではすべての機能が「可逆的なエフェクトを持つ Cordis プラグイン」です。この構造を突き詰めると、**「エージェント自身が会話の途中で新しいプラグイン（ツールや UI パネル）を書いて、自分のセッションにマウントし、不要になったら綺麗にアンマウントする」** ことが可能になります（参照：[docs/subsystems/extensions.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/extensions.md)）。

これを担うのが `packages/extensions/` の 2 つのサービスです。

### 1.1 `ctx.cordisInspect`：コードを書く前の安全な自己検査

モデルが当てずっぽうでプラグインを書くと、存在しないサービス名や誤った API で失敗します。`ctx.cordisInspect`（[`packages/extensions/cordis-host-runner/src/inspect-registry.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/extensions/cordis-host-runner/src/inspect-registry.ts)）は、承認済みの Host プロバイダと Client プロバイダ（ブラウザ側マニフェストのミラー）を横断的にクエリできる検査レジストリです。モデルはまず `list()` / `query()` で現在のランタイム状態を確認してからプラグインを実装できます。

### 1.2 `ctx.dynamicCordisRunner`：バージョン管理・承認・自動ロールバック

エージェントがプラグインを作成・起動する流れは以下の通りです。

1. **`define()`**：セッション所有のプラグイン ID と、不変（immutable）なバージョン付きパッケージ（Host コード／Client コード）を定義。
2. **`run()`**：指定バージョンをアクティベート。未承認の Client パッケージや権限が必要な実行は人間承認を待ち、プラグイン単位で「今後のバージョンも承認（`approveFutureVersions`）」可能。
3. **Fiber 隔離と自動ロールバック**：[`packages/extensions/cordis-host-runner/src/lifecycle.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/extensions/cordis-host-runner/src/lifecycle.ts#L22-L45) では、動的プラグインを `guardedPlugin()` でラップして専用グループ Fiber（`cordis-dynamic`）配下にマウントします。

```ts
// packages/extensions/cordis-host-runner/src/lifecycle.ts (L22-L45)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/extensions/cordis-host-runner/src/lifecycle.ts#L22-L45

export async function startHostHalf(
  group: Fiber,
  plugin: Plugin,
  reportGuardFailure: (error: Error) => void,
): Promise<Fiber> {
  // 1. 動的プラグインを束ねる親グループ(cordis-dynamic)の起動完了を待機
  await group.await()

  // 2. サンドボックスが生成したプラグインを登録ガード(guardedPlugin)でラップし、
  //    子 Fiber としてマウントする（以後このプラグインの全登録は fiber に紐づく）
  const fiber = group.ctx.plugin(guardedPlugin(plugin, reportGuardFailure))
  try {
    // 3. プラグインの初期化完了（または未解決 inject の待機状態への遷移）を待つ
    await fiber.await()
  } catch (error) {
    // 4. 起動失敗時は必ず fiber.dispose() で途中登録されたエフェクトを完全ロールバックする
    await fiber.dispose()
    const message = error instanceof Error ? error.message : String(error)

    // よくある失敗（旧バージョンを止めずに新バージョンを起動した名前衝突）には、
    // モデル自身が次のステップで修正できるよう具体的な対処手順をエラー文へ添える
    if (message.includes('already registered')) {
      throw new Error(
        `${message} — to REPLACE something an earlier dynamic package registered, first stop that package through its runner or the Cordis panel `
        + 'before running the new version.',
      )
    }
    throw error instanceof Error ? error : new Error(message)
  }
  return fiber
}
```

- **失敗したプラグインは絶対に残留しない**：起動に失敗した場合は `catch` 内で即座に `await fiber.dispose()` され、途中まで登録されたエフェクトがすべて巻き戻されます。
- **停止処理もゼロコード**：正常起動したプラグインを止めるとき（`undefine` / `undefineFromPanel`）も、特別なクリーンアップコードを書く必要はなく、通常の `fiber.dispose()` を呼ぶだけで全登録が綺麗に解除されます。

---

## 2. Typert RPC：型グラフ生成による Host ↔ Client 通信

プラグインが 50 以上のグループに分かれているシステムでは、REST API や手書きの WebSocket プロトコルを保守するのは現実的ではありません。DeepSeek Harness は **Typert**（[`docs/subsystems/typert.md`](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/typert.md)）という独自の型グラフ生成＋RPC ランタイムを採用しています。

### 2.1 Typert の仕組み

1. **Host サービスに `@Remote` デコレータを付けるだけ**
   - 例：Part 6 の `undefineFromPanel` や Part 5 の `agentTeams.setEnabled` のように、既存の Cordis サービスのメソッドに `@Remote` を宣言します。
2. **ビルド時に型グラフから 3 種のアーティファクトを自動生成**
   - Host 側ディスパッチ用の `./typert`
   - コンシューマ向けの `./remote`
   - ブラウザクライアント向けの `./client/typert`
3. **Lookup と Context 解決の自動化**
   - Host メソッドの引数が `agent: Agent` のようなメモリ上のオブジェクトであっても、`TypertLookupMap` の宣言によりワイヤー上は `sessionId` 等の ID に自動変換され、Gateway が Host 側でライブオブジェクトへ安全に逆引き（Lookup）します。
   - バイナリ（`Uint8Array`）の添付エンコード、双方向ストリーム（`RemoteStream<Out, In>`）、およびワイヤー引数に混ぜない帯域外キャンセル（`signal`）もすべて生成コーデックが透過的に扱います。

---

## 3. Web Client・Desktop・SDK へのマルチサーフェス展開

### 3.1 ブラウザ側も「Everything is a Plugin（Cordis）」

驚くべきことに、DeepSeek Harness の **Web UI（ブラウザ側）自体も独立した Cordis アプリケーション** です（参照：[docs/subsystems/web-client.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/web-client.md)）。

{% mermaid %}
graph TB
    Host["Host アプリケーション<br/>ビジネスサービス & packages/api/*-controller"]
    Gateway["API Gateway & Typert Remotes<br/>ctx.remote.&lt;namespace&gt; / $events 論理ストリーム"]
    Model["Client Models (React 非依存)<br/>ClientSessions / ClientWorkspaceModel"]
    Adapter["UI Adapters & Conversation<br/>ui-session / ui-conversation → ui-chat & ui-trajectory"]
    Slots["UI Slots & Renderer<br/>ui-slots / ui-renderer (useSyncExternalStore)"]
    React["React DOM レンダリング"]

    Host -->|"Typert RPC & Follow ストリーム"| Gateway
    Gateway -->|"生成済み型安全クライアント"| Model
    Model -->|"Observable スナップショット"| Adapter
    Adapter -->|"ターゲット別ビュー & Slot 登録"| Slots
    Slots --> React
    React -.->|"ユーザー操作コールバック"| Model

    style Host fill:#e1f5ff
    style Gateway fill:#fff9c4
    style Model fill:#e8f5e9
    style Slots fill:#fce4ec
{% endmermaid %}

- **厳格な単方向依存**：`Host state → Remote transport → Client model → UI adapter → Conversation → Slots → React`。プレゼンテーション用 React コンポーネントは Cordis の `ctx` やトランスポートを直接受け取らず、純粋な Props とコールバックだけで描画します。
- **Chat と Trajectory の独立投影**：同じセッションイベント列から、`ui-chat`（人間向けチャット表示）と `ui-trajectory`（詳細な実行軌跡ビュー）が互いを `import` せずに独立したビューを構築します。
- **物理再接続と論理復旧の分離**：WebSocket が切れても、再接続後の新しい Generation の先頭フレームで `follow()` が完全なベースラインスナップショットを送り直し、ローカルウィンドウを原子的に置換するため、汎用的な `resync()` の複雑さを排除しています。

### 3.2 Electron Desktop と TypeScript / Python SDK

- **Electron Desktop（`apps/desktop`）**：署名済みリソース内に同じ `dsh` 本番ランタイムを同梱し、Electron Node モードでプライベート Host を起動します。ウィンドウはパッケージ済み Web アセットを読み込み、Node IPC 経由で認証トークンとブート情報を受け取って同じ Web プラグイン群を動かします。
- **TypeScript / Python SDK**：どちらも裏側で同じ `dsh --profile sdk`（または最小構成の `sdk-minimal`）をサブプロセス起動し、JSON-RPC 越しにエージェントを制御します。Python 側はプラットフォーム別ランタイムホイール（`deepseek-harness-sdk-runtime-<platform>-<arch>`）として `dsh` CLI を同梱するため、言語が違ってもプラグイン構成やパッチモデルの挙動が 100% 一致します。

---

## シリーズ全体のまとめ

全 6 回にわたって、[DeepSeek Harness (`dsh`)](https://github.com/deepseek-ai/deepseek-harness) の設計理論をソースコードとともに解剖してきました。

- **Part 1**：Turn / Step / Inbox の 2 階層ループ、KV キャッシュを守る追記型プロンプト、Fail-Closed ツールパイプラインと異常系リカバリ
- **Part 2**：`Model-Visible ⟺ Logged` のイベントソーシング、Surface 層、同期 CQRS Projection、非破壊マイグレーションと Fork
- **Part 3**：Cordis の可逆エフェクト・`waterfall`、`dsh-scope` によるエージェント単位の隔離、Profiles / Bundles / Patch Layers 合成
- **Part 4**：Capability Seam によるローカル／SSH 実行世界の差し替え、Per-Call OS サンドボックス、Fail-Closed 人間承認
- **Part 5**：PTC Runtime によるコード実行型ツール合成、統一 Subagent Seam、共有タスク DAG の Agent Teams
- **Part 6**：`cordisInspect` / `dynamicCordisRunner` によるランタイム自己拡張、Typert 型グラフ RPC、ブラウザ側 Cordis によるマルチサーフェス展開

DeepSeek Harness が示しているのは、**「AI エージェントの複雑さは、巨大なモノリシックループではなく、可逆的なプラグイン合成・追記専用ログ・明示的な Capability Seam の組み合わせによって制御できる」** という明確な設計哲学です。独自のエージェント基盤を設計・拡張する際の最高のリファレンスとして、ぜひリポジトリのソースコードも直接覗いてみてください。
