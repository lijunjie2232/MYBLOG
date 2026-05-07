---
title: DeepSeek Harness 技術解析 Part 3：「Everything is a Plugin」を支える Cordis・スコープ付きライフタイム・レイヤー合成

date: 2026-04-23 19:05:31
categories: [AI]
tags: [Deep Learning, TypeScript, Node.js, LLM, AI Agent, DeepSeek, Cordis, Plugin Architecture, 機械学習, AI, 人工知能, 生成AI]
lang: ja
description: DeepSeek Harness の根幹である「Everything is a Plugin」アーキテクチャを解剖します。Cordis の DI・可逆エフェクト・5 種のイベント配信モード、エージェント単位の隔離を実現する dsh-scope（ScopedLayers / scopeTarget）、そして Profiles・Bundles・Patch Layers による宣言的起動合成をソースコードと図解で解説します。
---

## はじめに

[Part 1](/2026/04/03/deepseek_harness_part1/) で LLM エージェントフロー、[Part 2](/2026/04/12/deepseek_harness_part2/) でイベントソース型セッションを解説しました。

しかし、ここである疑問が湧くはずです。**「エージェントループ自体も、ツールも、権限チェックも、コンテキスト圧縮もすべてプラグインなら、どうやって依存順序・隔離・ホットリロードを破綻させずに制御しているのか？」**

Part 3 では、DeepSeek Harness の土台となる **3 つの合成メカニズム** を解き明かします。

1. **Cordis**：DI・可逆エフェクト・型付きイベントバス
2. **`dsh-scope`**：エージェント単位のスコープ付き登録と自動クリーンアップ
3. **Profiles / Bundles / Patch Layers**：YAML パッチの積層によるアプリケーション起動合成

---

## 1. Cordis の 5 つの基本理論

DeepSeek Harness は、ベンダリングされた [Cordis](https://github.com/cordiverse/cordis)（`vendor/` 配下）の上で動作します（参照：[docs/cordis-primer.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.md)）。Cordis の思想はたった 5 点に集約されます。

1. **プラグインは `Service` または `apply(ctx)` 関数**
   - すべての機能はコンテキスト `ctx` にマウントされるプラグインとして記述します。
2. **コンテキスト `ctx` はサービスのリポジトリ**
   - サービスは `ctx.tools`、`ctx.llm`、`ctx.sessions` のように安定したキーを登録します。利用側は具象パッケージを `import` せず、`ctx.<key>` 越しにアクセスします。
3. **起動順序ではなく `inject` で依存を宣言**
   - プラグインが `inject = ['sessions', 'llm']` と宣言すると、Cordis はそれらのサービスが揃うまでマウントを自動的に待機します。手動のブート順序管理は不要です。
4. **宣言マージによる型付きイベントと 5 種の配信モード**
   - イベントは用途に応じた配信契約（`@mode`）を持ちます。
5. **すべての登録は可逆的エフェクト（`ctx.effect()` / `ctx.on()`）**
   - ツール登録、プロンプト追加、イベントリスナー、サブスクリプションは必ず disposer を返し、プラグインのアンロードや HMR 時に自動で巻き戻されます。

### 1.1 5 種類のイベント配信モードと `waterfall` の重要性

| モード | `await` | 実行順序 | 戻り値 | 主な用途 |
|---|---|---|---|---|
| `emit` | しない | 登録順 | なし | 状態通知（`agent/status`, `session/event`） |
| `waterfall` | **async 対応** | 登録順（around-middleware） | **あり** | 介入・書き換え・ポリシー判定（`agent/pre-step`, `tools/pre-execute`, `llm/stream`） |
| `parallel` | する | 全リスナー並列 | なし | 独立した非同期処理のファンアウト |
| `serial` | する | 登録順 | あり | 順序保証が必要な終了チェック（`agent/turn-stopping`, `agent/created`） |
| `bail` | しない | 最初の非 `undefined` で停止 | あり | 最初に応答したプロバイダの採用 |

特に DeepSeek Harness で決定的な役割を果たすのが **`waterfall`** です。リスナーは `(...args, next)` を受け取り、`next()` を呼べば次のリスナーへ委譲（ラップ）、`next()` を呼ばずに値を返せば **短絡（short-circuit）** します。これにより、ループ本体のコードを 1 行も変えずに、権限承認・サンドボックス・リトライ・コンテキスト圧縮を安全に挿し込めます。

---

## 2. エージェント単位の隔離：`dsh-scope`（Scoped Registration）

複数のエージェント（親エージェント＋複数のサブエージェント、あるいは異なるプリセットを持つセッション）が **同一プロセス・同一 Cordis コンテキストツリー** の中で同時に動くとき、次の課題が生じます。

- 「エージェント A 専用のツールやプロンプトを登録したいが、エージェント B には見せたくない」
- 「サブエージェントのイベントは、その親スコープのリスナーには届けたいが、無関係なエージェントには届けたくない」
- 「エージェントが終了（`dispose`）した瞬間に、そのエージェントに紐づく全ツール・リスナー・リソースを漏れなく解放したい」

これを解決するライブラリプリミティブが [`packages/core/scope`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/scope/src/index.ts) です（参照：[docs/subsystems/scope.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/scope.md)）。

### 2.1 双方向のスコープ階層（下向き継承・上向きイベント伝播）

`dsh-scope` は `WeakMap<ScopeKey, ScopeKey>`（`scopeParents`）で親子関係を管理し、**1 つの親子リンクで 2 方向のセマンティクス** を実現しています（[`packages/core/scope/src/index.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/scope/src/index.ts#L32-L59)）。

{% mermaid %}
graph TB
    Global["グローバル層<br/>全エージェント共通のツール・プロンプト・リスナー"]
    Parent["親エージェント Scope<br/>key = ParentAgent<br/>親専用ツール / 親リスナー"]
    Child["子エージェント Scope<br/>key = ChildAgent<br/>子専用ツール / 子リスナー"]

    Global -->|"下向き継承: 登録ビューの可視性 (ScopedLayers)"| Parent
    Parent -->|"下向き継承: 親＋グローバルのツールが見える"| Child

    Child -.->|"上向き伝播: イベント到達 (scopeTarget)"| Parent
    Parent -.->|"上向き伝播: グローバルリスナーにも到達"| Global

    style Global fill:#e1f5ff
    style Parent fill:#fff9c4
    style Child fill:#e8f5e9
{% endmermaid %}

実際のコードでは、循環参照を防ぎつつ親リンクを張る `bindScopeParent()` と、Cordis の子 Fiber を 1 つ発行して `kScope` シンボルを刻印する `createScope()` が中核です（[`packages/core/scope/src/index.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/scope/src/index.ts#L130-L140)）。

```ts
// packages/core/scope/src/index.ts (L137-L140 抜粋)
// https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/scope/src/index.ts#L137-L140

export function createScope(ctx: Context, key: ScopeKey, options?: CreateScopeOptions): Scope {
  // 1. 親スコープ（例：親エージェント）が指定されていれば循環参照チェック付きで親子リンクを登録
  //    この 1 本のリンクが「下向きのツール可視性継承」と「上向きのイベント到達」の両方を駆動する
  if (options?.parent !== undefined) bindScopeParent(key, options.parent)

  // 2. 空のダミープラグイン(scope)をマウントして専用の Cordis Fiber を 1 つ発行する
  //    以後このスコープ経由で行われた全登録は、この Fiber が寿命（disposer）を一括所有する
  const fiber = ctx.plugin(scope)

  // 3. Fiber のコンテキストに不透明タグ [kScope]: key を刻印した派生コンテキストを生成
  //    各レジストリ(ScopedLayers)はこのタグを見て「どのエージェント専用レイヤーか」を自動判別する
  const scoped: Context = fiber.ctx.extend({ [kScope]: key })

  // 4. rawDispose / dispose() は fiber.dispose() と非同期 inertia 完了待ちに紐づけて返却
```

- **可視性と寿命の統一**：プラグインが `agent.ctx` 経由で `ctx.tools.register(...)` を呼ぶと、その `ctx` には `[kScope]: agent` が刻まれているため、`ScopedLayers`（[`packages/core/scope/src/store.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/scope/src/store.ts)）はその登録を自動的に「このエージェント専用レイヤー」へ格納します。同時に、登録エフェクトの所有者は `scope` の Cordis Fiber になるため、`agent.scope.dispose()` を呼ぶだけで全登録が一括アンワインドされ、空になった `ScopeLayer` は即座に GC 回収されます。
- **安全なイベントルーティング（`Scoped<T>`）**：`scopeTarget(base, key)` は主体プロパティを露出しない不透明キャリアオブジェクトを生成し、コンパイル時ブランド `Scoped<T>` によって「スコープ対象外への誤配信」を型レベルで防ぎます。

---

## 3. Profiles・Bundles・Patch Layers による宣言的起動合成

コードレベルでプラグインが疎結合になっていても、製品として起動するには「どのプラグインをどんな設定で組み立てるか」を決める必要があります。DeepSeek Harness では、**すべてのアプリ起動が `dsh --profile <name>` に統一** され、空のエントリリストに対して YAML パッチを順番に重ねる（Layer Composition）ことでプラグインツリーを構築します（参照：[docs/architecture.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md#profiles-and-bundles) / [`packages/boot/app-boot/src/profile.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/boot/app-boot/src/profile.ts)）。

### 3.1 レイヤー積層の優先順位

{% mermaid %}
graph TB
    Empty["空のプラグインエントリリスト []"] --> Base["Layer 1: ベースバンドル<br/>例: @deepseek-ai/dsh-base<br/>モデル・ツール・永続化・サンドボックス"]
    Base --> AppBundle["Layer 2: サーフェスバンドル<br/>dsh-web-app / dsh-headless / dsh-sdk-app / dsh-acp-app"]
    AppBundle --> ProfilePatch["Layer 3: プロファイル固有パッチ<br/>$DSH_HOME/profiles/&lt;name&gt;/cordis.patch.yml"]
    ProfilePatch --> HomePatch["Layer 4: ホーム共通パッチ<br/>$DSH_HOME/cordis.patch.yml"]
    HomePatch --> CliOverlay["Layer 5: 起動時オーバーレイ<br/>--patch foo.overlay.yml / フラグ由来パッチ"]
    CliOverlay --> Tree["最終的な Cordis プラグインツリー<br/>dsh --profile web --dump-config で確認可能"]

    style Empty fill:#f5f5f5
    style Base fill:#e1f5ff
    style AppBundle fill:#fff9c4
    style ProfilePatch fill:#e8f5e9
    style Tree fill:#c8e6c9
{% endmermaid %}

- **Bundle（バンドル）**：`package.json` の `dsh.bundle.patch` で YAML パッチファイル群を宣言する npm パッケージ。
- **Profile（プロファイル）**：`$DSH_HOME/profiles/<name>/` に置かれ、`package.json` の `dsh.profile.bundles` で積層するバンドル順序を宣言し、ユーザー自身の `cordis.patch.yml` と追加インストールした外部プラグインを保持します。
- **パッチ操作**：既存行を `id` で指定して設定まるごと置換、あるいは新規行を挿入できます。`preset` パッチを使えば、特定エージェントプリセットの `config.plugins` 内部も同様に上書き可能です。
- **設定の式展開と HMR**：`@deepseek-ai/cordis-plugin-include` が YAML 内の `!!js` 式を評価し、`dsh-hmr` が `cordis.patch.yml` の変更を検知して安全にモジュール・設定をリロードします（Headless / SDK / ACP では無効化）。

### 3.2 2 アンカー解決（Two-Anchor Resolution）

[`packages/boot/app-boot/src/profile.ts`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/boot/app-boot/src/profile.ts#L16-L20) のモジュールコメントにある通り、モジュール解決は **2 アンカー方式** です。

1. まず `dsh` 本体のインストール先（ランチャー自身のパッケージ）から公式バンドルを解決
2. 次にプロファイルディレクトリ（`$DSH_HOME/profiles/<name>/node_modules`）からユーザーが `dsh plugin` で追加した外部プラグインを解決

これにより、CLI・Web・Electron Desktop・Python SDK ランタイムのすべてが「同一の起動パイプライン」を共有しつつ、プロファイルごとに独立したプラグイン構成とロックファイルを維持できます。

---

## まとめと次回予告

Part 3 では、「Everything is a Plugin」を絵空事で終わらせない 3 つの技術基盤を解説しました。

- **Cordis**：`inject` による宣言的依存解決、可逆エフェクト、`waterfall` ミドルウェアにより、特権的コアなしで拡張性を実現。
- **`dsh-scope`**：`ScopeKey` と `ScopedLayers` により、同一プロセス上の複数エージェント間で可視性継承・イベントルーティング・一括解放を完全自動化。
- **Profiles / Bundles / Patch Layers**：空リストへのパッチ積層という単一モデルで、Web・Desktop・CLI・SDK・ユーザーカスタマイズを統一。

次回の **Part 4** では、このプラグイン基盤の上で「実行世界（ローカル / SSH / リモート）」「OS サンドボックス（bwrap / Landlock / Seatbelt）」「人間承認と権限プリセット」をどう抽象化しているか——**Capability Seam（能力の継ぎ目）** の設計を掘り下げます。
