<div align="center">

# Hexo-Theme-ParticleXMod

[![Readme JP](https://img.shields.io/badge/README-JP-white)](./README.md)
[![Readme CN](https://img.shields.io/badge/README-CN-red)](./README_CN.md)
[![Readme EN](https://img.shields.io/badge/README-EN-blue)](./README_EN.md)

</div>

<br />

[ParticleXMod](https://github.com/lijunjie2232/particlexMod) は、 [ParticleX](https://github.com/theme-particlex/hexo-theme-particlex) から、編集したテーマです。

注：スタイルの調整は個人的な審美に基づいています。ブログの見た目に関する変更はすべて筆者の好みによるものであり、もしデザインに不満がある場合は自由にお好きなようにカスタマイズしてください。

- [Hexo-Theme-ParticleXMod](#hexo-theme-particlexmod)
- [ChangeLog \&\& TODO](#changelog--todo)
  - [1. デモ](#1-%E3%83%87%E3%83%A2)
  - [2. インストール](#2-%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
    - [2.1. 自動コードハイライトを無効にする](#21-%E8%87%AA%E5%8B%95%E3%82%B3%E3%83%BC%E3%83%89%E3%83%8F%E3%82%A4%E3%83%A9%E3%82%A4%E3%83%88%E3%82%92%E7%84%A1%E5%8A%B9%E3%81%AB%E3%81%99%E3%82%8B)
    - [2.2. 年別・月別のアーカイブを無効にする](#22-%E5%B9%B4%E5%88%A5%E3%83%BB%E6%9C%88%E5%88%A5%E3%81%AE%E3%82%A2%E3%83%BC%E3%82%AB%E3%82%A4%E3%83%96%E3%82%92%E7%84%A1%E5%8A%B9%E3%81%AB%E3%81%99%E3%82%8B)
  - [3. 設定](#3-%E8%A8%AD%E5%AE%9A)
    - [3.1. 基本設定](#31-%E5%9F%BA%E6%9C%AC%E8%A8%AD%E5%AE%9A)
    - [3.2. コンテンツ設定](#32-%E3%82%B3%E3%83%B3%E3%83%86%E3%83%B3%E3%83%84%E8%A8%AD%E5%AE%9A)
      - [3.2.1. ナビゲーションバー](#321-%E3%83%8A%E3%83%93%E3%82%B2%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%90%E3%83%BC)
      - [3.2.2. ホームページ情報カード](#322-%E3%83%9B%E3%83%BC%E3%83%A0%E3%83%9A%E3%83%BC%E3%82%B8%E6%83%85%E5%A0%B1%E3%82%AB%E3%83%BC%E3%83%89)
      - [3.2.3. フッター](#323-%E3%83%95%E3%83%83%E3%82%BF%E3%83%BC)
    - [3.3. 機能設定](#33-%E6%A9%9F%E8%83%BD%E8%A8%AD%E5%AE%9A)
      - [3.3.1. Polyfill](#331-polyfill)
      - [3.3.2. コードハイライト](#332-%E3%82%B3%E3%83%BC%E3%83%89%E3%83%8F%E3%82%A4%E3%83%A9%E3%82%A4%E3%83%88)
      - [3.3.3. 数式レンダリング](#333-%E6%95%B0%E5%BC%8F%E3%83%AC%E3%83%B3%E3%83%80%E3%83%AA%E3%83%B3%E3%82%B0)
      - [3.3.4. 画像プレビュー](#334-%E7%94%BB%E5%83%8F%E3%83%97%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC)
      - [3.3.5. 記事の要約表示](#335-%E8%A8%98%E4%BA%8B%E3%81%AE%E8%A6%81%E7%B4%84%E8%A1%A8%E7%A4%BA)
      - [3.3.6. 記事の固定表示](#336-%E8%A8%98%E4%BA%8B%E3%81%AE%E5%9B%BA%E5%AE%9A%E8%A1%A8%E7%A4%BA)
      - [3.3.7. 記事の暗号化](#337-%E8%A8%98%E4%BA%8B%E3%81%AE%E6%9A%97%E5%8F%B7%E5%8C%96)
      - [3.3.8. 検索機能](#338-%E6%A4%9C%E7%B4%A2%E6%A9%9F%E8%83%BD)
  - [3.4. コメント設定](#34-%E3%82%B3%E3%83%A1%E3%83%B3%E3%83%88%E8%A8%AD%E5%AE%9A)
    - [3.4.1. giscus](#341-giscus)
    - [3.4.2. Gitalk](#342-gitalk)
    - [3.4.3. Waline](#343-waline)
    - [3.4.4. Twikoo](#344-twikoo)
  - [4. 最後に](#4-%E6%9C%80%E5%BE%8C%E3%81%AB)
  - [5. サンクス](#5-%E3%82%B5%E3%83%B3%E3%82%AF%E3%82%B9)


# ChangeLog && TODO

- [x] (fix)数式機能の修正
- [x] ホームページの背景を固定（`fixed`）に設定
- [x] 目次機能を追加

---

## 1. デモ

-   [My Blog](https://blog.lijunjie.dpdns.org)
-   [GitHub Pages](https://argvchs.github.io)
-   [Netlify](https://argvchs.netlify.app)
-   [Vercel](https://argvchs.vercel.app)

## 2. インストール

```bash
cd themes
git clone https://github.com/lijunjie2232/particlexMod.git particlexmod --depth=1
```

その後、ルートディレクトリの `_config.yml` でテーマを ParticleX に設定してください。

```yaml
theme: particlexmod
```

### 2.1. 自動コードハイライトを無効にする

Hexo には組み込みのコードハイライト機能がありますが、ParticleX とは互換性がありません。

```yaml
highlight:
    enable: false
prismjs:
    enable: false
```

Hexo 7.0.0 以降を使用している場合、以下のように設定するだけで十分です。

```yaml
syntax_highlighter:
```

Pandoc を使用する場合、以下の設定も必要です。

```yaml
pandoc:
    extra:
        - no-highlight:
```

### 2.2. 年別・月別のアーカイブを無効にする

Hexo はデフォルトで年別・月別のアーカイブを生成しますが、ParticleX テーマにはこの機能はありません。~~めんどくさかったので実装しませんでした~~

```yaml
archive_generator:
    enabled: true
    per_page: 0
    yearly: false
    monthly: false
    daily: false
```

変更後は `hexo cl` を使ってキャッシュをクリアしてください。

## 3. 設定

### 3.1. 基本設定

`background` パラメータはリスト形式であり、ページを開いたときにランダムに背景画像が選ばれます。

```yaml
# アバター画像
avatar: /images/avatar.jpg

# ホームページの背景画像
background:
    - /images/background.jpg

# ローディング画像
loading: /images/loading.gif

# カテゴリとタグに使用されるオプションカラー
colors:
    - "#ffa2c4"
    - "#00bcd4"
    - "#03a9f4"
    - "#00a596"
    - "#ff7d73"
```

### 3.2. コンテンツ設定

#### 3.2.1. ナビゲーションバー

使いやすさのため、テーマアイコンには Font Awesome 6 を採用しています。

```yaml
# ParticleX テーマのアイコンは Font Awesome 6 を使用
# https://fontawesome.com

# メインメニュー
menu:
    Home:
        name: house
        theme: solid
        link: /
    About:
        name: id-card
        theme: solid
        link: /about
    Archives:
        name: box-archive
        theme: solid
        link: /archives
    Categories:
        name: bookmark
        theme: solid
        link: /categories
    Tags:
        name: tags
        theme: solid
        link: /tags
```

#### 3.2.2. ホームページ情報カード

`description` は Markdown 形式をサポートしています。

アイコンリンク `iconLinks` の設定はナビゲーションバーと同じ構造です。

```yaml
# サイド情報カード
card:
    enable: true
    description: |
        Description
        ...
    iconLinks:
    friendLinks:
        Argvchs: https://argvchs.github.io
```

#### 3.2.3. フッター

ブログをサーバーにホストし独自ドメインを使用する場合、法的に記載が必要なICP番号などを含む備考情報を表示する必要があります。

必要がない場合は非表示にできます。

```yaml
# フッター情報
footer:
    since: 2022
    # カスタムサーバードメインの ICP 情報
    ICP:
        enable: false
        code:
        link:
```

### 3.3. 機能設定

#### 3.3.1. Polyfill

[Polyfill.io](https://polyfill.io) を使用してブラウザごとに新しい JS API の互換性を自動で処理します。

[Hexo-Babel](https://github.com/theme-particlex/hexo-babel) プラグインと併用することで JS 文法の互換性も対応可能です。

Polyfill は中国の一部地域でブロックされているため、阿里の [Polyfill](https://polyfill.alicdn.com) を使用しています。

```yaml
# Polyfill
# https://polyfill.io
polyfill:
    enable: true
    features:
        - default
```

#### 3.3.2. コードハイライト

Highlight.js を使用してコードにハイライトを追加します。

スタイルは[こちら](https://highlightjs.org/static/demo)から選択可能で、デフォルトは GitHub スタイルです。

```yaml
# Highlight.js
# https://highlightjs.org
highlight:
    enable: true
    style: github
```

#### 3.3.3. 数式レンダリング

KaTeX を使用して数式を描画します。

```yaml
# KaTeX 数式描画
math:
    enable: false
```

#### 3.3.4. 画像プレビュー

単純なクリックでの画像拡大・縮小プレビュー機能です。

```yaml
# 画像プレビュー
preview:
    enable: true
```

#### 3.3.5. 記事の要約表示

通常、記事の要約表示は `<!-- more -->` を挿入すれば可能です。しかし、要約部分を本文内に含めたくない場合もあります。

その場合は Front-Matter に [description](file://d:\code\MYBLOG\themes\volantis\scripts\helpers\head\generate_title__keywords__description.js#L3-L3) を設定してください。Markdown 形式をサポートしています。

```yaml
description: |
    Normal _Italic_ **Strong**
```

#### 3.3.6. 記事の固定表示

Front-Matter に `pinned` を設定すると固定表示されます。値が大きいほど上部に表示されます。デフォルトは `0` です。

#### 3.3.7. 記事の暗号化

AES 暗号化アルゴリズムを使用します。Front-Matter に `secret` を設定してパスワードとして利用します。**使用するには [Hexo-Helper-Crypto](https://github.com/theme-particlex/hexo-helper-crypto) プラグインのインストールが必要です。**

```yaml
# 記事暗号化
crypto:
    enable: false
```

#### 3.3.8. 検索機能

アーカイブページに埋め込まれた検索機能です。

現状ではタイトルのみの検索に対応しています。

```yaml
# 検索
search:
    enable: false
```

## 3.4. コメント設定

### 3.4.1. giscus

giscus は GitHub Discussions をベースにしたコメントシステムです。

[giscus.app](https://giscus.app) で設定を完了すると `<script>` タグが生成されますので、テーマに貼り付けてください。

```yaml
# giscus
# https://github.com/giscus/giscus
giscus:
    enable: false
    src: https://giscus.app/client.js
    repo:
    repoID:
    category:
    categoryID:
    mapping: pathname
    strict: 0
    reactionsEnabled: 1
    emitMetadata: 0
    inputPosition: bottom
    theme: preferred_color_scheme
    lang:
```

### 3.4.2. Gitalk

Gitalk は GitHub Issue と Preact を基盤としたコメントシステムです。

Gitalk 公式の CORS プロキシは Cloudflare を使用しており遅いため、CORS プロキシの自前運用方法はこちらの[記事](https://argvchs.github.io/2022/07/04/build-cors-anywhere)をご覧ください。

```yaml
# Gitalk
# https://github.com/gitalk/gitalk
gitalk:
    enable: false
    clientID: # デフォルトClientID
    clientSecret: # デフォルトClientSecret
    repo: # コメントを保存するリポジトリ名
    owner: # GitHub のリポジトリ所有者
    admin: # 所有者およびコラボレーターのみ初期化可能
    language: # en, zh-CN, zh-TW, es-ES, fr, ru, de, pl, ko が使用可能
    proxy: # CORS プロキシ
```

### 3.4.3. Waline

Waline はシンプルで安全なコメントシステムです。

詳しくは：[ParticleX で Waline を使う | Yuzi's Blog](https://blog.yuzi.dev/posts/bcb4ff00.html)

```yaml
# Waline
# https://github.com/walinejs/waline
waline:
    enable: false
    serverURL: # Waline サーバーのURL
    locale: # 詳細: https://waline.js.org/guide/client/i18n.html#locale-option
    commentCount: true # false にするとホーム画面に表示されず、投稿ページのみ表示
    pageview: false # 閲覧数カウント（注意：waline.pageview と leancloud_visitors の両方を同時に有効にしないでください）
    emoji: # 絵文字
        - https://unpkg.com/@waline/emojis@1.2.0/weibo
        - https://unpkg.com/@waline/emojis@1.2.0/alus
        - https://unpkg.com/@waline/emojis@1.2.0/bilibili
        - https://unpkg.com/@waline/emojis@1.2.0/qq
        - https://unpkg.com/@waline/emojis@1.2.0/tieba
        - https://unpkg.com/@waline/emojis@1.2.0/tw-emoji
    meta: # コメント情報 (nick, mail, link)
        - nick
        - mail
        - link
    requiredMeta: # 必須項目 [nick] or [nick, mail]
        - nick
    lang: # 言語設定: en-US, zh-CN, zh-TW, pt-BR, ru-RU, jp-JP
    wordLimit: 0 # 文字制限（0で無制限）
    login: enable # ログイン機能 'enable', 'disable', 'force'
    pageSize: 10 # 1ページあたりのコメント数
```

### 3.4.4. Twikoo

Twikoo はシンプルで安全、無料の静的サイトコメントシステムです。

```yaml
# Twikoo
# https://github.com/imaegoo/twikoo
twikoo:
    enable: false
    envID:
    region:
    path: location.pathname
    lang:
```

## 4. 最後に

このプロジェクトは MIT ライセンスに基づいています。自由に貢献を受け付けています。Issue を開いて質問したり、Pull Request を送って改善案を提案することも可能です！お待ちしています！

## 5. サンクス

以下のプロジェクトに感謝します：
- [Hexo](https://hexo.io/)
- [Hexo-Theme-Butterfly](https://github.com/theme-particlex/hexo-theme-particlex)
- [Hexo-Theme-redefine](https://github.com/EvanNotFound/hexo-theme-redefine)