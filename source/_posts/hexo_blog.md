---
title: Hexoでブログを構築する方法
date: 2022-7-6 17:36:00
categories: [blog]
tags: [hexo, blog, server]
lang: ja
---

GitHub Pagesへの再デプロイを機に、Hexoをブログフレームワークとして採用しました。  
HexoはNode.jsベースの高速でシンプルな静的ブログ生成ツールです。詳細な使い方を解説します。

---

- [**1. インストール方法**](#1-インストール方法)
  - [前提条件](#前提条件)
- [**2. プロジェクト初期化**](#2-プロジェクト初期化)
  - [新規ブログ作成](#新規ブログ作成)
  - [フォルダ構造](#フォルダ構造)
- [**3. 基本設定**](#3-基本設定)
- [**4. Hexo 基本コマンド一覧**](#4-hexo-基本コマンド一覧)
  - [1. プロジェクト管理](#1-プロジェクト管理)
  - [2. 記事管理](#2-記事管理)
  - [3. ローカルプレビュー](#3-ローカルプレビュー)
  - [4. デプロイ](#4-デプロイ)
  - [5. ヘルプ](#5-ヘルプ)
  - [6. 基本ワークフロー](#6-基本ワークフロー)
- [**5. 重要な設定ファイル**](#5-重要な設定ファイル)
- [**6. 便利な小技**](#6-便利な小技)
- [**7. 高度なカスタマイズ** 🔧](#7-高度なカスタマイズ-)
  - [プラグイン活用例](#プラグイン活用例)
  - [テーマ改造テクニック](#テーマ改造テクニック)
- [**8. SEO最適化 🔍**](#8-seo最適化-)
  - [基本設定](#基本設定)
  - [メタタグ強化](#メタタグ強化)
- [**9. 自動デプロイ設定 🤖**](#9-自動デプロイ設定-)
  - [**cloudflareのワーカー機能を使った推奨デプロイメント**](#cloudflareのワーカー機能を使った推奨デプロイメント)
  - [GitHub Actions例](#github-actions例)
- [**10. トラブルシューティング 🚨**](#10-トラブルシューティング-)
- [公式ドキュメント](#公式ドキュメント)


---

## **1. インストール方法**
### 前提条件
- **Node.js**（v12.0.0以上）と**Git**をインストール
- Hexo CLIのグローバルインストール:
  ```bash
  npm install -g hexo-cli
  ```


## **2. プロジェクト初期化**
### 新規ブログ作成
```bash
hexo init <フォルダ名>
cd <フォルダ名>
npm install
```

### フォルダ構造
```
.
├── _config.yml       # メイン設定ファイル
├── package.json      # 依存関係
├── scaffolds         # 投稿テンプレート
├── source            # Markdownコンテンツ
├── themes            # テーマフォルダ
└── public            # 生成ファイル（自動生成）
```

## **3. 基本設定**
`_config.yml`の主要設定（日本語訳例）:
```yaml
# サイト設定
title: マイブログ
subtitle: Hexoで作るブログ
description: 個人技術ブログです
keywords: Hexo, ブログ, チュートリアル
author: あなたの名前

# パーマリンク設定
permalink: :year/:month/:day/:title/

# テーマ設定
theme: landscape

# デプロイ設定
deploy:
  type: git
  repo: <リポジトリURL>
  branch: main
```

## **4. Hexo 基本コマンド一覧**


### 1. プロジェクト管理
| コマンド                 | 説明                     | 例                               |
| ------------------------ | ------------------------ | -------------------------------- |
| `hexo init <フォルダ名>` | 新しいプロジェクトを作成 | `hexo init my-blog`              |
| `npm install`            | 依存関係をインストール   | （プロジェクトフォルダ内で実行） |

### 2. 記事管理
| コマンド                   | 説明                      |
| -------------------------- | ------------------------- |
| `hexo new "タイトル"`      | 新しい記事を作成          |
| `hexo new page "ページ名"` | タグ/カテゴリページを作成 |
| `hexo generate` / `hexo g` | 静的ファイルを生成        |
| `hexo clean`               | キャッシュを削除          |

### 3. ローカルプレビュー
| コマンド                 | 説明                           |
| ------------------------ | ------------------------------ |
| `hexo server` / `hexo s` | ローカルサーバー起動           |
| `hexo server -p 5000`    | ポート指定（例: 5000番ポート） |

### 4. デプロイ
| コマンド                 | 説明                     |
| ------------------------ | ------------------------ |
| `hexo deploy` / `hexo d` | サイトをデプロイ         |
| `hexo g -d`              | 生成＋デプロイを同時実行 |

### 5. ヘルプ
| コマンド       | 説明                   |
| -------------- | ---------------------- |
| `hexo help`    | ヘルプを表示           |
| `hexo version` | Hexoのバージョンを確認 |


### 6. 基本ワークフロー
1. 記事作成
```bash
hexo new "はじめての投稿"
```

2. ローカル確認
```bash
hexo clean && hexo g && hexo s
```

3. デプロイ
```bash
hexo clean && hexo deploy --generate
```


## **5. 重要な設定ファイル**
- `_config.yml`：メイン設定ファイル
- `themes/[テーマ名]/_config.yml`：テーマ設定ファイル


## **6. 便利な小技**
- 下書き機能：`hexo new draft "下書きタイトル"`
- デバッグモード：`hexo generate --debug`
- 特定ファイルのみ生成：`hexo g --watch`

以下是为您的Hexo日语博客添加的完善建议，使用Markdown格式呈现：

---

## **7. 高度なカスタマイズ** 🔧
### プラグイン活用例
```yaml
# _config.yml に追加
plugins:
  - hexo-generator-search  # 検索機能
  - hexo-related-posts  # 関連記事表示
  - hexo-autonofollow  # 外部リンク対策
```

### テーマ改造テクニック
1. ナビゲーションメニュー追加：
```html
<!-- themes/landscape/_partial/header.ejs -->
<nav>
  <a href="<%- url_for('/about') %>">自己紹介</a>
  <a href="<%- url_for('/projects') %>">プロジェクト</a>
</nav>
```

2. カスタムCSS追加：
```bash
mkdir -p source/css
echo '.custom-class { color: #ff0000; }' > source/css/custom.css
```

---

## **8. SEO最適化 🔍**
### 基本設定
```yaml
# _config.yml
sitemap:
  path: sitemap.xml
google_analytics: UA-XXXXX-X
```

### メタタグ強化
```html
<!-- themes/landscape/layout/_partial/head.ejs -->
<meta name="keywords" content="<%= config.keywords %>">
<meta property="og:image" content="<%- url_for('/images/ogp.png') %>">
```


## **9. 自動デプロイ設定 🤖**
### [**cloudflareのワーカー機能を使った推奨デプロイメント**](https://www.cloudflare.com/ja-jp/developer-platform/products/workers/)
### GitHub Actions例
```yaml
# .github/workflows/deploy.yml
name: Deploy
on: [push]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v2

    - name: Setup Node
      uses: actions/setup-node@v2
      with:
        node-version: '16'

    - name: Install Dependencies
      run: |
        npm install -g hexo-cli
        npm install

    - name: Deploy
      run: |
        hexo clean
        hexo deploy --generate
```


## **10. トラブルシューティング 🚨**
| 現象               | 解決方法                              |
| ------------------ | ------------------------------------- |
| デプロイ失敗       | `hexo clean` を実行後再試行           |
| 画像が表示されない | パスを`/images/example.jpg`形式で記述 |
| スタイルが崩れる   | `hexo g` 実行後にハードリフレッシュ   |


## [公式ドキュメント](https://hexo.io/ja/docs/)