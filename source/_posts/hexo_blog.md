---
title: Hexoでブログを構築する方法
date: 2023-7-6 17:36:00
categories: [blog]
tags: [hexo, blog, server]
lang: ja

description: 
  GitHub Pagesへの再デプロイを機に、Hexoをブログフレームワークとして採用しました。
  HexoはNode.jsベースの高速でシンプルな静的ブログ生成ツールです。詳細な使い方を解説します。
---

GitHub Pagesへの再デプロイを機に、Hexoをブログフレームワークとして採用しました。  
HexoはNode.jsベースの高速でシンプルな静的ブログ生成ツールです。詳細な使い方を解説します。

---
## 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [**インストール方法**](#%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E6%96%B9%E6%B3%95)
  - [前提条件](#%E5%89%8D%E6%8F%90%E6%9D%A1%E4%BB%B6)
- [**プロジェクト初期化**](#%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E5%88%9D%E6%9C%9F%E5%8C%96)
  - [新規ブログ作成](#%E6%96%B0%E8%A6%8F%E3%83%96%E3%83%AD%E3%82%B0%E4%BD%9C%E6%88%90)
  - [フォルダ構造](#%E3%83%95%E3%82%A9%E3%83%AB%E3%83%80%E6%A7%8B%E9%80%A0)
- [**基本設定**](#%E5%9F%BA%E6%9C%AC%E8%A8%AD%E5%AE%9A)
- [**Hexo基本コマンド一覧**](#hexo%E5%9F%BA%E6%9C%AC%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E4%B8%80%E8%A6%A7)
  - [プロジェクト管理](#%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E7%AE%A1%E7%90%86)
  - [記事管理](#%E8%A8%98%E4%BA%8B%E7%AE%A1%E7%90%86)
  - [ローカルプレビュー](#%E3%83%AD%E3%83%BC%E3%82%AB%E3%83%AB%E3%83%97%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC)
  - [デプロイ](#%E3%83%87%E3%83%97%E3%83%AD%E3%82%A4)
  - [ヘルプ](#%E3%83%98%E3%83%AB%E3%83%97)
  - [基本ワークフロー](#%E5%9F%BA%E6%9C%AC%E3%83%AF%E3%83%BC%E3%82%AF%E3%83%95%E3%83%AD%E3%83%BC)
- [**重要な設定ファイル**](#%E9%87%8D%E8%A6%81%E3%81%AA%E8%A8%AD%E5%AE%9A%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB)
- [**便利な小技**](#%E4%BE%BF%E5%88%A9%E3%81%AA%E5%B0%8F%E6%8A%80)
- [**高度なカスタマイズ**🔧](#%E9%AB%98%E5%BA%A6%E3%81%AA%E3%82%AB%E3%82%B9%E3%82%BF%E3%83%9E%E3%82%A4%E3%82%BA%F0%9F%94%A7)
  - [プラグイン活用例](#%E3%83%97%E3%83%A9%E3%82%B0%E3%82%A4%E3%83%B3%E6%B4%BB%E7%94%A8%E4%BE%8B)
  - [テーマ改造テクニック](#%E3%83%86%E3%83%BC%E3%83%9E%E6%94%B9%E9%80%A0%E3%83%86%E3%82%AF%E3%83%8B%E3%83%83%E3%82%AF)
- [**SEO最適化🔍**](#seo%E6%9C%80%E9%81%A9%E5%8C%96%F0%9F%94%8D)
  - [基本設定](#%E5%9F%BA%E6%9C%AC%E8%A8%AD%E5%AE%9A-1)
  - [メタタグ強化](#%E3%83%A1%E3%82%BF%E3%82%BF%E3%82%B0%E5%BC%B7%E5%8C%96)
- [**自動デプロイ設定🤖**](#%E8%87%AA%E5%8B%95%E3%83%87%E3%83%97%E3%83%AD%E3%82%A4%E8%A8%AD%E5%AE%9A%F0%9F%A4%96)
  - [**Cloudflare Pagesを使った自動デプロイ**](#cloudflare-pages%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E8%87%AA%E5%8B%95%E3%83%87%E3%83%97%E3%83%AD%E3%82%A4)
  - [**既存設定との統合例**](#%E6%97%A2%E5%AD%98%E8%A8%AD%E5%AE%9A%E3%81%A8%E3%81%AE%E7%B5%B1%E5%90%88%E4%BE%8B)
  - [**主なメリット**](#%E4%B8%BB%E3%81%AA%E3%83%A1%E3%83%AA%E3%83%83%E3%83%88)
  - [公式ガイド](#%E5%85%AC%E5%BC%8F%E3%82%AC%E3%82%A4%E3%83%89)
  - [GitHub Actions例](#github-actions%E4%BE%8B)
- [**トラブルシューティング🚨**](#%E3%83%88%E3%83%A9%E3%83%96%E3%83%AB%E3%82%B7%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0%F0%9F%9A%A8)
- [公式ドキュメント](#%E5%85%AC%E5%BC%8F%E3%83%89%E3%82%AD%E3%83%A5%E3%83%A1%E3%83%B3%E3%83%88)


---

## **インストール方法**
### 前提条件
- **Node.js**（v12.0.0以上）と**Git**をインストール
- Hexo CLIのグローバルインストール:
  ```bash
  npm install -g hexo-cli
  ```


## **プロジェクト初期化**
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

## **基本設定**
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

## **Hexo基本コマンド一覧**


### プロジェクト管理
| コマンド                 | 説明                     | 例                               |
| ------------------------ | ------------------------ | -------------------------------- |
| `hexo init <フォルダ名>` | 新しいプロジェクトを作成 | `hexo init my-blog`              |
| `npm install`            | 依存関係をインストール   | （プロジェクトフォルダ内で実行） |

### 記事管理
| コマンド                   | 説明                      |
| -------------------------- | ------------------------- |
| `hexo new "タイトル"`      | 新しい記事を作成          |
| `hexo new page "ページ名"` | タグ/カテゴリページを作成 |
| `hexo generate` / `hexo g` | 静的ファイルを生成        |
| `hexo clean`               | キャッシュを削除          |

### ローカルプレビュー
| コマンド                 | 説明                           |
| ------------------------ | ------------------------------ |
| `hexo server` / `hexo s` | ローカルサーバー起動           |
| `hexo server -p 5000`    | ポート指定（例: 5000番ポート） |

### デプロイ
| コマンド                 | 説明                     |
| ------------------------ | ------------------------ |
| `hexo deploy` / `hexo d` | サイトをデプロイ         |
| `hexo g -d`              | 生成＋デプロイを同時実行 |

### ヘルプ
| コマンド       | 説明                   |
| -------------- | ---------------------- |
| `hexo help`    | ヘルプを表示           |
| `hexo version` | Hexoのバージョンを確認 |


### 基本ワークフロー
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


## **重要な設定ファイル**
- `_config.yml`：メイン設定ファイル
- `themes/[テーマ名]/_config.yml`：テーマ設定ファイル


## **便利な小技**
- 下書き機能：`hexo new draft "下書きタイトル"`
- デバッグモード：`hexo generate --debug`
- 特定ファイルのみ生成：`hexo g --watch`

以下是为您的Hexo日语博客添加的完善建议，使用Markdown格式呈现：

---

## **高度なカスタマイズ**🔧
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

## **SEO最適化🔍**
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


## **自動デプロイ設定🤖**

### **Cloudflare Pagesを使った自動デプロイ**
1. **プロジェクト作成手順**
   - Cloudflareダッシュボード → [Pages]を選択
   - [プロジェクトを作成] ボタンをクリック
   - GitHubアカウントを接続（全リポジトリor特定リポジトリを選択）

2. **ビルド設定**
   ```yaml
   ビルドコマンド: npx hexo generate
   公開ディレクトリ: public
   フレームワークプリセット: None
   ```

3. **デプロイ完了**
   - ビルド成功後、自動で`*.pages.dev`ドメインが割り当てられます
   - カスタムドメインの設定も可能

### **既存設定との統合例**
```yaml
# _config.yml
url: https://your-domain.pages.dev
deploy:
  type: git
  repo: git@github.com:yourname/yourrepo.git
```
### **主なメリット**
✅ GitHub連動での自動ビルド  
✅ 無料SSL証明書自動発行  
✅ グローバルCDN配信  
✅ プレビュー機能付きPRデプロイ

### [公式ガイド](https://developers.cloudflare.com/pages/)

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
        node-version: '22'

    - name: Install Dependencies
      run: |
        npm install -g hexo-cli
        npm install

    - name: Deploy
      run: |
        hexo clean
        hexo deploy --generate
```


## **トラブルシューティング🚨**
| 現象               | 解決方法                              |
| ------------------ | ------------------------------------- |
| デプロイ失敗       | `hexo clean` を実行後再試行           |
| 画像が表示されない | パスを`/images/example.jpg`形式で記述 |
| スタイルが崩れる   | `hexo g` 実行後にハードリフレッシュ   |


## [公式ドキュメント](https://hexo.io/ja/docs/)