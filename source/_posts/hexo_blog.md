---
title: Hexoでブログを構築する方法
date: 2022-7-6 17:36:00
categories: [blog]
tags: [hexo, blog, server]
lang: ja  # 言語指定
---

GitHub Pagesへの再デプロイを機に、Hexoをブログフレームワークとして採用しました。  
HexoはNode.jsベースの高速でシンプルな静的ブログ生成ツールです。詳細な使い方を解説します。

---

## **1. インストール方法**
### 前提条件
- **Node.js**（v12.0.0以上）と**Git**をインストール
- Hexo CLIのグローバルインストール:
  ```bash
  npm install -g hexo-cli
  ```

---

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

---

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
