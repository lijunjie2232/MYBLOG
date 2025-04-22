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

---

### 3. **Language File Configuration**
Create `themes/landscape/languages/ja.yml` (if using landscape theme):
```yaml
index:
  recent_posts: 最新記事
menu:
  home: ホーム
  archive: アーカイブ
  about: このサイトについて
```

---

### 4. **Multi-language Workflow**
For bilingual support (Japanese/English):
1. Install i18n plugin:
   ```bash
   npm install hexo-generator-i18n --save
   ```
2. Update `_config.yml`:
   ```yaml
   language: 
     - ja
     - en
   i18n:
     type: [page, post]
     generator: true
   ```

---

### 5. **Deployment Notes**
When deploying to GitHub Pages:
- Set branch to `gh-pages`
- Add CNAME file (if using custom domain):
  ```bash
  echo "yourdomain.com" > source/CNAME
  ```

---

### 6. **Translation Tips**
- Use DeepL/Google Translate API for batch content translation
- Maintain consistency in technical terms (e.g., "permalink" →「パーマリンク」)
- Add ruby annotations for complex terms:  
  `Node.js（ノードジェイエス）`

---

**Complete Japanese Version Example**:  
[View Full Translated File](https://gist.github.com/sample/hexo_ja_translation) (仮想サンプルリンク)

Let me know if you need help with specific sections or encounter translation inconsistencies! 🇯🇵