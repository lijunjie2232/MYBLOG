---
title: Hexoでカテゴリとタグの追加
date: 2022-7-6 21:40:00
categories: [blog]
tags: [hexo, blog, server]
lang: ja

description: Hexoでカテゴリとタグの追加方法を解説します。カテゴリページ、タグページの生成から記事へのcategories属性・tags属性の追加まで網羅しています。
---

## 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [カテゴリの作成](#%E3%82%AB%E3%83%86%E3%82%B4%E3%83%AA%E3%81%AE%E4%BD%9C%E6%88%90)
  - [カテゴリページの生成と type 属性の追加](#%E3%82%AB%E3%83%86%E3%82%B4%E3%83%AA%E3%83%9A%E3%83%BC%E3%82%B8%E3%81%AE%E7%94%9F%E6%88%90%E3%81%A8-type-%E5%B1%9E%E6%80%A7%E3%81%AE%E8%BF%BD%E5%8A%A0)
  - [categories 属性を追加](#categories-%E5%B1%9E%E6%80%A7%E3%82%92%E8%BF%BD%E5%8A%A0)
- [タグの作成](#%E3%82%BF%E3%82%B0%E3%81%AE%E4%BD%9C%E6%88%90)
  - [タグページの生成と type 属性の追加](#%E3%82%BF%E3%82%B0%E3%83%9A%E3%83%BC%E3%82%B8%E3%81%AE%E7%94%9F%E6%88%90%E3%81%A8-type-%E5%B1%9E%E6%80%A7%E3%81%AE%E8%BF%BD%E5%8A%A0)
  - [tags 属性を追加](#tags-%E5%B1%9E%E6%80%A7%E3%82%92%E8%BF%BD%E5%8A%A0)

---

## カテゴリの作成

### カテゴリページの生成と type 属性の追加

コマンドラインを開き、ブログフォルダに移動して次のコマンドを実行:

```bash
$ hexo new page categories
```

成功すると次のメッセージが表示されます:

```bash
INFO  Created: ...(略)/source/categories/index.md
```

表示されたパスから index.md ファイルを開き、デフォルトの内容に type: "categories"を追加:

```markdown
---
title: 記事カテゴリ
date: 2018-10-31 13:47:40
type: "categories"
---
```

### categories 属性を追加

対象の記事ファイルを開き、categories 属性を追加。下記の例では記事を「web フロントエンド」カテゴリに分類。注意: 1 記事につき 1 カテゴリのみ指定可能。複数指定した場合、ネスト構造として扱われます

```markdown
---
title: Hexoブログ+Githubブログチュートリアル：03カテゴリ・タグの追加
date: 2018-11-01 14:17:46
categories:
  - hexo
---
```

## タグの作成

### タグページの生成と type 属性の追加

次のコマンドを実行:

```bash
$ hexo new page tags
```

生成された index.md に type: "tags"を追加:

```markdown
---
title: タグ
date: 2018-10-31 13:47:40
type: "tags"
---
```

### tags 属性を追加

タグを複数設定可能（例では hexo, github, ブログの 3 タグ）:

```markdown
---
title: Hexoブログ+Githubブログチュートリアル：03カテゴリ・タグの追加
date: 2018-11-01 14:17:46
categories:
  - 基本知識
  - hexo
tags:
  - hexo
  - github
  - ブログ
---
```

補足: scaffolds/post.md テンプレートを編集すると、新規記事作成時にデフォルトで categories/tags 項目が追加されるようになります。
