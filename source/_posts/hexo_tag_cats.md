---
title: Hexoでカテゴリとタグの追加
date: 2022-7-6 21:40:00
categories: [blog]
tags: [hexo, blog, server]
lang: ja
---

- [1. カテゴリの作成](#1-カテゴリの作成)
  - [1.1 カテゴリページの生成とtype属性の追加](#11-カテゴリページの生成とtype属性の追加)
  - [1.2 categories属性を追加](#12-categories属性を追加)
- [2. タグの作成](#2-タグの作成)
  - [2.1 タグページの生成とtype属性の追加](#21-タグページの生成とtype属性の追加)
  - [2.2 tags属性を追加](#22-tags属性を追加)


## 1. カテゴリの作成
### 1.1 カテゴリページの生成とtype属性の追加
コマンドラインを開き、ブログフォルダに移動して次のコマンドを実行:

```bash
$ hexo new page categories
```

成功すると次のメッセージが表示されます:
```bash
INFO  Created: ...(略)/source/categories/index.md
```

表示されたパスからindex.mdファイルを開き、デフォルトの内容にtype: "categories"を追加:

```markdown
---
title: 記事カテゴリ
date: 2018-10-31 13:47:40
type: "categories"
---
```

### 1.2 categories属性を追加
対象の記事ファイルを開き、categories属性を追加。下記の例では記事を「webフロントエンド」カテゴリに分類。注意: 1記事につき1カテゴリのみ指定可能。複数指定した場合、ネスト構造として扱われます

```markdown
---
title: Hexoブログ+Githubブログチュートリアル：03カテゴリ・タグの追加
date: 2018-11-01 14:17:46
categories: 
- hexo
---
```

## 2. タグの作成
### 2.1 タグページの生成とtype属性の追加
次のコマンドを実行:

```bash
$ hexo new page tags
```

生成されたindex.mdにtype: "tags"を追加:

```markdown
---
title: タグ
date: 2018-10-31 13:47:40
type: "tags"
---
```

### 2.2 tags属性を追加
タグを複数設定可能（例ではhexo, github, ブログの3タグ）:

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

補足: scaffolds/post.mdテンプレートを編集すると、新規記事作成時にデフォルトでcategories/tags項目が追加されるようになります。
