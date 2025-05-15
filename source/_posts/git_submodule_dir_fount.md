---
title: "git: A git directory for 'xxxx' is found locally with remote(s)"

date: 2023-1-7 12:00:00
categories: [AI]
tags: [git]
lang: ja
---

# 問題

```bash
❯ git submodule add https://github.com/volantis-x/hexo-theme-volantis .\themes\volantis
fatal: A git directory for 'themes/volantis' is found locally with remote(s):
  origin        https://github.com/volantis-x/hexo-theme-volantis
If you want to reuse this local git directory instead of cloning again from
  https://github.com/volantis-x/hexo-theme-volantis
use the '--force' option. If the local git directory is not the correct repo
or you are unsure what this means choose another name with the '--name' option.
```

<!--more-->

# 解決策

- `.git/module`に対応のフォルダを削除する

```bash
❯ rm -r .\.git\modules\themes\volantis\
```

# 原因

前回の`git submodule add`で追加しでも、ダウンロードは完全しなく、途中で中断した可能性がある。こうして、`.git/modules`に`themes/volantis`フォルダがキャッシュとして残っても、目標のフォルダ`themes/volantis`が作成されなかった。
