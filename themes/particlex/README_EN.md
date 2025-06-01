<div align="center">

# Hexo-Theme-ParticleXMod

[![Readme JP](https://img.shields.io/badge/README-JP-white)](./README.md)
[![Readme CN](https://img.shields.io/badge/README-CN-red)](./README_CN.md)
[![Readme EN](https://img.shields.io/badge/README-EN-blue)](./README_EN.md)

</div>

<br />

[ParticleXMod](https://github.com/lijunjie2232/particlexMod) is a modified version of [ParticleX](https://github.com/theme-particlex/hexo-theme-particlex).

Note: The style adjustments reflect my personal aesthetic preferences only. If you find the design unattractive, it's definitely a matter of taste — feel free to customize it as you like.

- [Hexo-Theme-ParticleXMod](#hexo-theme-particlexmod)
  - [ChangeLog \&\& TODO](#changelog--todo)
  - [1. Demo](#1-demo)
  - [2. Installation](#2-installation)
    - [2.1. Disable Built-in Code Highlighting](#21-disable-built-in-code-highlighting)
    - [2.2. Disable Yearly/Monthly Archives](#22-disable-yearlymonthly-archives)
  - [3. Configuration](#3-configuration)
    - [3.1. Basic Configuration](#31-basic-configuration)
    - [3.2. Content Configuration](#32-content-configuration)
      - [3.2.1. Navigation Bar](#321-navigation-bar)
      - [3.2.2. Homepage Info Card](#322-homepage-info-card)
      - [3.2.3. Footer](#323-footer)
    - [3.3. Feature Configuration](#33-feature-configuration)
      - [3.3.1. Polyfill](#331-polyfill)
      - [3.3.2. Code Highlighting](#332-code-highlighting)
      - [3.3.3. Math Rendering](#333-math-rendering)
      - [3.3.4. Image Preview](#334-image-preview)
      - [3.3.5. Article Excerpt](#335-article-excerpt)
      - [3.3.6. Pin Articles](#336-pin-articles)
      - [3.3.7. Article Encryption](#337-article-encryption)
      - [3.3.8. Search](#338-search)
    - [3.4. Comment Systems](#34-comment-systems)
      - [3.4.1. giscus](#341-giscus)
      - [3.4.2. Gitalk](#342-gitalk)
      - [3.4.3. Waline](#343-waline)
      - [3.4.4. Twikoo](#344-twikoo)
  - [4. Final Words](#4-final-words)
  - [5. Acknowledgments](#5-acknowledgments)



---

## ChangeLog && TODO

-   [x] Fix math rendering
-   [x] Set homepage background to `fixed`
-   [ ] Add table of contents

---

## 1. Demo

-   [My Blog](https://blog.lijunjie.dpdns.org)
-   [GitHub Pages](https://argvchs.github.io)
-   [Netlify](https://argvchs.netlify.app)
-   [Vercel](https://argvchs.vercel.app)

---

## 2. Installation

```bash
cd themes
git clone https://github.com/lijunjie2232/particlexMod.git particlexmod --depth=1
```

Then, in your root `_config.yml`, set the theme:

```yaml
theme: particlexmod
```

### 2.1. Disable Built-in Code Highlighting

Hexo has its own code highlighting system which conflicts with ParticleX's.

```yaml
highlight:
    enable: false
prismjs:
    enable: false
```

If using Hexo 7.0.0 or newer:

```yaml
syntax_highlighter:
```

For Pandoc support:

```yaml
pandoc:
    extra:
        - no-highlight:
```

### 2.2. Disable Yearly/Monthly Archives

Hexo automatically generates yearly/monthly archives, but this feature is not supported in the ParticleX theme.~~I was too lazy to implement it~~

```yaml
archive_generator:
    enabled: true
    per_page: 0
    yearly: false
    monthly: false
    daily: false
```

After modifying, run `hexo cl` to clear cache.

---

## 3. Configuration

### 3.1. Basic Configuration

The `background` parameter is a list; a random background will be loaded on each visit.

```yaml
# Avatar image
avatar: /images/avatar.jpg

# Home page background image
background:
    - /images/background.jpg

# Loading image
loading: /images/loading.gif

# Optional colors for category and tag
colors:
    - "#ffa2c4"
    - "#00bcd4"
    - "#03a9f4"
    - "#00a596"
    - "#ff7d73"
```

### 3.2. Content Configuration

#### 3.2.1. Navigation Bar

For convenience, the theme uses Font Awesome 6 icons.

```yaml
# ParticleX theme icon adopts Font Awesome 6
# https://fontawesome.com

# Main menu navigation
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

#### 3.2.2. Homepage Info Card

The [description](file://d:\code\MYBLOG\themes\volantis\scripts\helpers\head\generate_title__keywords__description.js#L3-L3) supports Markdown format.  
The configuration for `iconLinks` is the same as for the navigation bar.

```yaml
# Side info card
card:
    enable: true
    description: |
        Description
        ...
    iconLinks:
    friendLinks:
        Argvchs: https://argvchs.github.io
```

#### 3.2.3. Footer

If you're deploying your blog on a server with a custom domain, you may need to add an ICP备案 message at the bottom of the site.

You can disable it if not needed.

```yaml
# Footer info
footer:
    since: 2022
    # Customize the server domain name ICP
    ICP:
        enable: false
        code:
        link:
```

### 3.3. Feature Configuration

#### 3.3.1. Polyfill

Use [Polyfill.io](https://polyfill.io) to handle JS API compatibility based on user agent.

Works well with [Hexo-Babel](https://github.com/theme-particlex/hexo-babel) plugin for JS syntax compatibility.

Since Polyfill is blocked in some provinces in China, we use Alibaba's [Polyfill](https://polyfill.alicdn.com).

```yaml
# Polyfill
# https://polyfill.io
polyfill:
    enable: true
    features:
        - default
```

#### 3.3.2. Code Highlighting

Uses Highlight.js for code highlighting.

Styles can be selected [here](https://highlightjs.org/static/demo), default is GitHub.

```yaml
# Highlight.js
# https://highlightjs.org
highlight:
    enable: true
    style: github
```

#### 3.3.3. Math Rendering

Renders math formulas using KaTeX.

```yaml
# KaTeX math rendering
math:
    enable: false
```

#### 3.3.4. Image Preview

Simple click-to-preview functionality for images (zoom in/out).

```yaml
# Image preview
preview:
    enable: true
```

#### 3.3.5. Article Excerpt

Usually, you only need to insert `<!-- more -->` in a post to generate an excerpt. This content also appears in the full view.

However, if you want to define the excerpt separately from the body, use this parameter in [Front-Matter](https://hexo.io/docs/front-matter). Supports Markdown.

```yaml
description: |
    Normal _Italic_ **Strong**
```

#### 3.3.6. Pin Articles

Set `pinned` in [Front-Matter](https://hexo.io/docs/front-matter) to pin articles. Higher values appear earlier. Default is `0`.

#### 3.3.7. Article Encryption

Uses AES encryption. Set `secret` in [Front-Matter](https://hexo.io/docs/front-matter) as password.  
**Requires installing plugin [Hexo-Helper-Crypto](https://github.com/theme-particlex/hexo-helper-crypto)**.

```yaml
# Article encryption
crypto:
    enable: false
```

#### 3.3.8. Search

Embedded search within Archives.

Currently only supports searching by title.

```yaml
# Search
search:
    enable: false
```

### 3.4. Comment Systems

#### 3.4.1. giscus

A comment system powered by GitHub Discussions.

After configuring at [giscus.app](https://giscus.app), a `<script>` tag will be generated. You can paste it here.

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

#### 3.4.2. Gitalk

A comment system based on GitHub Issues and Preact.

Gitalk’s official CORS proxy uses Cloudflare, which is slow in China. To build your own CORS proxy, see [this article](https://argvchs.github.io/2022/07/04/build-cors-anywhere).

```yaml
# Gitalk
# https://github.com/gitalk/gitalk
gitalk:
    enable: false
    clientID: # Default ClientID
    clientSecret: # Default ClientSecret
    repo: # Repository name to store comments
    owner: # GitHub repo owner
    admin: # GitHub repo owner and collaborators, only these users can initialize GitHub issues
    language: # Supported: en, zh-CN, zh-TW, es-ES, fr, ru, de, pl, ko
    proxy: # CORS proxy
```

#### 3.4.3. Waline

A simple and secure comment system.

See: [Using Waline with ParticleX | Yuzi's Blog](https://blog.yuzi.dev/posts/bcb4ff00.html)

```yaml
# Waline
# https://github.com/walinejs/waline
waline:
    enable: false
    serverURL: # Waline server address URL, set to your own
    locale: # https://waline.js.org/guide/client/i18n.html#locale-option
    commentCount: true # If false, comment count shown only on post pages
    pageview: false # Pageviews count (do not enable both waline.pageview and leancloud_visitors)
    emoji: # Custom emoji
        - https://unpkg.com/@waline/emojis@1.2.0/weibo
        - https://unpkg.com/@waline/emojis@1.2.0/alus
        - https://unpkg.com/@waline/emojis@1.2.0/bilibili
        - https://unpkg.com/@waline/emojis@1.2.0/qq
        - https://unpkg.com/@waline/emojis@1.2.0/tieba
        - https://unpkg.com/@waline/emojis@1.2.0/tw-emoji
    meta: # Valid fields: nick, mail, link
        - nick
        - mail
        - link
    requiredMeta: # e.g.: [nick] or [nick, mail]
        - nick
    lang: # Available: en-US, zh-CN, zh-TW, pt-BR, ru-RU, jp-JP
    wordLimit: 0 # Word limit (0 = unlimited)
    login: enable # Options: enable, disable, force
    pageSize: 10 # Number of comments per page
```

#### 3.4.4. Twikoo

A lightweight, secure, free static site comment system.

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

---

## 4. Final Words

This project is licensed under the MIT open-source license. Contributions are welcome! Feel free to open an issue, fork the project, and submit pull requests for any improvements you have in mind!

---

## 5. Acknowledgments

Thanks to the following projects:

- [Hexo](https://hexo.io/)
- [Hexo-Theme-Butterfly](https://github.com/theme-particlex/hexo-theme-particlex)
- [Hexo-Theme-redefine](https://github.com/EvanNotFound/hexo-theme-redefine)
