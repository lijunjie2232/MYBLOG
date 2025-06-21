---
title: 生成敵対ネットワーク (GAN) 理論と実装 (PyTorch)

date: 2024-4-3 11:15:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習, 生成AI, 画像生成]
lang: ja
description: GAN（生成敵対ネットワーク）は、2014年にIan Goodfellowによって提案された深層生成モデルの一種です。このモデルは、生成器 (Generator) と判別器 (Discriminator) の2つのニューロンネットワークから構成され、互いに競争しながら学習を進めます。
---

## 目次
- [目次](#%E7%9B%AE%E6%AC%A1)
- [GAN紹介](#gan%E7%B4%B9%E4%BB%8B)
  - [基本的な構造](#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E6%A7%8B%E9%80%A0)

---

## GAN紹介

GAN（生成敵対ネットワーク）は、2014年にIan Goodfellowによって提案された深層生成モデルの一種です。このモデルは、生成器 (Generator) と判別器 (Discriminator) の2つのニューロンネットワークから構成され、互いに競争しながら学習を進めます。


### 基本的な構造

![gan_structure](/assert/GAN/gan_structure.png)

- **生成器 (Generator)**: 
  - 入力としてランダムノイズ $ z $ を受け取ります。
  - 出力として疑似データ $ G(z) $ を生成します。
  - 目標：本物のデータ分布に近づけ、判別器をだます。

- **判別器 (Discriminator)**:
  - 入力として本物のデータ $ x $ または生成器からの出力 $ G(z) $ を受け取ります。
  - 出力として、入力が本物である確率 $ D(x) $ を返します。
  - 目標：正しく本物と偽物を識別すること。
