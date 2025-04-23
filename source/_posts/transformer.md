---
title: Transformer
date: 2022-8-15 10:17:00
categories: [AI]
tags: [deep learning, transformer]
lang: ja
---

Transformerは、RNNやCNNを用いたモデルの代わりに、Attentionを用いたモデルです。

- [Attention](#attention)
  - [基本的な概念](#基本的な概念)
  - [Attentionの計算式](#attentionの計算式)
    - [Query,Key,Valueの計算](#querykeyvalueの計算)
    - [Attentionスコアの計算](#attentionスコアの計算)
    - [スケーリングと正規化](#スケーリングと正規化)
    - [重み付き和の計算](#重み付き和の計算)


## Attention

- [Attention](https://arxiv.org/abs/1706.03762) は、ある入力の特徴量が特定の出力の特徴量にどれだけ関連しているかを学習するメカニズムです。これは、シーケンスデータ（例えば、文章や音声）の処理において非常に重要な役割を果たします。

### 基本的な概念

- Query (Q): 出力の特徴量を表します。
- Key (K): 入力の特徴量を表します。
- Value (V): 入力の特徴量に関連する情報を表します。
- softmax: ソフトマックス関数

### Attentionの計算式

元の論文に、Attentionは"Scaled Dot-Product Attention"

<center>$\text{Attention} = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right)V$</center>

![Scaled Dot-Product Attention](/assert/transformer/image/attention_original.png)

#### Query,Key,Valueの計算
入力データから Query, Key, Value を計算します。通常、これらは線形変換（行列乗算）によって生成されます。

#### Attentionスコアの計算
<center>$\text{Attention Score} = Q \cdot K^T$</center>

Query と Key の間の関連性を計算します。これは一般的に内積（Dot Product）を使用して行われます。

#### スケーリングと正規化
<center>$\text{Attention Weights} = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right)$</center>

Attentionスコアはスケーリング（通常は Key の次元数の平方根で割る）と正規化（ソフトマックス関数を適用）によって調整されます。

#### 重み付き和の計算
<center>$\text{Output} = \text{Attention Weights} \cdot V$</center>

Attention Weights を Value に適用し、重み付き和を計算します。

