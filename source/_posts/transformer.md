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
  - [計算式の仕組み](#計算式の仕組み)
    - [Query,Key,Valueの計算](#querykeyvalueの計算)
    - [Attentionスコアの計算](#attentionスコアの計算)
    - [スケーリングと正規化](#スケーリングと正規化)
    - [重み付き和の計算](#重み付き和の計算)
- [Multi-Head Attention](#multi-head-attention)
  - [コード](#コード)
- [FNN](#fnn)


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

### 計算式の仕組み
![The attention mechanism](/assert/transformer/image/attention_calc.png)
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


## Multi-Head Attention

Multi-Head Attention は、複数の Head を結合したものを表します。

<center>$head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$</center>
<center>$MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O$</center>

- ここで、Headの数は$h$です。

![Multi-Head Attention](/assert/transformer/image/multi_head_attention.png)

### コード
```python
# Multi-Head Attention implementation in PyTorch
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
```

## FNN
FNN(Feed-Forward Network)は、入力から出力を計算する非線形関数です。
<center>FFN(x) = max \left(0, xW_1 + b_1 \right)W_2 + b_2</center>