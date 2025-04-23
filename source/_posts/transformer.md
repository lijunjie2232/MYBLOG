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
- [Casual Mask](#casual-mask)
- [Multi-Head Attention](#multi-head-attention)
  - [コード](#コード)
- [FNN](#fnn)
- [Positional Encoding](#positional-encoding)


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

Attention Score (scaled):

<center>$\text{Scores} = \frac{Q K^T}{\sqrt{d_k}}$</center>

softmax正規化:

<center>$A_{i,j} = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) = \frac{exp(Scores_{i,j})}{\sum_{k=1}^{n}exp(Scores_{i,k})}$</center>

Attentionスコアはスケーリング（通常は Key の次元数の平方根で割る）と正規化（ソフトマックス関数を適用）によって調整されます。


#### 重み付き和の計算
<center>$\text{Output} = \text{A} \cdot V$</center>

Attention Weights を Value に適用し、重み付き和を計算します。

###　コード
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
```

## Casual Mask

<center>
$
M_{i,j} = 
\begin{cases}
0  & \text{ if } j \le i \\
-\infty   & \text{ if } j \lt i
\end{cases}
$
</center>

エンコーダーでは、全ての位置の情報が利用できますが、デコーダーでは未来の位置の情報が利用できないようにするため、因果掩码が使用されます。
因果掩码は、未来の位置のスコアを $-\infty$ に設定し、Softmax後の重みを0にします。

## Multi-Head Attention

Multi-Head Attention は、複数の Head を結合したものを表します。

<center>$head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$</center>
<center>$MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O$</center>

- ここで、Headの数は$h$です。

![Multi-Head Attention](/assert/transformer/image/multi_head_attention.png)

### コード
```python
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(seq_len):
    # return tensor with size [seq_len, seq_len], masking all the future tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

class MaskedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention implementation in PyTorch '''

    def __init__(self, emb_size, num_heads):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, x):  # x: [B, T, emb]
        B, T, E = x.shape
        qkv = self.qkv(x)  # [B, T, 3*E]
        q, k, v = qkv.chunk(3, dim=-1)
        # Multi Head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Score
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # add casual mask
        mask = causal_mask(T).to(scores.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        out = attn @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.fc_out(out), attn
```

## FNN

FNN(Feed-Forward Network)は、入力から出力を計算する非線形関数です。

<center>$FFN(x) = max \left(0, xW_1 + b_1 \right)W_2 + b_2$</center>

入力の次元数は、入力と出力の次元数に等しい必要があります。

## Positional Encoding

Transformerモデルは、再帰や畳み込みを使用しないため、モデルがシーケンスの順序を利用できるように位置情報を組み込む必要があります。そのため、エンコーダとデコーダのスタックの入力埋め込みに "位置エンコーディング" を追加します。位置エンコーディングは、埋め込みと同じ次元 $ d_{\text{model}} $ を持つため、二つを足し合わせることができます。

位置エンコーディングには学習型と固定型の選択肢がありますが、この研究では正弦とコサイン関数を使用します。これらの関数は異なる周波数を持ちます：

<center>
$PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})$
</center>
<center>
$PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})$
</center>

- $\text{PE}_{(pos, 2i)}$ は偶数次元の位置エンコーディング
- $\text{PE}_{(pos, 2i+1)}$ は奇数次元の位置エンコーディング
- $pos$ はシーケンス内の位置
- $i$ は次元のインデックス
- $d_{\text{model}}$ はモデルの次元数（通常は512など）

ここで、$ \text{pos} $ は位置、$ i $ は次元です。つまり、位置エンコーディングの各次元は異なる周波数を持つ正弦波に対応します。波長は $ 2\pi $ から $ 10000 \cdot 2\pi $ までの幾何級数を形成します。

この関数を選択した理由は、モデルが相対的な位置に簡単に注意を向けることができると仮定したからです。任意の固定オフセット $ k $ に対して、$ \text{PE}\text{({pos}+k)} $ は $ \text{PE}{({pos})} $ の線形関数として表現できるためです。

また、学習型位置エンコーディングも試しましたが、両バージョンの結果はほぼ同じでした。正弦関数を選択した理由は、モデルが訓練中に見なかったよりも長いシーケンス長に外挿できる可能性があるためです。
