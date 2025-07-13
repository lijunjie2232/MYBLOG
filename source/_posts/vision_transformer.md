---
title: Vision Transformer(ViT)
date: 2024-4-10 10:17:00
categories: [AI]
tags: [Deep Learning, transformer, 機械学習, AI, 人工知能, 深層学習]
lang: ja
description: ViTはTransformerアーキテクチャを画像認識に応用したもので、パッチ単位で画像を処理します。CNNの帰納的バイアスをTransformerに持ち込み、より強力な特徴抽出能力を実現します。

---

## 目次

---


## Vision Transformer (ViT)
ViTは[Transformer](/2023/08/15/transformer/)アーキテクチャを画像認識に応用したもので、パッチ単位で画像を処理します。

![ViT](/assert/vision_transformer/vision_transformer.png)

###  画像のパッチ化（Patch）

- 入力画像（例：224×224×3）を固定サイズのパッチに分割。
- 例: 16×16×3 のパッチに分割 → 合計196個のパッチ（14×14）。
- 各パッチを768次元のベクトルに変換（=> `Token`）。

### 位置エンコーディング（Positional Encoding）

自然言語処理（NLP）において、Transformerなどのモデルは入力されたトークン列を順序なしの集合として扱います。しかし、言語には「単語の並び順」が意味を左右するという重要な性質があります。

ViT（Vision Transformer）におけるPositional Encodingの目的は、画像を複数のパッチ（小さな領域）に分割して処理する際に、各パッチが画像内で持つ空間的な位置情報をモデルに与えることです。Transformerアーキテクチャは入力データの順序情報を持たないため、この位置情報を明示的に追加しないと、どのパッチがどこにあるかという空間構造に関する重要な情報が失われてしまいます。

ViTでは、自然言語処理で使われるTransformerと同様に、sin/cos関数によるPositional Encodingを用いて、各パッチに位置情報を埋め込みます。

#### 数式

位置 `pos`、次元 `i` に対して、

<center>
$
PE(\text{pos},\ 2i) = \sin\left( \frac{\text{pos}}{10000^{2i / d_{\text{model}}}} \right)
$
</center>

<center>
$
PE(\text{pos},\ 2i+1) = \cos\left( \frac{\text{pos}}{10000^{2i / d_{\text{model}}}} \right)
$
</center>

- `pos`: トークンまたはパッチの位置（0, 1, 2,...）
- `i`: 埋め込みベクトルの次元（0 ≤ i < d_model/2）
- `d_model`: 入力ベクトルの次元（例: 768）

### cls token

ViT（Vision Transformer）における<cls>トークンの目的は、画像全体の意味的な情報を集約して分類タスクに利用することです。

Transformerでは各パッチが局所的な特徴を表す一方で、グローバルな情報や画像全体の文脈を直接捉えることは困難です。この問題を解決するために、ViTでは入力される各画像パッチの先頭に特別なトークンである<cls>トークンを追加します。このトークンは、Transformer内部での自己注意機構（Self-Attention）を通じて、他のすべてのパッチトークンからの情報を収集し、最終的に画像全体を表現する特徴ベクトルとなります。分類時には、この<cls>トークンに対応する出力を分類ヘッド（通常は全結合層）に入力することで、画像のカテゴリを予測します。

### Encoder

ViTにTransformerのEncoder層を利用しが、画像分類任務には特徴の理解であるから、Decoder層は省略します。以下の図は、ViTのEncoder層を示したものです。

![ViT Encoder](/assert/vision_transformer/vit_encoder.png)

以下の図は、TransformerのEncoderとDecoderを示したものです。

![Transformer Encoder](/assert/transformer/image/transformer.svg)

この二つの図を比べて、ViTのEncoderは、TransformerのEncoder`Multi-Head Attention / Feed Forward Network`後`Normalization`の流れと異なって、まずは`Layer Normalization`、次に`Multi-Head Attention / Feed Forward Network`を行うと示しています。


### ViTで画像処理の流れ
1. 画像を16×16などのサイズで分割 → パッチ化
2. 各パッチを線形変換して埋め込みベクトルに
3. `<cls>`トークンを追加（分類用）
4. 位置エンコーディングを各Tokenに加算
5. Transformer Encoderへ入力


## CNNにかえりみる

### 平移等価性（Translation Equivariance）

![CNN weight shares](/assert/vision_transformer/cnn_weights_share.png)

- 畳み込みは「窓関数」のように画像上をスライドしながら演算を行います。

- 平移等価性とは、**画像全体を平行移動させた場合でも、その特徴マップも同じように平行移動する**という性質です。

#### 数式

数学的に表現すると：

<center>
$
f(g(x)) = g(f(x))
$
</center>

- `f`：畳み込み操作
- `g`：平行移動操作

#### 利点

- この性質により、物体が画像内のどこにあっても、その特徴を一貫して検出できます。

- 並列計算が容易で、GPUなどのハードウェアを活用して高速化が可能です。

### 局所性（Locality）

- 畳み込みカーネルは通常3×3などの小さなサイズを使い、画像の**局所的な領域のみを観測**します。

- 局所性があることで、隣接するピクセル間の関係性を強調し、エッジやテクスチャなどの局所的特徴を効果的に抽出できます。

#### 利点

- 隣接する画素が意味的に関連していることが多いという前提に基づいており、自然画像の構造に適しています。

## 帰納的バイアス

帰納的バイアス（Inductive Bias）とは、**モデルが持つ仮定や先験知識**のことです。これにより、特定のタスクに対してより効率的に学習できるようになります。

### CNNにおける帰納的バイアス

- 帰納的バイアスとは、**モデルが持つ仮定や先験知識**のことです。これにより、特定のタスクに対してより効率的に学習できるようになります。

#### 主なバイアス:

1. **空間的局所性 (Spatial Locality)**  
   - 画像では近い位置にあるピクセルが相互に関係していると考えます。
   - 例: 太陽と空はよく一緒に現れます。

2. **平移等価性 (Translation Equivariance)**  
   - 物体が画像内で移動しても、それが特徴マップ上で同様に移動することを保証します。
   - 例: 左上にあった太陽が右上に移動しても、畳み込み結果も同様に移動します。

これらのバイアスにより、CNNは画像認識において非常に効果的なモデルとなっています。

### ViTの特徴と実め

- 入力画像を固定サイズのパッチに分割し、それぞれをベクトルとして扱います。
- Attention機構によって、**すべてのパッチ間の関係性を同時に考慮**できます。

![ViT bias](/assert/vision_transformer/vision_transformer_bias.png)

図中の矢印で示された2つの部分は、同じ建物の一部です。CNN（畳み込みニューロンネットワーク）では、適切なサイズの畳み込みカーネルを使えば、これらの領域を一緒に捉えることができます。しかし、ViT（Vision Transformer）では、これらのパッチ間の位置が遠く離されてしまい、さらにパッチを細かく分割すると、その距離はより一層広がってしまいます。Attention機構によってベクトル間の関係性を学ぶことは可能ですが、**空間的な局所性**という点では、ViTはCNNほど優れていないと言えます。

また、「**平移等価性**」に注目すると、ViTは各パッチの位置情報を学習する必要があるため、同じ内容のパッチでも場所が変わると出力結果も変わってしまうという問題があります。このように、<font color="#e3008c">**ViTは画像認識における「帰納的バイアス（inductive bias）」の仮定をうまく維持できていない**</font>とも言えます。

しかし、だからといってViTには未来がないわけではありません。むしろ逆です。**Transformer系のモデルには有名な特徴があります。「データがあれば、何とかなる」** つまり、<font color="#e3008c">十分な量の学習データがあれば、ViTはピクセル単位の関係性を十分に学習し、帰納的バイアスの問題を解消することができます</font>。

要するに、「**ViTは少ないデータでは弱いが、データが多ければ驚くべき力を発揮する**」ということです。

## 参考文献
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)