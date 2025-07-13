---
title: Vision Transformer(ViT)
date: 2024-4-10 10:17:00
categories: [AI]
tags: [Deep Learning, transformer, ViT, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]

lang: ja
description: Vision Transformer（ViT）が画像認識において空間的な局所性や平移等価性といった帰納的バイアスを明示的に持たないことによる課題を指摘しつつも、大規模データでの学習を通じてAttention機構がこれらの問題を補完できると述べています。また、モデルの性能は構造だけでなくデータ量にも依存しており、十分なデータがあればTransformer系のモデルでも優れた結果が得られると強調しています。要するに、「ViTは帰納的バイアスが弱いが、データがあればその欠点を克服できる」という主張がまとめられます。

---

## 目次
- [目次](#%E7%9B%AE%E6%AC%A1)
- [Vision Transformer (ViT)](#vision-transformer-vit)
  - [画像のパッチ化（Patch）](#%E7%94%BB%E5%83%8F%E3%81%AE%E3%83%91%E3%83%83%E3%83%81%E5%8C%96patch)
  - [Position Embedding](#position-embedding)
    - [数式](#%E6%95%B0%E5%BC%8F)
  - [cls token](#cls-token)
  - [Encoder](#encoder)
  - [ViTで画像処理の流れ](#vit%E3%81%A7%E7%94%BB%E5%83%8F%E5%87%A6%E7%90%86%E3%81%AE%E6%B5%81%E3%82%8C)
- [CNNにかえりみる](#cnn%E3%81%AB%E3%81%8B%E3%81%88%E3%82%8A%E3%81%BF%E3%82%8B)
  - [平移等価性（Translation Equivariance）](#%E5%B9%B3%E7%A7%BB%E7%AD%89%E4%BE%A1%E6%80%A7translation-equivariance)
    - [数式](#%E6%95%B0%E5%BC%8F-1)
    - [利点](#%E5%88%A9%E7%82%B9)
  - [局所性（Locality）](#%E5%B1%80%E6%89%80%E6%80%A7locality)
    - [利点](#%E5%88%A9%E7%82%B9-1)
- [帰納的バイアス](#%E5%B8%B0%E7%B4%8D%E7%9A%84%E3%83%90%E3%82%A4%E3%82%A2%E3%82%B9)
  - [CNNにおける帰納的バイアス](#cnn%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8B%E5%B8%B0%E7%B4%8D%E7%9A%84%E3%83%90%E3%82%A4%E3%82%A2%E3%82%B9)
    - [主なバイアス](#%E4%B8%BB%E3%81%AA%E3%83%90%E3%82%A4%E3%82%A2%E3%82%B9)
  - [ViTの特徴と実め](#vit%E3%81%AE%E7%89%B9%E5%BE%B4%E3%81%A8%E5%AE%9F%E3%82%81)
- [参考文献](#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)


---


## Vision Transformer (ViT)
ViTは[Transformer](/2023/08/15/transformer/)アーキテクチャを画像認識に応用したもので、パッチ単位で画像を処理します。

![ViT](/assert/vision_transformer/vision_transformer.png)

###  画像のパッチ化（Patch）

- 入力画像（例：224×224×3）を固定サイズのパッチに分割。
- 例: 16×16×3 のパッチに分割 → 合計196個のパッチ（14×14）。
- 各パッチを768次元のベクトルに変換（=> `Token`）。

### Position Embedding

自然言語処理（NLP）において、Transformerなどのモデルは入力されたトークン列を順序なしの集合として扱います。しかし、言語には「単語の並び順」が意味を左右するという重要な性質があります。

![position embedding](/assert/vision_transformer/position_embeding.png)

ViT（Vision Transformer）におけるposition embeddingの目的は、画像を複数のパッチ（小さな領域）に分割して処理する際に、各パッチが画像内で持つ空間的な位置情報をモデルに与えることです。Transformerアーキテクチャは入力データの順序情報を持たないため、この位置情報を明示的に追加しないと、どのパッチがどこにあるかという空間構造に関する重要な情報が失われてしまいます。

ViTでは、自然言語処理で使われるTransformerと同様に、sin/cos関数によるposition embeddingを用いて、各パッチに位置情報を埋め込みます。

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

![cls](/assert/vision_transformer/cls.png)

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
4. Position Embeddingを各Tokenに加算
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

#### 主なバイアス

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


![ViT samples acc](/assert/vision_transformer/samples_acc.png)

1. 異なるサイズのデータセットでの事前学習

    ImageNet（約140万枚）、ImageNet-21k（約1400万枚）、JFT-300M（約3億枚）でViTを事前学習させた結果、データ量が少ないImageNetではViT-LargeはResNet系のBiTに劣るが、データ量が多くなるにつれてViTの性能が向上し、最終的にBiTを上回ることが確認されています。
    このことから、「ViTは画像における局所性や平移等価性といった帰納的バイアスを持たないが、大量のデータがあれば、それらをAttention機構を通じて学ぶことができる」という結論が導かれます。
2. サブセットを使ったFew-shot実験

    JFT-300Mからランダムに抽出した9M、30M、90Mなどの部分集合で学習を行い、正則化なし・ハイパーパラメータ固定の条件下で評価しました。結果として、ViTは小規模データではResNetよりも過学習が起きやすく性能も低いですが、データ量が増えるとResNetを上回る性能を発揮します。

これは「畳み込みによる帰納的バイアスは小規模データでは有効だが、大規模データではむしろ制約となり得る」という重要な知見を示しています。**Transformer系のモデルには有名な特徴があります。「データがあれば、何とかなる」** つまり、<font color="#e3008c">十分な量の学習データがあれば、ViTはピクセル単位の関係性を十分に学習し、帰納的バイアスの問題を解消することができます</font>。

要するに、「**ViTは少ないデータでは弱いが、データが多ければ驚くべき力を発揮する**」ということです。

## 参考文献
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)