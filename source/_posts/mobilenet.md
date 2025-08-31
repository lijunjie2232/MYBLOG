---
title: MobileNetシリーズの説明
date: 2024-5-11 10:17:00
categories: [AI]
tags: [Deep Learning, CNN, mobilenet, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]
lang: ja　
description: 歴代のMobileNetシリーズモデルを用いる革新を解説。

---

# 深度方向分離畳み込み

MobileNetV1の核心的な概念は、従来の標準的な畳み込みの代わりに深度方向分離畳み込みを使用することです。標準的な畳み込み操作は、入力特徴マップの空間次元（幅と高さ）とチャネル次元の両方に対して同時に情報抽出を行いますが、これは計算コストが非常に高くなります。深度方向分離畳み込みは、このプロセスを巧妙に2つのステップに分割します。

![Convolution compare](/assert/MobileNet/dsc.png)

## 核心的な概念：2段階の分解

### 深度方向畳み込み (Depthwise Convolution)

![Convolution compare](/assert/MobileNet/dc.png)

第一段階は深度方向畳み込みです。これは「空間フィルタリング」を担当し、入力の各チャネルに対して独立して1つの畳み込みカーネルを使用します。入力にM個のチャネルがあると仮定すると、深度方向畳み込みはM個の畳み込みカーネルを使用し、各カーネルは対応するチャネルのみを処理します。このステップでは特徴マップのサイズのみが変更され（stride>1またはpaddingがある場合）、チャネル数は変更されません。

### ポイントワイズ畳み込み (Pointwise Convolution)

![Convolution compare](/assert/MobileNet/sc.png)

第二段階はポイントワイズ畳み込みです。これは「チャネルの組み合わせ」を担当し、本質的には1x1の標準的な畳み込みです。前のステップで各チャネルの情報は独立して処理されるため、チャネル間の交流はありません。ポイントワイズ畳み込みの役割は、1x1の畳み込みカーネルを使用して、前の深度方向畳み込みの出力であるM個のチャネルの特徴マップを重み付けして組み合わせ、新しい特徴を生成することです。このステップでは特徴マップのサイズは変更されず、チャネル数のみが変更されます。

### Pointwise Convolutionの二つの役割

ポイントワイズ畳み込み(Pointwise Convolution)は実際には1×1畳み込みであり、DSC(深度方向分離畳み込み)において二つの重要な役割を果たします。

#### 第一の役割：出力チャネル数の自由な変更

単独の深度方向畳み込みでは出力チャネル数を変更できないため、1×1畳み込みを用いて出力チャネル数を変更することは直感的で簡単な方法です。

#### 第二の役割：チャネル融合

深度方向畳み込みのみを使用してネットワークを構築した場合の問題を理解するために、以下のようなシナリオを考えてみましょう：

- 入力を$IN$とし、その第iチャネルを$IN_i$とする
- 第一層の深度方向畳み込みの出力を$DC1$とし、その第$i$チャネルを$DC1_i$とする
- 第二層の深度方向畳み込みの出力を$DC2$とし、その第$i$チャネルを$DC2_i$とする

深度方向畳み込みの動作原理により：

- $DC1_i$は$IN_i$のみに関連
- $DC2_i$は$DC1_i$のみに関連
- 結果として、$DC2_i$も$IN_i$のみに関連

つまり、入力と出力の各チャネル間には何の計算的関連も存在しません。$1 \times 1$畳み込みはチャネル融合能力を持っているため、深度方向畳み込みの後にポイントワイズ畳み込みを接続することで、この問題を効果的に解決できます。

## 数式解析と計算複雑度の比較

![Convolution compare](/assert/MobileNet/dsc.png)

以下の仮定に基づいて分析します：

- 入力特徴マップのサイズ：$D_k \times D_k \times M$
- 畳み込みカーネルのサイズ：$D_F \times D_F \times M$、その数：$N$

### 標準的な畳み込み (Standard Convolution)

単一の畳み込みに対する計算量：
$$
D_k \times D_k \times D_F \times D_F \times M
$$

これは特徴マップの空間次元に含まれる$D_k \times D_k$個の点と、各点での畳み込み操作の計算量$D_F \times D_F \times M$の積です。

$N$個の畳み込みに対する総計算量：
$$
D_k \times D_k \times D_F \times D_F \times M \times N
$$

ここで：
- $D_k$：入力特徴マップのサイズ
- $D_F$：カーネルのサイズ
- $M$：入力特徴マップのチャネル数
- $N$：カーネル数

## 深度方向分離畳み込み (DSC / Depthwise Separable Convolution)

深度方向畳み込みの計算総量：
$$
D_k \times D_k \times D_F \times D_F \times M
$$

ポイントワイズ畳み込みの計算総量：
$$
M \times N \times D_K \times D_K
$$

Depthwise Separable Convolutionの計算総量：
$$
D_k \times D_k \times D_F \times D_F \times M + M \times N \times D_K \times D_K
$$

この分析から、Depthwise Separable Convolutionは通常の畳み込みよりも計算効率がはるかに優れていることがわかります。特に、$D_F = 3$という典型的なカーネルサイズでは、計算量は約1/9に削減されます。

# 参考
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)