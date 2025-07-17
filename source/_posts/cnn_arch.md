---
title: CNN基本建築
date: 2024-4-1 10:17:00
categories: [AI]
tags: [Deep Learning, CNN, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]
lang: ja　
description: CNN (Convolutional Neural Networks / 畳み込みニューラルネットワーク) とは、画像認識や画像分類などのコンピュータビジョンタスクで広く使用される深層学習モデルです。

---

## 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [基本構造](#%E5%9F%BA%E6%9C%AC%E6%A7%8B%E9%80%A0)
- [入力層](#%E5%85%A5%E5%8A%9B%E5%B1%A4)
- [畳み込み層](#%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E5%B1%A4)
  - [畳み込み演算の手順](#%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E6%BC%94%E7%AE%97%E3%81%AE%E6%89%8B%E9%A0%86)
    - [順伝播（Forward）の数式](#%E9%A0%86%E4%BC%9D%E6%92%ADforward%E3%81%AE%E6%95%B0%E5%BC%8F)
  - [パディング（Padding）](#%E3%83%91%E3%83%87%E3%82%A3%E3%83%B3%E3%82%B0padding)
    - [パディングの仕組み](#%E3%83%91%E3%83%87%E3%82%A3%E3%83%B3%E3%82%B0%E3%81%AE%E4%BB%95%E7%B5%84%E3%81%BF)
  - [多チャンネル画像の処理](#%E5%A4%9A%E3%83%81%E3%83%A3%E3%83%B3%E3%83%8D%E3%83%AB%E7%94%BB%E5%83%8F%E3%81%AE%E5%87%A6%E7%90%86)
    - [処理手順](#%E5%87%A6%E7%90%86%E6%89%8B%E9%A0%86)
    - [図の説明](#%E5%9B%B3%E3%81%AE%E8%AA%AC%E6%98%8E)
- [プーリング層](#%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0%E5%B1%A4)
  - [主なプーリング手法](#%E4%B8%BB%E3%81%AA%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0%E6%89%8B%E6%B3%95)
    - [**最大プーリング（Max Pooling）**](#%E6%9C%80%E5%A4%A7%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0max-pooling)
    - [**平均プーリング（Average Pooling）**](#%E5%B9%B3%E5%9D%87%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0average-pooling)
- [全結合層](#%E5%85%A8%E7%B5%90%E5%90%88%E5%B1%A4)
  - [処理の流れ](#%E5%87%A6%E7%90%86%E3%81%AE%E6%B5%81%E3%82%8C)
  - [コード例 (AlexNet)](#%E3%82%B3%E3%83%BC%E3%83%89%E4%BE%8B-alexnet)


---

CNN (Convolutional Neural Networks / 畳み込みニューラルネットワーク) とは、画像認識や画像分類などのコンピュータビジョンタスクで広く使用される深層学習モデルです。

## 基本構造

![CNN arch](/assert/CNN/cnn_arch.png)

1. **畳み込み層（Convolutional Layer）**:
   - 主な役割: 特徴抽出
   - 入力画像に対してフィルター（カーネル）を適用して特徴マップを作成します。
   - 例: エッジ検出、形状認識など。

2. **プーリング層（Pooling Layer）**:
   - 主な役割: ダウンサンプリング
   - 特徴マップのサイズを縮小し、計算量を減らすとともに過学習を防ぎます。
   - 最大プーリング（Max Pooling）や平均プーリング（Average Pooling）が一般的です。

3. **活性化関数（Activation Function）**:
   - 非線形性を導入するために使用されます。
   - 一般的にはReLU（Rectified Linear Unit）が使われます。

4. **全結合層（Fully Connected Layer）**:
   - 畳み込み層とプーリング層で抽出された特徴を利用して、最終的に分類を行います。
   - 多層パーセプトロンとして機能します。

5. **ドロップアウト層（Dropout Layer）**:
   - 過学習を防ぐために一部のニューロンをランダムに無効化します。


## 入力層

コンピュータは画像を数値として処理するため、画像は次のような形式に変換されます：


![入力](/assert/CNN/input.gif)

- グレースケール画像（Gray-scale Image）:
  - 各ピクセルの値は0〜255の範囲で、0が黒、255が白を表します。
  - 画像は2次元の行列として表現されます。
  - 例：手書き数字「8」の画像は、以下のようにピクセル値の行列として表現されます。
  
- 二値画像（Binary Image）:
  - 各ピクセル値は0（黒）または255（白）のどちらかです。
  - 単純なパターン認識に適しています。

- RGB画像（カラー画像）:
  - 赤（R）、緑（G）、青（B）の3つのチャネルを持ち、それぞれのチャネルのピクセル値は0〜255の範囲です。
  - 3次元の行列として表現されます。
  - 例：画像サイズが28×28ピクセルの場合、RGB画像の行列は (28, 28, 3) となります。


## 畳み込み層

畳み込み層は、入力画像の2次元行列に対して**畳み込み演算（Convolution Operation）**を行い、特徴マップ（Feature Map）を生成します。この特徴マップには、画像内の特定のパターン（エッジ、角、形状など）が強調されて現れます。


### 畳み込み演算の手順

![特徴マップの生成](/assert/CNN/conv.gif)

1. **畳み込みカーネル（フィルター）の適用**

- 畳み込みカーネル（Convolution Kernel）は小さな2次元行列で、画像の特定の特徴（例：エッジ、角）を検出するために設計されています。
- 例：カーネルのサイズは通常 `3×3` や `5×5` です。

2. **スライディング（移動）**

- カーネルは、入力画像の2次元行列上を左上から右下に向かってスライドします（移動量はストライドと呼ばれます）。
- スライドするたびに、カーネルと対応する画像領域の要素ごとの積を計算し、総和を取ります。

3. **特徴マップの生成**

- 各位置での畳み込み演算の結果を結合して、新しい2次元行列（特徴マップ）を作成します。


#### 順伝播（Forward）の数式
畳み込み演算は、入力画像 $ X $ とフィルター（カーネル） $ W $ を使って行われます。以下にその基本的な数式を示します。

<center>
$
Y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m, j+n} \cdot W_{m,n} + b
$
</center>

- $ Y_{i,j} $：出力特徴マップの位置 ` (i,j) ` の値 
- ` X `：入力画像（2次元行列）
- ` W `：フィルター（カーネル）$ k \times k $
- ` b `：バイアス項
- ` k `：カーネルのサイズ

逆伝播では、損失関数 $ L $ の勾配をフィルター $ W $ と入力 $ X $ に対して求めます。

(1) フィルター $ W $ に関する勾配
$$
\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j}} \cdot X_{i+m, j+n}
$$

(2) 入力 $ X $ に関する勾配
<center>
$
\frac{\partial L}{\partial X_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial Y_{i-m,j-n}} \cdot W_{m,n}
$
</center>

(3) バイアス $ b $ に関する勾配
<center>
$
\frac{\partial L}{\partial b} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j}}
$
</center>

### パディング（Padding）

![padding operation](/assert/CNN/padding_0.gif)

畳み込み演算では、画像の端の情報が特徴マップに反映されにくいという問題があります。これを補うために、**パディング**という処理が使われます。

#### パディングの仕組み

- 入力画像の周囲に0などの値で1層または複数層を追加します。
- これにより、画像の端の情報も特徴マップに十分に反映されます。


**図6：Padding = 1 の場合**
![padding operation](/assert/CNN/padding_1.gif)

- 画像の周囲に1層の0を追加し、畳み込み演算を行います。
- これにより、特徴マップのサイズが入力画像とほぼ同じになります。

**図7：Padding = 2 の場合**
![padding operation](/assert/CNN/padding_2.gif)

- 周囲に2層の0を追加します。
- より広範囲の端の情報を保持することができます。

### 多チャンネル画像の処理

カラー画像（RGB画像）のように、複数のチャンネルを持つ画像の場合、以下のように処理されます。

#### 処理手順

1. 各チャンネル（R, G, B）ごとに畳み込み演算を行います。
2. 各チャンネルに対応するカーネルを使用し、それぞれで畳み込み演算を行います。
3. 各カーネルの結果を足し合わせ、最終的な特徴マップを生成します。

#### 図の説明

**図8：2つのカーネルを使用して畳み込みを行う過程**

![channels convolution process](/assert/CNN/channel_conv.gif)

- 入力画像がRGB画像（7×7×3）であると仮定します。
- 2つのカーネルを使用して、それぞれで畳み込み演算を行い、2つの特徴マップを生成します。
- 各カーネルにはバイアス（偏置項）が含まれており、演算結果に加算されます。

## プーリング層

1. **ダウンサンプリング（Downsampling）**:
   - 特徴マップの解像度を下げることで、後の層での計算量を削減します。
2. **位置変動へのロバスト性の向上**:
   - 少し位置がずれても同じ特徴が検出されるようにします。
3. **過学習の防止**:
   - データの空間的冗長性を減らすことで、過学習を抑制します。

### 主なプーリング手法
#### **最大プーリング（Max Pooling）**
![Max Pooling](/assert/CNN/pooling_max.png)

- 特徴マップの一部領域（例：2×2）から最大値を取り出す方法。
- よく使われ、エッジや重要な特徴を強調します。

**数式**：
$$
Y_{i,j} = \max(X_{i:i+k, j:j+k})
$$
- $ k $：プーリング窓のサイズ

**例**：
```
入力:
[
   [2, 5, 6, 1],
   [9, 3, 7, 3],
   [8, 5, 7, 8],
   [6, 6, 1, 1],
]

最大プーリング (2x2):
[
   [9, 7],
   [8, 8],
]
```


#### **平均プーリング（Average Pooling）**

![Average Pooling](/assert/CNN/pooling_avg.png)

- 特徴マップの一部領域から平均値を取り出す方法。
- 最大プーリングよりも滑らかな特徴を抽出します。

**数式**：
$$
Y_{i,j} = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m, j+n}
$$

**例**：
```
入力:
[
   [2, 5, 6, 1],
   [9, 3, 7, 3],
   [8, 5, 7, 8],
   [6, 6, 1, 1],
]

平均プーリング (2x2):
[
   [5, 5],
   [7, 5],
]
```

## 全結合層

![Fully Connected Layer](/assert/CNN/fc.png)

畳み込み層やプーリング層では、画像の局所的な特徴（例：目、鼻、口）が特徴マップとして抽出されます。  
しかし、これらの特徴を使って「これは人の顔だ」と判断するには、**すべての特徴を統合して評価する必要があります**。

### 処理の流れ

1. **展平（Flattening）**
   - 特徴マップを1次元のベクトルに変換します。
   - 例：特徴マップが `7×7×3` の場合 → `1×147` の1次元ベクトルに変換。

2. **全結合演算**
   - すべての要素に対して重み（Weight）とバイアス（Bias）を使った線形変換を行います。
   - 数式：  
     $$
     Y = W \cdot X + b
     $$
     - $ X $：入力ベクトル（展平された特徴）
     - $ W $：重み行列
     - $ b $：バイアス
     - $ Y $：出力ベクトル

3. **活性化関数による非線形変換**
   - ReLUやSigmoid、Softmaxなどの活性化関数を使って、非線形な関係を学習します。
   - 最終的には、どのクラスに属するかの**確率**を出力します。


### コード例 (AlexNet)

```python
import torch
import torch.nn as nn
import torchvision

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=96,out_channels=192,kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classifier(x)
        return x


if __name__ =='__main__':
    # model = torchvision.models.AlexNet()
    model = AlexNet()
    print(model)

    input = torch.randn(8,3,224,224)
    out = model(input)
    print(out.shape)
```

