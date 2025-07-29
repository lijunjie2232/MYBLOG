---
title: ResNet
date: 2024-4-5 10:17:00
categories: [AI]
tags: [Deep Learning, CNN, resnet, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]
lang: ja　
description: Resnet

---

## 目次
- [目次](#%E7%9B%AE%E6%AC%A1)
  - [ネットワークの特徴](#%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%81%AE%E7%89%B9%E5%BE%B4)
- [CNN従来ネットワークの課題](#cnn%E5%BE%93%E6%9D%A5%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%81%AE%E8%AA%B2%E9%A1%8C)
  - [問題1：勾配消失と勾配爆発](#%E5%95%8F%E9%A1%8C1%E5%8B%BE%E9%85%8D%E6%B6%88%E5%A4%B1%E3%81%A8%E5%8B%BE%E9%85%8D%E7%88%86%E7%99%BA)
  - [問題2：劣化問題（Degradation Problem）](#%E5%95%8F%E9%A1%8C2%E5%8A%A3%E5%8C%96%E5%95%8F%E9%A1%8Cdegradation-problem)
  - [ResNetの解決策](#resnet%E3%81%AE%E8%A7%A3%E6%B1%BA%E7%AD%96)
- [Residual Block](#residual-block)
  - [劣化問題の直感的理解](#%E5%8A%A3%E5%8C%96%E5%95%8F%E9%A1%8C%E3%81%AE%E7%9B%B4%E6%84%9F%E7%9A%84%E7%90%86%E8%A7%A3)
  - [残差学習のアイディア](#%E6%AE%8B%E5%B7%AE%E5%AD%A6%E7%BF%92%E3%81%AE%E3%82%A2%E3%82%A4%E3%83%87%E3%82%A3%E3%82%A2)
  - [順伝播の特徴](#%E9%A0%86%E4%BC%9D%E6%92%AD%E3%81%AE%E7%89%B9%E5%BE%B4)
  - [勾配伝播の改善](#%E5%8B%BE%E9%85%8D%E4%BC%9D%E6%92%AD%E3%81%AE%E6%94%B9%E5%96%84)
- [ResNet](#resnet)
  - [コード例](#%E3%82%B3%E3%83%BC%E3%83%89%E4%BE%8B)


---

ResNetは2015年に何凱明氏ら（マイクロソフト研究機関）より開発された畳み込みネットワーク（CNN）をベースにした深層学習モデルであり、深層ニューラルネットワークにおける重要なブレークスルーを提供する。

### ネットワークの特徴

1. 極めて深いネットワーク構造（1000層以上）
2. residual（残差）モジュールの提案
3. Batch Normalizationを用いた学習の高速化（dropoutを廃止）

## CNN従来ネットワークの課題

ResNetが提案される前は、すべてのニューラルネットワークは畳み込み層とプーリング層の積み重ねで構成されていた。
研究者たちは、畳み込み層とプーリング層をより多く重ねることで、画像の特徴情報をより完全に取得でき、学習効果も良くなると考えていた。しかし、実際の実験では、畳み込み層とプーリング層を重ねていくにつれて、学習効果が良くなるどころか...

### 問題1：勾配消失と勾配爆発

**勾配消失**：各層の誤差勾配が1未満の場合、逆伝播時にネットワークが深くなるほど勾配は0に近づく
**勾配爆発**：各層の誤差勾配が1より大きい場合、逆伝播時にネットワークが深くなるほど勾配は大きくなる

### 問題2：劣化問題（Degradation Problem）

CNNは、層数が増加するにつれて、予測性能が悪化する。

![score down with depth](/assert/CNN/deep_sdown.png)

図から、20層の時の誤差率が最も低く、56層に増加されても悪化する。


### ResNetの解決策

- ショートカット接続：層を飛び越えて接続することで勾配の逆伝播を容易にする
- 残差構造：入力を直接出力に加算することで学習を容易にする
- Batch Normalization：データの前処理とネットワーク内の正規化により学習安定化


## Residual Block

残差学習は、ResNetが解決しようとする**ネットワークの劣化問題**に対する核心的なアプローチです。

### 劣化問題の直感的理解

- 浅層ネットワークがあるとき、新たな層を追加して深層ネットワークを構築することを考える
- 最悪の場合、追加された層は何も学習せず、浅層ネットワークの特徴をそのまま複製する（恒等写像）
- この場合でも、深層ネットワークは少なくとも浅層ネットワークと同じ性能を持つはず
- しかし実際には性能が劣化するため、従来の学習方法に問題があると判断

### 残差学習のアイディア

![res block](/assert/CNN/res_block.png)

従来の学習目標の変更：

$$y_l = h(x_l) + F(x_l, W_l)$$

$$x_{l+1} = f(y_l)$$

ここで：

- $x_l$: l番目の残差ユニットの入力
- $x_{l+1}$: l番目の残差ユニットの出力
- $F$: 残差関数（学習対象）
- $h(x_l) = x_l$: 恒等写像（ショートカット接続）
- $f$: ReLU活性化関数

- 元の目標: 入力 `x` から特徴 `H(x)` を直接学習
- 新しい目標: 残差 `F(x) = H(x) - x` を学習し、最終出力を `F(x) + x` とする

### 順伝播の特徴

浅層 $l$ から深層 $L$ までの特徴は以下のように表されます：

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$

これは、出力が入力と各層の残差関数の和であることを示しており、ショートカット接続により情報が直接伝播されることを意味します。

### 勾配伝播の改善

$$ \frac{\partial loss}{\partial x_l} = \frac{\partial loss}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l}=\frac{\partial loss}{\partial x_l}\cdot ( 1+\frac{\partial }{\partial x_l}\sum\limits_{i=l}^{L-1}{F(x_i,w_i)} ) $$

この式：
1. **第1項** $\frac{\partial loss}{\partial x_L}$: 損失関数からL層までの勾配
2. **第2項の1**: ショートカット接続（恒等写像）による勾配の無損失伝播を示す。この1があることで、勾配が直接入力層まで流れることができます。
3. **第3項** $\frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i, W_i)$: 重みを介した残差勾配で、通常のネットワークパスを通る勾配を表す。

## ResNet

ResNetには2種類の残差ユニットが存在し、ネットワークの深さに応じて使い分けられています：

![diff res block](/assert/CNN/res_block_two_diff.png)

1. 基本的な残差ユニット（浅層ネットワーク用）
     - 18層や34層のResNetで使用
     - 比較的シンプルな構造
2. ボトルネック残差ユニット（深層ネットワーク用）
     - 50層、101層、152層のResNetで使用
     - 計算効率を考慮した複雑な構造
     - ショートカット接続の処理方法
     - ショートカット接続はResNetの核心的な要素ですが、入出力の次元が異なる場合に特別な処理が必要です：

### コード例

![resnet 18](/assert/CNN/resnet_18.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# from modules.bn import InPlaceABNSync as BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        # state_dict = torch.load('/apdcephfs/share_1290939/kevinyxpang/STIT/resnet18-5c106cde.pth')
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
```