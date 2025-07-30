---
title: RNN、GRU、LSTMの説明
date: 2024-4-7 10:17:00
categories: [AI]
tags: [Deep Learning, CNN, resnet, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]
lang: ja　
description: rnn

---
## 目次

---

## リカレントニューラルネットワーク (RNN)

リカレントニューラルネットワーク（RNN）は、**隠れ状態**（hidden state）を持つニューラルネットワークです。この隠れ状態により、ネットワークは時系列データの履歴情報を保持し、現在の出力を計算することができます。

![RNN](/assert/rnn/rnn.png)

### 隠れ状態を持たないニューラルネットワーク

まず、隠れ状態を持たない多層パーセプトロン（MLP）を考えてみましょう。入力 $\mathbf{X}$ に対して、隠れ層の出力 $\mathbf{H}$ は次のように計算されます。

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h})$$

ここで、$\phi$ は活性化関数、$\mathbf{W}_{\textrm{xh}}$ は重み、$\mathbf{b}_\textrm{h}$ はバイアスです。

### 隠れ状態を持つRNN

RNNでは、隠れ層の出力 $\mathbf{H}_t$ は現在の入力 $\mathbf{X}_t$ だけでなく、**1つ前の時刻の隠れ状態** $\mathbf{H}_{t-1}$ にも依存します。

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h})$$

この式により、RNNは時系列の履歴情報を保持し、現在の出力を計算することができます。隠れ状態 $\mathbf{H}_t$ は、時系列の現在の「状態」または「記憶」として機能します。

### 出力層

RNNの出力層は、通常のMLPと同様に計算されます。

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}$$

### 特徴

- **時系列情報の保持**: RNNは隠れ状態を通じて、過去の入力情報を持つことができます。
- **パラメータ数の固定**: 時刻が増加しても、RNNのパラメータ数は増加しません。同じパラメータが各時刻で再利用されます。
- **言語モデルへの応用**: RNNは、文字レベルや単語レベルの言語モデルを構築するために使用されます。
