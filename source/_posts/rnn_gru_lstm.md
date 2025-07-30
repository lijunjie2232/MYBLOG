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

### RNNをもちいる言語モデル

![RNN Language Model](/assert/rnn/rnn_lm.png)

言語モデルの目的は、現在および過去のトークン（単語や文字）に基づいて、次のトークンを予測することです。RNNを用いた文字レベル言語モデルでは、テキストを文字単位でトークン化し、1文字ずつ予測していきます。

例えば、"machine"という単語のシーケンスを考えると、入力シーケンスは"machin"、ターゲットシーケンスは"achine"となります。各時刻tにおいて、RNNは過去の文字列に基づいて次の文字を予測します。

#### RNNによる予測プロセス
RNNは各時刻tにおいて、以下の手順で処理を行います：

1. 現在の入力文字 $\mathbf{X}t$ と前の隠れ状態 $\mathbf{H}{t-1}$ から新しい隠れ状態 $\mathbf{H}_t$ を計算
2. 隠れ状態 $\mathbf{H}_t$ から出力 $\mathbf{O}_t$ を生成
3. 出力に対してsoftmax関数を適用し、各文字の出現確率を算出
4. クロスエントロピー損失を用いて、予測結果と正解ラベル（次の文字）との誤差を計算
5. 例えば、時刻3では、入力シーケンス「m」「a」「c」に基づいて出力 $\mathbf{O}_3$ が生成され、正解の文字「h」と比較して損失が計算されます。

## RNNの限界

### 勾配消失・勾配爆発問題（Vanishing/Exploding Gradients）

#### 問題の概要

RNNの学習では誤差逆伝播（Backpropagation Through Time: BPTT）が用いられますが、これが長期的な依存関係の学習において深刻な問題を引き起こします。

#### 数学的背景

時刻$t$の損失$L_t$から時刻$h_\tau(h_\tau < t)$のパラメータWに対する勾配は次のように表されます：

$$\nabla W_{L_t} \approx \sum (\frac{\partial h_t}{\partial h_\tau } )*(\frac{\partial h_\tau }{\partial W} )$$

ここで、$\frac{\partial h_t}{\partial h_\tau}$は時刻τからtへの勾配フローであり、これは連鎖律により次のように展開されます：

$$\frac{\partial h_t}{\partial h_\tau } = \prod_{k=\tau+1}^{t}\frac{\partial t_k}{\partial h_{k-1}} $$

RNNの更新式 h_t = tanh(W_h h_{t-1} + W_x x_t) を用いると：

$$\frac{\partial h_k}{\partial h_{k-1} } = diag(1-h_{k-1}^2) * W_h$$

したがって、勾配は行列$W_h$の累乗に比例して変化します。$W_h$の最大特異値が1より大きい場合、勾配は指数関数的に増加し（勾配爆発）、1より小さい場合、勾配は指数関数的に減少します（勾配消失）。

#### 実際の影響

- **勾配消失**：初期の時刻の情報が学習されにくくなり、長期依存関係を捉えられない
- **勾配爆発**：勾配が非常に大きくなり、学習が不安定になる

### 長期依存関係の学習困難

RNNは理論上、任意の長さの時系列情報を保持できますが、実際には勾配消失のため、数百〜数千ステップ前の情報は効果的に学習できません。

#### 具体例

例えば、言語モデルで以下の文を考えます：
"フランスの首都は...パリです。"

「パリ」を予測するには「フランスの首都は」という文脈が必要ですが、RNNは長い文脈の場合、この関係性を学習するのが困難です。

### 計算の逐次性

RNNの更新は逐次的に行われるため、並列化が難しく、学習や推論が遅くなります。

$$h_t = f(h_{t-1}, x_t)$$

この式からわかるように、時刻tの計算は時刻$t-1$の結果に依存するため、並列処理ができません。

### 固定長の隠れ状態

RNNの隠れ状態h_tは固定長のベクトルであるため、時系列が長くなるにつれて情報を圧縮する必要があります。これは情報の損失を引き起こします。

### 学習の不安定性

- 初期値に敏感
- ハイパーパラメータの調整が難しい
- 学習率の選択が重要で、適切でないと学習が進まない

## LSTM

従来のRNNでは、勾配消失・爆発問題により長期依存関係を学習するのが困難でした。LSTM(Hochreiter & Schmidhuber, 1997)はこの問題を解決するため、以下のような革新的な構造を導入しました

### 核心コンセプト

- **メモリセル**：時系列情報を保持する中間記憶領域
- **ゲート機構**：情報の流入・流出を制御する3つの多重ノード
- **定数誤差フロー**：勾配が消失/爆発しない設計

#### 勾配消失問題への対処

1. **定数誤差ループ**：$\frac{\partial C_t}{\partial C_{t-1}} = F_t \in (0,1) $ を満たすような関数 $f(x)$ を定義により、勾配が指数関数的に減衰しない
2. **多重ノード構造**：各ゲートが独立して学習されるため、複雑な依存関係をモデル化可能
3. **非線形活性化の分離**：tanhとsigmoid関数が異なる役割を分担し、数値安定性を確保


### 三重のゲート構造

![LSTM](/assert/rnn/lstm_1.png)

1. **入力ゲート** ($\mathbf{I}_t$): 新しい情報がセル状態にどれだけ影響を与えるかを制御
2. **忘却ゲート** ($\mathbf{F}_t$): 以前のセル状態の情報をどれだけ保持するかを制御
3. **出力ゲート** ($\mathbf{O}_t$): 現在のセル状態が隠れ状態にどれだけ影響を与えるかを制御

| ゲート                      | 機能                 |
| --------------------------- | -------------------- |
| **入力ゲート(input gate)**  | 新しい情報の流入許可 |
| **忘却ゲート(forget gate)** | 古い情報の保持/破棄  |
| **出力ゲート(output gate)** | セル状態の出力制御   |


### LSTMのゲート計算の数式


数学的に、$h$ 個の隠れユニット、バッチサイズ $n$、入力数 $d$ があると仮定します。したがって、入力は $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ で、前の時間ステップの隠れ状態は $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ です。これに対応して、時間ステップ $t$ におけるゲートは次のように定義されます：入力ゲートは $\mathbf{I}_t \in \mathbb{R}^{n \times h}$、忘却ゲートは $\mathbf{F}_t \in \mathbb{R}^{n \times h}$、出力ゲートは $\mathbf{O}_t \in \mathbb{R}^{n \times h}$ です。これらは次のように計算されます：

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$

ここで、$\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$ および $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$ は重みパラメータで、$\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$ はバイアスパラメータです。合計中にブロードキャストがトリガーされることに注意してください。入力値を区間 $(0, 1)$ にマッピングするためにシグモイド関数を使用します。

各成分の意味を以下に示します：

- $\mathbf{X}_t$: 時間ステップ $t$ における入力ベクトル（バッチサイズ $n$、入力次元 $d$）
- $\mathbf{H}_{t-1}$: 時間ステップ $t-1$ における隠れ状態（バッチサイズ $n$、隠れユニット数 $h$）
- $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}}$: 入力から各ゲートへの重み行列（入力次元 $d$ × 隠れユニット数 $h$）
- $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}}$: 前の隠れ状態から各ゲートへの重み行列（隠れユニット数 $h$ × 隠れユニット数 $h$）
- $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o}$: 各ゲートのバイアス項（1 × 隠れユニット数 $h$）
- $\sigma$: シグモイド活性化関数（出力を0から1の間に制限）


### 入力ノード (input node)

![LSTM Input Node](/assert/rnn/lstm_input_node.png)

$$
\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c})
$$

- セル状態に追加される候補情報を生成
- tanh活性化関数で[-1,1]の範囲に正規化

### メモリセル状態 (memory cell state)

![LSTM Memory Cell State](/assert/rnn/lstm_memory_cell_state.png)]

$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t
$$

- Hadamard積(要素ごとの積)で情報の更新を制御
- 忘却ゲートが1の時、過去の情報が維持される

