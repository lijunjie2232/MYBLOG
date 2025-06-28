---
title: AutoEncoder、DAE(Denoising Autoencoder) と VAE(Variational Autoencoder)

date: 2024-4-6 11:15:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習, 生成AI, 画像生成, 画像処理, 画像生成AI, 自動エンコーダー, 可変自動エンコーダー]
lang: ja
description: オートエンコーダーは、主に教師なし学習に使われるニューラルネットワークの一種で、データの効率的な表現を学ぶことを目的とします。特に、次元削減や特徴抽出に用いられます。一方、VAE（Variational Autoencoder）はオートエンコーダーの一種であり、確率的表現を持つため、より柔軟なデータ生成が可能です。

---

## 目次

---

## オートエンコーダー（Autoencoder）

オートエンコーダーは、主に教師なし学習に使われるニューラルネットワークの一種で、データの効率的な表現を学ぶことを目的とします。特に、次元削減や特徴抽出に用いられます。

![Autoencoder](/assert/AutoEncoder_and_VAE/AE.png)


- **エンコーダー（Encoder）**：入力データを低次元空間に圧縮。
- **潜在ベクトル（Bottleneck/Code）**：圧縮された情報が格納される部分。
- **デコーダー（Decoder）**：潜在ベクトルから元の入力データを復元。

### 数式による表現

- 入力データ：$ x $  
- 再構成されたデータ：$ x' $  
- 潜在空間での表現（圧縮された特徴）：$ z $


#### エンコード関数（Encoder）：

$$
z = f(x)
$$

- $ f(x) $ はニューラルネットワークで構成される関数であり、入力 $ x $ を低次元の潜在変数 $ z $ に変換します。

#### デコード関数（Decoder）：

$$
x' = g(z)
$$

- $ g(z) $ もニューラルネットワークで構成され、潜在変数 $ z $ から入力 $ x $ と似たデータを再構成します。

#### 目的関数（損失関数）：

AE の目的は、出力 $ x' $ が入力 $ x $ にできるだけ近づくように学習することです。一般的には次のような損失関数を使用します：

- **MSE（平均二乗誤差）**：

$$
\mathcal{L} = \|x - x'\|^2_2
$$

- **Binary Cross Entropy（バイナリ交差エントロピー）**：
  画像が0〜1の範囲の値を持つ場合などに使用されます。


### 応用例

- 次元削減（PCAの非線形版）
- ノイズ除去（Denoising AE）
- 異常検知（再構成誤差が大きい＝異常）
- 特徴抽出（潜在空間 $ z $ を他のタスクに利用）

### コード例

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

## DAE(Denoising Autoencode)

DAE（**Denoising Autoencoder**）は、**ノイズ除去**を目的としたオートエンコーダー（Autoencoder）の一種です。通常のオートエンコーダーとは異なり、**意図的にノイズを加えた入力データ**から、**ノイズを取り除いた元のデータを再構成する**ように学習します。

![DAE](/assert/AutoEncoder_and_VAE/DAE.png)


### 基本的な仕組み

1. **入力にノイズを加える**  
   入力データ $ x $ に対して、ある種のノイズ（例：ガウシアンノイズ、マスクノイズ）を加えて、損なわれたデータ $ \tilde{x} $ を作ります。

2. **エンコーダーで潜在表現を抽出**  
   損なわれたデータ $ \tilde{x} $ をエンコーダー関数 $ f $ に入力し、潜在ベクトル $ z $ を得ます：
   $$
   z = f(\tilde{x})
   $$

3. **デコーダーで元のデータを再構成**  
   潜在ベクトル $ z $ をデコーダー関数 $ g $ を使って復元し、元のノイズなしデータ $ x $ に近づける：
   $$
   \hat{x} = g(z)
   $$

4. **損失関数で誤差を最小化**  
   再構成された $ \hat{x} $ とオリジナルの $ x $ の間の誤差（MSEやBCEなど）を最小化することで、モデルがノイズを除去できるように学習させます。

### 数式による表現

- ノイズ入り入力：
  $$
  \tilde{x} = x + \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \sigma^2)
  $$
- エンコード：
  $$
  z = f(\tilde{x})
  $$
- デコード：
  $$
  \hat{x} = g(z)
  $$
- 損失関数(MSE)：
  $$
  \mathcal{L} = \|x - \hat{x}\|_2^2
  $$

### 実装例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ノイズを加える例（訓練時）
def add_noise(x, noise_factor=0.2):
    return x + noise_factor * torch.randn_like(x)

# 使用例
model = DenoisingAutoencoder()
x_clean = ... # clean image tensor
x_noisy = add_noise(x_clean)
x_recon = model(x_noisy)

loss = F.mse_loss(x_recon, x_clean)  # 損失計算
```

## DAE vs 普通の AE の違い

| 項目     | 普通の Autoencoder     | Denoising Autoencoder          |
| -------- | ---------------------- | ------------------------------ |
| 入力     | 生データ $ x $         | ノイズ付きデータ $ \tilde{x} $ |
| 出力     | 再構成データ $ x' $    | 元の生データ $ x $             |
| 学習目標 | データの圧縮・再構成   | ノイズ除去                     |
| 特徴     | 潜在空間に構造を捉える | ノイズに頑健な表現を学ぶ       |



## 変分オートエンコーダー（Variational Autoencoder, VAE）

VAE とは、対数尤度を最大化するように学習するオートエンコーダーのことです。
**新しいデータを生成できるのが大きな特徴です。**

![VAE](/assert/AutoEncoder_and_VAE/VAE.png)

### 直感的な理解

- **普通のAutoencoder**：入力 $ x $ → 固定値 $ z $ → 再構成 $ x' $
- **VAE(Variational Autoencoder)**：入力 $ x $ → 正規分布 $ (\mu, \sigma) $ → ノイズ$ \varepsilon \in \mathcal{N}(0,1) $を加え → $ z $ をサンプリング → 再構成 $ x' $

> **例**：顔画像を入力すると、VAE は「笑顔の強さ」「髪型」「年齢」などの特徴を表す**確率分布**として潜在空間に表現します。デコーダーはその分布からランダムに値を取り出し、**新しい顔画像を生成**できます。

### VAEの基本構造と原理

VAEは次の2つのネットワークで構成されます：

1. **推論ネットワーク（Encoder）**
   - 入力 $ x $ を与えると、潜在変数 $ z $ の近似事後分布 $ q(z|x) $ を出力。
   - 出力は平均 $ \mu $ と分散 $ \sigma^2 $。

2. **生成ネットワーク（Decoder）**
   - 潜在変数 $ z $ から入力 $ x $ に類似したデータ $ x' $ を復元。
   - 分布 $ p(x|z) $ をモデル化。

### 数式による表現

#### 1. Encoder（推論ネットワーク）

$$
q_{\phi}(z|x) = \mathcal{N}(z \mid \mu_{\phi}(x), \sigma_{\phi}^2(x))
$$

- 平均 $ \mu $ と標準偏差 $ \sigma $ をニューラルネットワークで出力。

- 潜在変数 $ z $ は以下の方法でサンプリング：
  $$
  z = \mu + \sigma \cdot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, 1)
  $$
  → この手法は「**再パラメータ化トリック**」と呼ばれます。

#### 2. Decoder（生成ネットワーク）

$$
p_{\theta}(x|z)
$$

- 一般的にはベルヌーイ分布（画像など0/1値）やガウシアン分布を使用。
- データ $ x $ を $ z $ から復元する条件付き分布。

### VAEの目的関数：ELBO（Evidence Lower Bound）

VAEの学習目標は、対数尤度 $ \log p(x) $ を最大化することですが、直接計算は困難です。そのため、**ELBO（Evidence Lower Bound）** を最大化します。

$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

この右辺が ELBO であり、次のように分解されます：

1. **再構成損失（Reconstruction Loss）**  
   $$
   \mathbb{E}_{q(z|x)}[\log p(x|z)]
   $$

   - デコーダーが出力する $ x' $ がどれだけ $ x $ に近いかを評価。
   - 実装上は BCE（Binary Cross Entropy）や MSE を使用。

2. **KL散逸（KL Divergence）**  
   $$
   D_{KL}(q(z|x) \| p(z))
   $$

   - 潜在変数の分布 $ q(z|x) $ が事前分布 $ p(z) $（通常は標準正規分布）に近づくように制約を与える。


### VAEの利点・応用

- **滑らかな潜在空間**：隣接する $ z $ 値は似たような出力を生成。
- **新規データ生成**：潜在空間からのサンプリングで新しいデータを生成可能。
- **内挿・外挿**：潜在空間上で線形補間することで、中間的なデータを生成。
- **応用分野**：
  - 画像生成（例：顔、風景）
  - 音声合成
  - 半教師あり学習
  - 異常検知（KL項 or 再構成誤差を利用）

### 実装例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mu
        self.fc22 = nn.Linear(400, 20)  # log-variance
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```


## AEとVAEの違い

| 項目           | Autoencoder                | VAE                                |
| -------------- | -------------------------- | ---------------------------------- |
| 学習方法       | 再構成誤差最小化           | ELBO最大化（再構成誤差 + KL散逸）  |
| 潜在空間       | 固定値                     | 確率分布（μ, σ）                   |
| データ生成能力 | ×                          | ◯                                  |
| 応用先         | 圧縮、ノイズ除去、異常検知 | 生成モデル、画像生成、潜在空間探索 |
