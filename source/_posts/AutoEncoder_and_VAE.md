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

### AEとの違い

| 項目           | Autoencoder                | VAE                                |
| -------------- | -------------------------- | ---------------------------------- |
| 学習方法       | 再構成誤差最小化           | ELBO最大化（再構成誤差 + KL散逸）  |
| 潜在空間       | 固定値                     | 確率分布（μ, σ）                   |
| データ生成能力 | ×                          | ◯                                  |
| 応用先         | 圧縮、ノイズ除去、異常検知 | 生成モデル、画像生成、潜在空間探索 |


### 実装例
```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mu
        self.fc22 = nn.Linear(400, 20)  # logvar
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```