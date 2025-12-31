---
title: VITS論文の解読
date: 2024-4-9 10:17:00
categories: [AI]
tags: [Deep Learning, 機械学習, AI, 人工知能, 深層学習, GAN, VAE, 音声変換, 音声合成, 音声認識, 音声処理, TTS, ]
lang: ja
description: VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）は、変分推論（variational inference）、正規化フロー（normalizing flows）、および敵対的学習を組み合わせた、表現力の高い音声合成モデルです。VITSは、音声合成における音響モデルとボコーダーをスペクトログラムではなく潜在変数で連結し、潜在変数上で確率モデリングを行い、確率的デュレーション予測器を利用することで、合成音声の多様性を向上させています。同じテキストを入力しても、異なるトーンやリズムの音声を合成することが可能になります。

---

目次

- [VITS](#vits)
- [主な貢献点](#%E4%B8%BB%E3%81%AA%E8%B2%A2%E7%8C%AE%E7%82%B9)
- [モデルアーキテクチャ](#%E3%83%A2%E3%83%87%E3%83%AB%E3%82%A2%E3%83%BC%E3%82%AD%E3%83%86%E3%82%AF%E3%83%81%E3%83%A3)
- [トレーニング](#%E3%83%88%E3%83%AC%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0)
  - [変分推論（Variational Inference）](#%E5%A4%89%E5%88%86%E6%8E%A8%E8%AB%96variational-inference)
  - [重建損失（Reconstruction Loss）](#%E9%87%8D%E5%BB%BA%E6%90%8D%E5%A4%B1reconstruction-loss)
  - [KLダイバージェンス（KL Divergence）](#kl%E3%83%80%E3%82%A4%E3%83%90%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%82%B9kl-divergence)
  - [アライメント推定（Alignment Estimation）](#%E3%82%A2%E3%83%A9%E3%82%A4%E3%83%A1%E3%83%B3%E3%83%88%E6%8E%A8%E5%AE%9Aalignment-estimation)
    - [単調アライメント探索（Monotonic Alignment Search: MAS）](#%E5%8D%98%E8%AA%BF%E3%82%A2%E3%83%A9%E3%82%A4%E3%83%A1%E3%83%B3%E3%83%88%E6%8E%A2%E7%B4%A2monotonic-alignment-search-mas)
    - [テキストからの持続時間予測](#%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%8B%E3%82%89%E3%81%AE%E6%8C%81%E7%B6%9A%E6%99%82%E9%96%93%E4%BA%88%E6%B8%AC)
  - [敵対的学習（Adversarial Training）](#%E6%95%B5%E5%AF%BE%E7%9A%84%E5%AD%A6%E7%BF%92adversarial-training)
  - [理解](#%E7%90%86%E8%A7%A3)
- [参考文献](#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)


---

## VITS

![VITS Architecture](/assert/VITS/arch.png)

VITS([Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech（ICML 2021）](https://proceedings.mlr.press/v139/kim21f/kim21f.pdf))は、変分推論（variational inference）、正規化フロー（normalizing flows）、および敵対的学習を組み合わせた、表現力の高い音声合成モデルです。VITSは、音声合成における音響モデルとボコーダーをスペクトログラムではなく潜在変数で連結し、潜在変数上で確率モデリングを行い、確率的デュレーション予測器を利用することで、合成音声の多様性を向上させています。同じテキストを入力しても、異なるトーンやリズムの音声を合成することが可能になります。

論点アドレス：https://proceedings.mlr.press/v139/kim21f/kim21f.pdf

コードアドレス：https://github.com/jaywalnut310/vits

demoアドレス：https://jaywalnut310.github.io/vits

## 主な貢献点

1. **並列エンドツーエンドTTS手法の提案**
   - 従来の2段階モデルよりも自然な音声を生成可能
   - 全体が一つのネットワークで構成され、効率的な学習と推論が実現

2. **変分推論の強化**
   - 正規化フローと敵対的訓練プロセスにより、生成モデルの表現能力を向上
   - より高品質な音声合成が可能に

3. **確率的持続時間予測器の導入**
   - 入力テキストから異なるリズムを持つ音声を合成可能
   - 自然なone-to-many関係を表現（同じテキストを異なるトーンとリズムで複数の方法で発話可能）

## モデルアーキテクチャ

![VITS Architecture](/assert/VITS/arch.png)

- **音声情報入力**: メルスペクトログラム: 音声データの短時フーリエ変換から得られる: `Posterior Encoder` (WaveNetモデル) で潜在変数zの事後分布を取得

- **Decoder**: HiFiGANのジェネレーターと同等: 生の音声データを直接生成し、中間特徴量(メルスペクトログラム)の生成やボコーダーの訓練を不要に

- **Flow**: 事後分布qから事前分布pへの変換関数fとして機能: 信号の表現能力を強化

- **Monotonic Alignment Search**: 音声とテキストのベクトル系列の長さを一致させる

- **Text Encoder**: 多頭注意transformer構造を使用: 入力テキストを音素系列に変換し、さらにベクトル系列に変換

- **Projection**: ベクトル系列に基づいて$p_{θ}(z|c)$の事前分布パラメータ($μ_{θ}$, $σ_{θ}$)を取得

- **Stochastic Duration Predictor**: Flowモデルを使用して周期予測を行う

- **Discriminator**: 生成されたRaw Waveformと実際の音声波形を識別する一組の識別器

## トレーニング

![VITS Architecture](/assert/VITS/train.png)


### 変分推論（Variational Inference）

VITSの生成器は、変分下界（ELBO: Evidence Lower Bound）を最大化する条件付きVAEとして考えることができます。これは以下の目的関数を最適化することを意味します：

$$
L_{vae} = L_{recon} + L_{kl} + L_{dur} + L_{adv} + L_{fm}(G)
$$

### 重建損失（Reconstruction Loss）

訓練時には、モデルの学習を導くためにメルスペクトログラムを生成します。重建損失の目標サンプルには、生の波形ではなくメルスペクトログラムが使用されます：

$$
L_{recon}=||x_{mel}-\hat{x}_{mel}||_1
$$

推論時にはメルスペクトログラムの生成は不要で、この損失は訓練中のみ計算に使用されます。

### KLダイバージェンス（KL Divergence）

事前エンコーダーの入力$c$には、テキストから生成された音素$c_{text}$と音素・潜在変数間のアライメント$A$が含まれます。アライメントとは、$|c_{text}|×|z|$サイズの厳密単調注意行列で、各音素の発音時間を表します。KLダイバージェンスは以下の通りです：

$$
L_{kl}=log\ q_{\phi}(z|x_{lin})-log\ p_{\theta}(z|c_{text},A)
$$

ここで：
- $q_{\phi}(z|x)$は線形スペクトログラム$x$が与えられたときの潜在変数$z$の事後分布
- $p_{\theta}(z|c)$は条件$c$が与えられたときの潜在変数$z$の事前分布
- 潜在変数$z$は以下に従います：$z\sim q_{\phi}(z|x_{lin})=N(z;\mu_{\phi}(x_{lin}),\sigma_{\phi}(x_{lin}))$


より高解像度の情報を事後エンコーダー$q_{\phi}$に提供するために、メルスペクトログラムではなく線形スペクトログラムを入力として使用します。よりリアルなサンプルを生成するために、事前分布の表現能力を向上させることが重要であるため、正規化フローを導入して、テキストエンコーダーが生成する単純な分布と潜在変数$z$に対応する複雑な分布の間で可逆変換を行います：

$$
p_{\theta}(z|c)=N(f_{\theta}(z);\mu_{\theta}(c),\sigma_{\theta}(c))|det\frac{\partial f_{\theta}(z)}{\partial z}|
$$

ここで入力$c$はアップサンプリングされたエンコーダー出力です：$c=[c_{text},A]$


### アライメント推定（Alignment Estimation）

訓練時には「アライメント」の真のラベルがないため、訓練フェーズの各イテレーションでテキストと音声の間のアライメントを推定する必要があります。

#### 単調アライメント探索（Monotonic Alignment Search: MAS）

テキストと音声の間のアライメント$A$を推定するために、VITSはGlow-TTSと同様の単調アライメント探索（MAS）手法を採用しています。この手法は、正規化フロー$f$でパラメータ化されたデータの対数尤度を最大化する最適なアライメントパスを探すことを試みます：

$$
A=\mathop{argmax}\limits_{\hat A}\mathop{log}p(x|c_{text},\hat A)\\=\mathop{argmax}\limits_{\hat A}\mathop{log}N(f(x);\mu(c_{text},\hat A),\sigma(c_{text},\hat A))
$$


MASによって得られる最適アライメントは単調かつスキップなしである必要がありますが、VITSの最適化目標は決定論的な潜在変数$z$の対数尤度ではなくELBOであるため、MASを直接VITSに適用することはできません。そのため、ELBOを最大化する最適なアライメントパスを探すためにMASを若干変更しています：

$$
\mathop{argmax}\limits_{\hat A}\mathop{log}p_{\theta}(x_{mel}|z)-\mathop{log}\frac{q_\phi(z|x_{lin})}{p_\theta(z|c_{text},\hat A)}=\mathop{argmax}\limits_{\hat A}\mathop{log}p_\theta(z|c_{text},\hat A)=\mathop{log}N(f_{\theta}(z);\mu_{\theta}(c_{text},\hat A),\sigma_{\theta}(c_{text},\hat A))
$$

#### テキストからの持続時間予測 

確率的持続時間予測器はフローに基づく生成モデルであり、持続時間シーケンスと同じ時間解像度と次元を持つ確率変数$u$と$v$を導入します。近似事後分布$q_{\phi}(u,v|d,c_{text})$を使用してこれらの変数をサンプリングし、訓練目標は音素持続時間の対数尤度の変分下界です：

$$
\mathop{log}p_\theta(d|c_{text})\geq \mathbb{E}_{q_{\phi}(u,v|d,c_{text})}[\mathop{log}\frac{p_{\theta}(d-u,v|c_{text})}{q_{\phi}(u,v|d,c_{text})}]
$$

訓練の時には、他のモジュールに影響を与えるのを防ぐために、確率的持続時間予測器からの勾配伝播を遮断します。音素の持続時間は、確率的持続時間予測器の可逆変換を通じてランダムノイズからサンプリングされ、その後整数値に変換されます。

### 敵対的学習（Adversarial Training）

判別器$D$を導入して、出力がデコーダー$G$からの出力か、実際の波形$y$かを判断します。VITSは2種類の損失関数を使用します：

1. 敵対的学習用の最小二乗損失関数：
   $$
   L_{adv}(D)=\mathbb{E}_{(y,z)}[(D(y)-1)^2+(D(G(z)))^2]
   $$
   $$
   L_{adv}(G)=\mathbb{E}_z[(D(G(z))-1)^2]
   $$

2. 生成器に特別に適用される特徴マッチング損失（feature-matching loss）：
   $$
   L_{fm}(G)=\mathbb{E}_{(y,z)}[\sum_{l=1}^{T}\frac{1}{N_l}||D^l(y)-D^l(G(z))||_1]
   $$

ここで$T$は判別器の層数、$D^l$は第$l$層の判別器出力特徴マップ、$N_l$は特徴マップの数です。特徴マッチング損失は、判別器の中間層出力を制約する再構成損失と見なすことができます。

### 理解

この訓練プロセスにより、VITSは以下のような特徴を持つことができます：

1. **高品質な音声合成**: VAEとGANの結合により、自然でリアルな音声生成が可能
2. **多様性の向上**: 潜在変数と確率的持続時間予測により、同じテキストから異なるトーンやリズムの音声を生成可能
3. **エンドツーエンド学習**: 伝統的な2段階モデルよりも効率的な学習と推論が実現
4. **自己教師あり学習**: アライメントラベルなしでの学習が可能

## 参考文献
[Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech（ICML 2021）](https://proceedings.mlr.press/v139/kim21f/kim21f.pdf)

[细读经典：VITS，用于语音合成带有对抗学习的条件变分自编码器](https://zhuanlan.zhihu.com/p/419883319)