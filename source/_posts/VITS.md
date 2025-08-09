---
title: VITS論文の解読
date: 2024-4-9 10:17:00
categories: [AI]
tags: [Deep Learning, 機械学習, AI, 人工知能, 深層学習, GAN, VAE, 音声変換, 音声合成, 音声認識, 音声処理, TTS, ]
lang: ja
description: VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）は、変分推論（variational inference）、正規化フロー（normalizing flows）、および敵対的学習を組み合わせた、表現力の高い音声合成モデルです。VITSは、音声合成における音響モデルとボコーダーをスペクトログラムではなく潜在変数で連結し、潜在変数上で確率モデリングを行い、確率的デュレーション予測器を利用することで、合成音声の多様性を向上させています。同じテキストを入力しても、異なるトーンやリズムの音声を合成することが可能になります。

---

目次

---

VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）は、変分推論（variational inference）、正規化フロー（normalizing flows）、および敵対的学習を組み合わせた、表現力の高い音声合成モデルです。VITSは、音声合成における音響モデルとボコーダーをスペクトログラムではなく潜在変数で連結し、潜在変数上で確率モデリングを行い、確率的デュレーション予測器を利用することで、合成音声の多様性を向上させています。同じテキストを入力しても、異なるトーンやリズムの音声を合成することが可能になります。

![VITS Architecture](/assert/VITS/arch.png)

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


##　トレーニング

![VITS Architecture](/assert/VITS/train.png)


