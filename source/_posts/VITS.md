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

