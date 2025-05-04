---
title: pytorch 実践
date: 2022-8-3 15:10:00
categories: [AI]
tags: [deep learning, pytorch, python]
lang: ja
---

この記事は、画像の顔表情認識に例をして、Pytorchを用いた実践な任務を解説する。

- [Pytorchインストール](#pytorchインストール)
- [Tips](#tips)
- [基本的な任務](#基本的な任務)
- [データセット情報](#データセット情報)


## Pytorchインストール

[アーカイブ　リース](https://pytorch.org/get-started/previous-versions)

- use pip: `pip3 install torch torchvision torchaudio`　(CUDA12.6 by default)

最新のpytorchはcondaでインストールのをできません。

## Tips


## 基本的な任務

要するに、画像のファイルには顔がある、その画像を読み込んて、pytorchモデルで顔表情を判断します。

1. データ準備と画像前処理
2. モデルアーキテクチャ設計
3. モデルトレーニング
4. モデル推論


## データセット情報

今回のデータセットは、Kaggleのデータセットを利用します。

link：[https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset)

ラベル: 7種類の感情（angry, disgust, fear, happy, neutral, sad, surprise）

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("aadityasinghal/facial-expression-dataset")

print("Path to dataset files:", path)
```
