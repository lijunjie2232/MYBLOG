---
title: pytorch 実践
date: 2022-8-3 15:10:00
categories: [AI]
tags: [deep learning, pytorch, python]
lang: ja
---

この記事は、画像の表情認識に例して、Pytorchを用いた実践な任務を解説する。

- [Pytorchインストール](#pytorchインストール)
- [Tips](#tips)
- [データセット情報](#データセット情報)
- [基本的な任務](#基本的な任務)


## Pytorchインストール

[アーカイブ　リース](https://pytorch.org/get-started/previous-versions)

- use pip: `pip3 install torch torchvision torchaudio`　(CUDA12.6 by default)

最新のpytorchはcondaでインストールのをできません。

## Tips

## データセット情報

今回のデータセットは、Kaggleのデータセットを利用します。

link：[https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset)

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("aadityasinghal/facial-expression-dataset")

print("Path to dataset files:", path)
```


## 基本的な任務

