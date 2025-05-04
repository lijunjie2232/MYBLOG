---
title: pytorch 実践
date: 2022-8-12 15:10:00
categories: [AI]
tags: [deep learning, pytorch, python]
lang: ja
---

この記事は、画像の顔表情認識に例をして、Pytorchを用いた実践な任務を解説する。

- [Pytorchインストール](#pytorchインストール)
- [Tips](#tips)
- [基本的な任務](#基本的な任務)
- [データセット情報](#データセット情報)
- [データセットtransoforms](#データセットtransoforms)
  - [主な特徴](#主な特徴)
  - [よく使われるクラス一覧](#よく使われるクラス一覧)
  - [使用例](#使用例)
  - [適用方法の例（`ImageFolder`）](#適用方法の例imagefolder)


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

## データセットtransoforms

`torchvision.transforms` は、PyTorchで画像データを処理するための一般的な変換（前処理・水増し）を行うモジュールです。主に `torchvision.datasets` と組み合わせて使用され、画像をテンソルに変換したり、正規化や拡張などの操作を行います。

### 主な特徴
- **画像変換**：画像をテンソルに変換します。
- **正規化**：画像のピクセル値を特定の範囲に正規化します。
- **データ水増し**（Data Augmentation）：ランダムな回転・反転などを行い学習データを多様化します。
- **パイプライン構築**：複数の変換を順番に適用する処理を簡単に構成できます（`Compose` を使用）。


### よく使われるクラス一覧

| クラス名                                         | 機能                                                          |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `ToTensor()`                                     | PIL形式の画像を PyTorch の Tensor に変換（0〜255 → 0.0〜1.0） |
| `Normalize(mean, std)`                           | テンソル画像に対して平均 `mean`、標準偏差 `std` で正規化する  |
| `Resize(size)`                                   | 画像を指定されたサイズにリサイズする                          |
| `CenterCrop(size)`                               | 画像の中心部分を指定されたサイズに切り出す                    |
| `RandomHorizontalFlip(p=0.5)`                    | 画像を確率 `p` で左右反転する（デフォルトは 50%）             |
| `RandomRotation(degrees)`                        | 画像をランダムに回転させる（角度は `degrees` 以内）           |
| `RandomAffine(degrees, translate, scale, shear)` | 画像をランダムに回転・平行移動・拡大縮小・傾斜させる          |
| `RandomCrop(size)`                               | 画像をランダムに切り出す                                      |

### 使用例

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),         # 256x256にリサイズ
    transforms.CenterCrop(224),     # 中心を224x224に切り抜き
    transforms.ToTensor(),          # Tensorに変換
    transforms.Normalize(           # 正規化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

このようにして作成した `transform` は、`ImageFolder` や自作の `Dataset` クラスで画像に適用されます。


### 適用方法の例（`ImageFolder`）

```python
from torchvision import datasets

dataset = datasets.ImageFolder(root='path/to/data', transform=transform)
```
