---
title: pytorch 実践（二）
date: 2022-8-12 11:15:00
categories: [AI]
tags:
  [
    Deep Learning,
    PyTorch,
    Python,
    Computer Vision,
    機械学習,
    AI,
    人工知能,
    深層学習,
    画像処理,
    画像認識,
    表情認識,
  ]
lang: ja
---

この記事では、画像を用いた顔表情認識を例に、PyTorch を使った実践的なタスクの開発プロセスを一から解説します。
第二部となる今回は、PyTorch を使った実装的な技術ポイントを紹介します。

code の例：[main.ipynb](https://colab.research.google.com/github/lijunjie2232/MYBLOG/blob/master/source/assert/dl_pytorch_prct/main.ipynb)

プロジェクトアドレス：[https://github.com/lijunjie2232/emotion_analyse_pytorch](https://github.com/lijunjie2232/emotion_analyse_pytorch)

## 目次

- [目次](#目次)
- [Tips](#tips)
- [基本的な任務](#基本的な任務)
- [データの transoforms](#データの-transoforms)
  - [主な特徴](#主な特徴)
  - [よく使われるクラス一覧](#よく使われるクラス一覧)
  - [使用例](#使用例)
  - [適用方法の例（`ImageFolder`）](#適用方法の例imagefolder)

## Tips

## 基本的な任務

要するに、画像のファイルには顔がある、その画像を読み込んて、pytorch モデルで顔表情を判断します。

1. データ準備と画像前処理
2. モデルアーキテクチャ設計
3. モデルトレーニング
4. モデル推論

## データの transoforms

`torchvision.transforms` は、PyTorch で画像データを処理するための一般的な変換（前処理・水増し）を行うモジュールです。主に `torchvision.datasets` と組み合わせて使用され、画像をテンソルに変換したり、正規化や拡張などの操作を行います。

### 主な特徴

- **画像変換**：画像をテンソルに変換します。
- **正規化**：画像のピクセル値を特定の範囲に正規化します。
- **データ水増し**（Data Augmentation）：ランダムな回転・反転などを行い学習データを多様化します。
- **パイプライン構築**：複数の変換を順番に適用する処理を簡単に構成できます（`Compose` を使用）。

### よく使われるクラス一覧

| クラス名                                         | 機能                                                           |
| ------------------------------------------------ | -------------------------------------------------------------- |
| `ToTensor()`                                     | PIL 形式の画像を PyTorch の Tensor に変換（0〜255 → 0.0〜1.0） |
| `Normalize(mean, std)`                           | テンソル画像に対して平均 `mean`、標準偏差 `std` で正規化する   |
| `Resize(size)`                                   | 画像を指定されたサイズにリサイズする                           |
| `CenterCrop(size)`                               | 画像の中心部分を指定されたサイズに切り出す                     |
| `RandomHorizontalFlip(p=0.5)`                    | 画像を確率 `p` で左右反転する（デフォルトは 50%）              |
| `RandomRotation(degrees)`                        | 画像をランダムに回転させる（角度は `degrees` 以内）            |
| `RandomAffine(degrees, translate, scale, shear)` | 画像をランダムに回転・平行移動・拡大縮小・傾斜させる           |
| `RandomCrop(size)`                               | 画像をランダムに切り出す                                       |

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
## データ変換の構築
train_transformer = transforms.Compose(
    [
        transforms.Resize((256, 256)),         # 画像を256x256にリサイズ
        transforms.RandomCrop(224),           # 224x224の範囲でランダムに切り抜き
        transforms.RandomHorizontalFlip(),    # 水平方向にランダムに反転（データ拡張）
        transforms.RandomRotation(degrees=15),# 最大15度の範囲でランダムに回転
        transforms.RandomVerticalFlip(),      # 垂直方向にランダムに反転
        transforms.ToTensor(),                # PIL画像をテンソルに変換
        transforms.Normalize(                 # 正規化
            mean=[0.485, 0.456, 0.406],       # ImageNetの平均値を使用
            std=[0.229, 0.224, 0.225],        # ImageNetの標準偏差を使用
        ),
    ]
)
val_transformer = transforms.Compose(
    [
        transforms.Resize((256, 256)),         # 画像を256x256にリサイズ
        transforms.CenterCrop(224),           # 中心を基準に224x224に切り抜き
        transforms.ToTensor(),                # PIL画像をテンソルに変換
        transforms.Normalize(                 # 正規化
            mean=[0.485, 0.456, 0.406],       # ImageNetの平均値を使用
            std=[0.229, 0.224, 0.225],        # ImageNetの標準偏差を使用
        ),
    ]
)

## データセットの構築
train_dataset = ImageFolder(
    root=data_root / "train" / "train",       # 学習用データのルートディレクトリ
    transform=train_transformer,              # 学習用の変換を適用
)
val_dataset = ImageFolder(
    root=data_root / "test" / "test",         # 検証用データのルートディレクトリ
    transform=val_transformer,                # 検証用の変換を適用
)
```

つつく．．．
