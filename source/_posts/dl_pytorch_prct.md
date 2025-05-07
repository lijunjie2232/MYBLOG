---
title: pytorch 実践
date: 2022-8-12 10:15:00
categories: [AI]
tags: [deep learning, pytorch, python]
lang: ja
---


この記事は、画像の顔表情認識に例をして、Pytorchを用いた実践な任務を解説する。

codeの例：[main.ipynb](https://colab.research.google.com/github/lijunjie2232/MYBLOG/blob/master/source/assert/dl_pytorch_prct/main.ipynb)

プロジェクトアドレス：[https://github.com/lijunjie2232/emotion_analyse_pytorch](https://github.com/lijunjie2232/emotion_analyse_pytorch)


## 目次
- [目次](#目次)
- [Pytorchインストール](#pytorchインストール)
  - [requirements](#requirements)
- [Tips](#tips)
- [基本的な任務](#基本的な任務)
- [データセット情報](#データセット情報)
- [データのtransoforms](#データのtransoforms)
  - [主な特徴](#主な特徴)
  - [よく使われるクラス一覧](#よく使われるクラス一覧)
  - [使用例](#使用例)
  - [適用方法の例（`ImageFolder`）](#適用方法の例imagefolder)
- [モーデルの作成](#モーデルの作成)


## Pytorchインストール

[アーカイブ　リース](https://pytorch.org/get-started/previous-versions)

- use pip: `pip3 install torch torchvision torchaudio`　(CUDA12.6 by default)

最新のpytorchはcondaでインストールのをできません。

### requirements
```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install kagglehub
```

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

## データのtransoforms

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

## モーデルの作成

```python
class EmotionNet(nn.Module):
    def __init__(self, c1: int = 3, nc: int = 7):
        """
        Initialize classification model

        Args:
            c1 (int): Input channel size
            nc (int): Number of output classes
        """
        super().__init__()

        # Calculate scaling parameters from config

        # Build backbone
        self.backbone = nn.ModuleList(
            [
                # 0-P1/2
                Conv(c1=3, c2=16, k=3, s=2),
                # 1-P2/4
                Conv(c1=16, c2=32, k=3, s=2),
                # 2-C3k2 block
                C3k2(c1=32, c2=64, n=1, e=0.25),
                # 3-P3/8
                Conv(c1=64, c2=128, k=3, s=2),
                # 4-C3k2 block
                C3k2(c1=128, c2=128, n=1, e=0.25),
                # 5-P4/16
                Conv(c1=128, c2=128, k=3, s=2),
                # 6-A2C2f block
                A2C2f(c1=128, c2=128, n=2, a2=True, area=4, e=0.5),
                # 7-P5/32
                Conv(c1=128, c2=256, k=3, s=2),
                # 8-A2C2f block
                A2C2f(c1=256, c2=256, n=2, a2=True, area=1, e=0.5),
            ]
        )

        # Build classification head
        self.classify = Classify(256, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Pass through backbone
        for layer in self.backbone:
            x = layer(x)

        # Pass through classification head
        x = self.classify(x)

        return x
```

つつく．．．