---
title: Pytorch Lightning の使い方

date: 2022-10-18 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description:
---

## 目次

---

## PyTorch Lightning

PyTorch Lightning は、PyTorch をより効率的に使えるようにする軽量ライブラリです。以下に基本的な使い方を説明します。

## インストール

まず PyTorch Lightning をインストールします:

```bash
pip install lightning
```

```conda
conda install lightning -c conda-forge
```

## モデルの定義

`LightningModule` を継承してモデルを定義します。以下は簡単な例です:

```python
import pytorch_lightning as pl

# LightningModule を継承したクラス定義
class LightningModule(L.LightningModule):
    def __init__(self, model, criterion, lr):
        super().__init__()
        self.model = model      # モデル
        self.lr = lr            # 学習率
        self.criterion = criterion  # 損失関数

    # 学習ステップの定義
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)  # 学習損失の記録
        return loss

    # 検証ステップの定義
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, _ = self.model(inputs)  # モデルによる予測
        loss = self.criterion(outputs, labels)
        # 正解率（accuracy）を計算
        acc = torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)
        self.log("val_loss", loss, prog_bar=True)  # 検証損失の記録
        self.log("val_acc", acc, prog_bar=True)    # 検証精度の記録
        return loss

    # オプティマイザと学習率スケジューラーの定義
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.lr,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=-1,
            gamma=0.1,
        )
        return [optimizer], [scheduler]
```

## データセットとデータローダーの準備

### pytorch の dataloader を使う場合

```python
# ## build dataset
train_dataset = ImageFolder(
    root=data_root / train_data_path,
    transform=train_transformer,
)
val_dataset = ImageFolder(
    root=data_root / val_data_path,
    transform=val_transformer,
)

# ## build dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
)
```

### LightningDataModule を使う場合

```python
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # データのダウンロードなど、1回だけ実行される処理
        MNIST(root='./data', train=True, download=True)
        MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        # データセットの分割や前処理を定義
        full_dataset = MNIST(root='./data', train=True, transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [55000, 5000])
        self.test_dataset = MNIST(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
```

#### 🔁 各メソッドの役割

| メソッド名           | 説明                                                                               |
| -------------------- | ---------------------------------------------------------------------------------- |
| `prepare_data()`     | データのダウンロードなど、1 度だけ実行される処理（マルチ GPU 環境でも 1 回のみ）   |
| `setup()`            | 学習/検証/テストデータセットの作成・分割を行う（分散環境では各ワーカーで呼ばれる） |
| `train_dataloader()` | 学習用のデータローダーを返す                                                       |
| `val_dataloader()`   | 検証用のデータローダーを返す                                                       |
| `test_dataloader()`  | テスト用のデータローダーを返す                                                     |
