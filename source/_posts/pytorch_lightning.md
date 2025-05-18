---
title: Pytorch Lightning の使い方

date: 2022-10-18 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description:
---

## 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [PyTorch Lightning](#pytorch-lightning)
- [インストール](#%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
- [モデルの定義](#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E5%AE%9A%E7%BE%A9)
    - [各メソッドの役割](#%E5%90%84%E3%83%A1%E3%82%BD%E3%83%83%E3%83%89%E3%81%AE%E5%BD%B9%E5%89%B2)
- [データセットとデータローダーの準備](#%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%81%A8%E3%83%87%E3%83%BC%E3%82%BF%E3%83%AD%E3%83%BC%E3%83%80%E3%83%BC%E3%81%AE%E6%BA%96%E5%82%99)
  - [pytorch の dataloader を使う場合](#pytorch-%E3%81%AE-dataloader-%E3%82%92%E4%BD%BF%E3%81%86%E5%A0%B4%E5%90%88)
  - [LightningDataModule を使う場合](#lightningdatamodule-%E3%82%92%E4%BD%BF%E3%81%86%E5%A0%B4%E5%90%88)
    - [各メソッドの役割](#%E5%90%84%E3%83%A1%E3%82%BD%E3%83%83%E3%83%89%E3%81%AE%E5%BD%B9%E5%89%B2-1)
- [トレーナーの作成と学習の実行](#%E3%83%88%E3%83%AC%E3%83%BC%E3%83%8A%E3%83%BC%E3%81%AE%E4%BD%9C%E6%88%90%E3%81%A8%E5%AD%A6%E7%BF%92%E3%81%AE%E5%AE%9F%E8%A1%8C)
  - [主な引数一覧](#%E4%B8%BB%E3%81%AA%E5%BC%95%E6%95%B0%E4%B8%80%E8%A6%A7)
  - [主なメソッド](#%E4%B8%BB%E3%81%AA%E3%83%A1%E3%82%BD%E3%83%83%E3%83%89)
  - [コードの例](#%E3%82%B3%E3%83%BC%E3%83%89%E3%81%AE%E4%BE%8B)

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

- ここで、`self.log("val_acc", acc, prog_bar=True)` などの `self.log` は、学習中に記録したい値を指定します。これにより、TensorBoard や CSVLogger などで確認できます。`prog_bar` を True にすると、進捗バーで確認できます。
- `configure_optimizers()` でオプティマイザと学習率スケジューラーを定義します。ここでは、AdamW を使用していますが、任意のオプティマイザや学習率スケジューラーを指定。

#### 各メソッドの役割

- `training_step` : 1 epoch あたりの学習ステップで呼ばれます。ここで、損失を計算して返します。
- `validation_step` : 1 epoch あたりの検証ステップで呼ばれます。ここで、損失と正解率を計算して返します。
- `test_step` : 1 epoch あたりのテストステップで呼ばれます。ここで、損失と正解率を計算して返します。
- `configure_optimizers` : オプティマイザと学習率スケジューラーを定義します。これは、学習開始時に一度だけ呼ばれます。

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

#### 各メソッドの役割

| メソッド名           | 説明                                                                               |
| -------------------- | ---------------------------------------------------------------------------------- |
| `prepare_data()`     | データのダウンロードなど、1 度だけ実行される処理（マルチ GPU 環境でも 1 回のみ）   |
| `setup()`            | 学習/検証/テストデータセットの作成・分割を行う（分散環境では各ワーカーで呼ばれる） |
| `train_dataloader()` | 学習用のデータローダーを返す                                                       |
| `val_dataloader()`   | 検証用のデータローダーを返す                                                       |
| `test_dataloader()`  | テスト用のデータローダーを返す                                                     |

## トレーナーの作成と学習の実行

```python
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    logger=True,
    log_every_n_steps=10,
    default_root_dir="./logs"
)

```

### 主な引数一覧

| 引数名              | 説明                                                                           |
| ------------------- | ------------------------------------------------------------------------------ |
| `max_epochs`        | 学習する最大エポック数                                                         |
| `accelerator`       | 使用するデバイス (`"auto"` / `"cpu"` / `"gpu"` / `"tpu"` など)                 |
| `devices`           | 使用するデバイス数 or 番号指定（例: `1`, `[0,1]`）                             |
| `logger`            | ロガーを使用するかどうか（`True` または TensorBoardLogger などのインスタンス） |
| `callbacks`         | コールバック関数のリスト（例: `EarlyStopping`, `ModelCheckpoint`）             |
| `log_every_n_steps` | 何ステップごとにログ出力を行うか                                               |
| `default_root_dir`  | チェックポイント保存先ディレクトリ                                             |
| `precision`         | 学習精度（`32`, `16`（FP16）, `bf16` など）                                    |
| `fast_dev_run`      | 実験用：`True` にすると 1 バッチだけ実行される                                 |

### 主なメソッド

| メソッド名           | 説明                         |
| -------------------- | ---------------------------- |
| `trainer.fit()`      | 学習＋検証を実行             |
| `trainer.validate()` | 検証のみを実行               |
| `trainer.test()`     | テストのみを実行             |
| `trainer.predict()`  | 推論を実行（予測結果を得る） |

### コードの例

```python
import lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# コールバックの追加例（早期終了、モデル保存など）
callbacks = [
    EarlyStopping(monitor="val_acc", patience=10),
    ModelCheckpoint(monitor="val_acc", save_top_k=5)
]

# Trainer のインスタンス化
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    logger=True,
    callbacks=callbacks,
    log_every_n_steps=10,
    default_root_dir="./logs"
)

# 学習 + 検証
trainer.fit(model, train_dataloader, val_dataloader)

# 検証のみ
trainer.validate(model, val_dataloader)

# テストのみ
trainer.test(model, test_dataloader)

```

- ここで、EarlyStopping とは、モデルのパラメータを保存するためのコールバックです。EarlyStopping は、指定したエポック数以上にモデルのパラメータが更新されない場合、モデルのパラメータを保存します。つまり、EarlyStopping は、モデルのパラメータを保存するためのコールバックです。
- ここで、ModelCheckpoint とは、モデリングのパラメータを保存するためのコールバックです。ModelCheckpoint は、指定したエポック数以上にモデルのパラメータが更新される場合、モデルのパラメータを保存します。
- `monitor="val_acc"` は、EarlyStopping と ModelCheckpoint が監視する指標を指定します。"val_acc"は、検証精度を指し、`LightningModule`に定義した`validation_step()`に`self.log("val_acc", acc, prog_bar=True)`で記録した検証精度を監視します。
- `trainer.fit`は、モデルの`training_step`を呼び、もし`val_dataloader`が指定されていれば`validation_step`も呼びます。つまり、モデルの`training_step`と`validation_step`が定義べきです。
- `trainer.validate`は、モデルの`validation_step`を呼びます。
- `trainer.test`は、モデルの`test_step`を呼びます。
