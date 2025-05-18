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