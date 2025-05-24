---
title: timmライブラリ入門 - PyTorch Image Models (timm) ライブラリの紹介と使い方

date: 2023-10-18 12:00:00
categories: [AI]
tags:
  [Deep Learning, PyTorch, Lightning, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description: timm
---

## 目次

---

## 概要

timm (PyTorch Image Models) は Ross Wightman 氏によって開発された PyTorch ベースのコンピュータビジョンライブラリ。ImageNet クラスの画像分類モデルを中心に、物体検出、セグメンテーション、特徴抽出などのタスクをサポート。

## 主な機能

- **200+以上の事前学習済みモデル** (ResNet, EfficientNet, ViT など)
- モデルアーキテクチャの柔軟なカスタマイズ
- 特徴抽出の簡易化
- 転移学習の効率化
- モデルアンサンブル機能

## インストール

```bash
pip install timm
```

## 基本的な使い方

### モデルのロード

```python
import timm

# 事前学習済みモデルのロード
model = timm.create_model('resnet50', pretrained=True)

# カスタムクラス数に変更
model = timm.create_model('resnet50', pretrained=True, num_classes=10)

# 利用可能なモデル一覧表示
model_names = timm.list_models(pretrained=True)
```

### データ前処理
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 画像分類の訓練例 (CIFAR-10)
```python
# モデル構築
model = timm.create_model('resnet50', pretrained=True, num_classes=10)

# 損失関数と最適化手法
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練ループ
for epoch in range(num_epochs):
    for inputs, labels in trainloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

