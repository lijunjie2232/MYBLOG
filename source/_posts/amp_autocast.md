---
title:

date: 2022-10-8 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description:
---

## 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [AMP（Automatic Mixed Precision）とは](#ampautomatic-mixed-precision%E3%81%A8%E3%81%AF)
- [`torch.cuda.amp.autocast` 機能概要](#torchcudaampautocast-%E6%A9%9F%E8%83%BD%E6%A6%82%E8%A6%81)
  - [役割](#%E5%BD%B9%E5%89%B2)
  - [基本的な使い方](#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E4%BD%BF%E3%81%84%E6%96%B9)
- [`autocast` の原理](#autocast-%E3%81%AE%E5%8E%9F%E7%90%86)
- [`GradScaler` との併用](#gradscaler-%E3%81%A8%E3%81%AE%E4%BD%B5%E7%94%A8)
  - [なぜ必要か](#%E3%81%AA%E3%81%9C%E5%BF%85%E8%A6%81%E3%81%8B)
  - [使用方法](#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)
- [利点（メリット）](#%E5%88%A9%E7%82%B9%E3%83%A1%E3%83%AA%E3%83%83%E3%83%88)
- [注意点（デメリット）](#%E6%B3%A8%E6%84%8F%E7%82%B9%E3%83%87%E3%83%A1%E3%83%AA%E3%83%83%E3%83%88)
- [一般的な問題と対処法](#%E4%B8%80%E8%88%AC%E7%9A%84%E3%81%AA%E5%95%8F%E9%A1%8C%E3%81%A8%E5%AF%BE%E5%87%A6%E6%B3%95)
  - [Loss が NaN になるケース](#loss-%E3%81%8C-nan-%E3%81%AB%E3%81%AA%E3%82%8B%E3%82%B1%E3%83%BC%E3%82%B9)
  - [モデルが深すぎて勾配消失](#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%8C%E6%B7%B1%E3%81%99%E3%81%8E%E3%81%A6%E5%8B%BE%E9%85%8D%E6%B6%88%E5%A4%B1)
- [サンプルコード](#%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E3%82%B3%E3%83%BC%E3%83%89)
  - [基本的な AMP 訓練ループ](#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA-amp-%E8%A8%93%E7%B7%B4%E3%83%AB%E3%83%BC%E3%83%97)
- [まとめ](#%E3%81%BE%E3%81%A8%E3%82%81)
- [参考文献](#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)

---

## AMP（Automatic Mixed Precision）とは

- **定義**:  
  モデルの計算を**FP32（単精度）**と**FP16（半精度）**で効率的に切り替えて実行することで、**パフォーマンス向上**と**メモリ削減**を実現する技術。
- **目的**:
  - 訓練速度の向上
  - GPU メモリ使用量の削減
  - 数値精度の維持

## `torch.cuda.amp.autocast` 機能概要

### 役割

- 推論・訓練中の演算に対して、**自動的に適切な精度（FP16/FP32）を選択して実行**。
- 特に数値安定性が重要な部分は FP32 を使用し、性能が重要な部分は FP16 を使用。

### 基本的な使い方

```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
    loss = loss_fn(output, target)
```

## `autocast` の原理

- **内部処理フロー**:

  1. 算子呼び出し時に、入力テンソルの型に基づいて**自動で精度変換**。
  2. FP16 で安全な演算のみ実行、不安定な演算は FP32 で実行。
  3. 出力は必要に応じて元の精度に戻す。

- **適用範囲**:
  - 多くの PyTorch 演算（例: `Linear`, [Conv](file://d:\code\MYBLOG\source\assert\dl_pytorch_prct\layers.py#L16-L76), `Matmul`）が対象。
  - GPU でのみ利用可能（CUDA サポートが必要）。

## `GradScaler` との併用

### なぜ必要か

- FP16 では**勾配消失や inf/nan**が発生しやすいため、数値安定性を確保するために使用。
- 勾配を一定倍率（scale）で拡大して保存し、更新前に縮小。

### 使用方法

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():    # 自動精度変換を有効化する。
        output = model(data)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward() # 自動精度変換と勾配スケーリングを行う。
    scaler.step(optimizer) # 勾配を更新する。
    scaler.update() # 勾配スケーリングを更新する。
```

## 利点（メリット）

| 項目           | 内容                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| **高速化**     | FP16 演算により、特に NVIDIA Ampere アーキテクチャ以降の GPU で顕著に高速化。 |
| **メモリ削減** | 半精度使用により、モデルサイズやバッチサイズの上限が緩和。                    |
| **自動管理**   | 手動でのデータ型指定不要で、開発負荷が軽減される。                            |
| **精度保持**   | クリティカルな演算は FP32 で行われるため、精度低下が最小限。                  |

## 注意点（デメリット）

| 問題点               | 対応策                                                         |
| -------------------- | -------------------------------------------------------------- |
| **ハードウェア依存** | FP16 対応 GPU（例: NVIDIA Volta 以降）が必要。                 |
| **NaN/Inf 問題**     | `GradScaler` を使用して勾配スケーリングを行う。                |
| **特定演算の非対応** | 一部の演算は FP16 未対応の場合あり（ドキュメント参照）。       |
| **デバッグ難易度**   | 精度が自動選択されるため、中間出力の確認が複雑化する場合あり。 |

## 一般的な問題と対処法

### Loss が NaN になるケース

- **原因**:
  - FP16 による数値誤差
  - log(0) や除算ゼロなど数学的エラー
- **解決策**:
  - `GradScaler` を必ず併用
  - 安全な値で clamp（例: `x.clamp(min=1e-8)`）

### モデルが深すぎて勾配消失

- **対処法**:
  - 層ごとの勾配クリッピング
  - 活性化関数や正則化手法を見直し

## サンプルコード

### 基本的な AMP 訓練ループ

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## まとめ

| 項目           | 内容                                                   |
| -------------- | ------------------------------------------------------ |
| **AMP**        | FP32 と FP16 を自動で切り替える技術                    |
| **autocast**   | 自動で演算精度を最適化するコンテキストマネージャー     |
| **GradScaler** | FP16 勾配のスケーリングを行い、数値不安定性を回避      |
| **推奨環境**   | NVIDIA GPU（Volta 以降）、CUDA 対応アプリケーション    |
| **主な利点**   | パフォーマンス向上、メモリ節約、自動管理による簡潔さ   |
| **注意点**     | NaN/Inf 問題、一部演算の FP16 非対応、デバッグの複雑化 |

## 参考文献

- [PyTorch Automatic Mixed Precision Docs](https://pytorch.org/docs/stable/amp.html)
- [Mixed Precision Training Paper (2017)](https://arxiv.org/abs/1710.03740)
