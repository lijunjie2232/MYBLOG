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
- [モデルの保存と読み込み](#モデルの保存と読み込み)
  - [Pytorch モデルの保存と読み込み](#pytorch-モデルの保存と読み込み)
  - [Pytorch 状態辞書の保存と読み込み（推奨）](#pytorch-状態辞書の保存と読み込み推奨)
  - [ベストプラクティス](#ベストプラクティス)
  - [主な注意点](#主な注意点)
  - [実践的な保存方法](#実践的な保存方法)
    - [チェックポイントとして保存](#チェックポイントとして保存)
    - [チェックポイントから読み込み](#チェックポイントから読み込み)
    - [主な注意点](#主な注意点-1)
- [AMP を使った学習](#amp-を使った学習)
  - [概要](#概要)
  - [優位性](#優位性)
  - [使用方法](#使用方法)
  - [注意点](#注意点)

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
        transforms.Resize((224, 224)),         # 画像を224x224にリサイズ
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

## モデルの保存と読み込み

### Pytorch モデルの保存と読み込み

```python
# モデル全体（構造＋パラメータ）を保存
torch.save(model, 'model.pth')

# モデル全体を読み込み
model = torch.load('model.pth')
model.eval()  # 推論モードに設定
```

### Pytorch 状態辞書の保存と読み込み（推奨）

```python
# パラメータのみ保存
torch.save(model.state_dict(), 'model_state.pth')

# パラメータを読み込み（事前にモデル定義が必要）
model = MyModelClass()  # モデルクラスのインスタンス化
model.load_state_dict(torch.load('model_state.pth'))
model.eval()
```

### ベストプラクティス

- ファイル拡張子: `.pth` または `.pt` を使用
- 推論時は必ず `model.eval()` を呼び出す
- デバイス指定の例:
  ```python
  # CPUで読み込み
  model.load_state_dict(torch.load('model_state.pth', map_location='cpu'))
  ```

### 主な注意点

- **`state_dict` の利点**: モデル構造の変更に柔軟に対応可能（バージョン管理に適す）
- **モデル全体保存の制約**: 再現性に依存（同じコード環境でしか使えない）

### 実践的な保存方法

#### チェックポイントとして保存

トレーニング状態を再開するために、モデルとオプティマイザの状態を一緒に保存する場合：

```python
# チェックポイント保存例
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

#### チェックポイントから読み込み

保存したチェックポイントを読み込み、トレーニングを再開する場合：

```python
# チェックポイント読み込み
checkpoint = torch.load('checkpoint.pth')

# モデルとオプティマイザの状態を復元
model = MyModelClass()  # 事前にモデル定義が必要
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters())  # オプティマイザを再構築
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# その他の情報も復元可能
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

#### 主な注意点

- **オプティマイザの再構築**:  
  オプティマイザの状態を読み込むには、事前に同じアルゴリズム（例: `Adam`）とパラメータでオプティマイザを再構築する必要があります。
- **デバイス指定**:  
  `map_location='cpu'`で保存したモデルを CPU で読み込むと、GPU の RAM を節約できる：

  ```python
  model.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))
  ```

- **strict=False オプション**:  
  モデル構造が変更された場合に部分的に読み込み可能にする：

  ```python
  model.load_state_dict(checkpoint['model_state_dict'], strict=False)
  ```

- **学習再開時の注意**:  
  読み込み後は `model.train()` を呼び出し、学習モードに戻す必要があります。

## AMP を使った学習

AMP とは、自動混合精度（Automatic Mixed Precision）の略であり、PyTorch 1.6 以降から公式にサポートされる機能です。これは、モデルを浮動小数点数精度（通常は 32 ビットの float 型）と半精度で学習するようにします。つまり、モデルの計算を半精度で行い、最終的な出力を 32 ビットの float 型で出力します。この方法により、GPU のメモリ使用量を削減し、学習速度を向上させることができます。

### 概要

AMP は混合精度計算を自動化する技術で、FP32（単精度）と FP16（半精度）を動的に切り替えて計算効率を向上させます。
PyTorch では `torch.cuda.amp` モジュールが提供されており、以下の 2 つのコンポーネントで構成されます：

- **`autocast`**: 計算を FP16 で実行できるか自動判定し、適切な精度を選択します。
- **`GradScaler`**: FP16 による勾配消失を防ぐため、損失（loss）をスケーリングして勾配更新を安定化させます。

### 優位性

- **メモリ削減**: FP16 は FP32 の半分のメモリを使用するため、バッチサイズを増やせる。
- **高速化**: GPU の Tensor Cores を活用し、行列演算が高速化されます。
- **エネルギー効率向上**: 計算量の削減により消費電力が低下。

### 使用方法

以下は典型的な AMP トレーニングのコード例です：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # グラジエントスケーラーを初期化

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast():  # autocastコンテキスト内で順伝播
        output = model(data)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()  # スケーリングされた損失で逆伝播
    scaler.step(optimizer)         # オプティマイザのステップ
    scaler.update()                # スケーラーの更新
```

### 注意点

- **サポートされるハードウェア**: AMP は NVIDIA GPU（Tensor Cores 対応）で最大限の効果を発揮します。
- **数値不安定性**: FP16 ではオーバーフロー/アンダーフローが発生する可能性があるため、`GradScaler` が必要です。
- **非対応操作**: 一部の演算（例: 損失関数の log など）は FP32 で実行されるため、パフォーマンス改善の効果が限定的な場合があります。

つつく．．．
