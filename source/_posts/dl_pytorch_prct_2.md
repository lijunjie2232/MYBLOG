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

description:
  この記事では、画像を用いた顔表情認識を例に、PyTorch を使った実践的なタスクの開発プロセスを一から解説します。
  第二部となる今回は、PyTorch を使った実装的な技術ポイントを紹介します。
---

この記事では、画像を用いた顔表情認識を例に、PyTorch を使った実践的なタスクの開発プロセスを一から解説します。
第二部となる今回は、PyTorch を使った実装的な技術ポイントを紹介します。

code の例：[main.ipynb](https://colab.research.google.com/github/lijunjie2232/MYBLOG/blob/master/source/assert/dl_pytorch_prct/main.ipynb)

プロジェクトアドレス：[https://github.com/lijunjie2232/emotion_analyse_pytorch](https://github.com/lijunjie2232/emotion_analyse_pytorch)

## 目次

- [目次](#目次)
- [Tips](#tips)
- [基本的な任務](#基本的な任務)
- [データセットの割当り](#データセットの割当り)
  - [testset と validationset の違い](#testset-と-validationset-の違い)
    - [目的の違い](#目的の違い)
    - [データの扱いの違い](#データの扱いの違い)
  - [実装例](#実装例)
    - [注意点](#注意点)
    - [実践的なワークフロー](#実践的なワークフロー)
- [データの transoforms](#データの-transoforms)
  - [主な特徴](#主な特徴)
  - [よく使われるクラス一覧](#よく使われるクラス一覧)
  - [使用例](#使用例)
  - [適用方法の例](#適用方法の例)
- [モデルの保存と読み込み](#モデルの保存と読み込み)
  - [Pytorch モデルの保存と読み込み](#pytorch-モデルの保存と読み込み)
  - [Pytorch 状態辞書の保存と読み込み](#pytorch-状態辞書の保存と読み込み)
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
  - [注意点](#注意点-1)
- [Early Stopping](#early-stopping)
  - [概要](#概要-1)
  - [優位性](#優位性-1)
  - [実装手順](#実装手順)
    - [初期化](#初期化)
    - [各エポックでの処理](#各エポックでの処理)
    - [コード例](#コード例)
  - [主な注意点](#主な注意点-2)
  - [pytorchlighting の応用](#pytorchlighting-の応用)
- [トレニングの並行化](#トレニングの並行化)
  - [DDP を使用したモデルの並列化](#ddp-を使用したモデルの並列化)
  - [使う手順](#使う手順)
    - [分散環境の初期化](#分散環境の初期化)
    - [モデルのラップ](#モデルのラップ)
    - [データローダーの設定](#データローダーの設定)
    - [学習ループ](#学習ループ)
    - [終了処理](#終了処理)
    - [torchrun を使用した実行](#torchrun-を使用した実行)
      - [主なオプション](#主なオプション)
      - [実行例:](#実行例)

## Tips

## 基本的な任務

要するに、画像のファイルには顔がある、その画像を読み込んて、pytorch モデルで顔表情を判断します。

1. データ準備と画像前処理
2. モデルアーキテクチャ設計
3. モデルトレーニング
4. モデル推論

## データセットの割当り

データセットは、通常に以下の 3 つに分かれます。

1. 訓練用(train)データセット

   - モデルの学習に使用するデータ
   - 通常に 8 割〜9 割のデータを訓練用として使用する。

2. 検証用(validation)データセット

   - モデルの性能を評価するデータ
   - 通常に 1 割〜2 割のデータを検証用として使用する。

3. テスト用(test)データセット（可選択）
   - テスト用データセット
   - 通常に 1 割〜2 割のデータをテスト用として使用する。
   - 通常に最後の評価として使用する。

### testset と validationset の違い

- Validation Set は学習中の「チューニング用」であり、Test Set は「最終評価用」。
- Test Set を Validation として再利用すると、評価結果がバイアスされる。

#### 目的の違い

| 項目               | Validation Set                                                                                     | Test Set                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **目的**           | モデルのハイパーパラメータ調整や早期停止の判断に利用。学習過程でモデル性能を監視し、過学習を防止。 | モデルの最終的な汎化性能を評価。学習・調整が完了した後の「未知のデータ」に対する性能を測定。 |
| **使用タイミング** | 学習中に定期的に評価。例えば、Early Stopping や学習率の調整に活用。                                | モデルの学習・調整が完全に終了した後のみ使用。                                               |

#### データの扱いの違い

- **Validation Set**:
  - 学習プロセスに間接的に影響を与える（例: 検証損失が改善しなければ学習を停止）。
  - ハイパーパラメータ（例: ニューロン数、学習率）の最適化に活用。
- **Test Set**:
  - 学習プロセスに一切関与しない。
  - 最終評価のみに使用されるため、「未知のデータ」として完全に独立している必要がある。

### 実装例

```python
# データセットの分割例（train/val/test）
from torch.utils.data import random_split

# 全データを8:1:1に分割
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# データローダーの構築
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
```

#### 注意点

- **データリークの防止**:
  - Test Set は**学習・検証プロセスで絶対に使用しない**。リークすると評価結果が過剰に楽観視される。
- **サイズの目安**:
  - Validation Set: 全データの 10〜20%程度。
  - Test Set: 同様に 10〜20%程度（タスクによって調整）。

#### 実践的なワークフロー

1. **学習**: Train Set でモデルを学習。
2. **検証**: Val Set でハイパーパラメータ調整や Early Stopping の判定。
3. **評価**: Test Set で最終的な性能を測定（例: Accuracy, F1 Score）。

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

### 適用方法の例

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

### Pytorch 状態辞書の保存と読み込み

モデルのパラメータ（重みとバイアス）を保存する際の推奨方法。モデル構造が変わった場合でも、同じ構造の新しいモデルにパラメータを読み込むことが可能になる。

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

USE_AMP = True  # AMP を使用するかどうかのフラグ

scaler = GradScaler()  # グラジエントスケーラーを初期化

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast(enabled=USE_AMP):  # autocastコンテキスト内で順伝播
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

## Early Stopping

### 概要

Early Stopping は、過学習（Overfitting）を防止するためのテクニックで、**検証データ（Validation Data）の性能が改善しなくなった時点でトレーニングを自動的に停止**します。  
通常、検証損失（Validation Loss）や精度（Accuracy）を監視し、一定のエポック数（`patience`）改善が見られなければトレーニングを終了します。

### 優位性

- **過学習の防止**: 検証データの性能が悪化した時点で停止し、モデルの汎化性能を維持。
- **リソース効率化**: 無駄なエポックを実行せず、計算時間とメモリを節約。
- **最適なモデル選択**: 最良の検証性能を達成した時点のモデルを保存可能。

### 実装手順

以下に Early Stopping の基本的な実装フローを示します：

#### 初期化

- 検証損失の最小値を保存する変数（`min_val_loss`）を初期化。
- 改善が停止したカウンター（`counter`）を初期化。

#### 各エポックでの処理

- 学習データでモデルを更新。
- 検証データで損失と精度を評価。
- 検証損失が最小値を更新した場合：
  - `min_val_loss`を更新。
  - 最良のモデルパラメータを保存。
  - カウンターをリセット。
- 検証損失が改善しない場合：
  - カウンターをインクリメント。
  - カウンターが`patience`に達した場合、トレーニングを停止。

#### コード例

```python
import torch
from torch.utils.data import DataLoader

# パラメータ設定
patience = 5  # 改善が停止するまでのエポック数
min_val_loss = float('inf')  # 最小検証損失の初期化
counter = 0  # 改善停止カウンター

# ダミーのデータローダー
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Early Stopping付きトレーニングループ
for epoch in range(100):
    # 学習フェーズ
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 検証フェーズ
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += loss_fn(outputs, labels).item()

    val_loss /= len(val_loader)

    # Early Stopping判定
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # 最良モデルを保存
    else:
        counter += 1

    if counter >= patience:
        print(f"Early Stopping at epoch {epoch+1}")
        break
```

### 主な注意点

- **`patience`の設定**: 小さすぎると学習が早すぎる段階で停止する可能性があり、大きすぎると過学習のリスクが生じます。
- **検証指標の選定**: 検証損失ではなく精度や F1 スコアなど、タスクに合った指標を使用することも可能です。
- **モデル保存のタイミング**: 最良のモデルを保存しておき、推論時に読み込みます。
- **ランダム性の影響**: シード値（Seed）を固定することで再現性を確保。

### pytorchlighting の応用

[PyTorch Lightning](https://www.pytorchlightning.ai/) や [ignite](https://pytorch.org/ignite/) には組み込みの Early Stopping 機能があります。

```python
# PyTorch Lightningの例
from pytorch_lightning.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=5)
trainer = Trainer(callbacks=[early_stop])
```

## トレニングの並行化

- PyTorch の `DataParallel` は、単一ノード内の複数 GPU を利用してモデルを並列化するためのクラスです。データ並列化（Data Parallelism）を実現し、学習速度の高速化やバッチサイズの拡大に役立ちます。

- PyTorch の `DistributedDataParallel` (DDP) は、複数の GPU または複数ノードでモデルを分散して学習するためのフレームワークです。データ並列処理を実現し、大規模なモデルやデータセットの学習を効率化します。

- `DataParallel`と`DistributedDataParallel`もは，GPU を利用してモデルを並列処理するためのフレームワークですが、主な違いは、通信のオーバーヘッドや分散学習の設定方法にあります。通常、大規模システムで複数ノードを使用する場合

### DDP を使用したモデルの並列化

今回の実装例は、`DDP`を使用して複数 GPU でモデルを並列化するため、`torch.nn.parallel.DistributedDataParallel`を説明、ほかのてんは別の文章で説明する．

- **目的**:  
  複数の GPU/ノードでモデルの並列化を行い、学習速度の高速化とメモリ負荷の分散を実現。
- **特徴**:
  - 各 GPU に独立したプロセスを生成し、データを分割して並列処理。
  - 勾配の同期（AllReduce）により、分散環境でも正確な更新を行う。
  - `torch.distributed` パッケージを基盤に動作。

### 使う手順

#### 分散環境の初期化

```python
import torch.distributed as dist

# 分散プロセスグループの初期化
dist.init_process_group(backend='nccl')  # backend は 'nccl' または 'gloo' など
# nccl は NVIDIA GPU 用、gloo は CPU/多種の環境用、mpi もあります．
```

#### モデルのラップ

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# モデルを DDP でラップ
model = DDP(model, device_ids=[local_rank])  # local_rank は GPU のローカル ID
```

#### データローダーの設定

```python
from torch.utils.data.distributed import DistributedSampler

# 分散用サンプラーを使用
sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### 学習ループ

```python
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # 勾配計算と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 終了処理

```python
# 分散プロセスグループの終了
dist.destroy_process_group()
```

#### torchrun を使用した実行

torchrun を使用することで、PyTorch のモデルを分散プロセスグループで実行することができます。以下は torchrun を使用してモデルを実行するための基本的なコマンドの例です。

##### 主なオプション

- `--nproc_per_node`: 1 ノードあたりのプロセス数を指定。
- `--nnodes`: 使用するノード数を指定。
- `--start_method`: プロセスの起動方法を指定、デフォルトは spawn、fork はプロセスを fork する方法で起動します。
- `--log_dir`: ログ出力先のディレクトリを指定。
- `--master_addr`: マスターノードのアドレスを指定。
- `--master_port`: マスターノードのポートを指定。
- `--monitor_interval`: ログの出力間隔を秒単位で指定。
- `--run_path`: 実行ファイルのパスを指定。

##### 実行例:

```bash
torchrun --nproc_per_node=2 --master_port=23450 train.py --batch_size=32 --epochs=100 --lr=0.001
```
