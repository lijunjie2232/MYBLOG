---
title: pytorch 実践（一）
date: 2022-8-12 10:15:00
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
第一部となる今回は、プロジェクト構築の全流程を実践的に説明します。

code の例：[main.ipynb](https://colab.research.google.com/github/lijunjie2232/MYBLOG/blob/master/source/assert/dl_pytorch_prct/main.ipynb)

プロジェクトアドレス：[https://github.com/lijunjie2232/emotion_analyse_pytorch](https://github.com/lijunjie2232/emotion_analyse_pytorch)

## 目次

- [目次](#目次)
- [Pytorch インストール](#pytorch-インストール)
  - [requirements](#requirements)
- [基本的な任務](#基本的な任務)
- [データセット情報](#データセット情報)
- [データの準備と前処理](#データの準備と前処理)
- [モーデルの作成](#モーデルの作成)
- [モーデルのトレーニングと検証](#モーデルのトレーニングと検証)
  - [train\_epoch 関数](#train_epoch-関数)
  - [val\_epoch 関数](#val_epoch-関数)
  - [メインの訓練ループ](#メインの訓練ループ)
  - [全体の流れ](#全体の流れ)
  - [主な技術ポイント](#主な技術ポイント)
- [結果](#結果)
- [参考資料](#参考資料)

## Pytorch インストール

[アーカイブ　リース](https://pytorch.org/get-started/previous-versions)

- use pip: `pip3 install torch torchvision torchaudio`　(CUDA12.6 by default)

最新の pytorch は conda でインストールのをできません。

### requirements

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install kagglehub
```

## 基本的な任務

要するに、画像のファイルには顔がある、その画像を読み込んて、pytorch モデルで顔表情を判断します。

1. データ準備と画像前処理
2. モデルアーキテクチャ設計
3. モデルトレーニング
4. モデル推論

## データセット情報

今回のデータセットは、Kaggle のデータセットを利用します。

link：[https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset)

ラベル: 7 種類の感情（angry, disgust, fear, happy, neutral, sad, surprise）

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("aadityasinghal/facial-expression-dataset")

print("Path to dataset files:", path)
```

## データの準備と前処理

```python
# ## build data transforms
train_transformer = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),  # 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
val_transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
# ## build dataset
train_dataset = ImageFolder(
    root=data_root / "train" / "train",
    transform=train_transformer,
)
val_dataset = ImageFolder(
    root=data_root / "test" / "test",
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

## モーデルのトレーニングと検証

### train_epoch 関数

訓練用のデータローダー (`train_loader`) を使って、モデルを 1 エポック訓練します。

- **主な処理:**
  - `model.train()` でモデルを訓練モードに設定。
  - `tqdm` を使用してプログレスバーを表示（`progress=True` の場合）。
  - データとターゲットを指定デバイス（デフォルトは `"cuda"`）に移動。
  - `optimizer.zero_grad()` で勾配をリセット。
  - `torch.amp.autocast` を使用して自動混合精度（FP16）を適用（`fp16=True` の場合）。
  - 損失を計算し、逆伝播（`scaler.scale(loss).backward()` または `loss.backward()`）。
  - オプティマイザとスケジューラを更新。
  - 精度（`acc`）と損失（`loss`）を計算し、プログレスバーに表示。
  - 最終的な訓練精度と平均損失を返す。

```python
def train_epoch(
    model,               # 訓練対象のモデル
    train_loader,        # 訓練データのDataLoader
    optimizer,          # オプティマイザ（例: Adam）
    scheduler,          # 学習率スケジューラ
    criterion,          # 損失関数（例: 交差エントロピー）
    scaler=None,        # FP16混合精度訓練用のスケーラ
    device="cuda",      # 使用デバイス（デフォルトはCUDA/GPU）
    fp16=True,          # FP16混合精度使用フラグ
    progress=True,      # 進捗表示の有無
):
    model.train()  # モデルを訓練モードに設定
    # tqdmでプログレスバーを表示（progress=Falseなら表示しない）
    loop = tqdm(train_loader, desc="train", leave=False) if progress else train_loader
    total_loss = 0.0    # 累積損失
    total_count = 0     # 累計データ数
    ateru = 0          # 正解数（「当てる」から命名）
    assert scaler is not None or not fp16  # FP16時はscaler必須

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device)    # データをGPUに転送
        target = target.to(device) # ラベルをGPUに転送
        optimizer.zero_grad()      # 勾配をリセット

        # 自動混合精度（FP16）コンテキスト
        with torch.amp.autocast(
            device_type=device,
            enabled=fp16,
            dtype=torch.float16 if fp16 else torch.float32,
        ):
            output = model(data)           # モデルの順伝播
            loss = criterion(output, target)  # 損失計算

        # FP16対応の逆伝播
        if fp16:
            scaler.scale(loss).backward()  # スケール付き逆伝播
            scaler.step(optimizer)         # 勾配更新
            scaler.update()                # スケーラ更新
        else:
            loss.backward()     # 通常の逆伝播
            optimizer.step()    # 勾配更新

        # メトリクス計算
        total_count += len(data)
        ateru += (output.argmax(dim=1) == target).sum().item()  # 正解数集計
        total_loss += loss.item()

        # 進捗表示更新
        if progress:
            loop.set_postfix({
                "acc": ateru / total_count,               # 精度
                "loss": total_loss / (batch_idx + 1),     # 平均損失
            })

    scheduler.step()  # 学習率更新
    return ateru / total_count, total_loss / (batch_idx + 1)  # 精度, 平均損失

```

### val_epoch 関数

検証用のデータローダー (`val_loader`) を使って、モデルの性能を評価します。

- **主な処理:**
  - `model.eval()` でモデルを評価モードに設定。
  - `torch.no_grad()` で勾配計算を無効化。
  - 検証データをモデルに入力し、損失と精度を計算。
  - バッチごとに損失と正解数を累積。
  - 最終的な検証精度と平均損失を返す。

```python
@torch.no_grad()  # 勾配計算無効化（メモリ節約）
def val_epoch(
    model,              # 評価対象モデル
    val_loader,         # 検証データのDataLoader
    criterion,          # 損失関数
    device="cuda",      # 使用デバイス
    progress=True,      # 進捗表示の有無
):
    model.eval()  # モデルを評価モードに設定
    loop = tqdm(val_loader, desc="val", leave=False) if progress else val_loader
    val_loss = 0.0
    total_num = 0
    total_correct = 0

    for i, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logist, _ = model(inputs)  # モデル推論（多くの場合、logistは予測スコア）
        loss = criterion(logist, labels)

        # メトリクス集計
        val_loss += loss.item() * inputs.size(0)  # 損失をバッチサイズで重み付け
        total_num += labels.size(0)
        total_correct += (logist.argmax(dim=1) == labels).sum().item()  # 正解数

        if progress:
            loop.set_postfix(
                loss=val_loss / (i + 1),           # 現在の平均損失
                acc=total_correct / total_num      # 現在の精度
            )

    # 最終結果計算
    val_loss = val_loss / len(val_loader.dataset)  # データセット全体での平均損失
    return total_correct / total_num, val_loss / (i + 1)  # 精度, 平均損失
```

### メインの訓練ループ

エポックごとに `train_epoch` と `val_epoch` を実行し、モデルの訓練と検証を行います。

- **主な処理:**
  - `tqdm` でエポックの進行状況を表示。
  - 訓練と検証の精度・損失をリストに保存。
  - チェックポイントを保存（`save` 関数）。
  - 訓練/検証の精度・損失をプロット（`plot` 関数）。
  - 検証精度が向上した場合、ベストモデルを保存（`best_checkpoint`）。
  - `patience` を使った早期停止（Early Stopping）を実装（精度が向上しなければ `patience` を減らし、`0` になったら訓練終了）。

```python
# 訓練進捗表示の初期化
loop = tqdm(range(start_epoch + 1, epochs))
train_acc_list = []  # 訓練精度の履歴
train_loss_list = [] # 訓練損失の履歴
val_acc_list = []    # 検証精度の履歴
val_loss_list = []   # 検証損失の履歴
best_acc = 0         # ベスト精度記録用
patience = train_patience  # Early Stopping用のカウンタ

for epoch in loop:
    loop.set_description(f"Epoch [{epoch+1}/{epochs}]")  # 現在のエポック表示

    # 1エポックの訓練実行
    train_acc, train_loss = train_epoch(
        model, train_dataloader, optimizer,
        scheduler, criterion, scaler, device
    )
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # 検証実行
    val_acc, val_loss = val_epoch(model, val_dataloader, criterion, device)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

    # 進捗表示更新
    loop.set_postfix(
        train_acc=train_acc,
        train_loss=train_loss,
        val_acc=val_acc,
        val_loss=val_loss,
    )

    # チェックポイント保存
    save(model, optimizer, epoch, last_checkpoint)

    # 学習曲線描画
    plot(
        train_acc_list, train_loss_list,
        val_acc_list, val_loss_list,
        fig_save_dir="./",
    )

    # Early Stopping ロジック
    if val_acc > best_acc:  # 精度が向上した場合
        shutil.copyfile(last_checkpoint, best_checkpoint)  # ベストモデルを保存
        best_acc = val_acc
        patience = train_patience  # カウンタをリセット
    else:
        patience -= 1  # 精度向上なしの場合カウンタを減らす

    if patience == 0:  # 指定エポック数精度が向上しなかった場合
        break  # 訓練終了
```

### 全体の流れ

1. **訓練フェーズ** (`train_epoch`):
   - モデルを訓練データで学習。
   - 損失と精度を記録。
2. **検証フェーズ** (`val_epoch`):
   - モデルを検証データで評価。
   - 過学習を防ぐため、検証精度を監視。
3. **モデルの保存と早期停止**:
   - 検証精度が向上した場合、モデルを保存。
   - 一定エポック（`patience`）精度が向上しなければ訓練を終了。

### 主な技術ポイント

1. **混合精度訓練（FP16）**
   - `scaler`を使用して勾配のアンダーフローを防止
   - メモリ使用量削減と計算速度向上
2. **Early Stopping**
   - `patience`回数だけ検証精度の向上を待つ
   - 過学習を防止するための重要な仕組み
3. **モード切り替え**
   - `model.train()` / `model.eval()`で BatchNorm や Dropout の挙動を変更
4. **チェックポイント管理**
   - 最高精度モデルを`best_checkpoint`として別途保存
   - 訓練中断時の再開が可能

## 結果

![結果](/assert/dl_pytorch_prct/accuracy_and_loss.png)

## 参考資料

- ultralytics: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
