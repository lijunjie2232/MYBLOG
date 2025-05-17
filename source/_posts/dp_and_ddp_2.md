---
title: DP と DDP 実践

date: 2022-10-6 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja
---

目次

- [DP の実装](#dp-%E3%81%AE%E5%AE%9F%E8%A3%85)
  - [コード](#%E3%82%B3%E3%83%BC%E3%83%89)
  - [主な手順](#%E4%B8%BB%E3%81%AA%E6%89%8B%E9%A0%86)
- [DDP の実装](#ddp-%E3%81%AE%E5%AE%9F%E8%A3%85)
  - [モデル並列実装手順](#%E3%83%A2%E3%83%87%E3%83%AB%E4%B8%A6%E5%88%97%E5%AE%9F%E8%A3%85%E6%89%8B%E9%A0%86)
    - [プロセスグループの初期化](#%E3%83%97%E3%83%AD%E3%82%BB%E3%82%B9%E3%82%B0%E3%83%AB%E3%83%BC%E3%83%97%E3%81%AE%E5%88%9D%E6%9C%9F%E5%8C%96)
    - [モデルの定義と GPU への分割配置](#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E5%AE%9A%E7%BE%A9%E3%81%A8-gpu-%E3%81%B8%E3%81%AE%E5%88%86%E5%89%B2%E9%85%8D%E7%BD%AE)
    - [DDP でモデルをラップ](#ddp-%E3%81%A7%E3%83%A2%E3%83%87%E3%83%AB%E3%82%92%E3%83%A9%E3%83%83%E3%83%97)
    - [データローダーの設定](#%E3%83%87%E3%83%BC%E3%82%BF%E3%83%AD%E3%83%BC%E3%83%80%E3%83%BC%E3%81%AE%E8%A8%AD%E5%AE%9A)
    - [学習ループ](#%E5%AD%A6%E7%BF%92%E3%83%AB%E3%83%BC%E3%83%97)
    - [リソースの解放](#%E3%83%AA%E3%82%BD%E3%83%BC%E3%82%B9%E3%81%AE%E8%A7%A3%E6%94%BE)
    - [重要なポイント](#%E9%87%8D%E8%A6%81%E3%81%AA%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88)
    - [注意事項](#%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A0%85)
  - [データ並列化の実装手順](#%E3%83%87%E3%83%BC%E3%82%BF%E4%B8%A6%E5%88%97%E5%8C%96%E3%81%AE%E5%AE%9F%E8%A3%85%E6%89%8B%E9%A0%86)
    - [プロセスグループの初期化](#%E3%83%97%E3%83%AD%E3%82%BB%E3%82%B9%E3%82%B0%E3%83%AB%E3%83%BC%E3%83%97%E3%81%AE%E5%88%9D%E6%9C%9F%E5%8C%96-1)
    - [モデルの定義と DDP ラップ](#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E5%AE%9A%E7%BE%A9%E3%81%A8-ddp-%E3%83%A9%E3%83%83%E3%83%97)
    - [データローダーの設定](#%E3%83%87%E3%83%BC%E3%82%BF%E3%83%AD%E3%83%BC%E3%83%80%E3%83%BC%E3%81%AE%E8%A8%AD%E5%AE%9A-1)
    - [学習ループ](#%E5%AD%A6%E7%BF%92%E3%83%AB%E3%83%BC%E3%83%97-1)
    - [モデルの保存](#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E4%BF%9D%E5%AD%98)
    - [DDP のデータ並列の特徴](#ddp-%E3%81%AE%E3%83%87%E3%83%BC%E3%82%BF%E4%B8%A6%E5%88%97%E3%81%AE%E7%89%B9%E5%BE%B4)
  - [起動コマンド](#%E8%B5%B7%E5%8B%95%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89)
    - [torch.distributed.launch を使用](#torchdistributedlaunch-%E3%82%92%E4%BD%BF%E7%94%A8)
    - [torchrun を使用](#torchrun-%E3%82%92%E4%BD%BF%E7%94%A8)
    - [オプション説明](#%E3%82%AA%E3%83%97%E3%82%B7%E3%83%A7%E3%83%B3%E8%AA%AC%E6%98%8E)
  - [init\_process\_group](#initprocessgroup)
    - [init\_process\_group 関数の引数](#initprocessgroup-%E9%96%A2%E6%95%B0%E3%81%AE%E5%BC%95%E6%95%B0)

<!-- more -->

## DP の実装

### コード

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

n_sample = 100
n_dim = 10
batch_size = 10
X = torch.randn(n_sample, n_dim)
Y = torch.randint(0, 2, (n_sample, )).float()

dataset = SimpleDataset(X, Y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== 注：作成されたモデルはCPU上にある ===== #
device_ids = [0, 1, 2]
model = SimpleModel(n_dim).to(device_ids[0])
model = nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)

        loss = nn.BCELoss()(outputs, targets.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
```

### 主な手順

- `DataParallel` を使う手順はただ一つ：
  ```python
  model = nn.DataParallel(model, device_ids=device_ids)
  ```

## DDP の実装

### モデル並列実装手順

- **DistributedDataParallel (DDP)** は、複数 GPU や複数ノードで分散学習を行うための PyTorch 公式推奨手法です。
- **モデル並列**では、モデルの各層を異なる GPU に分割配置し、データを順次処理します。
- DDP モデル並列の場合は、モデル分割・勾配同期・データ分配を自動化し、効率的な分散学習を実現します。

#### プロセスグループの初期化

各プロセス間の通信を設定します。

```python
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(
        "nccl",              # NCCL通信バックエンド
        rank=rank,           # 現在のプロセスID
        world_size=world_size  # 総プロセス数
    )
    torch.cuda.set_device(rank)  # GPUデバイスを設定

def cleanup():
    dist.destroy_process_group()  # 通信終了
```

ここで、`rank` は現在のプロセスの ID、`world_size` は総プロセス数です。
`init_process_group` 関数は、プロセス間の通信を初期化します。`nccl` は NCCL 通信バックエンドを指定します。`rank` は現在のプロセスの ID、`world_size` は総プロセス数を指定します。
`torch.cuda.set_device(rank)` は、現在のプロセスが使用する GPU デバイスを設定します。

#### モデルの定義と GPU への分割配置

モデルの各層を異なる GPU に配置します。

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, rank):
        super(MyModel, self).__init__()
        self.part1 = nn.Linear(10, 10).to(rank*2)        # GPU0に配置
        self.part2 = nn.Linear(10, 5).to(rank*2 + 1)    # GPU1に配置

    def forward(self, x):
        x = self.part1(x)
        x = x.to(rank + 1)  # データをGPU1に転送
        return self.part2(x)

model = MyModel(rank=0)  # rank=0のGPUにモデルを初期配置
```

ここで、`MyModel` クラスの `forward` メソッドでデータを異なる GPU 間で転送します。

#### DDP でモデルをラップ

モデルを `DistributedDataParallel` でラップし、分散学習を有効化します。

```python
model = nn.parallel.DistributedDataParallel(model, device_ids=[0, 1])
```

- `device_ids`: 使用する GPU の ID を指定（例: `[0, 1]`）。

#### データローダーの設定

`DistributedSampler` を使用してデータを均等に分配します。

```python
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import TensorDataset

# データセットの作成
dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 5))

# 分散サンプラー
sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
data_loader = DataLoader(dataset, batch_size=10, sampler=sampler)
```

#### 学習ループ

各プロセスが割り当てられたデータで学習を行います。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for data, target in data_loader:
    optimizer.zero_grad()
    output = model(data.cuda(0))  # GPU0で入力データを処理
    loss = criterion(output, target.cuda(0))  # GPU0で損失計算
    loss.backward()  # 勾配計算
    optimizer.step()  # パラメータ更新
```

#### リソースの解放

学習終了時にプロセスグループを破棄します。

```python
cleanup()
```

#### 重要なポイント

| 項目             | 説明                                                       |
| ---------------- | ---------------------------------------------------------- |
| **プロセス構造** | 各 GPU に独立したプロセスを割り当て、GIL の影響を回避      |
| **通信方式**     | NCCL バックエンドを使用した AllReduce で勾配同期           |
| **モデル分割**   | モデルの各層を異なる GPU に手動で配置（例: `to(rank)`）    |
| **勾配同期**     | DDP が自動で勾配を全プロセス間で総和（AllReduce）          |
| **データ分配**   | `DistributedSampler`で均等にミニバッチを分割               |
| **メモリ効率**   | 勾配バケット化により通信と計算を並列化（パイプライン処理） |

#### 注意事項

- **プロセスと GPU の対応**  
  各プロセスは 1 つの GPU のみを使用（例: `rank=0 → GPU0`）。
- **モデルの手動分割**  
  モデルの各層を明示的に異なる GPU に配置する必要がある。
- **同期の保証**  
  DDP によりパラメータ更新が自動同期されるが、中間データの転送（例: `x.to(rank+1)`）は手動で管理。
- **マルチノード対応**  
  複数ノードでの実行にはネットワーク設定（例: IP アドレス・ポート番号）が必要。

### データ並列化の実装手順

- **DistributedDataParallel (DDP)** は、複数 GPU や複数ノードで分散学習を行うための PyTorch 公式推奨手法です。
- **データ並列**では、各プロセスが**完全なモデルコピー**を持ち、**異なるデータサブセット**で独立して学習します。
- プロセス間で勾配を同期（AllReduce）することで、モデルパラメータを統一します。

#### プロセスグループの初期化

各プロセス間の通信を設定します。

```python
import torch.distributed as dist

def setup(local_rank):
    dist.init_process_group(backend='nccl')  # NCCLバックエンドで通信を初期化
    torch.cuda.set_device(local_rank)        # 現在のプロセスが使用するGPUを設定
```

- `local_rank`: 現在のプロセスが使用する GPU のローカル ID（例: 0, 1）。
- `backend='nccl'`: NVIDIA GPU 向けの高速通信ライブラリ NCCL を使用。

#### モデルの定義と DDP ラップ

モデルを各 GPU に配置し、DDP でラップします。

```python
model = SimpleModel(n_dim).to(local_rank)  # モデルをGPUに配置
model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # DDPでラップ
```

- `device_ids`: 使用する GPU の ID を指定（例: `[0, 1]`）。
- `output_device`: 出力のデバイスを指定（通常、`local_rank`と同じ）。

#### データローダーの設定

`DistributedSampler` を使用してデータを均等に分配します。

```python
sampler = torch.utils.data.distributed.DistributedSampler(dataset)  # 分散サンプラー
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```

- **DistributedSampler の役割**:
  - 各プロセスが重複しないデータサブセットを取得。
  - シャッフル時にプロセス間で同じ乱数シードを維持。

#### 学習ループ

各プロセスが割り当てられたデータで学習を行い、勾配同期を行います。

```python
for epoch in range(num_epochs):
    data_loader.sampler.set_epoch(epoch)  # シャッフルのシード同期
    for data, label in data_loader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label.unsqueeze(1))
        loss.backward()  # 勾配計算（各プロセスで独立）
        optimizer.step()  # 勾配同期（AllReduce）後に更新
```

- **勾配同期の仕組み**:
  - 各プロセスが逆伝播で勾配を計算。
  - `optimizer.step()` で AllReduce により勾配総和を同期。

#### モデルの保存

プロセス 0 のみでモデルを保存します。

```python
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), "model.pt")  # model.moduleでラップ解除
```

ここで、`model.module` は DDP ラップ解除後のモデルへの参照です。

#### DDP のデータ並列の特徴

| 特徴                 | 説明                                                                      |
| -------------------- | ------------------------------------------------------------------------- |
| **プロセス構造**     | 各 GPU に独立したプロセスを割り当て（マルチプロセス）、GIL の影響を回避。 |
| **通信方式**         | NCCL バックエンドによる AllReduce で勾配総和を同期。                      |
| **データ分配**       | `DistributedSampler`でデータを均等に分割（重複なし）。                    |
| **勾配同期**         | バケット単位の AllReduce で通信と計算を並列化（パイプライン処理）。       |
| **メモリ効率**       | 勾配バケット化により通信と計算を最適化。                                  |
| **スケーラビリティ** | GPU 数増加時でも通信コストが固定に近い（固定オーバーヘッド）。            |

### 起動コマンド

#### torch.distributed.launch を使用

- **単一ノードの場合**:
  ```bash
  CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 train_ddp.py
  ```
- **複数ノードの場合**:

  ```bash
  # ノード0(ip=192.168.0.10):
  CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nnodes=2 --node_rank=0 --master_addr="192.168.0.10" --master_port=12345 --nproc_per_node=2 --use_env train_ddp.py --batch_size=64 --lr=0.01

  # ノード1(ip=192.168.0.11):
  CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nnodes=2 --node_rank=1 --master_addr="192.168.0.10" --master_port=12345 --nproc_per_node=2 --use_env train_ddp.py --batch_size=64 --lr=0.01
  ```

#### torchrun を使用

- **単一ノードの場合**:

  ```bash
  torchrun --nnodes=1 --nproc_per_node=2 train_ddp.py --batch_size=64 --lr=0.01
  ```

- **複数ノードの場合**:

  ```bash
  # ノード0(ip=192.168.0.10):
  torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.0.10" --master_port=12345 --nproc_per_node=2 train_ddp.py --batch_size=64 --lr=0.01

  # ノード1(ip=192.168.0.11):
  torchrun --nnodes=2 --node_rank=1 --master_addr="192.168.0.10" --master_port=12345 --nproc_per_node=2 train_ddp.py --batch_size=64 --lr=0.01
  ```

#### オプション説明

- `--nproc_per_node`: 1 ノードあたりのプロセス数。
- `--nnodes`: ノード数。
- `--master_addr`: マスターノードの IP アドレス。
- `--master_port`: マスターノードのポート番号。
- `--use_env`: 環境変数を参照してプロセスを起動する。(torchrun は必要ではない)

### init_process_group

#### init_process_group 関数の引数

```python
torch.distributed.init_process_group(backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name='', pg_options=None, device_id=None)
```

| パラメータ名  | 説明                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------- |
| `backend`     | 通信バックエンド (`nccl`, `gloo`, `mpi`, `ucc`) を指定。GPU なら`nccl`が推奨。                          |
| `init_method` | プロセス間接続の初期化方法 (`env://`, `tcp://<ip>:<port>`, `file://<path>`, `mpi://`)。                 |
| `timeout`     | 通信操作のタイムアウト時間（デフォルト: NCCL は 10 分）。                                               |
| `world_size`  | 参加プロセス総数（例: [4](file://d:\code\MYBLOG\themes\next\scripts\tags\group-pictures.js#L18-L23)）。 |
| `rank`        | 現在のプロセスの ID（0 から`world_size-1`まで）。                                                       |
| `store`       | 接続情報を共有するストア（`init_method`と排他使用）。                                                   |
| `device_id`   | 特定デバイス（例: GPU）にプロセスをバインド（バックエンド最適化用）。                                   |
