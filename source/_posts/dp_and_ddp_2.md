---
title: DP と DDP 実践

date: 2022-10-6 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja
---

目次

- [DP の実装](#dp-の実装)
  - [コード](#コード)
  - [主な手順](#主な手順)
- [DDP の実装](#ddp-の実装)
  - [モデル並列実装手順](#モデル並列実装手順)
    - [プロセスグループの初期化](#プロセスグループの初期化)
    - [モデルの定義と GPU への分割配置](#モデルの定義と-gpu-への分割配置)
    - [DDP でモデルをラップ](#ddp-でモデルをラップ)
    - [データローダーの設定](#データローダーの設定)

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

- **DistributedDataParallel (DDP)** は、複数 GPU や複数ノードで分散学習を行うための PyTorch 公式推奨手法です。
- **モデル並列**では、モデルの各層を異なる GPU に分割配置し、データを順次処理します。
- DDP モデル並列の場合は、モデル分割・勾配同期・データ分配を自動化し、効率的な分散学習を実現します。
- DDP データ並列の場合は、各プロセスが異なるデータを処理し、勾配同期を行います。

### モデル並列実装手順

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

つつく．．．
