---
title: DP と DDP 実践

date: 2022-10-6 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja
---

目次

- [DP の実装](#dp-の実装)
  - [主な手順](#主な手順)

## DP の実装

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
