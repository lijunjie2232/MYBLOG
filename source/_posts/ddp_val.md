---
title: Pytorch多GPU推論結果の統合方法

date: 2022-10-8 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description: torch.distributedの通信操作(broadcast, all_reduce, all_gather, reduce_scatter, scatter, gather, reduceなど)について解説します。通信の種類や用途、実装方法などを紹介します。
---

# 目次

---

## 単一ノード

### DataParallel の使用

- **特徴**: 簡単に並列化可能（シングルノード限定）。
- **自動集約**: 各 GPU の出力が自動でメイン GPU に統合される。
- **コード例**:
  ```python
  model = nn.DataParallel(model).to(device)  # 自動でデータ分割＋結果統合
  with torch.no_grad():
      outputs = model(inputs)  # outputs は全GPUの結果を含む
  ```

### 結果の評価

- **自動集約済みのため、通常の評価処理で OK**:
  ```python
  _, predicted = torch.max(outputs, 1)
  accuracy = (predicted == labels).sum().item() / len(labels)
  ```

## 分散環境 (DistributedDataParallel)

### DistributedDataParallel の使用

- **特徴**: 多ノード環境で高性能（手動で初期化・同期が必要）。
- **初期化**:
  ```python
  dist.init_process_group(backend='nccl')  # 分散環境初期化
  model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
  ```

### 推論 dataloader

- 多 GPU 推論の場合は、各 GPU にデータを割り当てる DistributedSampler を使用。

```python
sampler = DistributedSampler(val_dataset, rank=rank, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size//world_size, sampler=sampler, num_workers=num_workers)
```
