---
title: Pytorch多GPU推論結果の統合方法

date: 2022-10-8 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description: Pytorchで複数GPUを使用して推論した結果の実装例を紹介します
---

# 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [単一ノード(DataParallel)](#%E5%8D%98%E4%B8%80%E3%83%8E%E3%83%BC%E3%83%89dataparallel)
  - [DataParallel の使用](#dataparallel-%E3%81%AE%E4%BD%BF%E7%94%A8)
  - [結果の評価](#%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A9%95%E4%BE%A1)
- [分散環境(DistributedDataParallel)](#%E5%88%86%E6%95%A3%E7%92%B0%E5%A2%83distributeddataparallel)
  - [DistributedDataParallel の使用](#distributeddataparallel-%E3%81%AE%E4%BD%BF%E7%94%A8)
  - [推論 dataloader](#%E6%8E%A8%E8%AB%96-dataloader)
  - [推論](#%E6%8E%A8%E8%AB%96)
  - [結果の集約](#%E7%B5%90%E6%9E%9C%E3%81%AE%E9%9B%86%E7%B4%84)
- [参考](#%E5%8F%82%E8%80%83)

---

# 単一ノード(DataParallel)

## DataParallel の使用

- **特徴**: 簡単に並列化可能（シングルノード限定）。
- **自動集約**: 各 GPU の出力が自動でメイン GPU に統合される。
- **コード例**:
  ```python
  model = nn.DataParallel(model).to(device)  # 自動でデータ分割＋結果統合
  with torch.no_grad():
      outputs = model(inputs)  # outputs は全GPUの結果を含む
  ```

## 結果の評価

- **自動集約済みのため、通常の評価処理で OK**:
  ```python
  _, predicted = torch.max(outputs, 1)
  accuracy = (predicted == labels).sum().item() / len(labels)
  ```

# 分散環境(DistributedDataParallel)

## DistributedDataParallel の使用

- **特徴**: 多ノード環境で高性能（手動で初期化・同期が必要）。
- **初期化**:
  ```python
  dist.init_process_group(backend='nccl')  # 分散環境初期化
  model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
  ```

## 推論 dataloader

- 多 GPU 推論の場合は、各 GPU にデータを割り当てる DistributedSampler を使用。

```python
sampler = DistributedSampler(val_dataset, rank=rank, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size//world_size, sampler=sampler, num_workers=num_workers)
```

## 推論

- 推論は各プロシースに普通的に実行。結果の統合は別途必要。

```python
correct = 0
total = 0

model.eval()  # 推論モードへ切り替え

with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 推論実行
        outputs = model(inputs)

        # 当プロシースのデータの精度計算
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

```

## 結果の集約

- **集約方法**: `torch.distributed.all_reduce` で各 GPU の結果を合計。
- **コード例**:

  ```python
  # 各GPUの値をテンソルに変換
  correct_tensor = torch.tensor(correct, device='cuda')
  total_tensor = torch.tensor(total, device='cuda')

  # 全GPUの値を集約
  dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
  dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

  if rank == 0:  # 主プロシースのみ精度計算
    accuracy = correct_tensor.item() / total_tensor.item()
    print(f'Accuracy: {accuracy * 100:.2f}%')
  ```

# 参考

- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [torch.distributed の通信](/2022/10/08/torch_distributed/)
