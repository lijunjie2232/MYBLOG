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

### DataParallelの使用
- **特徴**: 簡単に並列化可能（シングルノード限定）。
- **自動集約**: 各GPUの出力が自動でメインGPUに統合される。
- **コード例**:
  ```python
  model = nn.DataParallel(model).to(device)  # 自動でデータ分割＋結果統合
  with torch.no_grad():
      outputs = model(inputs)  # outputs は全GPUの結果を含む
  ```

### 結果の評価
- **自動集約済みのため、通常の評価処理でOK**:
  ```python
  _, predicted = torch.max(outputs, 1)
  accuracy = (predicted == labels).sum().item() / len(labels)
  ```

