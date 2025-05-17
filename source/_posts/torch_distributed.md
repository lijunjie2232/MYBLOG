---
title: torch.distributedの通信と実装に運用

date: 2022-10-8 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description:
---

# 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [torch.distributed](#torchdistributed)
  - [初期化方法](#%E5%88%9D%E6%9C%9F%E5%8C%96%E6%96%B9%E6%B3%95)
- [集合通信操作](#%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E6%93%8D%E4%BD%9C)
  - [ブロードキャスト操作](#%E3%83%96%E3%83%AD%E3%83%BC%E3%83%89%E3%82%AD%E3%83%A3%E3%82%B9%E3%83%88%E6%93%8D%E4%BD%9C)
    - [dist.broadcast](#distbroadcast)
    - [dist.broadcast\_object\_list](#distbroadcastobjectlist)
  - [集約操作 (Reduce)](#%E9%9B%86%E7%B4%84%E6%93%8D%E4%BD%9C-reduce)
    - [dist.all\_reduce](#distallreduce)
    - [dist.reduce](#distreduce)
    - [非同期集約](#%E9%9D%9E%E5%90%8C%E6%9C%9F%E9%9B%86%E7%B4%84)
  - [データ収集 (Gather)](#%E3%83%87%E3%83%BC%E3%82%BF%E5%8F%8E%E9%9B%86-gather)
    - [dist.all\_gather](#distallgather)
    - [dist.gather](#distgather)
    - [オブジェクト収集](#%E3%82%AA%E3%83%96%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E5%8F%8E%E9%9B%86)
  - [データ配布 (Scatter)](#%E3%83%87%E3%83%BC%E3%82%BF%E9%85%8D%E5%B8%83-scatter)
    - [dist.scatter](#distscatter)
    - [dist.scatter\_object\_list](#distscatterobjectlist)
  - [複合操作](#%E8%A4%87%E5%90%88%E6%93%8D%E4%BD%9C)
    - [dist.reduce\_scatter/dist.reduce\_scatter\_tensor](#distreducescatterdistreducescattertensor)
    - [dist.all\_to\_all/dist.all\_to\_all\_single](#distalltoalldistalltoallsingle)
  - [同期操作](#%E5%90%8C%E6%9C%9F%E6%93%8D%E4%BD%9C)
    - [dist.barrier()](#distbarrier)
    - [dist.monitored\_barrier(timeout=10)](#distmonitoredbarriertimeout10)

---

# torch.distributed

`torch.distributed` は PyTorch の分散学習を実装するためのコアライブラリです。  
プロセス間通信（IPC）を効率化するための**集合通信**（AllReduce, AllGather, Broadcast）と**ポイントツーポイント通信**（Send/Recv）を提供します。

## 初期化方法

```python
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',          # GPU なら NCCL 推奨
    init_method='env://',    # 環境変数で初期化
    world_size=4,            # 総プロセス数
    rank=0                   # 現在のプロセスID
)
```

# 集合通信操作

## ブロードキャスト操作

### dist.broadcast

- **機能**: 特定プロセス（[src](file://d:\code\MYBLOG\themes\volantis\scripts\tags\media.js#L9-L9)）のテンソルを**全プロセスに配信**。
- **用途**: 初期重みやハイパーパラメータの共有。
- **例**:

  ```python
  dist.broadcast(tensor, src=0)  # ランク0から全プロセスに配信
  ```

### dist.broadcast_object_list

- **機能**: Python オブジェクトリストを全プロセスに配信。
- **用途**: 非テンソルデータ（文字列、辞書など）の共有。
- **例**:
  ```python
  dist.broadcast_object_list(obj_list, src=0)  # ランク0のオブジェクトを全プロセスに送信
  ```

## 集約操作 (Reduce)

### dist.all_reduce

- **機能**: 全プロセスのテンソルを**集約**（加算、平均など）し、結果を全プロセスに配布。
- **用途**: 勾配同期や損失関数の平均計算。
- **例**:
  ```python
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 全プロセスのテンソルを加算
  ```

### dist.reduce

- **機能**: 全プロセスのテンソルを**集約**し、結果を特定プロセス（`dst`）に送信。
- **用途**: マスターノードでの結果収集。
- **例**:
  ```python
  dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)  # 全プロセスのデータをランク0に集約
  ```

### 非同期集約

- **機能**: 非同期処理で通信と計算を並列化。
- **例**:
  ```python
  work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
  work.wait()  # 通信完了まで待機
  ```

## データ収集 (Gather)

### dist.all_gather

- **機能**: 全プロセスのテンソルを**収集**し、各プロセスに全データを複製。
- **用途**: 分散データの集約（例: 推論結果の統合）。
- **例**:
  ```python
  dist.all_gather(gather_list, tensor)  # 全プロセスのデータを各プロセスに複製
  ```

### dist.gather

- **機能**: 全プロセスのテンソルを**特定プロセス**（`dst`）に集約。
- **用途**: マスターノードでの結果集約。
- **例**:
  ```python
  dist.gather(tensor, gather_list, dst=0)  # 全プロセスのデータをランク0に集約
  ```

### オブジェクト収集

- **関数**:
  - `dist.all_gather_object(gather_list, obj)`
  - `dist.gather_object(obj, gather_list, dst=0)`

## データ配布 (Scatter)

### dist.scatter

- **機能**: 源プロセス（[src](file://d:\code\MYBLOG\themes\volantis\scripts\tags\media.js#L9-L9)）のテンソルリストを**全プロセスに分散**。
- **用途**: データ分割処理。
- **例**:
  ```python
  dist.scatter(tensor, scatter_list, src=0)  # ランク0のリストを全プロセスに分散
  ```

### dist.scatter_object_list

- **機能**: Python オブジェクトリストを全プロセスに分散。
- **例**:
  ```python
  dist.scatter_object_list(obj, scatter_list, src=0)
  ```

## 複合操作

### dist.reduce_scatter/dist.reduce_scatter_tensor

- **機能**: 各プロセスのデータを**集約＋分散**。
- **用途**: モデル並列化時の効率化。
- **例**:
  ```python
  dist.reduce_scatter(output_tensor, [input_tensor], op=dist.ReduceOp.SUM)
  ```

### dist.all_to_all/dist.all_to_all_single

- **機能**: 全プロセス間でテンソルの**部分交換**。
- **用途**: パイプライン並列処理。
- **例**:
  ```python
  dist.all_to_all_single(output_tensor, input_tensor)  # 全プロセス間でデータ交換
  ```

## 同期操作

### dist.barrier()

- **機能**: 全プロセスの**同期**（全プロセスが到達するまで待機）。

### dist.monitored_barrier(timeout=10)

- **機能**: タイムアウト付き同期（デバッグ時有効）。
