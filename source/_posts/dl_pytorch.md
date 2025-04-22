---
title: pytorch フレームワーク
date: 2022-8-3 15:10:00
categories: [AI]
tags: [deep learning, pytorch, python]
lang: ja
---

PyTorchは、Pythonで書かれたフレームワークです。

- [Pytorchインストール](#pytorchインストール)
- [Tips](#tips)
  - [torch.Tensor()とtorch.Tensor(\[\])　の違い](#torchtensorとtorchtensorの違い)
- [基本的な使い方](#基本的な使い方)
  - [import](#import)
  - [データ型](#データ型)
  - [説明](#説明)


## Pytorchインストール

[アーカイブ　リース](https://pytorch.org/get-started/previous-versions)

- use pip: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`　(**CUDAは必要！！！**)

最新のpytorchはcondaでインストールのをできません。

## Tips
### torch.Tensor()とtorch.Tensor([])　の違い
```python
torch.Tensor(2,3,4).size()  # torch.Size([2, 3, 4])
torch.Tensor([2,3,4]).size()  # torch.Size([3])
```
!(torch_tensor)[assert/dl_pytorch/image/torch_tensor.png]

## 基本的な使い方

### import

```python

import torch
 
print(torch.__version__) # pytorchバージョン
print(torch.version.cuda) # cudaバージョン
print(torch.cuda.is_available()) # cudaが利用可能かどうか
```

### データ型

| データ型               | CPU Tensor           | GPU Tensor                |
| ---------------------- | -------------------- | ------------------------- |
| 8ビット符号なし整数型  | `torch.ByteTensor`   | `torch.cuda.ByteTensor`   |
| 8ビット符号付き整数型  | `torch.CharTensor`   | `torch.cuda.CharTensor`   |
| 16ビット符号付き整数型 | `torch.ShortTensor`  | `torch.cuda.ShortTensor`  |
| 32ビット符号付き整数型 | `torch.IntTensor`    | `torch.cuda.IntTensor`    |
| 64ビット符号付き整数型 | `torch.LongTensor`   | `torch.cuda.LongTensor`   |
| 32ビット浮動小数点数型 | `torch.FloatTensor`  | `torch.cuda.FloatTensor`  |
| 64ビット浮動小数点数型 | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| ブール型               | `torch.BoolTensor`   | `torch.cuda.BoolTensor`   |

### 説明
- **CPU Tensor**: CPU上で動作するPyTorchのテンソル型です。
- **GPU Tensor**: CUDA対応GPU上で動作するPyTorchのテンソル型です。  
  （例: `torch.cuda.FloatTensor` はGPU上での32ビット浮動小数点数型を表します。）


