---
title: pytorch フレームワーク
date: 2022-8-3 15:10:00
categories: [AI]
tags: [deep learning, pytorch, python]
lang: ja
---

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

- [Pytorchインストール](#pytorchインストール)
- [Tips](#tips)
  - [torch.Tensor()とtorch.Tensor(\[\])の違い](#torchtensorとtorchtensorの違い)
  - [torch.tensor()とtorch.Tensor()の違い](#torchtensorとtorchtensorの違い-1)
  - [torch.mmとtorch.mulの違い](#torchmmとtorchmulの違い)
- [基本的な使い方](#基本的な使い方)
  - [import](#import)
- [Tensor](#tensor)
  - [データ型](#データ型)
  - [説明](#説明)
- [Tensorの作成](#tensorの作成)
  - [randxxx](#randxxx)
    - [randn](#randn)
    - [randint](#randint)
  - [tensorの運算](#tensorの運算)


## Pytorchインストール

[アーカイブ　リース](https://pytorch.org/get-started/previous-versions)

- use pip: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`　(**CUDA11.8は必要！！！**)

最新のpytorchはcondaでインストールのをできません。

## Tips
### torch.Tensor()とtorch.Tensor([])の違い
```python
torch.Tensor(2,3,4).size()  # torch.Size([2, 3, 4])
torch.Tensor([2,3,4]).size()  # torch.Size([3])
```
![torch.Tensor(2,3,4) と torch.Tensor([2,3,4])](/assert/dl_pytorch/image/torch_tensor.png)

### torch.tensor()とtorch.Tensor()の違い
- torch.Tensor()は、`dtype`, `requires_grad`を指定できない、既存のデータ（リストやNumPy配列など）からテンソルを作成または未初期化テンソルを作成のための関数
- torch.tensor()は、'dtype'などの引数を指定できる
```python
data = [1, 2, 3]
tensor = torch.tensor(data, dtype=torch.float32, device='cuda')

tensor = torch.Tensor([1, 2, 3])  # 指定されたデータでテンソルを作成
tensor = torch.Tensor(2, 3)       # 2x3の未初期化テンソルを作成
```

### torch.mmとtorch.mulの違い
✅ [torch.mm](https://pytorch.org/docs/stable/generated/torch.mm.html)：行列積 (Matrix Multiplication)

数学記号：
<center>$C = A \times B$</center>

ここで、$A \in \mathbb{R}^{m \times n}$、$B \in \mathbb{R}^{n \times p}$と、Cは$C \in \mathbb{R}^{m \times p}$行列です。

式展開：
<center>$C_{ij} = \sum_{k=1}^{n} A_{ik} \times B_{kj}$</center>

```python
>>> A = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
>>> B = torch.tensor([[1, 4], [2, 5], [3, 6]])  # shape: (3, 2)
>>> C = torch.mm(A, B)
>>> C
tensor([[14, 32],
        [32, 77]])
```
✅ [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html)：要素ごとの積 (Element-wise Multiplication)

数学記号：
<center>$C = A \odot B$</center>

ここで、$A,B \in \mathbb{R}^{m \times n}$ は同じ形の行列です。

式展開：
<center>$C_{ij} = A_{ij} \cdot B_{ij}$</center>

```python
>>> A = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
>>> B = torch.tensor([[5, 6], [7, 8]])  # shape: (2, 2)
>>> C = torch.mul(A, B)  # shape: (2, 2)
>>> C
tensor([[ 5, 12],
        [21, 32]])
```



## 基本的な使い方

### import

```python
import torch
 
print(torch.__version__) # pytorchバージョン
print(torch.version.cuda) # cudaバージョン
print(torch.cuda.is_available()) # cudaが利用可能かどうか
```

## Tensor
Tensorはnumpyのndarrayと同様にデータを格納するデータ構造です。

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

## Tensorの作成
![nparray2tensor](/assert/dl_pytorch/image/nparray2tensor.png)
```python
>>> a = np.random.randn(2,3)
>>> a
array([[ 0.55657307, -0.56752282, -1.29813938],
       [ 0.19568811, -1.58711887,  0.96234087]])
>>> t = torch.tensor(a, dtype=torch.float32, device=torch.device('cuda:2'))
>>> t
tensor([[ 0.5566, -0.5675, -1.2981],
        [ 0.1957, -1.5871,  0.9623]], device='cuda:2')
```

### randxxx
#### randn
```python
>>> torch.randn([2,3])
tensor([[-0.4271,  1.0660,  1.2755],
        [-1.5805,  0.4410,  0.4207]])
>>> torch.randn([2,3], dtype=torch.float64, device='cuda')
tensor([[-0.8559,  1.0472,  0.6330],
        [-0.5150, -0.8062, -2.4052]], device='cuda:0', dtype=torch.float64)
```
#### randint
```python
>>> torch.randint(0, 10, [2,3])
tensor([[5, 7, 5],
        [0, 7, 9]])
>>> torch.randint(0, 10, [2,3], dtype=torch.int8, device='cuda')
tensor([[7, 8, 6],
        [7, 5, 2]], device='cuda:0', dtype=torch.int8)
>>> torch.randint(0, 10, [2,3]).dtype
torch.int64
```

### tensorの運算

| 関数                          | 作用                                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `torch.abs(A)`                | 絶対値を計算する                                                                                                          |
| `torch.add(A, B)`             | 加算を行う。AとBはTensorまたはスカラーのいずれでも可能                                                                    |
| `torch.clamp(A, min, max)`    | クリップ（範囲制限）。Aのデータが`min`より小さいか`max`より大きい場合、`min`または`max`に変換され、範囲を[min, max]に保つ |
| `torch.div(A, B)`             | 除算を行う。A / B。AとBはTensorまたはスカラーのいずれでも可能                                                             |
| `torch.mul(A, B)`             | 要素ごとの乗算（点乗算）。A * B。AとBはTensorまたはスカラーのいずれでも可能                                               |
| `torch.pow(A, n)`             | べき乗を計算する。Aのn乗                                                                                                  |
| `torch.mm(A, B.T)`            | 行列の乗算（行列積）。`torch.mul`との違いに注意                                                                           |
| `torch.mv(A, B)`              | 行列とベクトルの乗算。Aは行列、Bはベクトル。Bは転置の必要はない                                                           |
| `A.item()`                    | Tensorを基本データ型に変換する。Tensorに要素が1つしかない場合に使用可能。主にTensorから数値を取り出すために使用           |
| `A.numpy()`                   | TensorをNumpy型に変換する                                                                                                 |
| `A.size()`                    | サイズを確認する                                                                                                          |
| `A.shape`                     | サイズを確認する                                                                                                          |
| `A.dtype`                     | データ型を確認する                                                                                                        |
| `A.view()`                    | テンソルのサイズを再構築する。Numpyの`reshape`に類似                                                                      |
| `A.transpose(0, 1)`           | 行と列を交換する                                                                                                          |
| `A[1:]`                       | スライス。Numpyのスライスに類似                                                                                           |
| `A[-1, -1] = 100`             | スライス。Numpyのスライスに類似                                                                                           |
| `A.zero_()`                   | ゼロ化する（すべての要素を0にする）                                                                                       |
| `torch.stack((A, B), dim=-1)` | テンソルを結合し、次元を増やす                                                                                            |
| `torch.diag(A)`               | Aの対角要素を取り出し、1次元ベクトルを形成する                                                                            |
| `torch.diag_embed(A)`         | 1次元ベクトルを対角線に配置し、残りの要素を0とするテンソルを作成する                                                      |

つづく...