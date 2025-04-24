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
  - [t.addとt.add\_の違い](#taddとtadd_の違い)
  - [torch.stackとtorch.cat](#torchstackとtorchcat)
- [torch.stack vs torch.cat](#torchstack-vs-torchcat)
  - [基本的な違い](#基本的な違い)
  - [詳細な使用例](#詳細な使用例)
    - [torch.stack](#torchstack)
    - [torch.cat](#torchcat)
  - [主な使用ケース](#主な使用ケース)
    - [torch.stackの典型的な用途](#torchstackの典型的な用途)
    - [torch.catの典型的な用途](#torchcatの典型的な用途)
  - [注意事項](#注意事項)
  - [パフォーマンス比較](#パフォーマンス比較)
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
- [CUDAの使い方](#cudaの使い方)
  - [使う例](#使う例)
  - [Tips](#tips-1)


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

### t.addとt.add_の違い

- t.addは元のテンソルを変更しない
- t.add_は元のテンソルを変更します

![add and add_](/assert/dl_pytorch/image/add_add_.png)

### torch.stackとtorch.cat

以下はPyTorchにおける`torch.stack`と`torch.cat`の使い分けと詳細な説明です：

---

## torch.stack vs torch.cat

### 基本的な違い
| 関数          | 動作                     | 入力条件                   | 出力形状                       | 主な用途               |
| ------------- | ------------------------ | -------------------------- | ------------------------------ | ---------------------- |
| `torch.stack` | 新しい次元を追加して結合 | **同じ形状**のテンソルのみ | 入力テンソルの形状に新次元追加 | バッチ処理・次元拡張   |
| `torch.cat`   | 既存の次元で結合         | **次元数が同じ**テンソル   | 指定次元のサイズが合算         | 特徴量結合・データ連結 |


| 関数                       | 作用                                                                             |
| -------------------------- | -------------------------------------------------------------------------------- |
| `torch.stack((A, B), dim)` | 新しい次元を追加してテンソルを結合（入力テンソルは完全に同一形状である必要あり） |
| `torch.cat((A, B), dim)`   | 既存の次元に沿ってテンソルを連結（指定次元のサイズ以外は同一形状である必要あり） |


### 詳細な使用例

#### torch.stack
```python
A = torch.tensor([[1, 2], [3, 4]])  # (2,2)
B = torch.tensor([[5, 6], [7, 8]])  # (2,2)

# dim=0でstack（バッチ次元追加）
C = torch.stack((A, B), dim=0)
print(C.shape)  # torch.Size([2, 2, 2])
"""
tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
"""

# dim=1でstack（行方向に結合）
D = torch.stack((A, B), dim=1)
print(D.shape)  # torch.Size([2, 2, 2])
"""
tensor([[[1, 2],
         [5, 6]],

        [[3, 4],
         [7, 8]]])
"""
```

#### torch.cat
```python
A = torch.randn(2, 3)  # (2,3)
B = torch.randn(2, 3)  # (2,3)

# dim=0で結合（行方向）
C = torch.cat((A, B), dim=0)
print(C.shape)  # torch.Size([4, 3])

# dim=1で結合（列方向）
D = torch.cat((A, B), dim=1)
print(D.shape)  # torch.Size([2, 6])
```

### 主な使用ケース

#### torch.stackの典型的な用途
- 複数の単一画像からバッチ次元を作成
```python
img1 = torch.randn(3, 224, 224)  # 画像1（CHW）
img2 = torch.randn(3, 224, 224)  # 画像2
batch = torch.stack((img1, img2), dim=0)  # バッチ次元追加 (2,3,224,224)
```

- 時系列データのフレームスタック
```python
frame1 = torch.randn(256)  # 特徴量ベクトル
frame2 = torch.randn(256)
sequence = torch.stack((frame1, frame2), dim=0)  # (2,256)
```

#### torch.catの典型的な用途
- マルチモーダルデータの結合
```python
image_features = torch.randn(10, 512)  # 画像特徴量
text_features = torch.randn(10, 256)   # テキスト特徴量
combined = torch.cat((image_features, text_features), dim=1)  # (10,768)
```

- レイヤー出力の連結
```python
conv_out1 = torch.randn(32, 64, 64)
conv_out2 = torch.randn(32, 64, 64)
merged = torch.cat((conv_out1, conv_out2), dim=0)  # (64, 64, 64)
```

### 注意事項

- **形状の一致要件**：
  ```python
  # torch.stackは完全な形状一致が必要
  A = torch.rand(2,3)
  B = torch.rand(2,3)
  torch.stack((A, B))  # OK

  A = torch.rand(2,3)
  B = torch.rand(3,2)
  torch.stack((A, B))  # RuntimeError
  ```

- **次元指定の範囲**：
  ```python
  A = torch.rand(2,3)
  B = torch.rand(2,3)
  
  torch.stack((A, B), dim=2)  # 有効な次元範囲外（0-1）
  # IndexError: Dimension out of range
  ```

### パフォーマンス比較
| 操作  | 実行時間（例） | メモリ使用量         |
| ----- | -------------- | -------------------- |
| stack | 1.2ms ± 15µs   | 高い（新次元作成）   |
| cat   | 850µs ± 10µs   | 低い（既存次元拡張） |


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
| `torch.unqueeze(A, dim)`      | テンソルの指定した次元に要素を追加する                                                                                    |

## CUDAの使い方

1. デバイスを設定する
2. テンソルをGPUに移動する
3. 計算する

### 使う例
```python
import torch

# GPU環境が利用可能かテスト
print(torch.__version__)      # PyTorchのバージョン確認
print(torch.version.cuda)     # CUDAのバージョン確認
print(torch.cuda.is_available())  # CUDAが利用可能か確認

# GPUまたはCPUのデバイス選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# オブジェクトの実行環境を確認
a.device  # テンソルaが存在するデバイスを表示

# オブジェクトを指定デバイスに転送
A = A.to(device)  # テンソルAを選択したデバイス環境に設定

# オブジェクトをCPU環境に転送
A.cpu().device  # テンソルAをCPU環境に移動

# 異なるデバイス間での演算時の自動変換
a + b.to(device)  # デバイス環境が混在する場合、bのデバイス環境に統一される

# CUDAテンソルからnumpyへの変換手順
a.cpu().numpy()  # CUDAテンソルはCPU経由でnumpyに変換する必要あり

# CUDA型テンソルの作成
torch.tensor([1,2], device=device)  # 指定デバイス上に直接テンソルを作成
```

### Tips
1. bashに`nvidia-smi`でGPUの状態を確認する
2. `torch.cuda.is_available()`でGPUが利用可能か確認する
3. `torch.cuda.empty_cache()`でメモリを解放する
4. `torch.cuda.device_count()`でGPUの数を確認する
5. `torch.cuda.get_device_name(0)`でGPUの名前を確認する
6. `torch.cuda.current_device()`で現在のGPUを確認する（おもにmulti-gpusをつかう場合）


つづく...