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
  - [epochとiterationとbatchsizeなどの関係](#epochとiterationとbatchsizeなどの関係)
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
  - [関するコマンド](#関するコマンド)
- [Autograd](#autograd)
  - [主な用途](#主な用途)
  - [基本原理](#基本原理)
  - [動的グラフ vs 静的グラフ](#動的グラフ-vs-静的グラフ)
  - [PyTorchでの実装方法](#pytorchでの実装方法)
  - [逆伝播（Backpropagation）](#逆伝播backpropagation)
  - [勾配計算の無効化](#勾配計算の無効化)
  - [主なメソッド](#主なメソッド)
- [DatasetとDataLoader](#datasetとdataloader)
  - [使用の例](#使用の例)
  - [Datasetの説明](#datasetの説明)
  - [DataLoaderの説明](#dataloaderの説明)
    - [主要パラメータの詳細説明](#主要パラメータの詳細説明)
    - [実践的な設定例](#実践的な設定例)
    - [重要な注意点](#重要な注意点)
- [nn.Module](#nnmodule)
  - [作成例](#作成例)
  - [主なポイント](#主なポイント)
  - [出力結果](#出力結果)
  - [簡単なトレーニングの例](#簡単なトレーニングの例)
- [Loss](#loss)
  - [損失関数の基本概念](#損失関数の基本概念)
  - [pytorch損失関数の使用例](#pytorch損失関数の使用例)
  - [カスタム損失関数の作成方法](#カスタム損失関数の作成方法)
  - [実践的な使用例](#実践的な使用例)
  - [損失関数選択のガイドライン](#損失関数選択のガイドライン)
  - [勾配計算の注意点](#勾配計算の注意点)
- [Optimizer](#optimizer)
  - [使用例](#使用例)
  - [PyTorchオプティマイザの主要機能](#pytorchオプティマイザの主要機能)
  - [実践的な使用パターン](#実践的な使用パターン)
    - [異なるレイヤーに異なる学習率を設定](#異なるレイヤーに異なる学習率を設定)
    - [勾配クリッピング](#勾配クリッピング)
  - [主要オプティマイザの数式比較](#主要オプティマイザの数式比較)
  - [パフォーマンス最適化Tips](#パフォーマンス最適化tips)
  - [よくあるエラーと対策](#よくあるエラーと対策)


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

### torch.stack vs torch.cat

#### 基本的な違い
| 関数          | 動作                     | 入力条件                   | 出力形状                       | 主な用途               |
| ------------- | ------------------------ | -------------------------- | ------------------------------ | ---------------------- |
| `torch.stack` | 新しい次元を追加して結合 | **同じ形状**のテンソルのみ | 入力テンソルの形状に新次元追加 | バッチ処理・次元拡張   |
| `torch.cat`   | 既存の次元で結合         | **次元数が同じ**テンソル   | 指定次元のサイズが合算         | 特徴量結合・データ連結 |


| 関数                       | 作用                                                                             |
| -------------------------- | -------------------------------------------------------------------------------- |
| `torch.stack((A, B), dim)` | 新しい次元を追加してテンソルを結合（入力テンソルは完全に同一形状である必要あり） |
| `torch.cat((A, B), dim)`   | 既存の次元に沿ってテンソルを連結（指定次元のサイズ以外は同一形状である必要あり） |


#### 詳細な使用例

##### torch.stack
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

##### torch.cat
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

#### 主な使用ケース

##### torch.stackの典型的な用途
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

##### torch.catの典型的な用途
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

#### 注意事項

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

#### パフォーマンス比較
| 操作  | 実行時間（例） | メモリ使用量         |
| ----- | -------------- | -------------------- |
| stack | 1.2ms ± 15µs   | 高い（新次元作成）   |
| cat   | 850µs ± 10µs   | 低い（既存次元拡張） |


### epochとiterationとbatchsizeなどの関係

ここで、画像のデータセットを例にして：


<center>$total\_iteration = iteration * epoch$</center>

<center>$image\_num = iteration * batch\_size$</center>

- `image_num`とは、画像はデータセットに何枚あるかを表す。
- `epoch`とは、データセットを何回学習させるかを表す。
- `iteration`とは、毎回epochはデータセットを何回読み込むかを表す。
- `batch_size`とは、毎回epochは画像を何枚読み込むかを表す。
- `total_iteration`とは、(トレーニング)全体はデータセットを読み込む回数を表す。



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

### 関するコマンド
1. bashに`nvidia-smi`でGPUの状態を確認する
2. `torch.cuda.is_available()`でGPUが利用可能か確認する
3. `torch.cuda.empty_cache()`でメモリを解放する
4. `torch.cuda.device_count()`でGPUの数を確認する
5. `torch.cuda.get_device_name(0)`でGPUの名前を確認する
6. `torch.cuda.current_device()`で現在のGPUを確認する（おもにmulti-gpusをつかう場合）


## Autograd

Autogradは自動微分のこと。つまり、関数の値や勾配を自動で計算する。
- Backpropagation → 誤差逆伝播法
- Chain Rule → 連鎖律
- Dynamic Graph → 動的計算グラフ
- Static Graph → 静的計算グラフ

自動微分（Automatic Differentiation）はディープラーニングフレームワークの中核機能で、数学関数の導関数を自動的に計算します。

### 主な用途
- ニューラルネットワークの訓練時の勾配計算
- 誤差逆伝播法（Backpropagation）の実装

### 基本原理
連鎖律（Chain Rule）に基づき、複合関数の導関数を効率的に計算します。PyTorchのモデルは多くの層で構成される複雑な関数のため、Autogradが各層の勾配を効率的に計算します。

### 動的グラフ vs 静的グラフ
| 種類           | 特徴                                         | フレームワーク例     |
| -------------- | -------------------------------------------- | -------------------- |
| 動的計算グラフ | 実行時にグラフを構築（デバッグ・変更が容易） | PyTorch              |
| 静的計算グラフ | 実行前にグラフを構築（最適化が可能）         | TensorFlow（初期版） |

### PyTorchでの実装方法
`torch.Tensor`オブジェクトの`requires_grad`属性で勾配計算の有無を制御：
```python
# 勾配計算が必要なテンソル作成
x = torch.randn(2, 2, requires_grad=True)
print(x)
# tensor([[-0.5736, -0.2012],
#         [ 1.3563,  0.5499]], requires_grad=True)

# 演算の追跡
y = x + 2
z = y * y * 3
out = z.mean()
print(out)
# tensor(17.2779, grad_fn=<MeanBackward0>)
```

### 逆伝播（Backpropagation）
`.backward()`メソッドで勾配計算：
```python
out.backward()        # 勾配計算
print(x.grad)         # 勾配値の確認
# tensor([[2.1396, 2.6982],
#         [5.0345, 3.8248]])
```

### 勾配計算の無効化
以下の方法で勾配計算を停止可能：
```python
# 方法1: コンテキストマネージャー
with torch.no_grad():
    y = x * 2

# 方法2: メソッドの引数
@torch.no_grad()
def inference():
    y = x * 2
    return y

# 方法3: 属性設定
print(x.requires_grad)
# True
x.requires_grad_(False)
# tensor([[-0.5736, -0.2012],
#         [ 1.3563,  0.5499]])
print(x.requires_grad)
# False
```

### 主なメソッド
| メソッド          | 説明                                 |
| ----------------- | ------------------------------------ |
| `.backward()`     | 勾配を自動計算                       |
| `.detach()`       | 勾配追跡から切り離したテンソルを生成 |
| `torch.no_grad()` | 勾配計算を無効にするコンテキスト     |
| `.retain_grad()`  | 非リーフテンソルの勾配を保持         |


## DatasetとDataLoader


### 使用の例
```python
from torch.utils.data import DataLoader, Dataset

# Datasetの基本構造
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

# DataLoaderの基本使用例
train_dataset = CustomDataset(data_samples, target_labels)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

# 実行ループ例
for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        ...
```

### Datasetの説明
1. **役割**  
   データセットの抽象基底クラス。データの読み込みと前処理をカプセル化し、データアクセスを一貫性のある方法で提供します。

2. **必須メソッド**  
   - `__getitem__(self, index)`: 指定インデックスのデータサンプルを返す
   - `__len__(self)`: データセットの総サンプル数を返す

3. **特徴**  
   - データ変換（transforms）の統合が可能
   - 自定义データ形式のサポート
   - データの並列読み込みとキャッシュの最適化


### DataLoaderの説明
1. **役割**  
   データセットをバッチ単位で効率的に読み込み、モデル訓練に最適なデータパイプラインを提供します。

```python
# DataLoaderの主要パラメータとその説明（橙色部分重点）

from torch.utils.data import DataLoader, Dataset

class SampleDataset(Dataset):
    def __init__(self):
        self.data = [...]  # データセットの初期化
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# データローダーの実用的設定例
train_loader = DataLoader(
    dataset=SampleDataset(),          # [必須] 使用するDatasetオブジェクト
    batch_size=32,                    # [重要] バッチサイズ（デフォルト1）
    shuffle=True,                     # [重要] エポックごとにデータをシャッフル（学習時はTrue推奨）
    num_workers=4,                    # [重要] データ読み込みの並列プロセス数（0でシングルスレッド）
    collate_fn=None,                  # [応用] カスタムバッチ作成関数（不規則データ用）
    pin_memory=True,                  # [GPU推奨] CUDAへの高速転送を可能にする
    drop_last=False                   # [調整] 最後の不完全バッチを削除するか（デフォルトFalse）
)
```

#### 主要パラメータの詳細説明

| パラメータ      | 役割と設定例                                     | 最適化のポイント                                                               |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------------ |
| **dataset**     | データソースとなるDatasetオブジェクト（必須）    | 自作Datasetクラスか torchvision.datasets等の既存データセットを使用する         |
| **batch_size**  | 1イテレーションあたりのサンプル数                | GPUメモリ容量とトレードオフ<br>（例：32/64/128/256）                           |
| **shuffle**     | エポックごとにデータをランダムシャッフルするか   | 学習時はTrueで過学習防止<br>評価時はFalseで安定化                              |
| **num_workers** | データ読み込み用プロセス数（I/O並列化）          | CPUリソース許す限り増やす（例：4-8）<br>`num_workers=0`はシングルスレッド      |
| **collate_fn**  | バッチ作成時のカスタム処理（不規則形状データ用） | 画像とテキストの組み合わせなど、形状が異なるデータをバッチ化する際に必要       |
| **pin_memory**  | CUDAへの転送速度向上（ページ锁定メモリ使用）     | GPU環境では必ずTrueに設定<br>（CPU→GPU転送速度が約2倍速くなる）                |
| **drop_last**   | 最後の不完全バッチを削除するか                   | バッチ処理が厳密なサイズ要求の場合はTrue<br>（例：特定のモデルアーキテクチャ） |


#### 実践的な設定例

```python
# GPU環境向け最適設定例
dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,    # GPUメモリ許容範囲で最大限設定
    shuffle=True,
    num_workers=4,    # CPUコア数に応じて調整
    pin_memory=True,  # CUDA環境必須
    drop_last=False   # データロスを許容できない場合
)

# データオーギメンテーション付き例
from torchvision import transforms
test_dataset = MyDataset(transform=transforms.Compose([...]))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    collate_fn=my_custom_collate  # 不規則データ用
)
```

#### 重要な注意点
1. **num_workersの最適化**  
   - 物理CPUコア数の半分を推奨（例：8コアなら4）
   - オーバー設定するとメモリ不足の可能性あり

2. **pin_memoryの効果**  
   - CUDAテンソルへの転送速度が約2倍向上
   - CPU→GPU転送がボトルネックの場合は必須

3. **バッチサイズのトレードオフ**  
   - 大きくすると学習安定性向上だがメモリ使用増加
   - 小さくすると学習速度向上だが不安定化の可能性

4. **collate_fnの使用ケース**  
   - 文字列と画像のペアデータ
   - 多可変長入力（例：異なる長さの時系列データ）
   - データ拡張をバッチレベルで実施する場合

5. **特徴**  
   - マルチプロセスデータローディングでI/Oボトルネックを軽減
   - サンプル加重やサブセット抽出のサポート
   - バッチ正規化やデータオーギメンテーションの統合

## nn.Module

PyTorchは`torch.nn.Module`クラスを通じてニューラルネットワーク構築の便利なインターフェースを提供します。nn.Moduleクラスを継承し、独自のネットワーク層を定義することができます。

### 作成例

```python
import torch.nn as nn
import torch.optim as optim

# 全結合ニューラルネットワークの定義
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1024, 224)  # 入力層から隠れ層へ
        self.fc2 = nn.Linear(224, 10)  # 隠れ層から出力層へ
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU活性化関数
        x = self.fc2(x)
        return x

# ネットワークインスタンスの作成
model = SimpleNN()

# モデル構造の表示
print(model)
```

### 主なポイント
- `nn.Module`を継承してネットワーククラスを定義
- `__init__`メソッドで層を初期化
- `forward`メソッドでデータの流れを定義

### 出力結果
```
SimpleNN(
  (fc1): Linear(in_features=1024, out_features=224, bias=True)
  (fc2): Linear(in_features=224, out_features=10, bias=True)
)
```

### 簡単なトレーニングの例

ニューラルネットワークはニューロン間の接続重みを調整することで予測結果を最適化し、このプロセスには以下の要素が含まれます：
1. 順伝播（Forward Propagation）
2. 損失計算（Loss Calculation）
3. 逆伝播（Backward propagation）
4. パラメータ更新（Parameter Update）

```python
# 実行ループ例
model.train()  # モデルを学習モードに設定
for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:  # バッチデータを取得
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_data)  # 順伝播
        loss = criterion(outputs, batch_labels)  # 損失計算
        loss.backward()  # 逆伝播とパラメータ更新（簡単の例）
```

主なニューラルネットワークの種類と用途：
| 種類                                    | 用途例                             |
| --------------------------------------- | ---------------------------------- |
| フィードフォワードネットワーク          | 基本的なパターン認識               |
| 畳み込みニューラルネットワーク（CNN）   | 画像認識、物体検出                 |
| リカレントニューラルネットワーク（RNN） | 時系列データ分析、自然言語処理     |
| LSTMネットワーク                        | 長期依存関係のある時系列データ処理 |


## Loss

### 損失関数の基本概念
1. **損失（Loss）の役割**  
   - モデル予測値と真値の差異を定量化
   - 逆伝播時の勾配計算の基準となる
   - 学習プロセスの収束状況を監視

2. **主な損失関数の種類**  
   | 損失関数クラス         | 数学式                                                                   | 主な用途     | 入力形式要件           |
   | ---------------------- | ------------------------------------------------------------------------ | ------------ | ---------------------- |
   | `nn.MSELoss`           | $\frac{1}{n}\sum(y_{pred}-y_{true})^2$                                   | 回帰問題     | 連続値（任意形状）     |
   | `nn.CrossEntropyLoss`  | $-\sum y_{true}\log(\text{softmax}(y_{pred}))$                           | 多クラス分類 | ロジット値（未正規化） |
   | `nn.BCELoss`           | $-\frac{1}{n}\sum[y_{true}\log y_{pred} + (1-y_{true})\log(1-y_{pred})]$ | 二値分類     | 確率値（0-1範囲）      |
   | `nn.L1Loss`            | $\frac{1}{n}\sum\|y_{pred}-y_{true}\|$                                   | ロバスト回帰 | 連続値（任意形状）     |
   | `nn.TripletMarginLoss` | $\max(d(a,p) - d(a,n) + \text{margin}, 0)$                               | 類似度学習   | 3組の埋め込みベクトル  |

### pytorch損失関数の使用例
```python
# 代表的な損失関数の使用例
import torch.nn as nn

# 平均二乗誤差（回帰問題用）
mse_loss = nn.MSELoss()
output = model(inputs)
loss = mse_loss(output, targets)

# 交差エントロピー（分類問題用）
ce_loss = nn.CrossEntropyLoss()
output = model(inputs)
loss = ce_loss(output, targets)

# バイナリ交差エントロピー（二値分類用）
bce_loss = nn.BCELoss()
output = torch.sigmoid(model(inputs))  # 確率値に変換
loss = bce_loss(output, targets.float())
```

### カスタム損失関数の作成方法
```python
# 方法1: nn.Moduleを継承
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = torch.mean((pred - target)**2)
        l1 = torch.mean(torch.abs(pred - target))
        return self.alpha * mse + (1 - self.alpha) * l1

# 方法2: 関数形式
def custom_loss(pred, target, beta=0.3):
    base_loss = F.binary_cross_entropy(pred, target)
    penalty = torch.mean(torch.relu(pred - 0.7))**2
    return base_loss + beta * penalty
```

### 実践的な使用例
```python
# マルチタスク学習用の複合損失
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred_class, pred_reg, target_class, target_reg):
        loss1 = self.ce(pred_class, target_class)
        loss2 = self.mse(pred_reg, target_reg)
        return loss1 + 0.5*loss2  # 重み付け加算

# 学習ループ内での利用
criterion = MultiTaskLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for data, (labels_class, labels_reg) in train_loader:
        outputs_class, outputs_reg = model(data)
        loss = criterion(outputs_class, outputs_reg, labels_class, labels_reg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 損失関数選択のガイドライン
| 問題タイプ       | 推奨損失関数               | 補足事項                                     |
| ---------------- | -------------------------- | -------------------------------------------- |
| 回帰問題         | `MSELoss` / `HuberLoss`    | 外れ値が多い場合はHuberLossが安定            |
| 多クラス分類     | `CrossEntropyLoss`         | 出力層の直前にSoftmax不要（組み込み済み）    |
| 二値分類         | `BCELoss`                  | 出力をシグモイドで確率値に変換する必要あり   |
| マルチラベル分類 | `BCEWithLogitsLoss`        | シグモイド+BCELossを組み合わせた効率的な実装 |
| 不均衡データ分類 | 重み付き`CrossEntropyLoss` | クラスごとのサンプル数に応じて重み設定       |
| 物体検出         | `SmoothL1Loss`             | バウンディングボックス回帰に有効             |

### 勾配計算の注意点
1. **カスタム損失の微分保証**  
   - PyTorchの組み込み関数のみ使用すれば自動微分可能
   - 独自の数学的操作を行う場合、微分可能性を確認する必要あり

2. **メモリ効率最適化**  
   ```python
   # メモリ節約テクニック
   loss = criterion(output, target)
   loss.backward()           # 勾配計算
   optimizer.step()          # パラメータ更新
   optimizer.zero_grad()     # 勾配リセット（重要！）
   ```

3. **複数損失の組み合わせ**  
   ```python
   # 複数損失を個別に計算する例
   loss1 = criterion1(output1, target1) * weight1
   loss2 = criterion2(output2, target2) * weight2
   total_loss = loss1 + loss2
   total_loss.backward()
   ```

## Optimizer

### 使用例
```python
# 主要オプティマイザの使用例
import torch.optim as optim

# モデルパラメータとオプティマイザの初期化
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学習ループでの使用
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()  # 勾配のリセット
        output = model(data)
        loss = criterion(output, target)
        loss.backward()        # 勾配計算
        optimizer.step()       # パラメータ更新
```

### PyTorchオプティマイザの主要機能

1. **パラメータ更新メカニズム**
   - 勾配降下法の変種を実装
   - モーメンタム、重み減衰などの拡張機能をサポート
   - 自動微分結果を利用した効率的なパラメータ更新

2. **主要オプティマイザの比較**
   | オプティマイザ | メリット                 | デメリット     | 最適な使用ケース   |
   | -------------- | ------------------------ | -------------- | ------------------ |
   | SGD            | シンプル、メモリ効率良好 | 収束が遅い     | 小規模データセット |
   | Adam           | 高速収束、ハイパラ調整少 | メモリ使用多   | 深層学習全般       |
   | RMSprop        | 不安定な勾配に強い       | 学習率調整必要 | RNN/LSTM           |
   | Adagrad        | 特徴頻度適応型           | 学習率急速低下 | スパースデータ     |
   | Adadelta       | 学習率自動調整           | 計算コスト高   | 長期依存関係       |

3. **共通パラメータ**
   ```python
   optimizer = optim.Adam(
       params=model.parameters(),  # 必須：最適化対象パラメータ
       lr=0.001,                   # 学習率（デフォルト値あり）
       betas=(0.9, 0.999),         # モーメンタム係数（Adam専用）
       weight_decay=0.01,          # L2正則化係数
       amsgrad=False               # AMSGrad変種（Adam専用）
   )
   ```

### 実践的な使用パターン

#### 異なるレイヤーに異なる学習率を設定
```python
# パラメータグループの分割
optimizer = optim.SGD([
    {'params': model.base.parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.01}
], momentum=0.9)

# 学習率スケジューラとの連携
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

#### 勾配クリッピング
```python
# 勾配のノルム制限
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 主要オプティマイザの数式比較

| オプティマイザ | 更新式（簡略版）                                                                                                                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SGD**        | $θ_{t+1} = θ_t - η \nabla J(θ_t)$                                                                                                                                                         |
| **Momentum**   | $v_{t+1} = γv_t + η \nabla J(θ_t)$<br>$θ_{t+1} = θ_t - v_{t+1}$                                                                                                                           |
| **Adam**       | $m_t = β_1m_{t-1} + (1-β_1)g_t$<br>$v_t = β_2v_{t-1} + (1-β_2)g_t^2$<br>$\hat{m}_t = m_t/(1-β_1^t)$<br>$\hat{v}_t = v_t/(1-β_2^t)$<br>$θ_{t+1} = θ_t - η\hat{m}_t/(\sqrt{\hat{v}_t} + ε)$ |

### パフォーマンス最適化Tips

1. **学習率のウォームアップ**
   ```python
   # 初期数エポックで学習率を徐々に増加
   warmup_epochs = 5
   for epoch in range(epochs):
       lr = base_lr * min(epoch / warmup_epochs, 1.0)
       for param_group in optimizer.param_groups:
           param_group['lr'] = lr
       # 通常の学習処理...
   ```

2. **勾配蓄積（メモリ不足時）**
   ```python
   accumulation_steps = 4
   for i, (data, target) in enumerate(train_loader):
       output = model(data)
       loss = criterion(output, target)
       loss = loss / accumulation_steps  # 損失のスケーリング
       loss.backward()
       
       if (i+1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **混合精度トレーニング**
   ```python
   scaler = torch.cuda.amp.GradScaler()
   
   for data, target in train_loader:
       optimizer.zero_grad()
       with torch.cuda.amp.autocast():
           output = model(data)
           loss = criterion(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

### よくあるエラーと対策

1. **勾配の未リセット**
   ```python
   # 誤り例
   output = model(input)
   loss.backward()
   optimizer.step()  # 勾配が蓄積される
   
   # 正しい例
   optimizer.zero_grad()
   output = model(input)
   loss.backward()
   optimizer.step()
   ```

2. **パラメータグループの誤指定**
   ```python
   # 誤り例（重み減衰の不適切適用）
   optimizer = Adam(model.parameters(), weight_decay=1e-4)  # BatchNorm層にも適用される
   
   # 正しい例
   params = [
       {'params': model.features.parameters(), 'weight_decay': 1e-4},
       {'params': model.bn.parameters(), 'weight_decay': 0}
   ]
   optimizer = Adam(params, lr=0.001)
   ```

3. **学習率スケジューラの誤使用**
   ```python
   # 誤り例（ステップ位置不適切）
   optimizer.step()
   scheduler.step()  # エポック終了時に実行するべき
   
   # 正しい例
   for epoch in range(epochs):
       for batch in train_loader:
           # ...学習処理...
           optimizer.step()
       scheduler.step()
   ```


つづく...