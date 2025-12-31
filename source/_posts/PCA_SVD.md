---
title: PCAとSVDの本質
date: 2024-6-4 21：30:04
categories: [AI]
tags: [Deep Learning, 機械学習, AI, 人工知能, 深層学習, machine learning]
lang: ja　
description: pass

---


# 数学基礎

## 基底変換

### 数式表現

$$
Y = PX = \begin{bmatrix}
p_1 \\
p_2 \\
\vdots \\
p_r
\end{bmatrix}_{r\times n}[
\begin{array}
{cccccc}x_1 & x_2 & \cdots & x_m
\end{array}]_{n\times m}=
\begin{bmatrix}
p_1x_1 & p_1x_2 & \cdots & p_1x_m \\
p_2x_1 & p_2x_2 & \cdots & p_2x_m \\
\vdots & \vdots & \ddots & \vdots \\
p_rx_1 & p_rx_2 & \cdots & p_rx_m
\end{bmatrix}_{r\times m}
$$

- 元の基底：$ {[p_1, p_2, ..., p_n]}^T $
- 新しい基底：$ {[x_1, x_2, ..., x_m]} $
- 座標変換：$ P $ が旧座標系でのベクトル、$ x $ が新座標系でのベクトルなら、$ Y = PX $

### 本質

![base change](/assert/PCA_SVD/base_change.png)

行列掛け算の本質は**基底変換**である。


## 分散

分散は数値が平均値からどれだけ散らばっているかを示す指標です。

数式：
$$ \text{Var}(X) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2 $$

ここで：

- $ x_i $：個々のデータ点
- $ \bar{x} $：データの平均
- $ n $：データ点の数

## 共分散

共分散は2つの変数が一緒にどう変化するかを測る指標で、同時に増加する傾向があるかを示します。

### 数式
$$ \text{Cov}(X,Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) $$

ここで：

- $ x_i, y_i $：2つの異なる変数のデータ点
- $ \bar{x}, \bar{y} $：それぞれの変数の平均


解釈：
- 正の共分散：変数が一緒に増加する傾向
- 負の共分散：一方が増えると他方は減る
- ゼロの共分散：線形関係なし

## 共分散行列とその対角化

共分散行列は多変量データセットの各次元ペア間の共分散を含む正方行列です。

### 数式

1. サンプル平均
$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{N}{x_{i}}
$$
2. サンプル分散
$$
Var(x)=S^{2}=\frac{1}{n-1}\sum_{i=1}^{n}{\left( x_{i}-\bar{x} \right)^2}
$$
3. サンプルの共分散
$$
Cov(x,y)=\frac{1}{n-1}\sum_{i=1}^{n}{\left( x_{i}-\bar{x} \right)\left( y_{i}-\bar{y} \right)}
$$
4. サンプルの共分散行列

ここで、$dim=2$のベクトル${[x,y]}^T$にしては：

$$
C=
\begin{bmatrix}
Var(x) & Cov(x,y) \\
Cov(x,y) & Var(y)
\end{bmatrix}
$$

$dim=n$のベクトル${[x_1,x_2,...,x_n]}^T$にしては：

$$
C=
\begin{bmatrix}
Var(x_1) & Cov(x_1,x_2) & \cdots & Cov(x_1,x_n) \\
Cov(x_2,x_1) & Var(x_2) & \cdots & Cov(x_1,x_n) \\
\vdots & \vdots & \ddots & \vdots \\
Cov(x_n,x_1) & Cov(x_n,x_2) & \cdots & Var(x_n)
\end{bmatrix}
$$


# SVD

## SVDの定義

特異値分解は、任意の行列に適用可能な分解方法であり、任意の行列Aに対して常に特異値分解が存在します。

## 数式表現
$$A = U \Sigma V^T$$

ここで：
- **A**：$m \times n$ の行列
- **U**：$m \times m$ の直交行列（左特異ベクトル）
- **Σ**：$m \times n$ の行列（対角線上以外はすべて0、対角線上の要素は特異値）
- **V^T**：$n \times n$ の直交行列の転置（右特異ベクトル）

## SVD分解の手順

1. $AA^T$ の固有値と固有ベクトルを求める
- $AA^T$ の固有値と固有ベクトルを計算
- 固有ベクトルを正規化して行列 **U** を構成
2. $A^TA$ の固有値と固有ベクトルを求める
- $A^TA$ の固有値と固有ベクトルを計算
- 固有ベクトルを正規化して行列 **V** を構成
3. 特異値の計算
- $AA^T$ または $A^TA$ の固有値の平方根を計算
- その値を対角要素として行列 **Σ** を構成

## 重要な特徴

- **特異値**は通常、大きい順に並べられます
- **U** は左特異ベクトル（$AA^T$ の固有ベクトル）
- **V** は右特異ベクトル（$A^TA$ の固有ベクトル）
- **Σ** の対角要素が特異値であり、行列Aの「重要度」を示します

# PCA

PCA（主成分分析）は主成分を抽出するための手法である。主成分はデータの分散を最大にするベクトルである。

## 共分散行列対角化

ここで、二つの行列$Y_{r \times m}$と$X_{n \times m}$がある。

そして、
1. $P_{r \times n}$があり、$$Y_{r \times m}=P_{r \times n}X_{n \times m}$$
2. $C$は$X$の共分散行列、$D$は$Y$の共分散行列である

<font color="red">目的: 元のデータ $X$ にPCAを適用した後、得られる $Y$ の共分散行列 $D$ の各方向の分散が最大になり、共分散が $0$ になることです。</font>

では、$C$ と $D$ はどのような関係にあるのでしょうか：
$$
D = 

$$