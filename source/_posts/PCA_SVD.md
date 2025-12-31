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

数式表現：

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

- 元の基底：$ {[p_1, p_2, ..., p_n]^T} $
- 新しい基底：$ {[x_1, x_2, ..., x_m]} $
- 座標変換：$ P $ が旧座標系でのベクトル、$ x $ が新座標系でのベクトルなら、$ Y = PX $

## 分散
分散は数値が平均値からどれだけ散らばっているかを示す指標です。


# PCA

PCA（主成分分析）は主成分を抽出するための手法である。主成分はデータの分散を最大にするベクトルである。

