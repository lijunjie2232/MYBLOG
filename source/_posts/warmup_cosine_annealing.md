---
title: Warm up と Cosine Anneal LR の組み合わせ


date: 2023-10-21 12:00:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description: あ
---

## 目次

---

## Linear Warmup とは

- **Linear Warmup**: 学習開始時に学習率を 0 から徐々に増加させる手法。初期の大きな更新による不安定性を軽減。


## Cosine Annealing とは

- **Cosine Annealing**: 学習率をコサイン関数のように減少させながら最適解に近づく手法。周期的に復活させる `SGDR`（Stochastic Gradient Descent with Warm Restarts）としても知られる。

### 基本の更新式

<center>$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min}) \left(1 + \cos\left(\frac{T_{cur}}{T_{max}} \pi\right)\right), T_{cur} \ne (2k+1)T_{max} $</center>
<center>$ \eta_{t+1} = \eta_t + \frac{1}{2}(\eta_{max} - \eta_{min}) \left(1 - \cos\left(\frac{T_{cur}}{T_{max}} \pi\right)\right), T_{cur} = (2k+1)T_{max} $</center>

- $t$: 現在のステップ数
- $k$: 現在の周期数（整数）
- $\eta_{t}$: 現在の学習率
- $\eta_{min}$: 最小学習率（デフォルト: 0）
- $\eta_{max}$: 初期学習率（optimizer で設定された値）
- $T_{cur}$: 現在の epoch 数 or step 数
- $T_{max}$: 最大学習ステップ数 or エポック数（半周期）

### コード例

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max,
    eta_min=0,
    last_epoch=-1
)
```

## 📚 参考リンク

- [PyTorch CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
