---
title: YOLOv8 リアルタイム物体検出3（損失関数の詳細解説）

date: 2024-7-21 11:15:00
categories: [AI]
tags: [Deep Learning, PyTorch, Python, Object Detection, 機械学習, AI, 人工知能, 深層学習, 物体検出, YOLO, リアルタイム検出, 損失関数]
lang: ja
description: YOLOv8の損失関数の詳細な実装と理論解説
---


# YOLOv8 の損失関数：詳細解説

この文書は、[YOLOv8 リアルタイム物体検出2（高度技術解析：Anchor-Free Object Detection / DFL / TAL）](../../../../2024/07/17/yolov8_advanced_anchor_free/) の補完としてYOLOv8 の損失関数について解説します。その中で細かい部分にはそちらの説明を参考にしてください。

## 概要

YOLOv8 の損失関数は、**3つの主要コンポーネント**で構成されています：

1. **Classification Loss**（分類損失）- BCE Loss
2. **Box Loss**（境界ボックス損失）- CIoU Loss
3. **DFL Loss**（Distribution Focal Loss）- 分布焦点損失

総損失は以下の式で表されます：

$$L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{dfl} L_{dfl}$$

ここで、$\lambda_{box}$、$\lambda_{cls}$、$\lambda_{dfl}$ は各損失の重み係数です。

## 損失計算の全体フロー

YOLOv8 の損失計算は4つの主要ステップで実行されます：

{% mermaid %}
graph TB
    Start["入力: preds, batch"] --> Step1["Step 1: 複数スケール特徴の統合"]
    
    Step1 --> concat_feats["3つの特徴マップを結合<br/>80x80, 40x40, 20x20 → 8400アンカー点<br/>pred_distri: [b, 8400, 64]<br/>pred_scores: [b, 8400, 80]"]
    
    concat_feats --> Step2["Step 2: 予測ボックス座標のデコード"]
    
    Step2 --> bbox_decode["bbox_decode関数<br/>DFL分布からx1y1x2y2座標へ変換<br/>softmax + matmulで期待値計算"]
    
    bbox_decode --> Step3["Step 3: TALによる正負样本分配"]
    
    Step3 --> assigner["TaskAlignedAssigner<br/>target_bboxes: [b, 8400, 4]<br/>target_scores: [b, 8400, 80]<br/>fg_mask: [b, 8400] (bool)"]
    
    assigner --> Step4["Step 4: 3種類の損失を計算"]
    
    Step4 --> calc_cls["Classification Loss<br/>BCE(pred_scores, target_scores)"]
    Step4 --> calc_box["Box Loss<br/>CIoU(pred_bboxes, target_bboxes)"]
    Step4 --> calc_dfl["DFL Loss<br/>分布間のクロスエントロピー"]
    
    calc_cls --> weighted["重み付けと合計<br/>loss *= hyp.gain<br/>return loss.sum() * batch_size"]
    calc_box --> weighted
    calc_dfl --> weighted
    
    weighted --> Output["出力: 総損失, 各項の損失"]
    
    style Start fill:#e1f5ff
    style Step1 fill:#fff9c4
    style Step2 fill:#fff9c4
    style Step3 fill:#fff9c4
    style Step4 fill:#fff9c4
    style concat_feats fill:#e8f5e9
    style bbox_decode fill:#e8f5e9
    style assigner fill:#e8f5e9
    style calc_cls fill:#ffe0b2
    style calc_box fill:#ffe0b2
    style calc_dfl fill:#ffe0b2
    style weighted fill:#ffcdd2
    style Output fill:#c8e6c9
{% endmermaid %}

## Step 1: 複数スケール特徴の統合

YOLOv8 は Feature Pyramid Network (FPN) を採用し、3つの異なるスケールで物体を検出します：

```python
# v8DetectionLoss.__call__ から抜粋
feats = preds[1] if isinstance(preds, tuple) else preds

# 3つの特徴マップを結合してフラット化
# feats[0]: [b, no, 80, 80] → [b, no, 6400]
# feats[1]: [b, no, 40, 40] → [b, no, 1600]
# feats[2]: [b, no, 20, 20] → [b, no, 400]
# 合計: [b, no, 8400]
pred_distri, pred_scores = torch.cat(
    [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
).split((self.reg_max * 4, self.nc), 1)

# pred_distri: [b, 64, 8400] - 境界ボックス分布
# pred_scores: [b, 80, 8400] - 分類スコア

# 次元を入れ替えてアンカー点基準に
pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [b, 8400, 80]
pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [b, 8400, 64]
```

**重要なポイント**：
- **80x80 特徴マップ**: 小物体検出用（stride=8）
- **40x40 特徴マップ**: 中規模物体検出用（stride=16）
- **20x20 特徴マップ**: 大物体検出用（stride=32）
- 合計 **8400 アンカー点** = 6400 + 1600 + 400
- `no = nc + reg_max * 4` = 80 + 16*4 = 144 （COCOデータセットの場合）

## Step 2: 予測ボックス座標のデコード

```python
def bbox_decode(self, anchor_points, pred_dist):
    """
    アンカー点と分布から予測された境界ボックス座標をデコードします。
    
    Args:
        anchor_points (Tensor): shape(num_anchors, 2) - アンカー中心座標
        pred_dist (Tensor): shape(b, num_anchors, reg_max*4) - 予測分布
    
    Returns:
        pred_bboxes (Tensor): shape(b, num_anchors, 4) - デコードされたボックス (x1y1x2y2)
    """
    if self.use_dfl:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        
        # 形状を変換: [b, a, 64] → [b, a, 4, 16]
        # 4方向 (left, top, right, bottom) × 16ビン
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        
        # softmax で確率分布に変換
        # matmul で期待値を計算: Σ P(yi) * yi
        pred_dist = pred_dist.softmax(3).matmul(self.proj.type(pred_dist.dtype))
        
        # pred_dist: [b, a, 4] - 各方向の相対オフセット
    
    # 相対オフセットから絶対座標へ変換
    return dist2bbox(pred_dist, anchor_points, xywh=False)
```

**DFL デコードの詳細**：

1. **形状変換**: `[b, 8400, 64]` → `[b, 8400, 4, 16]`
   - 4方向: left, top, right, bottom
   - 16ビン: reg_max=16

2. **Softmax**: 各方向の16ビンで確率分布に変換

3. **期待値計算**: 行列乗算で加重平均
   ```python
   offset_left = Σ_{i=0}^{15} P(left_i) * i
   offset_top = Σ_{i=0}^{15} P(top_i) * i
   offset_right = Σ_{i=0}^{15} P(right_i) * i
   offset_bottom = Σ_{i=0}^{15} P(bottom_i) * i
   ```

4. **座標変換**: `dist2bbox` でアンカー中心からの相対オフセットを絶対座標に変換
   ```python
   x1 = anchor_x - offset_left
   y1 = anchor_y - offset_top
   x2 = anchor_x + offset_right
   y2 = anchor_y + offset_bottom
   ```

## Step 3: TAL による正負样本分配

Task-Aligned Assigner を使用して、各アンカー点を正サンプルまたは負サンプルに割り当てます：

```python
# TAL による样本分配
_, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(),           # 分類スコア [b, 8400, 80]
    (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # 予測ボックス
    anchor_points * stride_tensor,             # アンカー点座標
    gt_labels,                                  # GT ラベル
    gt_bboxes,                                  # GT ボックス
    mask_gt,                                    # GT マスク
)

# 出力:
# target_bboxes: [b, 8400, 4] - 目標ボックス座標
# target_scores: [b, 8400, 80] - 目標スコア（正規化されたアライメント指標）
# fg_mask: [b, 8400] - 前景マスク（True=正サンプル, False=負サンプル）
```

**TAL の役割**：
- 高品質なアンカー点（高い分類スコア **かつ** 高い IoU）のみを正サンプルとして選択
- 各アンカー点に対して、目標ボックスと目標スコアを提供
- 损失計算時に正サンプルのみを使用（fg_mask でフィルタリング）

## Step 4: 3種類の損失を計算

### 1. Classification Loss (BCE)

二値交差エントロピー損失を使用して分類タスクを最適化します：

```python
# Cls loss
target_scores_sum = max(target_scores.sum(), 1)  # 正規化用の合計
loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
```

**BCE Loss の詳細**：

$$L_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \left[ t_i \log(p_i) + (1-t_i) \log(1-p_i) \right]$$

ここで：
- $p_i$: 予測された分類スコア（sigmoid 後）
- $t_i$: 目標スコア（TAL によって正規化されたアライメント指標）
- $N$: 正サンプルの数

**重要なポイント**：
- 目標ラベルは二元 {0, 1} ではなく、連続値 $\hat{t} \in [0, 1]$
- 高品質なサンプルほど大きな目標スコアを持つ
- `target_scores_sum` で正規化してバッチサイズの影響を軽減

### 2. Box Loss (CIoU)

Complete IoU 損失を使用して境界ボックスの位置回帰を最適化します：

```python
# Bbox loss
if fg_mask.sum():  # 正サンプルが存在する場合のみ計算
    target_bboxes /= stride_tensor  # ストライドで正規化
    loss[0], loss[2] = self.bbox_loss(
        pred_distri,      # 予測分布 [b, 8400, 64]
        pred_bboxes,      # 予測ボックス [b, 8400, 4]
        anchor_points,    # アンカー点
        target_bboxes,    # 目標ボックス [b, 8400, 4]
        target_scores,    # 目標スコア [b, 8400, 80]
        target_scores_sum, # 正規化定数
        fg_mask           # 前景マスク [b, 8400]
    )
```

**CIoU Loss の詳細**：

CIoU (Complete IoU) は、IoU に加えて以下の要素を考慮します：

$$L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

ここで：
- $IoU$: 予測ボックスと GT ボックスの交差比
- $\rho(b, b^{gt})$: 2つのボックス中心間のユークリッド距離
- $c$: 最小外接矩形の対角線長さ
- $v$: アスペクト比の一貫性
- $\alpha$: トレードオフパラメータ

**CIoU の進化**：
1. **IoU**: 基本的な重なり評価（ただし IoU=0 の場合勾配消失）
2. **GIoU**: IoU=0 の問題を解決
3. **DIoU**: 中心点距離を考慮
4. **CIoU**: アスペクト比の一貫性も考慮（最も包括的）

### 3. DFL Loss (Distribution Focal Loss)

分布焦点損失を使用して、境界ボックス分布の学習を最適化します：

```python
# BboxLoss クラス内
def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, 
            target_scores, target_scores_sum, fg_mask):
    """IoU loss と DFL loss を計算します。"""
    
    # IoU loss
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], 
                   xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    
    # DFL loss
    if self.use_dfl:
        # GT ボックスを相対オフセットに変換
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
        
        # DFL 損失を計算
        loss_dfl = self._df_loss(
            pred_dist[fg_mask].view(-1, self.reg_max + 1), 
            target_ltrb[fg_mask]
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.tensor(0.0).to(pred_dist.device)
    
    return loss_iou, loss_dfl

@staticmethod
def _df_loss(pred_dist, target):
    """
    DFL 損失を計算します。
    Generalized Focal Loss で提案された Distribution Focal Loss。
    
    Args:
        pred_dist (Tensor): shape(bs * num_anchors * 4, reg_max + 1)
                           予測された確率分布
        target (Tensor): shape(bs * num_anchors * 4,)
                        目標オフセット（連続値）
    
    Returns:
        loss (Tensor): shape(bs * num_anchors * 4, 1) - DFL 損失
    """
    tl = target.long()     # y_i (floor)
    tr = tl + 1            # y_{i+1} (ceil)
    wl = tr - target       # y_{i+1} - y (左側の重み)
    wr = 1 - wl            # y - y_i (右側の重み)
    
    return (
        F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
        + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
    ).mean(-1, keepdim=True)
```

**DFL Loss の数学的理解**：

$$L_{DFL}(\mathcal{S}_i, \mathcal{S}_{i+1}) = -\left[(y_{i+1} - y) \log(\mathcal{S}_i) + (y - y_i) \log(\mathcal{S}_{i+1})\right]$$

ここで：
- $y$: 目標オフセット（連続値、例: 5.7）
- $y_i = \lfloor y \rfloor$: floor 値（例: 5）
- $y_{i+1} = \lceil y \rceil$: ceil 値（例: 6）
- $\mathcal{S}_i, \mathcal{S}_{i+1}$: 予測された確率
- $w_l = y_{i+1} - y$: 左側の重み（例: 0.3）
- $w_r = y - y_i$: 右側の重み（例: 0.7）

**DFL の利点**：
1. **連続ラベルの処理**: 離散ビンで連続値を表現
2. **分布形状の誘導**: 隣接ビンの確率を滑らかに
3. **計算効率**: 2つのクロスエントロピーのみ計算
4. **柔軟性**: 不確実性を確率分布で表現

## 損失の重み付けと合計

最終的に、3つの損失成分を重み付けして合計します：

```python
# 各損失にハイパーパラメータの重みを適用
loss[0] *= self.hyp.box  # box gain (デフォルト: 7.5)
loss[1] *= self.hyp.cls  # cls gain (デフォルト: 0.5)
loss[2] *= self.hyp.dfl  # dfl gain (デフォルト: 1.5)

# 総損失を返す
return loss.sum() * batch_size, loss.detach()
```

**重み係数の意味**：
- `hyp.box = 7.5`: 境界ボックス損失の重み（最も重要）
- `hyp.cls = 0.5`: 分類損失の重み
- `hyp.dfl = 1.5`: DFL 損失の重み

これらの重みは、データセットやタスクに応じて調整可能です。


## **関連資料**

- [YOLOv8 リアルタイム物体検出1 (基礎)](../../../../2024/07/11/yolov8_basic/) - YOLOv8 architecture and basic concepts
- [YOLOv8 リアルタイム物体検出2（高度技術解析：Anchor-Free Object Detection / DFL / TAL）](../../../../2024/07/17/yolov8_advanced_anchor_free/) - YOLOv8 advanced concepts and technical details
