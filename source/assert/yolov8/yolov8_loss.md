# YOLOv8代码详解（loss和模型层面）

[![平凡的兵](https://picx.zhimg.com/v2-9b337f74799ab40d39d77f43602ec834_l.jpg?source=32738c0c&needBackground=1)](//www.zhihu.com/people/a_bing_jiang)

[平凡的兵](//www.zhihu.com/people/a_bing_jiang)

[​![](https://picx.zhimg.com/v2-2ddc5cc683982648f6f123616fb4ec09_l.png?source=32738c0c)](https://www.zhihu.com/question/48510028)

中国科学技术大学 工学硕士

[

收录于 · 多模态学习

](https://www.zhihu.com/column/c_1659462298400915456)

4 人赞同了该文章

​

目录

收起

1.Loss

2.模型结构

纸上得来终觉浅，绝知此事要躬行。

\--------------------------------------------------------------------------

代码仓库：[https://github.com/ultralytics/ultralytics](https://link.zhihu.com/?target=https%3A//github.com/ultralytics/ultralytics)

![](https://pic2.zhimg.com/v2-1fbf2f04bfcd2a67951a26b892028f5d_1440w.jpg)

以‘yolov8n’模型为例，采用coco128.yaml进行调试，batchsize设为1。

## **1.Loss**

**Loss的计算逻辑有四步：**

（1）基于3个不同尺寸(80x80、40x40、20x20)的特征头进行汇总；

这就是常见的FPN的应用，采用不同尺寸的头，用于大、中、小目标的检测。

（2）基于预测框分布计算出预测框的坐标值x1y1x2y2；

预测框分布的概念有点不好理解，可参考

[![](https://pic1.zhimg.com/v2-166383a8b833f7f098b99ea54a9d8d3a.png?source=7e7ef6e2&needBackground=1)平凡的兵：重温经典（10）--Generalized focal loss1 赞同 · 0 评论](https://zhuanlan.zhihu.com/p/710528970) 文章

（3）基于TaskAlignedLearning进行正负样本分配；

正负样本筛选时，同时考虑定位和分类得分。

TOOD论文中详细介绍了基本原理，可参考

[![](https://picx.zhimg.com/v2-3894821fd74bef787887ca151d2bef4f.png?source=7e7ef6e2&needBackground=1)平凡的兵：重温经典（9）--TOOD2 赞同 · 1 评论](https://zhuanlan.zhihu.com/p/713720285) 文章

（4）计算三类损失：CIOU、DFL、Classification。

关于IOU系列损失，IOU-->GIOU（主要解决IOU=0时的问题）-->DIOU（考虑预测框和标准框的距离）-->CIOU（考虑预测框和标准框长宽一致性）

```python
class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution.""" 
        # 从预测的边界框坐标分布pred_dist和锚点坐标anchor_points，计算出预测框的坐标值x1y1x2y2。
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        ###1.基于3个不同尺寸(80x80、40x40、20x20)的特征头进行汇总；
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # feats是个列表，以‘yolov8n’模型为例，表示三个特征头输出，分别为[1,144,40,40]、[1,144,40,40]、[1,144,20,20]，其中batchsize=1，no=144，输出的特征图大小分别为80x80,40x40,20x20.
        feats = preds[1] if isinstance(preds, tuple) else preds  
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        ) #pred_distri和pred_scores分别为[1,64,8400]、[1,80,8400]

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() #[1,8400,84]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() #[1,8400,64]

        ###2.基于预测框分布计算出预测框的坐标值x1y1x2y2；
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w), (640,640)
        #anchor_points([8400,2])表示Anchor的中心点坐标,stride_tensor([8400,1]),表示中心点的stride
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5) 

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        ###3.基于TaskAlignedLearning进行正负样本分配；
        #target_bboxes,[1,8400,4]
        #target_scores,[1,8400,80]
        #fg_mask,[1,8400],值为True的为正样本，值为Fals的为负样本
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        ###4.计算三类损失：CIOU、DFL、Classification
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
```

  

## 2.模型结构

下面两张全景图展现的非常清晰：

可以先看图(a)再看图(b)，核心思想就几个模块:

（1）CBS（或ConvModule）：由Conv+BN+SiLU组成。

（2）Bottleneck（或DarknetBottleneck）：包含两个卷积层，先减少通道数，再增加通道数，类似颈部。

（3）C2f（或CSPLayer\_2Conv）: CBS和Bottleneck组合的深层网络。

（4）SPPF：快速金字塔特征池化层。

![](https://pic1.zhimg.com/v2-884cdd09ee6e4725c355d4c7712b5a88_1440w.jpg)![](https://pic3.zhimg.com/v2-11b85fcfc90cad3257fe95a6c8198d68_1440w.jpg)

```python
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
```

参考资料：

\[1\][【目标检测】YOLOv8算法实现(一)：模型搭建\_yolov8模型-CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_43676259/article/details/135626687)

\[2\][【目标检测】YOLOv8算法实现(二)：正样本匹配(TaskAlignedAssigner)和损失计算-CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_43676259/article/details/135746237)

\[3\] [https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8](https://link.zhihu.com/?target=https%3A//github.com/open-mmlab/mmyolo/tree/main/configs/yolov8)

编辑于 2024-10-23 12:52・安徽

[

目标检测

](//www.zhihu.com/topic/19596960)

​赞同 4​​添加评论​28 ​喜欢

​分享

​申请转载​

​