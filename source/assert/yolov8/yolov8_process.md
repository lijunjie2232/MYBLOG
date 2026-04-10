# YOLOv8解码流程完全解析

创建于： 2024-03-23

**内容目录**

- [YOLOv8解码流程完全解析](#yolov8解码流程完全解析)
- [YOLOv8解码流程完全解析](#yolov8解码流程完全解析-1)
  - [目录](#目录)
  - [1. 预测解码流程 (decode\_predictions)](#1-预测解码流程-decode_predictions)
    - [1.1 网格点生成](#11-网格点生成)
    - [1.2 特征图处理](#12-特征图处理)
    - [1.3 DFL解码实现](#13-dfl解码实现)
    - [1.4 坐标转换](#14-坐标转换)
  - [2. 后处理流程 (post\_process)](#2-后处理流程-post_process)
    - [2.1 置信度过滤](#21-置信度过滤)
    - [2.2 坐标格式转换](#22-坐标格式转换)
    - [2.3 非极大值抑制](#23-非极大值抑制)
    - [2.4 坐标缩放还原](#24-坐标缩放还原)
  - [3. 核心方法解析](#3-核心方法解析)
    - [3.1 dist2bbox方法](#31-dist2bbox方法)
    - [3.2 scale\_boxes方法](#32-scale_boxes方法)
  - [4. 完整代码参考](#4-完整代码参考)
    - [decode\_predictions 完整实现](#decode_predictions-完整实现)
    - [post\_process 完整实现](#post_process-完整实现)

# YOLOv8解码流程完全解析

> 本文详细分析了YOLOv8目标检测算法中的预测解码和后处理机制，包括DFL(Distribution Focal Loss)解码、非极大值抑制(NMS)等关键环节。

## 目录

*   [1\. 预测解码流程 (decode\_predictions)](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#1-%e9%a2%84%e6%b5%8b%e8%a7%a3%e7%a0%81%e6%b5%81%e7%a8%8b-decode_predictions)
    *   [1.1 网格点生成](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#11-%e7%bd%91%e6%a0%bc%e7%82%b9%e7%94%9f%e6%88%90)
    *   [1.2 特征图处理](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#12-%e7%89%b9%e5%be%81%e5%9b%be%e5%a4%84%e7%90%86)
    *   [1.3 DFL解码实现](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#13-dfl%e8%a7%a3%e7%a0%81%e5%ae%9e%e7%8e%b0)
    *   [1.4 坐标转换](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#14-%e5%9d%90%e6%a0%87%e8%bd%ac%e6%8d%a2)
*   [2\. 后处理流程 (post\_process)](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#2-%e5%90%8e%e5%a4%84%e7%90%86%e6%b5%81%e7%a8%8b-post_process)
    *   [2.1 置信度过滤](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#21-%e7%bd%ae%e4%bf%a1%e5%ba%a6%e8%bf%87%e6%bb%a4)
    *   [2.2 坐标格式转换](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#22-%e5%9d%90%e6%a0%87%e6%a0%bc%e5%bc%8f%e8%bd%ac%e6%8d%a2)
    *   [2.3 非极大值抑制](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#23-%e9%9d%9e%e6%9e%81%e5%a4%a7%e5%80%bc%e6%8a%91%e5%88%b6)
    *   [2.4 坐标缩放还原](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#24-%e5%9d%90%e6%a0%87%e7%bc%a9%e6%94%be%e8%bf%98%e5%8e%9f)
*   [3\. 核心方法解析](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#3-%e6%a0%b8%e5%bf%83%e6%96%b9%e6%b3%95%e8%a7%a3%e6%9e%90)
    *   [3.1 dist2bbox方法](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#31-dist2bbox%e6%96%b9%e6%b3%95)
    *   [3.2 scale\_boxes方法](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#32-scale_boxes%e6%96%b9%e6%b3%95)
*   [4\. 完整代码参考](/posts/yolov8%E8%A7%A3%E7%A0%81%E6%B5%81%E7%A8%8B%E5%AE%8C%E5%85%A8%E8%A7%A3%E6%9E%90/#4-%e5%ae%8c%e6%95%b4%e4%bb%a3%e7%a0%81%e5%8f%82%e8%80%83)

## 1\. 预测解码流程 (decode\_predictions)

YOLOv8采用anchor-free设计，预测解码过程将网络输出转换为标准的边界框格式。整个流程可分为以下几个关键步骤：

### 1.1 网格点生成

为每个特征图生成参考点坐标和对应的stride值：

```python
# 生成锚点和对应步长
anchors, strides = (x.transpose(0, 1) for x in 
                  self.make_anchors(predictions[1], self.stride, 0.5))
# anchors: 所有特征图的网格点坐标
# strides: 对应的stride值(8/16/32)
```

这一步完成了：

*   为三个特征图(P3/P4/P5)生成网格点
*   生成对应的stride值(P3:8, P4:16, P5:32)

### 1.2 特征图处理

将三个尺度的特征图预测结果统一处理，分离边界框和类别预测：

```python
# 将三个特征图的预测结果拼接
x_cat = torch.cat([xi.view(1, self.nc + 16 * 4, -1) for xi in predictions[1]], 2)
# P3: (1, nc+64, 6400)  # 80*80=6400
# P4: (1, nc+64, 1600)  # 40*40=1600
# P5: (1, nc+64, 400)   # 20*20=400

# 分离边界框预测和类别预测
box, cls = x_cat.split((16 * 4, self.nc), 1)  
# box: (1, 64, 8400)  # 64=16*4，每个坐标用16个值编码
# cls: (1, nc, 8400)  # nc是类别数
```

维度解包为DFL解码做准备：

```python
b, c, a = box.shape
# b: batch size，通常为1
# c: channels，等于 64 (16*4)，表示每个边界框的编码维度
# a: anchors，等于 8400 (6400+1600+400)，表示所有特征图的网格点总数

# 例如:
box.shape = (1, 64, 8400)
# 则:
b = 1      # 批次大小
c = 64     # 16*4，每个坐标(x,y,w,h)用16个值编码
a = 8400   # 总网格点数 = P3(80*80) + P4(40*40) + P5(20*20)
```

### 1.3 DFL解码实现

YOLOv8使用DFL(Distribution Focal Loss)将离散预测值转换为连续坐标。这一步骤的核心是通过1x1卷积实现加权平均：

```python
# 创建DFL解码用的1x1卷积
# - 输入通道：16（每个坐标的编码维度）
# - 输出通道：1（解码后的单个值）
# - 核大小：1x1
# - 不需要偏置
# - 不需要梯度（固定权重）
conv = nn.Conv2d(16, 1, 1, bias=False).requires_grad_(False)

# 创建数组 [0,1,2,...,15]
x = torch.arange(16, dtype=torch.float) # tensor([0., 1., 2., ..., 15.])

# 设置卷积权重为 [0,1,2,...,15]
conv.weight.data[:] = nn.Parameter(x.view(1, 16, 1, 1))
# 这样卷积操作就相当于加权平均：
# output = 0*p0 + 1*p1 + 2*p2 + ... + 15*p15
```

实现细节解析：

```python
# 1. 创建数组 [0,1,2,...,15]
x = torch.arange(16, dtype=torch.float)
# tensor([0., 1., 2., ..., 15.])

# 2. 重塑为卷积权重的形状
x = x.view(1, 16, 1, 1)
# 1: 输出通道数
# 16: 输入通道数
# 1,1: 卷积核大小
# shape: torch.Size([1, 16, 1, 1])

# 3. 将其设置为卷积层的权重
conv.weight.data[:] = nn.Parameter(x)
# conv.weight.shape = (1, 16, 1, 1)
```

DFL设计的核心原理：

*   使用1x1卷积实现加权平均
*   权重固定为\[0-15\]，不需要学习
*   将16个概率值转换为一个连续的坐标值

示例：

```python
# 如果预测概率分布是：
probs = [0.0, 0.7, 0.3, 0.0, ..., 0.0]
# 通过卷积（加权平均）得到：
result = 1*0.7 + 2*0.3 = 1.3
```

### 1.4 坐标转换

最后，使用DFL解码并转换为实际边界框坐标：

```python
# 重要：使用这些维度进行张量重塑
# 将每个坐标的16个值转换为实际预测值
dfl = conv(box.view(b, 4, 16, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# 转换为实际边界框坐标
dbox = self.dist2bbox(dfl, anchors.unsqueeze(0), xywh=True, dim=1) * strides

# 组合最终结果
return torch.cat((dbox, cls.sigmoid()), 1)
```

## 2\. 后处理流程 (post\_process)

后处理主要完成四个任务：

*   置信度过滤：去除低置信度的检测框
*   坐标转换：将中心点格式转为左上右下角格式
*   NMS处理：去除重叠的检测框，保留最优的
*   尺度还原：将坐标映射回原始图像尺寸

### 2.1 置信度过滤

首先处理输入格式并进行初步的置信度过滤：

```python
# 1. 处理输入格式
prediction = pred[0] if isinstance(pred, (list, tuple)) else pred

# 2. 第一轮置信度过滤
xc = prediction[:, 4:84].amax(1) > conf_thres  # 找出最大类别概率>阈值的框
```

### 2.2 坐标格式转换

将中心点格式转为左上右下角格式：

```python
# 3. 坐标格式转换
prediction = prediction.transpose(-1, -2)
prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # 中心点格式转左上右下角格式

# 4-5. 根据置信度过滤
x = prediction[0]  # 取第一个batch
x = x[xc[0]]      # 保留高置信度的框
```

获取最终的边界框和类别:

```python
# 6-7. 获取最终的边界框和类别
box, cls = x.split((4, self.nc), 1)  # 分离坐标和类别概率
conf, j = cls.max(1, keepdim=True)    # 找出最高概率的类别
x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
```

代码实现详解：

```python
# 1. 首先看数据形状
box.shape  # (N, 4)       - 边界框坐标 (x1,y1,x2,y2)
conf.shape # (N, 1)       - 最大置信度值
j.shape    # (N, 1)       - 对应的类别索引

# 2. torch.cat 拼接操作
x = torch.cat((box, conf, j.float()), 1)
# 结果: (N, 6)
# - 前4列是边界框坐标
# - 第5列是置信度
# - 第6列是类别索引

# 3. conf.view(-1)
conf.view(-1)  # 将形状从(N,1)变为(N,)

# 4. 根据置信度阈值筛选
mask = conf.view(-1) > conf_thres  # 布尔掩码
x = x[mask]  # 只保留置信度大于阈值的行

# 举例：
# 假设有3个检测框
box = torch.tensor([[10,20,30,40],   # 框1
                   [50,60,70,80],   # 框2
                   [90,100,110,120]])# 框3

conf = torch.tensor([[0.9],  # 框1的置信度
                    [0.3],  # 框2的置信度
                    [0.8]]) # 框3的置信度

j = torch.tensor([[0],  # 框1是类别0
                 [1],  # 框2是类别1
                 [2]]) # 框3是类别2

# 拼接并筛选(conf_thres=0.5)
x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > 0.5]
# 结果只保留了框1和框3(置信度>0.5)
```

### 2.3 非极大值抑制

执行非极大值抑制(NMS)去除重叠框：

```python
# 8. 非极大值抑制(NMS)
# 1. 计算类别偏移
c = x[:, 5:6] * max_wh  # max_wh=7680
# 例如:
# 类别0: 0 * 7680 = 0
# 类别1: 1 * 7680 = 7680
# 类别2: 2 * 7680 = 15360

# 2. 将偏移添加到边界框坐标
boxes, scores = x[:, :4] + c, x[:, 4]
# boxes = x[:, :4] + c
# 这样不同类别的框会被分开到不同的空间区域
# 3. 获取置信度分数
# scores = x[:, 4]

i = torchvision.ops.nms(boxes, scores, iou_thres)  # 执行NMS
```

### 2.4 坐标缩放还原

最后将边界框坐标缩放回原始图像尺寸：

```python
# 9-10. 缩放到原图尺寸
pred = x[i]  # 保留NMS后的框
pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
```

## 3\. 核心方法解析

### 3.1 dist2bbox方法

将预测的距离转换为边界框坐标：

```python
def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
    """将预测的距离转换为边界框坐标"""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)
```

### 3.2 scale\_boxes方法

将检测框坐标从模型输入尺寸缩放回原始图像尺寸：

```python
def scale_boxes(self, img1_shape, boxes, img0_shape):
    """
    img1_shape: 模型输入尺寸 (640, 640)
    boxes: 检测框坐标 (x1,y1,x2,y2)
    img0_shape: 原始图片尺寸 (h, w)
    """
    # 1. 计算缩放比例:gain =输入尺寸/原始尺寸
    gain = min(img1_shape[0] / img0_shape[0],    # 高度比
              img1_shape[1] / img0_shape[1])      # 宽度比
    
    # 2. 计算填充量
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,     # 宽度填充
           (img1_shape[0] - img0_shape[0] * gain) / 2)      # 高度填充
    
    # 3. 去除填充
    boxes[..., [0, 2]] -= pad[0]  # x 坐标减去水平填充
    boxes[..., [1, 3]] -= pad[1]  # y 坐标减去垂直填充
    
    # 4. 缩放回原始尺寸
    boxes[..., :4] /= gain
```

示例解析：

```python
# 假设:
img1_shape = (640, 640)    # 模型输入尺寸
img0_shape = (800, 600)    # 原始图片尺寸

# 1. 计算缩放比例
gain = min(640/800, 640/600)  # = min(0.8, 1.067) = 0.8

# 2. 计算填充量
pad_w = (640 - 600 * 0.8) / 2  # 宽度方向的填充
pad_h = (640 - 800 * 0.8) / 2  # 高度方向的填充

# 原始图片缩放后的尺寸: (800*0.8, 600*0.8) = (640, 480)
# 需要填充到 640x640，所以:
# - 宽度两边各填充 (640-480)/2 像素
# - 高度两边各填充 (640-640)/2 = 0 像素
```

缩放示例：

```python
# 假设:
# - 原始图片: 1000x800
# - 模型输入: 640x640
# - gain = 640/1000 = 0.64 (缩放比例)

# 1. 模型预测的检测框(在640x640尺度上)
box = [100, 200, 300, 400]  # [x1, y1, x2, y2]

# 2. 缩放回原始尺寸
box /= gain  # 等价于 box / 0.64
# [100/0.64, 200/0.64, 300/0.64, 400/0.64]
# = [156.25, 312.5, 468.75, 625]
```

## 4\. 完整代码参考

### decode\_predictions 完整实现

```python
def decode_predictions(self, predictions):
    """解码模型的原始输出为检测框和类别概率
    Args:
        predictions: 模型输出的原始预测结果
    """
    # 生成锚点和对应步长
    anchors, strides = (x.transpose(0, 1) for x in 
                      self.make_anchors(predictions[1], self.stride, 0.5))
    
    # 将三个特征图的预测结果拼接
    x_cat = torch.cat([xi.view(1, self.nc + 16 * 4, -1) for xi in predictions[1]], 2)
    # 分离边界框预测和类别预测
    box, cls = x_cat.split((16 * 4, self.nc), 1)  # 16*4用于DFL解码，nc为类别数
    b, c, a = box.shape
    conv = nn.Conv2d(16, 1, 1, bias=False).requires_grad_(False)
    x = torch.arange(16, dtype=torch.float)
    conv.weight.data[:] = nn.Parameter(x.view(1, 16, 1, 1))
    
    dfl = conv(box.view(b, 4, 16, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    dbox = self.dist2bbox(dfl, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    
    return torch.cat((dbox, cls.sigmoid()), 1)
```

### post\_process 完整实现

```python
def post_process(self, pred, img, orig_img, conf_thres=0.25, iou_thres=0.7, max_wh=7680):
    """后处理：执行非极大值抑制（NMS）"""
    # 1. 处理验证模式的输出
    prediction = pred[0] if isinstance(pred, (list, tuple)) else pred

    # 2. 根据置信度阈值筛选候选框
    xc = prediction[:, 4:84].amax(1) > conf_thres

    # 3. 调整预测结果的形状和格式
    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])

    # 4. 获取当前batch的预测结果
    xi = 0
    x = prediction[xi]
    
    # 5. 根据置信度过滤预测框
    x = x[xc[xi]]

    # 6. 分离边界框和类别信息
    box, cls = x.split((4, self.nc), 1)
    
    # 7. 获取每个框的最大置信度和对应的类别
    conf, j = cls.max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # 8. 执行NMS
    c = x[:, 5:6] * max_wh
    boxes, scores = x[:, :4] + c, x[:, 4]
    i = torchvision.ops.nms(boxes, scores, iou_thres)

    # 9. 获取最终预测结果
    pred = x[i]
    
    # 10. 将边界框坐标缩放到原始图片尺寸
    pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

    return pred
```