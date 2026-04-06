目标检测发展多年，YOLO 系列一直以实时、高效为特点。YOLOv8 不仅在普通目标检测上表现优异，同时支持 **旋转目标检测（Oriented Object Detection，OBB）** 和 **实例分割（Instance Segmentation），还有其他的pose和分类等等，这里暂不做介绍**。  
这两类任务本质上都是在 **原始 YOLO 检测头（Detect Head）基础上做扩展**，但扩展方式非常典型且具工程特色。

本文从理论 → 结构 → YOLOv8 代码 → 数据维度 逐层讲解其差异，一方面能巩固所学，另一方面帮刚学的萌新多一条了解的途径。

***

## 1\. YOLOv8 网络结构概览

![](https://i-blog.csdnimg.cn/direct/598a1aee9f1a442cbe652bb290b88f5f.png)

 YOLOv8 延续了 YOLOv5/YOLOX 的经典架构：

```bash
Input Image   ↓Backbone（CSPDarknet-lite） —— 特征提取   ↓Neck（FPN + PAN）     （中间那部分网络） —— 多尺度特征融合   ↓Head（Detect / Segment / OBB）
```

Backbone 负责提取高、中、低层语义特征；  
Neck 用特征融合确保多尺度感受野；  
最终的检测、分割或旋转框预测工作全部交给 **Head** 完成。

接下来介绍YOLOv8 的 Head 的三种形态（还有其他的类似于pose暂且不介绍）：

*   **Detect：普通目标检测（水平框 HBB）**
    
*   **Segment：实例分割**
    
*   **OBB：旋转目标检测（Oriented Bounding Box）**
    

***

## 2\. YOLOv8 基础检测头（Detect Head）原理

关于检测头的具体分析可以看我另外的博客：[https://blog.csdn.net/weixin\_44115575/article/details/148161298?fromshare=blogdetail&sharetype=blogdetail&sharerId=148161298&sharerefer=PC&sharesource=weixin\_44115575&sharefrom=from\_link![](https://csdnimg.cn/release/blog_editor_html/release2.4.4/ckeditor/plugins/CsdnLink/icons/icon-default.png?t=P9T8)https://blog.csdn.net/weixin\_44115575/article/details/148161298?fromshare=blogdetail&sharetype=blogdetail&sharerId=148161298&sharerefer=PC&sharesource=weixin\_44115575&sharefrom=from\_link](https://blog.csdn.net/weixin_44115575/article/details/148161298?fromshare=blogdetail&sharetype=blogdetail&sharerId=148161298&sharerefer=PC&sharesource=weixin_44115575&sharefrom=from_link "https://blog.csdn.net/weixin_44115575/article/details/148161298?fromshare=blogdetail&sharetype=blogdetail&sharerId=148161298&sharerefer=PC&sharesource=weixin_44115575&sharefrom=from_link")

YOLOv8 的检测任务本质是：

> 在特征图每个网格点（grid cell）上，预测对应的 **bbox（4 个分布回归参数）+ 分类（class logits）**。

YOLOv8 引入了 **DFL（Distribution Focal Loss）** 用于边界框回归，这使得 Head 的输出维度不同于传统的 `xywh` 直接回归。

#### ▪ 检测头的输出格式（DFL 表达）

对于每个预测点：

*   **4 个边界框坐标 → 每个坐标用 reg\_max=16 离散分布表示**
    
*   即每个坐标预测 16 个概率，共 **4×16 = 64 通道**
    
*   类别预测通道数 = nc（例如 COCO 为 80）
    

所以 Head 的输出通道为：

```cobol
no = 4 * reg_max + nc = 64 + 80 = 144
```

#### ▪ Detect Head 的结构（多分支卷积）

对应 YOLOv8 代码：

📍 **文件**：**`ultralytics/nn/modules/head.py`**  
📍 **类**：`class Detect(nn.Module)`  
📍 **关键代码行**：

```python
class Detect(nn.Module):    dynamic = False  # force grid reconstruction    export = False  # export mode    format = None  # export format    end2end = False  # end2end    max_det = 300  # max_det    shape = None    anchors = torch.empty(0)  # init    strides = torch.empty(0)  # init    legacy = False  # backward compatibility for v3/v5/v8/v9 models    xyxy = False  # xyxy or xywh output     def __init__(self, nc: int = 80, ch: Tuple = ()):        """        Initialize the YOLO detection layer with specified number of classes and channels.        Args:            nc (int): Number of classes.            ch (tuple): Tuple of channel sizes from backbone feature maps.        """        super().__init__()        self.nc = nc  # number of classes        self.nl = len(ch)  # number of detection layers        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)        self.no = nc + self.reg_max * 4  # number of outputs per anchor        self.stride = torch.zeros(self.nl)  # strides computed during build        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels        self.cv2 = nn.ModuleList(            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch        )        self.cv3 = (            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)            if self.legacy            else nn.ModuleList(                nn.Sequential(                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),                    nn.Conv2d(c3, self.nc, 1),                )                for x in ch            )        )        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()         if self.end2end:            self.one2one_cv2 = copy.deepcopy(self.cv2)            self.one2one_cv3 = copy.deepcopy(self.cv3)     def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:        """Concatenate and return predicted bounding boxes and class probabilities."""        if self.end2end:            return self.forward_end2end(x)         for i in range(self.nl):            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)        if self.training:  # Training path            return x        y = self._inference(x)        return y if self.export else (y, x)
```

```python
# bbox 回归分支self.cv2 = nn.ModuleList(    nn.Sequential(        Conv(x, c2, 3),        Conv(c2, c2, 3),        nn.Conv2d(c2, 4 * self.reg_max, 1)    ) for x in ch)   
```

```python
# 分类分支self.cv3 = nn.ModuleList(    nn.Sequential(        ...        nn.Conv2d(c3, self.nc, 1)    ) for x in ch)  
```

#### ▪ forward 输出（训练时）

Detect 的 forward 做了两件事：

1.  每个尺度拼接 box 和 cls：
    
    ```python
    x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    ```
    
2.  **训练状态**直接返回**，不 decode**：
    
    ```python
    if self.training:    return x
    ```
    

例如输入为 640×640 图像，FPN 输出三层：

| 层 | 特征图尺寸 | 输出 shape（B=1 时） |
| --- | --- | --- |
| P3 | 80×80 | \[1, 144, 80, 80\] |
| P4 | 40×40 | \[1, 144, 40, 40\] |
| P5 | 20×20 | \[1, 144, 20, 20\] |

总预测点数：  
`80×80 + 40×40 + 20×20 = 8400`。

***

## 3\. 旋转检测（OBB）在原检测头上做了什么改动？

### 🔥 关键思想：

> **在普通检测的基础上“多预测一个角度 θ”，输出从 4 个框参数变成 5 个框参数**

旋转框（Oriented Bounding Box, OBB）一般表示为：

![](https://i-blog.csdnimg.cn/direct/38b5692979694c2fbfdded997661f7e5.png)

θ 是旋转角度（常见定义：绕 z 轴逆时针旋转，范围 -90° ~ 90°）

所以检测头输出：

从：

![](https://i-blog.csdnimg.cn/direct/3419aa9d8cf841eab967259d0732dc4a.png)

变成：

![](https://i-blog.csdnimg.cn/direct/94541dde257e44338b35b713fecef718.png)

也就是多一个通道angle，Head 输出变成：

```bash
[ bbox(DFL) + obj + cls + angle ]
```

**损失函数改为支持旋转框**

普通 YOLO 的 IoU 是水平框 IoU  
旋转检测需要 **Rotated IoU (rIoU) / Skew IoU / GIoU / CIoU（旋转版）**。

具体就不细讲了，反正**loss 中的 IoU 要换成支持旋转框的版本**。

***

### 3.1 OBB 代码结构

📍 **文件路径**：`ultralytics/nn/modules/head.py`  
📍 **类名**：`class OBB(Detect)`

```python
class OBB(Detect):    def __init__(self, nc: int = 80, ne: int = 1, ch: Tuple = ()):        """        Initialize OBB with number of classes `nc` and layer channels `ch`.        Args:            nc (int): Number of classes.            ne (int): Number of extra parameters.            ch (tuple): Tuple of channel sizes from backbone feature maps.        """        super().__init__(nc, ch)        self.ne = ne  # number of extra parameters         c4 = max(ch[0] // 4, self.ne)        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)     def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:        """Concatenate and return predicted bounding boxes and class probabilities."""        bs = x[0].shape[0]  # batch size        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]        if not self.training:            self.angle = angle        x = Detect.forward(self, x)        if self.training:            return x, angle        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))     def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:        """Decode rotated bounding boxes."""        return dist2rbox(bboxes, self.angle, anchors, dim=1)
```

#### ▪ 在 **init** 中多了一个新分支 cv4：

```python
self.ne = ne  # extra parameters (angle)self.cv4 = nn.ModuleList(    nn.Sequential(        Conv(x, c4, 3),        Conv(c4, c4, 3),        nn.Conv2d(c4, self.ne, 1)    ) for x in ch)   
```

**OBB 用 `cv4` 预测 angle。**

#### ▪ forward 输出角度

```python
angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1)                  for i in range(self.nl)], 2)  angle = (angle.sigmoid() - 0.25) * math.pi        # 角度范围
```

#### ▪ decode 阶段，替换 decode\_bboxes

```python
return dist2rbox(bboxes, self.angle, anchors, dim=1)
```

把水平框 decode 换成**旋转框 decode**。

***

### 3.2 OBB 的数据例子（以 640×640 输入为例）

bbox + cls 还是：

```python
[1, 144, 8400]
```

OBB 的 angle 输出：

```python
angle = [1, 1, 8400]
```

最终推理时拼成：

```python
[1, 144+1, 8400] = [1, 145, 8400]
```

解码后得到：

```python
[x, y, w, h, θ, class_probs]
```

即旋转框 (OBB)。

***

## 4\. 分割头（Segment）在原检测头上增加了什么？

Segment 相比 Detect 多了 **两条分支**：

**（1）增加一个 Prototype Mask 分支（生成一堆共享的“基础掩码”）**

**（2）在检测头里为每个目标增加一组 Mask 系数（用来组合基础掩码）**

最终的 mask = **基础掩码 × 系数**。其它 backbone、neck 全都不动。

***

###  4.1 YOLOv8-seg 的结构示意图

```cobol
               +----------------------+Backbone ----> |        Neck          |               +----------+-----------+                          |                          +--------------------+                          |                    |                       Det Head            Proto Head                          |                    |   bbox + cls + mask_coeffs (32)      32 × H' × W' prototypes                          |                    |                          ------- Combine -----                                 (matrix multiply)                                       ↓                               Upsample / Crop                                       ↓                                  Final Mask
```

***

### 4.2 代码

📍 **文件路径**：`ultralytics/nn/modules/head.py`  
📍 **类名**：`class Segment(Detect)`

```python
class Segment(Detect):    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: Tuple = ()):        """        Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.        Args:            nc (int): Number of classes.            nm (int): Number of masks.            npr (int): Number of protos.            ch (tuple): Tuple of channel sizes from backbone feature maps.        """        super().__init__(nc, ch)        self.nm = nm  # number of masks        self.npr = npr  # number of protos        self.proto = Proto(ch[0], self.npr, self.nm)  # protos         c4 = max(ch[0] // 4, self.nm)        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)     def forward(self, x: List[torch.Tensor]) -> Union[Tuple, List[torch.Tensor]]:        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""        p = self.proto(x[0])  # mask protos        bs = p.shape[0]  # batch size         mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients        x = Detect.forward(self, x)        if self.training:            return x, mc, p        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p)) 
```

***

### 4.3 YOLOv8 分割增加了什么结构？

#### 🔹 **1）在 neck 输出的最高分辨率的 feature map 上，加一个 “Mask 原型头（Prototype Head）”**

示意图如下：

```scss
Backbone → Neck → (普通检测头)                 → (原型 mask head)  ← 这是新增的！
```

这个 Prototype Head 输出：

*   通常是 **32 个 mask prototype（P）**
    
*   大小一般是 160×160（取决于输入）
    

**对于任意一个目标的形状，都可以用这 32 张图 线性组合 得到。**形状一般是：

![](https://i-blog.csdnimg.cn/direct/3d4c8e90cb92496fac3ecba82d748a94.png)

这些是：

✔ 全图共享

✔ 不针对任何一个具体目标

✔ 只是一些“基础”可组合的模板

***

#### 🔹 **2）检测头（Detection Head）多输出 mask coefficients（系数）**

普通 YOLOv8 检测头输出：

```python
x, y, w, h, objectness, class_probs
```

YOLOv8-seg 多加：

```python
mask_coeff (32 个数)
```

所以：每个预测框多预测一个**长度 32 的向量** 

***

### 4.4 如何用 Prototypes + Coefficients 得到最终 mask？

![](https://i-blog.csdnimg.cn/direct/b3e8844b677a418cb073db6b04c06b57.png)

用矩阵形式：

```python
FinalMask = Prototypes (H'× W'×32) × coeff (32 × 1)
```

得到一张 **低分辨率(160 \* 160) mask** 。

然后再：

*   上采样（upsample）到原图大小
    
*   在 bbox 区域裁剪
    
*   sigmoid → 二值化
    

就成了 YOLOv8 的实例 mask。

***

### 4.5 为什么不能“每个实例单独预测一张 mask”？

因为：

*   图像里可能有几十个实例
    
*   每个实例单独用卷积生成一张 mask → 极其耗时（Mask R-CNN 就比较慢）
    
*   YOLO 系列目标是高效实时
    

所以 YOLOv8（YOLACT/YOLACT++ 的 idea）选了一个非常高效的做法：

**→ 一整张图只生成 K 张“基础模板 mask”（prototypes）**

**→ 每个实例只是输出 K 个权重（mask 系数）**

然后：

![](https://i-blog.csdnimg.cn/direct/986ffc6fbefa48d6af75ba4ecfcd1b21.png)

***

### 4.6 分割头数据维度例子（以 640×640 输入为例）

**用 640×640 的具体维度举例子：**

##### ① 原型 `p = self.proto(x[0])`

*   输入 **`x[0]（分辨率最高）`**: `[B, 256, 80, 80]`
    
*   `Proto` 一般是几层 Conv：
    
    *   中间通道 npr=256
        
    *   最终输出 nm=32 通道
        

所以：

**`p shape = [B, 32, 80, 80]`**

> 这是 **整张图共享的 32 张“基础 mask 模板”**（prototype）。

##### ② mask 系数 `mc`

对每个尺度 `x[i]`：

*   P3: `x[0] = [B, 256, 80, 80]`
    
    *   `cv4[0](x[0]) → [B, 32, 80, 80]`
        
    *   reshape → `[B, 32, 6400]`
        
*   P4: `x[1] = [B, 512, 40, 40]`
    
    *   `cv4[1](x[1]) → [B, 32, 40, 40]`
        
    *   reshape → `[B, 32, 1600]`
        
*   P5: `x[2] = [B, 1024, 20, 20]`
    
    *   `cv4[2](x[2]) → [B, 32, 20, 20]`
        
    *   reshape → `[B, 32, 400]`
        

三个尺度 concat 在最后一维：

**`mc = [B, 32, 6400+1600+400] = [B, 32, 8400]`**

> 也就是说：**8400 个预测点，每个点都有 32 个 mask 系数。**

这些预测点和 Detect 里 bbox/cls 的 8400 个点 1:1 对应。

##### ③ 调用 Detect.forward

```python
x = Detect.forward(self, x)
```

*   训练时 `x` 是每尺度 `[B, 144, H, W]` 的列表
    
*   推理时 `x` 是 decode 后的 bbox+cls
    

Segment 在训练时返回：

```python
return x, mc, p# x: Detect 的输出（bbox+cls）# mc: 所有预测点系数 [B, 32, 8400]# p: 原型 [B, 32, 80, 80]
```

后续 loss 代码会对正样本对应的索引取出对应的 `coeff`，再和 `p` 组合生成 GT mask，计算 BCE/Dice 等 loss。

🔥 总结一句话：

> **Segment = Detect + 一组全图共享的原型 (p) + 每个框自己的线性组合系数 (mc)**  
> 最终 mask = Σ (coeff\[k\] \* proto\[k\])。

***

## 5\. 总结性对比：Detect vs OBB vs Segment

| Head 类型 | 额外分支 | 新增输出维度 | 功能 |
| --- | --- | --- | --- |
| **Detect** | 无 | 无 | 水平框检测 |
| **OBB** | \+ cv4(angle) | angle: \[B,1,8400\] | 旋转框 θ |
| **Segment** | \+ Proto（原型）+ cv4(mask coeff) | proto: \[B,32,80,80\]mc: \[B,32,8400\] | 实例分割 mask |