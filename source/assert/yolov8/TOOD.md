# 重温经典（9）--TOOD

[![平凡的兵](https://pic1.zhimg.com/v2-9b337f74799ab40d39d77f43602ec834_l.jpg?source=32738c0c&needBackground=1)](//www.zhihu.com/people/a_bing_jiang)

[平凡的兵](//www.zhihu.com/people/a_bing_jiang)

[​![](https://pic1.zhimg.com/v2-2ddc5cc683982648f6f123616fb4ec09_l.png?source=32738c0c)](https://www.zhihu.com/question/48510028)

中国科学技术大学 工学硕士

[

收录于 · 多模态学习

](https://www.zhihu.com/column/c_1659462298400915456)

2 人赞同了该文章

​

目录

收起

1\. 引言

2\. 相关工作

3\. 任务对齐的单阶段目标检测

3.1 任务对齐头部

3.2 任务对齐学习

重温经典，找寻那惊鸿一瞥的灵动。

\--------------------------------------------------------------------------

原文链接：

TOOD：[https://arxiv.org/abs/2108.07755](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2108.07755)

**摘要**

单阶段目标检测通常通过优化两个子任务来实现：对象分类和定位，使用两个并行分支的头部进行操作，这可能导致两个任务之间的预测在空间上出现一定程度的不对齐。在这项工作中，提出了一种**任务对齐的单阶段目标检测（TOOD）**，它以基于学习的方式显式地对齐这两个任务。首先，我们设计了一个新颖的任务对齐头部（[T-Head](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=T-Head&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJULUhlYWQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDY3MzQ3NjUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.4Mm-qCd2u7hFeuNnsqYfnuFdu5S82SkpqKZCwiRULiA&zhida_source=entity)），它提供了一个更好的平衡，用于学习任务交互和任务特定特征，并且通过任务对齐预测器具有更大的灵活性来学习对齐。其次，提出了**任务对齐学习（TAL）**，通过设计的样本分配方案和任务对齐损失，在训练过程中显式地拉近（甚至统一）两个任务的最优锚点。在[MS-COCO](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=MS-COCO&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJNUy1DT0NPIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ2NzM0NzY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.PQFI2aNqmALj40Bs1-Kls72P4cvPQIarjWFNfBSwnSQ&zhida_source=entity)上进行了广泛的实验，其中TOOD在单模型单尺度测试中达到了51.1 AP。这在参数数量和浮点运算次数更少的情况下，远远超过了最近的单阶段检测器，例如ATSS \[31\]（47.7 AP）、[GFL](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=GFL&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJHRkwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDY3MzQ3NjUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Ojsfyr6WkC_PNOPV1realU6TTAcCrdyi2y_7vhNaF68&zhida_source=entity) \[14\]（48.2 AP）和[PAA](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=PAA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJQQUEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDY3MzQ3NjUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.3Q--sjqyJ7UCkq-PIyj8EH8BbqaV8EDqxp_rS3pgU5o&zhida_source=entity) \[9\]（49.0 AP）。定性结果还证明了TOOD在更好地对齐目标分类和定位任务方面的有效性。代码可在 [https://github.com/fcjian/TOOD](https://link.zhihu.com/?target=https%3A//github.com/fcjian/TOOD) 上获取。

## **1\. 引言**

目标检测旨在从自然图像中定位和识别感兴趣的对象，是计算机视觉中的一个基础但具有挑战性的任务。它通常被制定为一个多任务学习问题，通过联合优化目标分类和定位 \[4, 6, 7, 16, 22, 33\]。分类任务旨在学习区分性特征，这些特征专注于对象的关键或显著部分，而定位任务则致力于精确地定位整个对象及其边界。**由于分类和定位的学习机制存在差异，两个任务学习到的特征的空间分布可能会有所不同，导致在使用两个独立分支进行预测时出现一定程度的不对齐。**最近的单阶段目标检测器试图通过关注对象的中心 \[3, 10, 27, 31\] 来预测两个独立任务的一致输出。它们假设位于对象中心的**锚点（无锚检测器的锚点，或基于锚的检测器的锚框）**可能为分类和定位提供更准确的预测。例如，**最近的[FCOS](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=FCOS&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJGQ09TIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ2NzM0NzY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.tdw7sUOw5wz8pFOuWzj9JeTCPk4dRz11RUWqqfnZcLI&zhida_source=entity) \[27\] 和 ATSS \[31\] 都使用中心度分支来增强从靠近对象中心的锚点预测的分类得分，并对相应的锚点分配更大的定位损失权重。**此外，[FoveaBox](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=FoveaBox&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJGb3ZlYUJveCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0NjczNDc2NSwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.WcO9wsyea3OXlCizO_L6WZtGZaMt9hZTiXwsIWhfQ1s&zhida_source=entity) \[10\] 将对象预定义中心区域内的锚点视为正样本。这些启发式设计取得了优异的结果，但这些方法可能存在两个局限性：

**(1) 分类和定位的独立性。**最近的单阶段检测器通过并行使用两个独立的分支（即头部）来独立执行目标分类和定位。这种双分支设计可能导致两个任务之间缺乏交互，从而导致执行预测时的不一致性。如图1的“结果”列所示，ATSS检测器识别了一个“餐桌”对象（由带有红点的锚点指示），但更准确地定位了另一个“披萨”对象（红色边框）。

**(2) 与任务无关的样本分配。**大多数无锚检测器使用基于几何的分配方案来选择靠近对象中心的锚点进行分类和定位 \[3, 10, 31\]，而基于锚的检测器通常通过计算锚框和真实标注之间的IoU来分配锚框 \[22, 23, 31\]。然而，分类和定位的最优锚点往往不一致，可能会根据对象的形状和特征而显著变化。广泛使用的样本分配方案与任务无关，因此可能难以为两个任务做出准确且一致的预测，如**图1**中ATSS的“得分”和“IoU”分布所示。“结果”列还说明了最佳定位锚点（绿点）的空间位置可能不在对象的中心，并且与最佳分类锚点（红点）对齐得不好。因此，在非极大值抑制（NMS）期间，一个精确的边界框可能会被不太准确的一个抑制。

![](https://picx.zhimg.com/v2-3894821fd74bef787887ca151d2bef4f_1440w.jpg)

为了解决这些限制，我们提出了一种任务对齐的单阶段目标检测（TOOD），旨在通过设计新的头部结构和对齐导向的学习方法来更准确地对齐两个任务：

**任务对齐头部。**与传统的单阶段目标检测中分类和定位分别由两个并行分支实现不同，我们**设计了一个任务对齐头部（T-Head）来增强两个任务之间的交互**。这使得两个任务能够更协作地工作，从而更准确地对齐它们的预测。T-Head在概念上很简单：它计算任务交互特征，并通过新颖的任务对齐预测器（TAP）进行预测。然后，根据接下来描述的任务对齐学习提供的学习信号，对两个预测的空间分布进行对齐。

**任务对齐学习。**为了进一步克服不对齐问题，提出了任务对齐学习（TAL），通过设计样本分配方案和任务对齐损失来显式地拉近两个任务的最优锚点。样本分配通过在每个锚点计算任务对齐度来收集训练样本（即正样本或负样本），而任务对齐损失在训练过程中逐渐统一预测分类和定位的最佳锚点。因此，在推理过程中，可以保留具有最高分类得分并与最精确定位联合的边界框。提出的T-Head和学习策略可以协作地朝着在分类和定位方面都具有高质量预测的目标努力。

**主要贡献可以总结如下：**

(1) 设计了一个新的T-Head，以增强分类和定位之间的交互，同时保持它们的特性，并进一步在预测中对齐两个任务；

(2) 提出了TAL，在识别出的任务对齐锚点上显式地对齐两个任务，并为提出的预测器提供学习信号；

(3) 在MSCOCO \[17\] 上进行了广泛的实验，我们的TOOD达到了51.1 AP，远远超过了最近的单阶段检测器，如ATSS \[31\]、GFL \[14\] 和 PAA \[9\]。定性结果进一步验证了我们任务对齐方法的有效性。

## **2\. 相关工作**

**单阶段检测器。**[OverFeat](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=OverFeat&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJPdmVyRmVhdCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0NjczNDc2NSwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.MtU-al9M2wXyTYnS3gx2mohy2d0tQMVWKmQBoip1GM8&zhida_source=entity) 是最早的基于 CNN 的单阶段检测器之一。之后，[YOLO](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=YOLO&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJZT0xPIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ2NzM0NzY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.OrarXMD7nA5Mr3oedlzpPvSJBa6ECxYP5wq-yLoSZf8&zhida_source=entity) 被开发出来，它直接预测边界框和分类得分，无需额外的阶段来生成区域建议。[SSD](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=SSD&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJTU0QiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDY3MzQ3NjUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.raKv49O1Iulf37aIderCl5p2k4epSnogLJpVNlFC7cw&zhida_source=entity) 引入了带有多层卷积特征的多尺度预测的锚点，Focal loss 被提出来解决像 RetinaNet 这样的单阶段检测器的类别不平衡问题。基于关键点的检测方法，如 \[3, 11, 34\]，通过识别和组合边界框的多个关键点来解决检测问题。最近，FCOS 和 FoveaBox 被开发出来，通过锚点和点到边界的距离来定位感兴趣的对象。大多数主流的单阶段检测器由两个基于 FCN 的分支组成，用于分类和定位，这可能导致两个任务之间的不对齐。在本文中，我们通过一个新的头部结构和对齐导向的学习方法来增强两个任务之间的对齐。

**训练样本分配。 大多数基于锚点的检测器，如 \[22, 31\]，通过计算提议和真实标注之间的 IoUs 来收集训练样本，而无锚点检测器将对象中心区域内的锚点视为正样本** \[3, 10, 27\]。最近的研究尝试通过使用输出结果来选择更有意义的训练样本，以更有效地训练检测器。例如，[FSAF](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=FSAF&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJGU0FGIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ2NzM0NzY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.8Tr5pgXii3j3QRO3r9D2UYEACBynPXRpcRiRD4HhNYA&zhida_source=entity) 根据计算出的损失从特征金字塔中选择有意义的样本，同样，[SAPD](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=SAPD&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJTQVBEIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ2NzM0NzY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.HFTRlqQBXcGMcu983wbZo1q7Tg7-LUPGfeDOL5HByq4&zhida_source=entity) 通过设计一个元选择网络来提供 FSAF 的软选择版本。FreeAnchor 和 [MAL](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=MAL&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJNQUwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDY3MzQ3NjUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.0ae4dDiccWGvS50nKTy3S0fGBuYgur90JrItzIhd9Xw&zhida_source=entity) 通过计算损失来确定最佳锚框，以改进锚点和对象之间的匹配。PAA 通过拟合锚点得分的概率分布来适应性地将锚点分为正样本和负样本。Mutual Guidance 通过考虑另一任务的预测质量来改进一个任务的锚点分配。与正/负样本分配不同，[PISA](https://zhida.zhihu.com/search?content_id=246734765&content_type=Article&match_order=1&q=PISA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzU0NzY1NDcsInEiOiJQSVNBIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ2NzM0NzY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.2DsjNMrIlgeSkOyGpQy4CyiicYBeQYwQ5IxCC58bXL0&zhida_source=entity) 根据输出的精确度排名重新加权训练样本。Noisy Anchor 为训练样本分配软标签，并使用清洁度分数重新加权锚框，以减少二元标签带来的噪声。GFL 用 IoU 分数替换了二元分类标签，将定位质量整合到分类中。这些优秀的方法启发了当前的工作，从任务对齐的角度开发了新的分配机制。

## **3\. 任务对齐的单阶段目标检测**

**概述。**与最近的单阶段检测器如 \[14, 31\] 类似，提出的 TOOD 拥有一个“主干网络-FPN-头部”的总体流程。此外，考虑到效率和简洁性，TOOD 在每个位置使用单个锚点（与 ATSS \[31\] 相同），其中“锚点”对于无锚检测器而言是指锚点，对于基于锚的检测器则是指锚框。如前所述，现有的单阶段检测器在分类和定位任务之间存在不对齐的局限性，这是由于通常使用两个独立的头部分支来实现这两个任务的分歧。在这项工作中，我们提出使用设计的 Task-aligned head (T-head) 和新的 Task Alignment Learning (TAL) 更显式地对齐这两个任务。如**图 2** 所示，T-head 和 TAL 可以协同工作，提高两个任务的对齐度。具体来说，T-head 首先在 FPN 特征上对分类和定位进行预测。然后，TAL 根据新设计的任务对齐度量标准计算任务对齐信号，该标准衡量两个预测之间的对齐程度。最后，在反向传播期间，T-head 使用 TAL 计算出的学习信号自动调整其分类概率和定位预测。

![](https://pic4.zhimg.com/v2-bde34364690ace2649ce2cc2a2b3bbbf_1440w.jpg)

### 3.1 任务对齐头部

我们的目标是设计一个高效的头部结构，以改进单阶段检测器中头部的传统设计（如图 3(a) 所示）。在这项工作中，我们通过考虑两个方面来实现这一点：(1) 增加两个任务之间的交互；(2) 提高检测器学习对齐的能力。**所提出的 T-head 如图 3(b) 所示**，它具有一个简单的特征提取器和两个任务对齐预测器 (TAP)。

![](https://pic3.zhimg.com/v2-048b65ef036856f16e5cd95cd5a60638_1440w.jpg)

为了增强分类和定位之间的交互，我们使用特征提取器从多个卷积层学习一堆任务交互特征，如图 3(b) 中的蓝色部分所示。这种设计不仅促进了任务交互，还为两个任务提供了具有多尺度有效接受场的多级特征。正式地，设 Xfpn∈RH×W×CX^{fpn}∈R^{H×W×C}X^{fpn}∈R^{H×W×C} 表示 FPN 特征，其中 H、W 和 C 分别表示高度、宽度和通道数。特征提取器使用 N 个连续的卷积层和激活函数来计算任务交互特征：

![](https://pica.zhimg.com/v2-327d72515753f38a2cb1d4821944c684_1440w.jpg)

其中 convkconv\_kconv\_k 和_δ_ 分别表示第 k 个卷积层和 relu 函数。因此我们使用头部中的单个分支从 FPN 特征中提取丰富的多尺度特征。然后，计算出的任务交互特征将被送入两个 TAP，以对齐分类和定位。

**任务对齐预测器 (TAP)**。在计算出的任务交互特征上执行目标分类和定位，两个任务可以很好地感知彼此的状态。然而，由于单一分支设计，任务交互特征在两个不同任务之间不可避免地引入了一定程度的特征冲突，这在 \[26, 28\] 中也有讨论。直观地说，目标分类和定位的任务有不同的目标，因此关注不同类型的特征（例如，不同的层次或接受场）。因此，我们提出了一种层注意力机制，通过动态计算层级任务特定特征来鼓励任务分解。如图 3(c) 所示，每个分类或定位任务的特定特征分别计算：

其中 wkw\_kw\_k ​ 是学习到的层注意力 w∈ RNR^NR^N 的第 k 个元素。 w 是从跨层任务交互特征计算出来的，能够捕获层之间的依赖关系：

![](https://pic3.zhimg.com/v2-56daf0d9be344bf9b0d7a5be2ea5fa68_1440w.jpg)

其中 fc1 和 fc2 指的是两个全连接层。 σ 是 sigmoid 函数， xinterx^{inter}x^{inter} 是通过对 XinterX^{inter}X^{inter} 进行平均池化得到的， XinterX^{inter}X^{inter} 是 XkinterX^{inter}\_kX^{inter}\_k 连接的特征。最后，分类或定位的结果由每个 XtaskX^{task}X^{task} ​ 预测：

![](https://pic1.zhimg.com/v2-48576b6bd508f9022e51eae976e1f31a_1440w.jpg)

其中 XtaskX^{task}X^{task} 是 Xktask X^{task}\_k X^{task}\_k ​ 连接的特征，conv1 是用于降维的 1×1 卷积层。 ZtaskZ^{task} Z^{task} 然后被转换为密集分类得分 P∈RH×W×80P∈R^{H×W×80}P∈R^{H×W×80} ，使用 sigmoid 函数，或者通过应用于 \[27, 31\] 中的距离到边界框转换，转换为对象边界框 B∈RH×W×4B∈R^{H×W×4}B∈R^{H×W×4} 。

预测对齐。在预测步骤，我们通过调整两个预测的空间分布进一步显式地对齐两个任务：P 和 B。与以前的工作不同，以前的工作使用中心度分支 \[27\] 或 IoU 分支 \[9\] 只能根据分类特征或定位特征调整分类预测，我们使用计算出的任务交互特征联合考虑两个任务来对齐两个预测。值得注意的是，我们分别对两个任务执行对齐方法。如图 3(c) 所示，我们使用空间概率图 M∈RH×W×1M∈R^{H×W×1}M∈R^{H×W×1} 来调整分类预测：

![](https://picx.zhimg.com/v2-8cf53a4e1aaa73c5ca62b5758e7ca487_1440w.jpg)

其中 M 是从交互特征计算出来的，允许它在每个空间位置学习两个任务之间的一致性程度。同时，为了在定位预测上进行对齐，我们进一步从交互特征中学习空间偏移图 O∈RH×W×8O∈R^{H×W×8}O∈R^{H×W×8} ，这些偏移图用于在每个位置调整预测的边界框。具体来说，学习到的空间偏移使最对齐的锚点能够识别周围最准确的边界预测：

![](https://pica.zhimg.com/v2-627231b896e5ad2058e99753da396cee_1440w.jpg)

其中索引 (i,j,c)表示张量中第 (i,j)个空间位置的第 c个通道。公式 (6) 通过双线性插值实现，由于 B 的通道维度非常小，其计算开销可以忽略不计。值得注意的是，偏移是为每个通道独立学习的，这意味着对象的每个边界都有自己的学习偏移。这允许更准确地预测四个边界，因为每个边界都可以从它附近最精确的锚点单独学习。因此，我们的方法不仅对齐了两个任务，还通过为每个边界识别一个精确的锚点来提高了定位精度。对齐图 M 和 O 是从交互特征堆栈自动学习得到的：

![](https://pic2.zhimg.com/v2-da714ea78c5c975cb587c7c449a4279f_1440w.jpg)

其中 conv1 和 conv3 是用于降维的两个 1×1 卷积层。M 和 O 的学习是通过提出的任务对齐学习 (TAL) 来执行的，这将在下面描述。注意，我们的 T-head 是一个独立的模块，可以在没有 TAL 的情况下很好地工作。它可以以即插即用的方式应用于各种单阶段目标检测器，以提高检测性能。

### **3.2 任务对齐学习**

我们进一步引入了任务对齐学习（TAL），以指导我们的 T-head 进行任务对齐的预测。TAL 与之前的方法 \[1, 8, 9, 12, 14, 29, 32\] 在两个方面不同。首先，从任务对齐的角度出发，它基于设计好的度量标准动态选择高质量的锚点。其次，它同时考虑了锚点分配和权重计算。TAL 包括一个样本分配策略和为对齐两个任务而特别设计的新型损失函数。

3.2.1 任务对齐样本分配

为了应对 NMS（非极大值抑制），训练实例的锚点分配应该满足以下规则：(1) 一个对齐良好的锚点应该能够联合预测出高分类得分和精确的定位；(2) 一个对齐不良的锚点应该有低分类得分，并在随后被抑制。基于这两个目标，我们设计了一个新的锚点对齐度量标准，明确地在锚点层面衡量任务对齐的程度。对齐度量标准被整合到样本分配和损失函数中，以动态地优化每个锚点的预测。

锚点对齐度量。考虑到分类得分和预测边界框与真实标注之间的 IoU（交并比）指示了两个任务预测的质量，我们使用分类得分和 IoU 的高阶组合来衡量任务对齐的程度。具体来说，我们设计了以下度量标准来计算每个实例的锚点级对齐：

![](https://picx.zhimg.com/v2-64fe97d06c6dfa994745c0427f0c8b43_1440w.jpg)

其中 s 和 u 分别表示分类得分和 IoU 值。α 和 β 用来控制两个任务在锚点对齐度量标准中的影响。值得注意的是，t 在两个任务的联合优化中起着关键作用，鼓励网络从联合优化的角度动态地关注高质量的（即任务对齐的）锚点。

训练样本分配。如 \[31, 32\] 中所讨论的，训练样本分配对目标检测器的训练至关重要。为了提高两个任务的对齐度，我们专注于任务对齐的锚点，并采用简单的分配规则来选择训练样本：对于每个实例，我们选择具有最大 t 值的 m 个锚点作为正样本，而将其余锚点作为负样本。再次，通过计算特别为对齐分类和定位任务而设计的新型损失函数来进行训练。

3.2.2 任务对齐损失

分类目标。为了显式地提高对齐锚点的分类得分，并同时减少不对齐锚点（即具有小 t）的得分，我们使用 t 来代替正锚点的二元标签进行训练。然而，我们发现当正锚点的标签（即 t）随着 α 和 β 的增加而变小，网络无法收敛。因此，我们使用标准化的 t，即 \\hat{t}\\hat{t} ，来代替正锚点的二元标签，其中 \\hat{t}\\hat{t} 通过以下两个属性进行标准化：(1) 确保有效学习困难实例（这些实例通常对于所有对应的正锚点都有小 t）；(2) 保持基于预测边界框的精确度的实例间的排名。因此，我们采用简单的实例级标准化来调整 \\hat{t}\\hat{t} 的尺度： \\hat{t}\\hat{t} 的最大值等于每个实例中最大的 IoU 值。然后，针对分类任务的正锚点计算的二元交叉熵（BCE）可以重写为：

![](https://pic3.zhimg.com/v2-680f782826dadd2e6430036bb2d6f8a2_1440w.jpg)

其中 i 表示一个实例中对应的 N\_{pos}N\_{pos} 个正锚点的第 i 个锚点。按照 \[16\]，我们采用焦点损失来减轻训练期间负样本和正样本之间的不平衡。通过公式 (10) 重新定义焦点损失，并定义分类任务的最终损失函数如下：

![](https://picx.zhimg.com/v2-187607ab367a4202196342202aeb5643_1440w.jpg)

其中 j 表示 N\_{neg}N\_{neg} 个负锚点的第 j 个锚点，γ 是聚焦参数。

定位目标。由对齐良好的锚点预测的边界框通常既有高分类得分又有精确的定位，这样的边界框在 NMS 期间更有可能被保留。此外，t 可以应用于通过更仔细地加权损失来选择高质量的边界框，以改进训练。如 \[21\] 所讨论的，从高质量的边界框中学习对模型的性能有益，而低质量的边界框通常对训练有负面影响，因为它们会产生大量的不太有信息量和多余的信号来更新模型。在本例中，我们应用 t 值来衡量边界框的质量。因此，我们通过专注于对齐良好的锚点（具有大 t），同时在边界框回归期间减少不对齐锚点（具有小 t）的影响，来提高任务对齐和回归精度。与分类目标类似，我们根据 \\hat{t}\\hat{t} 重新加权每个锚点计算的边界框回归损失，并且 GIoU 损失 (LGIoU) 可以重写为：

其中 b 和 ¯b 分别表示预测的边界框和相应的真实标注框。TAL 的总训练损失是 Lcls 和 Lreg 的总和。

编辑于 2025-03-13 20:55・安徽

[

目标检测

](//www.zhihu.com/topic/19596960)

​赞同 2​​1 条评论​7 ​喜欢

​分享

​申请转载​

​