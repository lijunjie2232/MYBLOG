---
title: MobileNetシリーズの説明
date: 2024-5-11 10:17:00
categories: [AI]
tags: [Deep Learning, CNN, mobilenet, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]
lang: ja　
description: MobileNet

---

# 深度方向分離畳み込み

MobileNetV1の核心的な概念は、従来の標準的な畳み込みの代わりに深度方向分離畳み込みを使用することです。標準的な畳み込み操作は、入力特徴マップの空間次元（幅と高さ）とチャネル次元の両方に対して同時に情報抽出を行いますが、これは計算コストが非常に高くなります。深度方向分離畳み込みは、このプロセスを巧妙に2つのステップに分割します。
