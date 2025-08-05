---
title: swin transformer
date: 2024-4-11 10:17:00
categories: [AI]
tags: [Deep Learning, transformer, swin transformer, 機械学習, AI, 人工知能, 深層学習, 画像分類]
lang: ja　
description: swin transformer

---

目次

---

![swin transformer architecture](/assert/swin_transformer/swin_transformer_arch.png)

Swin Transformerは、Microsoft Researchチームが開発した視覚モデルで、従来のTransformerモデルがコンピュータビジョンタスクにおいて抱える計算複雑性の問題を解決することを目的としています。正式名称は「Shifted Window Transformer」で、階層アーキテクチャとシフトウィンドウメカニズムを導入することで、性能と効率のバランスを実現しています。

## Vision Transformerの課題

- **計算複雑性**: 画像データを細かいパッチに分割する必要があり、より多くの特徴を得るためには長いシーケンスを構築する必要があります。自己注意機構の計算複雑性は$O(n^2)$であり、高解像度画像を処理する際に急速に増加します。
- **局所特徴の捕捉**: 画像の視覚情報は多くの場合局所的な関係に依存していますが、標準的なVision Transformerはグローバルな関係を処理するため、局所特徴を効果的に捉えることができません。

### Swin Transformerの解決策

- **ウィンドウベースのアプローチ**: 長いシーケンスの代わりに、ウィンドウと階層的な形式を採用
- **階層処理**: 
  - 多くのトークンから開始（例：400トークン）
  - レイヤーごとにトークンを徐々にマージ（400→200→100トークン）
  - トークン数が減少するにつれてウィンドウサイズが増加
  - CNNの畳み込みとプーリング操作と同様の概念

