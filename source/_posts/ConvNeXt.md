---
title: ConvNeXt
date: 2024-5-10 10:17:00
categories: [AI]
tags: [Deep Learning, CNN, 機械学習, AI, 人工知能, 深層学習, 画像認識, 画像分類, image classification]
lang: ja　
description: ConvNeXtは、特に新しい構造やイノベーションがあるわけではなく、既存のネットワークで使われている細かい設計要素を適切に組み合わせることで、ImageNetのTop-1精度を向上させました。この設計の動機は非常にシンプルで、「TransformerやSwin-Transformerがどのようにしているかを参考にして、効果があれば採用する」という方針に従っています。

---


## ConvNeXt

![ConvNeXt Score](/assert/ConvNeXt/score.png)

ConvNeXtは、特に新しい構造やイノベーションがあるわけではなく、既存のネットワークで使われている細かい設計要素を適切に組み合わせることで、ImageNetのTop-1精度を向上させました。この設計の動機は非常にシンプルで、「TransformerやSwin-Transformerがどのようにしているかを参考にして、効果があれば採用する」という方針に従っています。

## ConvNeXtの進化経路

![ConvNeXt Design](/assert/ConvNeXt/design_score.png)

ConvNeXtはResNet-50やResNet-200を出発点として、以下の5つの観点から順次改善を行いました：

1. **マクロ設計**
2. **深度方向分離畳み込み**（ResNeXt）
3. **逆ボトルネック層**（MobileNet v2）
4. **大きな畳み込みカーネル**
5. **他のの改善点**

### マクロ設計
Swin Transformerのマクロネットワーク設計を分析し、それに基づいてConvNeXtの設計を改善します。Swin Transformerは従来のConvNetsと同様にマルチステージ設計を採用しており、各ステージで異なる特徴マップ解像度を持っています。

#### ステージ計算比率の変更

ResNetにおけるオリジナルの計算分布設計は主に経験則に基づいていました。「res4」ステージが重めに設計されているのは、物体検出などの下流タスクとの互換性を考慮した結果で、検出器ヘッドが14×14の特徴平面で動作するためです。

一方、Swin-Tは同じ原則に従いながらも、わずかに異なるステージ計算比率（1:1:3:1）を採用しています。より大きなSwin Transformerでは、この比率は1:1:9:1になります。

この設計に従い、ResNet-50の各ステージのブロック数を(3, 4, 6, 3)から(3, 3, 9, 3)に調整しました。これにより、Swin-Tと同程度のFLOPsとなり、モデル精度が78.8%から79.4%に向上しました。計算の分布については多くの研究が行われており、より最適な設計が存在する可能性があります。

今後はこのステージ計算比率を使用します。

#### Stemを「Patchify」に変更

通常、stem cell設計は入力画像がネットワークの最初でどのように処理されるかに関係しています。自然画像に内在する冗長性のため、標準的なConvNetsとVision Transformersの両方で、一般的なstem cellは入力画像を適切な特徴マップサイズに積極的にダウンサンプリングします。

標準的なResNetのstem cellは、ストライド2の7×7畳み込み層と最大プール層を含み、これにより入力画像が4倍ダウンサンプリングされます。Vision Transformersでは、より積極的な「patchify」戦略がstem cellとして使用され、これは大きなカーネルサイズ（例：カーネルサイズ=14または16）と非オーバーラップ畳み込みに対応します。

```python
stem = nn.Sequential(
    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
)
```

Swin Transformerは同様の「patchify」層を使用しますが、アーキテクチャのマルチステージ設計に対応するため、パッチサイズは4と小さくなっています。ResNetスタイルのstem cellを、4×4、ストライド4の畳み込み層で実装されたpatchify層に置き換えました。これにより、精度は79.4%から79.5%に変化しました。これは、ResNetのstem cellがViT風のよりシンプルな「patchify」層に置き換えられても、同様の性能が得られることを示唆しています。

ネットワークでは「patchify stem」（4×4非オーバーラップ畳み込み）を使用します。


### 逆ボトルネック層と深度方向分離畳み込み

![ResNeXt Block](/assert/ConvNeXt/resnext.png)

この図は、異なるブロック構造の変更とその仕様を示しています。

a. ResNeXtブロック
   - 従来のResNeXtアーキテクチャのブロック構造
   - グループ化された畳み込みを使用して計算効率を向上
   - 通常のボトルネック構造に基づく設計

b. 逆ボトルネックブロック
   - 逆ボトルネック（Inverted Bottleneck）構造を採用
   - Transformerブロックと同様の設計思想
   - MLPブロックの隠れ次元が入力次元の4倍の幅を持つ
   - この構造によりネットワーク全体のFLOPsが4.6Gに削減
   - パフォーマンスが80.5%から80.6%にわずかに改善

c. 空間的深度方向畳み込み層の位置変更
   - 空間的な深度方向畳み込み層（spatial depthwise conv layer）の位置を上方に移動
   - ブロック内の処理順序を変更した変種構造

**効果**:
   - GFLOPs: 4.4 → 2.4（削減）
   - 精度: 79.5% → 78.3%（低下）

**精度低下の補償**:
   - ResNet-50の基本チャネル数を64から96に増加

**結果**:
  - GFLOPs: 5.3に増加
  - 精度: 80.5%に向上

### 大きな畳み込みカーネル

![ConvNeXt Design](/assert/ConvNeXt/block.png)

この図は、ResNet、Swin Transformer、およびConvNeXtのブロック設計を比較しています。

1. ResNetブロック
- 従来のResNetアーキテクチャのブロック構造
- 3×3の標準畳み込み層を使用
- 単純な構造で、主にボトルネック構造に基づく設計

2. Swin Transformerブロック
- Swin Transformerのブロック構造はより洗練されている
- 複数の専用モジュールが存在
- 2つの残差接続（residual connections）を持つ
- マスクされた自己注意（MSA）ブロックとMLPブロックを含む

3. ConvNeXtブロック
- 提案されたConvNeXtのブロック設計
- Swin Transformerの設計思想を取り入れながら簡素化
- Transformer MLPブロック内の線形層も「1×1畳み込み」として表現（等価であるため）

#### 深度方向畳み込み層の位置変更
- Transformerの設計と一致させるため、深度方向畳み込み層の位置を上方に移動
- 逆ボトルネック構造を持つ場合、これは自然な設計選択となる
- 複雑/非効率なモジュール（大規模カーネル畳み込み）はチャネル数を減らし、効率的な1×1層が主要な処理を行う

#### カーネルサイズの拡大
様々なカーネルサイズ(3, 5, 7, 9, 11)で実験を実施:

| カーネルサイズ | 性能 | FLOPs |
|---------------|------|-------|
| 3×3 | 79.9% | 4.1G |
| 7×7 | 80.6% | 同程度 |

- 7×7のカーネルサイズで性能が顕著に向上
- 7×7を超えると性能向上が頭打ちになることが確認されている
- 大容量モデル(ResNet-200)でも同様の傾向を確認

### 他の変更点

![ConvNeXt Design](/assert/ConvNeXt/block.png)

## 参考

https://arxiv.org/abs/2201.03545

https://github.com/facebookresearch/ConvNeXt