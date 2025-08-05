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

Swin Transformerは、Microsoft Researchチームが開発した視覚モデルで、従来のTransformerモデルがコンピュータビジョンタスクにおいて抱える計算複雑性の問題を解決することを目的としています。正式名称は「Shifted Window Transformer」で、階層アーキテクチャとシフトウィンドウメカニズムを導入することで、性能と効率のバランスを実現しています。

![swin transformer architecture](/assert/swin_transformer/swin_transformer_arch.png)

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

## Patch Embedding

Patch Embeddingは、入力画像を複数の小さなパッチに分割し、これらのパッチのピクセル値を高次元空間に埋め込むことで、Transformerが処理可能な特徴表現を形成します。

![swin patch embeding](/assert/swin_transformer/swin_vit.png)

### 処理手順

1. **画像の分割**: 入力画像(224×224×3)を小さなパッチに分割
2. **畳み込み操作**: Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))を使用して各パッチを96次元の特徴ベクトルに変換
   - カーネルサイズ: 4×4
   - ストライド: 4
   - 入力チャネル数: 3(RGB画像)
   - 出力チャネル数: 96
3. **出力サイズの計算**: (224 - 4) / 4 + 1 = 56 → 56×56×96の特徴マップを生成

そこで：
  - `kernel_size`: 各パッチの空間サイズを決定
  - `stride`: パッチ間の間隔(ストライド)を決定

### 特徴と利点

- **パッチの表現**: 56×56×96の出力は3,136個のパッチを含み、各パッチは96次元のベクトルとして表現される
- **パラメータの制御**: 
- **柔軟な設定**: 畳み込みパラメータを変更することで、パッチの数と各パッチの次元を制御可能

## ウィンドウ分割 (Window Partition)

Swin Transformerでは、Patch Embeddingで得られた特徴表現に加えて、ウィンドウ分割(Window Partition)によってさらに細分化・処理を行い、ウィンドウ内での局所的アテンションメカニズムにより計算効率を向上させ、局所特徴を捉えることを目的としています。

![swin patch embeding](/assert/swin_transformer/swin_vit.png)

### 処理手順と計算

**入力**: 畳み込み処理後の特徴マップ (56×56×96)
**ウィンドウサイズ**: 7×7

1. **ウィンドウ数の計算**:
   - 空間次元(高さと幅)におけるウィンドウ分割数: 56 ÷ 7 = 8
   - 総ウィンドウ数: 8 × 8 = 64個のウィンドウ

2. **分割後の特徴マップ次元**: (64, 7, 7, 96)
   - `64`: ウィンドウの数 (8×8 = 64個のウィンドウ)
   - `7×7`: 各ウィンドウの空間次元
   - `96`: 各ウィンドウ内の特徴チャネル数

### Tokenの概念の変化

- **従来のToken**: 画像の局所特徴を表し、各Tokenは画像の1つの位置に対応
- **ウィンドウ分割後のToken**: 各Tokenはウィンドウの内部特徴に対応し、より広範な局所情報を捉える

### 利点

- **局所構造への注目**: モデルが画像の局所構造に集中できる
- **計算量の削減**: 各ウィンドウ内でのみアテンション計算を行うため、計算効率が向上
- **情報の捕捉範囲拡大**: 元の各空間位置が表す画像情報の一部から、ウィンドウ分割によりより広範な局所情報を捉えることが可能に


## W-MSA (Windwow multi-head self attentio)

W-MSA (Window Multi-Head Self Attention) はSwin Transformerの中心的なアテンションメカニズムであり、<font color=red>各ウィンドウ内で独立に自己注意(Self-Attention)を計算することで計算複雑性を削減し、局所特徴を捉えることを目的としています。</font>

![swin W-MSA](/assert/swin_transformer/swin_sma.png)

### 入力データ構造

ウィンドウ分割(Window Partition)を経て、特徴マップは以下のようになります：

- ウィンドウ数: 64個
- 各ウィンドウのサイズ: 7×7
- 各位置の特徴チャネル数: 96
- 各ウィンドウの形状: (7, 7, 96)

### Multi-Head Self-Attentionの処理手順

1. **線形変換**:
   - 入力特徴行列を3つの異なる行列を用いて線形変換
   - 結果としてQuery (Q)、Key (K)、Value (V) を取得

2. **マルチヘッド分割**:
   - ヘッド数: 3個 (例)
   - 各ヘッドの入力特徴次元: 96 ÷ 3 = 32次元
   - 96次元の入力が3つのヘッドに均等に分割

### W-MSAの計算プロセス

各ウィンドウに対して以下の計算を独立に実行:

1. **Query, Key, Valueの計算**:

   - 各ウィンドウ内の49個のピクセル点(7×7)に対してQ, K, Vを計算

2. **アテンションスコアの計算**:

   $$Attention Score = \frac{Q \cdot K^T}{\sqrt{d_k}}$$

   - $d_k$: 各ヘッドの次元数 (この例では32)
   - $Q \cdot K^T$: 各位置間の類似性を測定

3. **Softmax処理**:

   - スコアを正規化して確率分布に変換
   - 各位置間の相関関係を確率として表現

4. **加重和の計算**:

   - スコアを用いてValue (V) の加重和を計算
   - 各位置の最終出力表現を取得

### 出力の形状

各ヘッドの自己注意計算結果の形状: **(64, 3, 49, 49)**

- `64`: ウィンドウ数
- `3`: ヘッド数
- `49`: 各ウィンドウ内の位置数 (7×7)
- `49`: 各位置から他の位置へのアテンションスコア (自己アテンション行列)




## コード例
```python
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    split image into non-overlapping patches   即将图片划分成一个个没有重叠的patch
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # パッチサイズをタプル形式に変換し、高さと幅の両方に同じ値を適用
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        # 入力チャネル数と埋め込み次元数をクラス属性として保存
        self.in_chans = in_c
        self.embed_dim = embed_dim
        # 畳み込み層を定義: 入力画像をパッチに分割し、指定された次元に埋め込む
        # kernel_size = stride により、パッチが重ならないように分割される
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 正規化層の設定: 指定されていればその層を使用、なければIdentity(何もしない)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
 
    def forward(self, x):
        # 入力テンソルの形状からバッチサイズ、チャネル数、高さ、幅を取得
        _, _, H, W = x.shape
 
        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        # 入力画像の高さまたは幅がパッチサイズの整数倍でない場合、パディングが必要
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            # 最後の3次元(幅、高さ、チャネル)にパディングを適用
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],   # 表示宽度方向右侧填充数
                          0, self.patch_size[0] - H % self.patch_size[0],   # 表示高度方向底部填充数
                          0, 0))
 
        # 下采样patch_size倍
        # パッチサイズ分だけダウンサンプリングし、パッチへの分割を実行
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # テンソルの形状を変更: パッチの系列を1次元に平坦化し、チャネル次元を最後に移動
        x = x.flatten(2).transpose(1, 2)
        # 正規化層を適用
        x = self.norm(x)
        # 埋め込み特徴、出力の高さ、出力の幅を返す
        return x, H, W
```