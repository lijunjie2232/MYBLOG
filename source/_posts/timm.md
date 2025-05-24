---
title: timmライブラリ入門 - PyTorch Image Models (timm) ライブラリの紹介と使い方

date: 2023-10-18 12:00:00
categories: [AI]
tags:
  [Deep Learning, PyTorch, Lightning, Python, 機械学習, AI, 人工知能, 深層学習]
lang: ja

description: timm
---

## 目次

- [目次](#%E7%9B%AE%E6%AC%A1)
- [概要](#%E6%A6%82%E8%A6%81)
- [主な機能](#%E4%B8%BB%E3%81%AA%E6%A9%9F%E8%83%BD)
- [インストール](#%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
- [基本的な使い方](#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E4%BD%BF%E3%81%84%E6%96%B9)
  - [モデルのロード](#%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E3%83%AD%E3%83%BC%E3%83%89)
  - [データ前処理](#%E3%83%87%E3%83%BC%E3%82%BF%E5%89%8D%E5%87%A6%E7%90%86)
  - [画像分類の訓練例 (CIFAR-10)](#%E7%94%BB%E5%83%8F%E5%88%86%E9%A1%9E%E3%81%AE%E8%A8%93%E7%B7%B4%E4%BE%8B-cifar-10)
- [高度な機能](#%E9%AB%98%E5%BA%A6%E3%81%AA%E6%A9%9F%E8%83%BD)
  - [特徴抽出](#%E7%89%B9%E5%BE%B4%E6%8A%BD%E5%87%BA)
  - [モデルアンサンブル](#%E3%83%A2%E3%83%87%E3%83%AB%E3%82%A2%E3%83%B3%E3%82%B5%E3%83%B3%E3%83%96%E3%83%AB)
- [timm.create\_model](#timmcreatemodel)
  - [主なパラメータの説明](#%E4%B8%BB%E3%81%AA%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AC%E6%98%8E)
  - [使い方](#%E4%BD%BF%E3%81%84%E6%96%B9)
    - [事前学習なしの MobileNetV3-Large モデル作成](#%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E3%81%AA%E3%81%97%E3%81%AE-mobilenetv3-large-%E3%83%A2%E3%83%87%E3%83%AB%E4%BD%9C%E6%88%90)
    - [事前学習ありの MobileNetV3-Large モデル作成](#%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E3%81%82%E3%82%8A%E3%81%AE-mobilenetv3-large-%E3%83%A2%E3%83%87%E3%83%AB%E4%BD%9C%E6%88%90)
    - [事前学習ありで分類層を 10 クラスに変更](#%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E3%81%82%E3%82%8A%E3%81%A7%E5%88%86%E9%A1%9E%E5%B1%A4%E3%82%92-10-%E3%82%AF%E3%83%A9%E3%82%B9%E3%81%AB%E5%A4%89%E6%9B%B4)
    - [Dinov2 モデルの重みをカスタムディレクトリに保存](#dinov2-%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E9%87%8D%E3%81%BF%E3%82%92%E3%82%AB%E3%82%B9%E3%82%BF%E3%83%A0%E3%83%87%E3%82%A3%E3%83%AC%E3%82%AF%E3%83%88%E3%83%AA%E3%81%AB%E4%BF%9D%E5%AD%98)
    - [モデル作成と重みロード](#%E3%83%A2%E3%83%87%E3%83%AB%E4%BD%9C%E6%88%90%E3%81%A8%E9%87%8D%E3%81%BF%E3%83%AD%E3%83%BC%E3%83%89)
      - [**features\_only**](#featuresonly)
      - [**output\_stride**](#outputstride)
      - [**out\_indices**](#outindices)
- [timm.list\_models](#timmlistmodels)
  - [ソースコード](#%E3%82%BD%E3%83%BC%E3%82%B9%E3%82%B3%E3%83%BC%E3%83%89)
  - [パラメータの説明](#%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AC%E6%98%8E)
  - [使用例](#%E4%BD%BF%E7%94%A8%E4%BE%8B)
    - [基本的な使用法](#%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E4%BD%BF%E7%94%A8%E6%B3%95)
    - [事前学習済みモデルのみ取得](#%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E3%81%BF%E5%8F%96%E5%BE%97)
    - [特定のサブモジュールに絞る](#%E7%89%B9%E5%AE%9A%E3%81%AE%E3%82%B5%E3%83%96%E3%83%A2%E3%82%B8%E3%83%A5%E3%83%BC%E3%83%AB%E3%81%AB%E7%B5%9E%E3%82%8B)
    - [フィルタ + 除外条件付き](#%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF--%E9%99%A4%E5%A4%96%E6%9D%A1%E4%BB%B6%E4%BB%98%E3%81%8D)
- [load\_pretrained](#loadpretrained)
  - [パラメータの説明](#%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AC%E6%98%8E-1)
  - [使用例](#%E4%BD%BF%E7%94%A8%E4%BE%8B-1)
    - [Vision Transformer をロードし、カスタムクラス数で事前学習済み重みをロード](#vision-transformer-%E3%82%92%E3%83%AD%E3%83%BC%E3%83%89%E3%81%97%E3%82%AB%E3%82%B9%E3%82%BF%E3%83%A0%E3%82%AF%E3%83%A9%E3%82%B9%E6%95%B0%E3%81%A7%E4%BA%8B%E5%89%8D%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E9%87%8D%E3%81%BF%E3%82%92%E3%83%AD%E3%83%BC%E3%83%89)
    - [グレースケール画像用に調整してロード](#%E3%82%B0%E3%83%AC%E3%83%BC%E3%82%B9%E3%82%B1%E3%83%BC%E3%83%AB%E7%94%BB%E5%83%8F%E7%94%A8%E3%81%AB%E8%AA%BF%E6%95%B4%E3%81%97%E3%81%A6%E3%83%AD%E3%83%BC%E3%83%89)
- [timm.data](#timmdata)
  - [ImageDataset](#imagedataset)
    - [使用例](#%E4%BD%BF%E7%94%A8%E4%BE%8B-2)
  - [IterableImageDataset](#iterableimagedataset)
    - [主な特徴](#%E4%B8%BB%E3%81%AA%E7%89%B9%E5%BE%B4)
    - [使用例](#%E4%BD%BF%E7%94%A8%E4%BE%8B-3)
  - [AugMixDataset](#augmixdataset)
    - [主な特徴](#%E4%B8%BB%E3%81%AA%E7%89%B9%E5%BE%B4-1)
    - [使用例](#%E4%BD%BF%E7%94%A8%E4%BE%8B-4)
    - [source code](#source-code)
    - [AugMix とは](#augmix-%E3%81%A8%E3%81%AF)
      - [特徴](#%E7%89%B9%E5%BE%B4)
    - [AugMix を使った訓練コード](#augmix-%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E8%A8%93%E7%B7%B4%E3%82%B3%E3%83%BC%E3%83%89)
- [参考](#%E5%8F%82%E8%80%83)

---

## 概要

timm (PyTorch Image Models) は Ross Wightman 氏によって開発された PyTorch ベースのコンピュータビジョンライブラリ。ImageNet クラスの画像分類モデルを中心に、物体検出、セグメンテーション、特徴抽出などのタスクをサポート。

## 主な機能

- **200+以上の事前学習済みモデル** (ResNet, EfficientNet, ViT など)
- モデルアーキテクチャの柔軟なカスタマイズ
- 特徴抽出の簡易化
- 転移学習の効率化
- モデルアンサンブル機能

## インストール

```bash
pip install timm
```

## 基本的な使い方

### モデルのロード

```python
import timm

# 事前学習済みモデルのロード
model = timm.create_model('resnet50', pretrained=True)

# カスタムクラス数に変更
model = timm.create_model('resnet50', pretrained=True, num_classes=10)

# 利用可能なモデル一覧表示
model_names = timm.list_models(pretrained=True)
```

### データ前処理

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 画像分類の訓練例 (CIFAR-10)

```python
# モデル構築
model = timm.create_model('resnet50', pretrained=True, num_classes=10)

# 損失関数と最適化手法
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練ループ
for epoch in range(num_epochs):
    for inputs, labels in trainloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 高度な機能

### 特徴抽出

```python
# 最終層直前の特徴抽出
model = timm.create_model('resnet50', pretrained=True)
features = model.forward_features(input)

# 多スケール特徴抽出
model = timm.create_model('resnet50', features_only=True, pretrained=True)
outputs = model(input)  # 各スケールの特徴がリストで返る
```

### モデルアンサンブル

```python
class ModelEnsemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.mean(torch.stack([m(x) for m in self.models]), dim=0)

# 使用例
model1 = timm.create_model('resnet18', pretrained=True)
model2 = timm.create_model('vgg16', pretrained=True)
ensemble = ModelEnsemble([model1, model2])
```

## timm.create_model

`timm.create_model` は、指定したモデル名のモデルをロードし、そのモデルを返す関数。

###　ソースコード

```python
def create_model(
        model_name: str,                  # モデル名（例: 'resnet50', 'vit_base_patch16_224'）
        pretrained: bool = False,         # 事前学習済み重みを使用するか
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,  # 外部のpretrained_cfgを指定
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,  # pretrained_cfgの一部パラメータを上書き
        checkpoint_path: Optional[Union[str, Path]] = None,       # チェックポイントファイルのパス
        cache_dir: Optional[Union[str, Path]] = None,             # キャッシュディレクトリ
        scriptable: Optional[bool] = None,  # JITスクリプタブル設定
        exportable: Optional[bool] = None,  # ONNXエクスポート可能設定
        no_jit: Optional[bool] = None,      # JIT最適化を無効化
        **kwargs,                          # その他のモデルパラメータ
):
    """Create a model.

    モデル名に対応するエントリポイント関数を呼び出し、新しいモデルを生成します。

    Tip:
        **kwargs はエントリポイント関数 → ``timm.models.build_model_with_cfg()`` → モデルクラス __init__() へと渡されます。
        None に設定された kwargs は渡す前に削除されます。

    Args:
        model_name: インスタンス化するモデル名（例: 'resnet50'）
        pretrained: True にすると ImageNet-1k の事前学習済み重みをロード
        pretrained_cfg: 外部の pretrained_cfg を指定
        pretrained_cfg_overlay: 基本 pretrained_cfg の一部を置き換え
        checkpoint_path: モデル初期化後にロードするチェックポイントパス
        cache_dir: Hugging Face Hub や Torch チェックポイントのキャッシュディレクトリを上書き
        scriptable: モデルを JIT スクリプタブルに設定（一部モデル未対応）
        exportable: モデルを ONNX エクスポート可能に設定（一部未実装）
        no_jit: JIT スクリプト層を利用しない設定（活性化関数のみ）

    キーワード引数:
        drop_rate (float): 学習用の分類器ドロップアウト率
        drop_path_rate (float): ストークastic depth ドロップ率
        global_pool (str): 分類器のグローバルプーリングタイプ
    """
    # すべてのモデルがサポートしないパラメータは None として扱い、デフォルト値を維持
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_id = parse_model_name(model_name)  # モデルソース（HuggingFaceなど）を解析
    if model_source:
        assert not pretrained_cfg, 'Hugging Face Hub からモデルをロードする際は pretrained_cfg は指定不可'
        if model_source == 'hf-hub':
            # Hugging Face Hub からモデル設定をロード
            pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_id, cache_dir=cache_dir)
        elif model_source == 'local-dir':
            # ローカルディレクトリからモデル設定をロード
            pretrained_cfg, model_name, model_args = load_model_config_from_path(model_id)
        else:
            assert False, f'不明な model_source {model_source}'
        if model_args:
            for k, v in model_args.items():
                kwargs.setdefault(k, v)  # モデル引数を kwargs に反映
    else:
        model_name, pretrained_tag = split_model_name_tag(model_id)
        if pretrained_tag and not pretrained_cfg:
            # pretrained_cfg が未指定なら、モデル名のタグを使用
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('不明なモデル (%s)' % model_name)  # モデル名チェック

    create_fn = model_entrypoint(model_name)  # モデル生成関数を取得
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            cache_dir=cache_dir,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)  # チェックポイントをロード

    return model
```

### 主なパラメータの説明

| パラメータ        | 説明                                                   |
| :---------------- | :----------------------------------------------------- |
| `model_name`      | モデル名（例: `'resnet50'`, `'vit_base_patch16_224'`） |
| `pretrained`      | 事前学習済み重みを使用するか（`True`/`False`）         |
| `num_classes`     | 分類クラス数の変更（例: `num_classes=10`）             |
| `pretrained_cfg`  | 外部の事前学習設定を指定                               |
| `checkpoint_path` | 事後チェックポイントファイルをロード                   |
| `drop_rate`       | ドロップアウト率（過学習防止用）                       |
| `drop_path_rate`  | ストーク astic depth のドロップ率                      |
| `global_pool`     | グローバルプーリングタイプ（例: `'avg'`, `'max'`）     |

### 使い方

```python
from timm import create_model
```

#### 事前学習なしの MobileNetV3-Large モデル作成

```python
>>> model = create_model('mobilenetv3_large_100')
```

#### 事前学習ありの MobileNetV3-Large モデル作成

```python
>>> model = create_model('mobilenetv3_large_100', pretrained=True)
>>> model.num_classes
1000
```

#### 事前学習ありで分類層を 10 クラスに変更

```python
>>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
>>> model.num_classes
10
```

#### Dinov2 モデルの重みをカスタムディレクトリに保存

```python
# データ保存先: `/data/my-models/models--timm--vit_small_patch14_dinov2.lvd142m/`
model = create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, cache_dir="/data/my-models")
```

#### モデル作成と重みロード

```python
m = timm.create_model(
    'ecaresnet101d',
    features_only=True,
    output_stride=8,
    out_indices=(0,1,2,3, 4),
    pretrained=True
)
```

##### **features_only**

- **意味**: 特徴抽出モードを有効にするオプション。
- **説明**: このフラグを `True` に設定すると、モデルの最終的な分類器（Classifier）を除いた部分のみが出力されます。つまり、特徴マップ（feature maps）だけを取得したい場合に使用します。
- **注意**: `output_stride` や `out_indices` を指定する際には、必ず `features_only=True` とする必要があります。

```python
# 例: 特徴抽出モードでResNet50をロード
model = timm.create_model('resnet50', features_only=True, pretrained=True)
```

##### **output_stride**

- **意味**: 出力ストライド（出力特徴マップのスケーリング率）
- **説明**: 入力画像に対して、最終的な特徴マップが何倍のダウンサンプリングされているかを制御します。特にセマンティックセグメンテーションや物体検出などのタスクで重要です。
- **サポート**: 一部のネットワークでは `output_stride=32` のみサポートされています。小さめの値（例: 8 や 16）に設定することで、より高解像度な特徴マップを得られます。

```python
# 例: output_stride=8 で特徴抽出モデルを作成
model = timm.create_model('resnet50', features_only=True, output_stride=8, pretrained=True)
```

##### **out_indices**

- **意味**: 取得したい特徴層のインデックス番号
- **説明**: モデル内のどの層から特徴を出力するかを指定します。複数のスケール（マルチスケール）の特徴が必要な場合（例：FPN、U-Net など）に便利です。
- **例**: `(0, 1, 2, 3, 4)` → 5 つの異なるレイヤーの特徴をリスト形式で返す

```python
# 例: 0〜4層目の特徴を抽出
model = timm.create_model(
    'resnet50',
    features_only=True,
    out_indices=(0, 1, 2, 3, 4),
    pretrained=True
)
```

## timm.list_models

`timm.list_models` は、利用可能なモデル名の一覧を取得する関数です。  
ワイルドカードによるフィルタリングや、特定のサブモジュール（例：`vision_transformer`）に絞り込むことが可能です。

### ソースコード

```python
def list_models(
    filter: Union[str, List[str]] = '',
    module: Union[str, List[str]] = '',
    pretrained: bool = False,
    exclude_filters: Union[str, List[str]] = '',
    name_matches_cfg: bool = False,
    include_tags: Optional[bool] = None,
) -> List[str]:
    """
    利用可能なモデル名一覧を取得する関数。アルファベット順にソートして返す。

    Args:
        filter (str or List[str]): ワイルドカードによるフィルタリング（例: 'resnet*'）
        module (str or List[str]): 特定サブモジュール内のモデルのみ表示（例: 'vision_transformer'）
        pretrained (bool): True のとき、事前学習済みモデルのみ表示
        exclude_filters (str or List[str]): フィルタ後に除外したいパターン
        name_matches_cfg (bool): モデル名が設定ファイルと一致するもののみ表示
        include_tags (Optional[bool]): モデル名に事前学習タグ（例：.in1k）を含めるか

    Returns:
        models (List[str]): ソート済みモデル名一覧

    Example:
        model_list('gluon_resnet*') -- 'gluon_resnet' から始まるすべてのモデルを取得
        model_list('*resnext*', 'resnet') -- 'resnext' を含む 'resnet' モジュールのモデルを取得
    """
```

1. **初期フィルタ適用**:

   - `filter` に指定されたワイルドカード（例：`'resnet*'`）でモデル名を絞り込み。

2. **サブモジュール制限**:

   - `module` が指定されていれば、そのサブモジュールに所属するモデルのみ選択。

3. **非推奨モデル除去**:

   - `_deprecated_models` に登録されている非推奨モデルは自動的に除外。

4. **事前学習タグの追加（オプション）**:

   - `include_tags=True` の場合、モデル名に `.in1k` などのタグも含めて表示。

5. **除外フィルタ適用**:

   - `exclude_filters` で指定されたパターンに合致するモデルを最終的に除外。

6. **事前学習ありのみ抽出**:

   - `pretrained=True` 時、`_model_has_pretrained` に含まれるモデルのみ残す。

7. **設定名とのマッチング**:

   - `name_matches_cfg=True` 時、`_model_pretrained_cfgs` と名前が一致するモデルのみ残す。

8. **自然順ソートして返却**:
   - 数字も正しく並べ替わるよう、`_natural_key` 関数でソート。

### パラメータの説明

| パラメータ         | 型                   | 説明                                                     |
| ------------------ | -------------------- | -------------------------------------------------------- |
| `filter`           | `str` or `List[str]` | ワイルドカードでフィルタ（例：`'resnet*'`）              |
| `module`           | `str` or `List[str]` | 特定のサブモジュールに限定（例：`'vision_transformer'`） |
| `pretrained`       | `bool`               | `True` のとき、事前学習済み重みがあるモデルのみ表示      |
| `exclude_filters`  | `str` or `List[str]` | フィルタ後に除外したいパターン（例：`'*efficientnet*'`） |
| `name_matches_cfg` | `bool`               | モデル名が設定ファイルと一致するもののみ表示             |
| `include_tags`     | `Optional[bool]`     | モデル名に事前学習タグ（例：`.in1k`）を含めるか          |

### 使用例

#### 基本的な使用法

```python
# 'gluon_resnet' から始まるすべてのモデルを取得
model_list = timm.list_models('gluon_resnet*')
```

```python
# 'resnext' を含む 'resnet' サブモジュールのモデルを取得
model_list = timm.list_models('*resnext*', 'resnet')
```

#### 事前学習済みモデルのみ取得

```python
# 事前学習済みのResNet系モデルのみ取得
model_list = timm.list_models('resnet*', pretrained=True)
```

#### 特定のサブモジュールに絞る

```python
# Vision Transformer 系モデルのみ取得
model_list = timm.list_models(module='vision_transformer')
```

#### フィルタ + 除外条件付き

```python
# EfficientNet-B0系だが、lite版は除外
model_list = timm.list_models('tf_efficientnet_b0*', exclude_filters='*lite*')
```

## load_pretrained

`timm.models.load_pretrained` は、事前学習済みモデルの重み（チェックポイント）を PyTorch モデルにロードするための関数です。  
カスタム設定（入力チャネル数、クラス数など）に応じて柔軟に適応させることができます。

```python
def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
):
    """ Load pretrained checkpoint

    Args:
        model: 重みをロードしたいPyTorchモデル（例：ResNet、Vision Transformerなど）
        pretrained_cfg: 事前学習設定情報
        num_classes: 対象モデルの分類クラス数。デフォルトはImageNetの1000クラス。異なる場合は自動調整される
        in_chans: 入力画像のチャネル数（RGBなら3、グレースケールなら1）。異なる場合、重みを調整してロード
        filter_fn: state_dict をロード前に加工する関数（例：特定層だけ除外、変更など）
        strict: ロード時にstate_dictとモデル構造が一致していないときにエラーを出すかどうか（True: 一致必須 / False: 柔軟にロード）
        cache_dir: チェックポイントファイルを保存するディレクトリ
    """
```

### パラメータの説明

| パラメータ       | 型                   | 説明                                                     |
| ---------------- | -------------------- | -------------------------------------------------------- |
| `model`          | `nn.Module`          | 重みをロードする対象の PyTorch モデル                    |
| `pretrained_cfg` | `Optional[Dict]`     | 事前学習済み重みやデータセットに関する設定情報           |
| `num_classes`    | `int`                | 対象モデルの分類クラス数（デフォルト：ImageNet の 1000） |
| `in_chans`       | `int`                | 入力画像のチャネル数（デフォルト：RGB 画像の 3）         |
| `filter_fn`      | `Optional[Callable]` | state_dict をロード前に加工するフィルタ関数              |
| `strict`         | `bool`               | モデルと state_dict が一致しないときにエラーを出すか     |
| `cache_dir`      | `str or Path`        | チェックポイントファイルを保存するディレクトリ           |

### 使用例

#### Vision Transformer をロードし、カスタムクラス数で事前学習済み重みをロード

```python
import timm
from timm.models import create_model

# ViT Base モデルを作成
model = create_model('vit_base_patch16_224', num_classes=10)

# 事前学習済み重みをロード
timm.models.load_pretrained(model, pretrained_cfg=model.default_cfg, num_classes=10)
```

#### グレースケール画像用に調整してロード

```python
# グレースケール画像（1チャネル）用に調整
model = create_model('resnet50', in_chans=1, num_classes=10)
timm.models.load_pretrained(model, pretrained_cfg=model.default_cfg, in_chans=1)
```

## timm.data

### ImageDataset

- **標準的な画像分類用データセット**を構築するためのクラス。
- ファイル構造に基づいて画像とラベルを読み込む（例: ImageNet 形式のフォルダ構造）。
- シンプルな使い方で、学習・検証用のデータローダーを作成可能。

#### 使用例

```python
from timm.data import ImageDataset
from torch.utils.data import DataLoader

dataset = ImageDataset(
    'path/to/train', # データセットのパス
    reader=None, # データを読み込む関数
    split='train', # 'train', 'valid', 'test' などの分割方法
    class_map=None, # クラス名のマッピング
    load_bytes=False, # バイト列として読み込むかどうか
    input_img_mode='RGB', # 入力画像のモード
    transform=None, # データを変換する関数
    target_transform=None, # ラベルを変換する関数
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### IterableImageDataset

- **大規模データセット**や **ストリーミングデータ** を扱うためのイテラブル型データセット。
- メモリ効率が高く、データ量が非常に多い場合に有効。
- `torch.utils.data.IterableDataset` を継承している。

#### 主な特徴

- **全データをメモリに載せずに逐次読み込み**
- 分散環境（DistributedDataParallel など）との親和性が高い
- サブセット（サブサンプリング）にも対応

#### 使用例

```python
from timm.data import IterableImageDataset

dataset = IterableImageDataset(
    'path/to/train', #  データセットのパス
    reader=None, #  データを読み込む関数
    split='train', #  'train', 'valid', 'test' などの分割方法
    class_map=None, #  クラス名のマッピング
    is_training=False, #   学習データか検証データかを指定
    batch_size=1, #  バッチサイズ
    num_samples=None, #  データセットのサンプル数
    seed=42, # サンプリングのシード
    repeats=0, #   データセットの繰り返し回数
    download=False, #   データをダウンロードするかどうか
    input_img_mode='RGB', #   入力画像のモード
    input_key=None, #   入力画像のキー
    target_key=None, #   ターゲット画像のキー
    transform=None, #   入力画像の変換
    target_transform=None, #  ターゲット画像の変換
    max_steps=None, #  データセットのステップ数
)
```

### AugMixDataset

- **AugMix** というデータ拡張手法をサポートしたラッパーデータセット。
- AugMix は、複数の合成拡張経路を混ぜたデータ拡張を行い、モデルの堅牢性（robustness）を向上させます。

#### 主な特徴

- 各バッチに対して、複数回の拡張（augmented copies）を生成し、それらを組み合わせて入力とする
- `__getitem__` ではなく `__iter__` によってデータを返す（PyTorch の `IterableDataset` と似た挙動）

#### 使用例

```python
from timm.data import ImageDataset, AugMixDataset
from torch.utils.data import DataLoader

# 通常の ImageDataset をベースにする
base_dataset = ImageDataset('path/to/train')

augmix_dataset = AugMixDataset(base_dataset, num_splits=2)

# データローダーに登録
loader = DataLoader(augmix_dataset, batch_size=32)

```

#### source code

```python
class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
```

- `num_splits` は AugMix の分割数を指定。通常 2 または 3 が使用される。

#### AugMix とは

AugMix は、ICML 2020 で提案された、**複数のデータ拡張パスを混合して使用することで、ノイズ・汚れ・ぼかしなどに対するモデルの堅牢性を高める手法**です。

##### 特徴

- 拡張パスを複数（デフォルトで 2 つ）作成し、その結果をランダムウェイトで合成
- 合成画像 + 元画像 から損失関数で一貫性制約を課すことで、汎化性能を向上

#### AugMix を使った訓練コード

```python
import timm
from timm.data import ImageDataset, IterableImageDataset, AugMixDataset, create_loader
import torch.nn as nn
import torch.optim as optim

dataset = ImageDataset('../../imagenet1K/')
dataset = AugMixDataset(dataset, num_splits=2)

# AugMix を使うためのデータローダー
loader_train = create_loader(
    dataset,
    input_size=(3, 224, 224), # モデルの入力サイズ
    batch_size=8, # バッチサイズ
    is_training=True, # 学習モード
    scale=[0.08, 1.], # 画像のスケール
    ratio=[0.75, 1.33], # 画像のアスペクト比
    num_aug_splits=2 # AugMixの分割数
)

# モデル・損失関数・オプティマイザ
model = timm.create_model('resnet50', pretrained=True, num_classes=1000)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練ループ
for inputs, targets in loader_train:
    print(inputs.shape) # >> torch.Size([16, 3, 224, 224]), 16=batch_size*num_splits
    output1 = model(inputs)
    loss = criterion(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 参考

- **AugMix 論文**: [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)
- **timm のドキュメント**: [https://huggingface.co/docs/timm/](https://huggingface.co/docs/timm/)
