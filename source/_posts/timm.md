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

timm.create_model は、指定したモデル名のモデルをロードし、そのモデルを返す関数。

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
>>> from timm import create_model

>>> # 事前学習なしの MobileNetV3-Large モデル作成
>>> model = create_model('mobilenetv3_large_100')

>>> # 事前学習ありの MobileNetV3-Large モデル作成
>>> model = create_model('mobilenetv3_large_100', pretrained=True)
>>> model.num_classes
1000

>>> # 事前学習ありで分類層を10クラスに変更
>>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
>>> model.num_classes
10

>>> # Dinov2 モデルの重みをカスタムディレクトリに保存
>>> model = create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, cache_dir="/data/my-models")
>>> # データ保存先: `/data/my-models/models--timm--vit_small_patch14_dinov2.lvd142m/`
```

