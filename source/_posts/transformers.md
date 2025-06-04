---
title: Transformers　フレームワーク
date: 2024-3-30 17:10:00
categories: [AI]
tags: [Deep Learning, transformers, 機械学習, AI, 人工知能, 深層学習]
lang: ja
description: Transformers は、PyTorch, TensorFlow, JAX に対応した機械学習ライブラリで、最先端の学習済みモデルを簡単にダウンロードして利用できるように設計されています。このフレームワークは、自然言語処理やコンピュータビジョン、音声認識などさまざまな分野でのタスクをサポートし、柔軟なフレームワーク間相互運用性と本番環境向けのデプロイメント機能（ONNX や TorchScript 形式へのエクスポート）を提供します。
---

## 目次

- [Transformers フレームワーク](#transformers-%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%83%AF%E3%83%BC%E3%82%AF)
  - [インストール](#%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
    - [pipでのインストール](#pip%E3%81%A7%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
    - [編集可能なインストール](#%E7%B7%A8%E9%9B%86%E5%8F%AF%E8%83%BD%E3%81%AA%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
    - [キャッシュの設定](#%E3%82%AD%E3%83%A3%E3%83%83%E3%82%B7%E3%83%A5%E3%81%AE%E8%A8%AD%E5%AE%9A)
    - [オフラインモード](#%E3%82%AA%E3%83%95%E3%83%A9%E3%82%A4%E3%83%B3%E3%83%A2%E3%83%BC%E3%83%89)
    - [オフラインでのモデルトークナイザー利用方法](#%E3%82%AA%E3%83%95%E3%83%A9%E3%82%A4%E3%83%B3%E3%81%A7%E3%81%AE%E3%83%A2%E3%83%87%E3%83%AB%E3%83%88%E3%83%BC%E3%82%AF%E3%83%8A%E3%82%A4%E3%82%B6%E3%83%BC%E5%88%A9%E7%94%A8%E6%96%B9%E6%B3%95)
      - [**Model Hub UI** から手動でダウンロード](#model-hub-ui-%E3%81%8B%E3%82%89%E6%89%8B%E5%8B%95%E3%81%A7%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89)
      - [PreTrainedModel.from\_pretrained() \& save\_pretrained() ワークフロー](#pretrainedmodelfrompretrained--savepretrained-%E3%83%AF%E3%83%BC%E3%82%AF%E3%83%95%E3%83%AD%E3%83%BC)
      - [huggingface\_hubライブラリを使用したプログラム的なダウンロード](#huggingfacehub%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%9F%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E7%9A%84%E3%81%AA%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89)
    - [参考](#%E5%8F%82%E8%80%83)


---

# Transformers フレームワーク

Transformers は、PyTorch, TensorFlow, JAX に対応した機械学習ライブラリで、最先端の学習済みモデルを簡単にダウンロードして利用できるように設計されています。このフレームワークは、自然言語処理やコンピュータビジョン、音声認識などさまざまな分野でのタスクをサポートし、柔軟なフレームワーク間相互運用性と本番環境向けのデプロイメント機能（ONNX や TorchScript 形式へのエクスポート）を提供します。


## インストール
### pipでのインストール

- 前提として、Pytorchまたはtensorflowをインストールしておく必要があります、さらに、flaxもサポートされています。


これで、次のコマンドで🤗 Transformersをインストールする準備が整いました:

```
pip install transformers
```

CPU対応のみ必要な場合、🤗 TransformersとDeep Learningライブラリを1行でインストールできるようになっていて便利です。例えば、🤗 TransformersとPyTorchを以下のように一緒にインストールできます:

```
pip install transformers[torch]
```

🤗 TransformersとTensorFlow 2.0:

```
pip install transformers[tf-cpu]
```

🤗 TransformersとFlax:

```
pip install transformers[flax]
```

最後に、以下のコマンドを実行することで🤗 Transformersが正しくインストールされているかを確認します。学習済みモデルがダウンロードされます:

```
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

その後、ラベルとスコアが出力されます:

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### 編集可能なインストール

必要に応じて、編集可能なインストールをします:

- ソースコードの`main`バージョンを使います。
- 🤗 Transformersにコントリビュートし、コードの変更をテストする必要があります。

以下のコマンドでレポジトリをクローンして、🤗 Transformersをインストールします:

```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

### キャッシュの設定
- 学習済みモデルはデフォルトで `~/.cache/huggingface/hub` にキャッシュされます。
- Windowsでは、`C:\Users\username\.cache\huggingface\hub` がデフォルトです。
- キャッシュディレクトリは以下の環境変数で変更可能（優先順位順）:
  1. `HF_HUB_CACHE` or `TRANSFORMERS_CACHE`
  2. `HF_HOME`
  3. `XDG_CACHE_HOME` + `/huggingface`

> ⚠️ 過去のバージョンで `PYTORCH_TRANSFORMERS_CACHE` または `PYTORCH_PRETRAINED_BERT_CACHE` を使用していた場合、それらが引き続き使用されます（明示的に `TRANSFORMERS_CACHE` を設定していない限り）。


### オフラインモード
- オフライン環境でも動作させるには、以下のように環境変数を設定します:
  - `HF_HUB_OFFLINE=1`: Hubからダウンロードしない
  - `HF_DATASETS_OFFLINE=1`: データセットもオフライン対応

例:
```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```


### オフラインでのモデルトークナイザー利用方法
#### **Model Hub UI** から手動でダウンロード
- Webインターフェース上の ↓ アイコンをクリックしてファイルを取得。

#### PreTrainedModel.from_pretrained() & save_pretrained() ワークフロー
前もってオンライン環境でダウンロード＆保存:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

tokenizer.save_pretrained("./your/path/bigscience_t0")
model.save_pretrained("./your/path/bigscience_t0")
```

オフライン環境で再読み込み:
```python
tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
model = AutoModel.from_pretrained("./your/path/bigscience_t0")
```

#### huggingface_hubライブラリを使用したプログラム的なダウンロード
インストール:
```bash
python -m pip install huggingface_hub
```

特定のファイルを指定してダウンロード:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
```

ダウンロード後にロード:
```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```


### 参考
- Hubからのファイルダウンロード詳細については [Transformers official doc](https://huggingface.co/docs/transformers) を参照。