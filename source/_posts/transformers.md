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
  - [使い方](#%E4%BD%BF%E3%81%84%E6%96%B9)
    - [主要コンポーネント概要](#%E4%B8%BB%E8%A6%81%E3%82%B3%E3%83%B3%E3%83%9D%E3%83%BC%E3%83%8D%E3%83%B3%E3%83%88%E6%A6%82%E8%A6%81)
      - [簡単な推論コード例:](#%E7%B0%A1%E5%8D%98%E3%81%AA%E6%8E%A8%E8%AB%96%E3%82%B3%E3%83%BC%E3%83%89%E4%BE%8B)
    - [Pipeline](#pipeline)
      - [サンプルコード:](#%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E3%82%B3%E3%83%BC%E3%83%89)
    - [Tokenizers](#tokenizers)
      - [分詞器の主な機能](#%E5%88%86%E8%A9%9E%E5%99%A8%E3%81%AE%E4%B8%BB%E3%81%AA%E6%A9%9F%E8%83%BD)
        - [**Tokenize**](#tokenize)
        - [**Encode**](#encode)
        - [**Encode + 特殊タグ付加**](#encode--%E7%89%B9%E6%AE%8A%E3%82%BF%E3%82%B0%E4%BB%98%E5%8A%A0)
        - [**Decode**](#decode)
      - [分詞器の高階機能](#%E5%88%86%E8%A9%9E%E5%99%A8%E3%81%AE%E9%AB%98%E9%9A%8E%E6%A9%9F%E8%83%BD)
        - [特殊トークンと語彙情報](#%E7%89%B9%E6%AE%8A%E3%83%88%E3%83%BC%E3%82%AF%E3%83%B3%E3%81%A8%E8%AA%9E%E5%BD%99%E6%83%85%E5%A0%B1)
        - [バッチ処理と長文対応](#%E3%83%90%E3%83%83%E3%83%81%E5%87%A6%E7%90%86%E3%81%A8%E9%95%B7%E6%96%87%E5%AF%BE%E5%BF%9C)
      - [実装例](#%E5%AE%9F%E8%A3%85%E4%BE%8B)
    - [実用的な設定と最適化技法](#%E5%AE%9F%E7%94%A8%E7%9A%84%E3%81%AA%E8%A8%AD%E5%AE%9A%E3%81%A8%E6%9C%80%E9%81%A9%E5%8C%96%E6%8A%80%E6%B3%95)
      - [モデルロード時の最適化](#%E3%83%A2%E3%83%87%E3%83%AB%E3%83%AD%E3%83%BC%E3%83%89%E6%99%82%E3%81%AE%E6%9C%80%E9%81%A9%E5%8C%96)
      - [バッチ処理の最適化](#%E3%83%90%E3%83%83%E3%83%81%E5%87%A6%E7%90%86%E3%81%AE%E6%9C%80%E9%81%A9%E5%8C%96)
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

## 使い方

### 主要コンポーネント概要

HuggingFace Transformers はモジュール化されたライブラリで、以下のような主要なコンポーネントが含まれています:

- **`AutoTokenizer`**: テキストの分詞やエンコーディングに使用されます。
- **`AutoModel`**: 学習済みモデルを読み込むための基本クラスです。
- **`Trainer`, `TrainingArguments`**: モデルのファインチューニングを行うための高レベルツールです。
- **`Pipeline`**: 前処理から推論、後処理までの全フローをカプセル化しており、素早く開発を開始できます。

#### 簡単な推論コード例:
```python
from transformers import AutoTokenizer, AutoModel

def basic_usage_example():
    # Tokenizerとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')
    
    # 入力テキストの前処理
    text = "こんにちは世界！"
    inputs = tokenizer(text, return_tensors="pt")  # PyTorchテンソルとして返す
    
    # 推論実行
    outputs = model(**inputs)
    
    # 隠れ層の最終出力を返す
    return outputs.last_hidden_state
```

### Pipeline

`pipeline` は非常に簡単にさまざまなNLPタスクを実行できるインターフェースです。

#### サンプルコード:
```python
from transformers import pipeline

def pipeline_examples():
    """代表的なタスクのPipeline例"""

    # 感情分析
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer("この製品はとても使いやすい！")
    print(f"感情分析結果：{result}")
    
    # テキスト生成（GPT-2）
    generator = pipeline("text-generation", model="gpt2-chinese")
    text = generator("人工知能は今", max_length=50)
    print(f"生成されたテキスト：{text}")
    
    # 固有表現抽出（NER）
    ner = pipeline("ner", model="bert-base-chinese")
    entities = ner("華為の本社は深圳にあります")
    print(f"認識された固有名詞：{entities}")
    
    # 質問応答システム
    qa = pipeline("question-answering", model="bert-base-chinese")
    context = "北京は中国の首都であり、上海は最大の経済都市です。"
    question = "中国の首都是はどこですか？"
    answer = qa(question=question, context=context)
    print(f"質問応答結果：{answer}")

if __name__ == "__main__":
    pipeline_examples()
```

### Tokenizers

Tokenizers は自然言語処理（NLP）においてテキストをモデルが理解できる形式に変換するための基本的なコンポーネントです。  
HuggingFace Transformers では、多様な言語やモデルに対応した柔軟で高性能な `Tokenizer API` を提供しています。

#### 分詞器の主な機能

##### **Tokenize**
- テキストを単語、サブワード、文字などの「トークン」に分割します。
- 例: `"これはテスト"` → `["これ", "は", "テスト"]`

```python
tokens = tokenizer.tokenize("これはテスト")
```

##### **Encode**
- トークンを数値 ID（語彙ID）に変換します。
- 例: `["これ", "は", "テスト"]` → `[345, 890, 1234]`

```python
token_ids = tokenizer.convert_tokens_to_ids(tokens)
```

##### **Encode + 特殊タグ付加**
- 入力に特殊タグ（[CLS], [SEP]など）を追加し、テンソル形式で出力します。

```python
encoded = tokenizer(text, return_tensors="pt")
```

##### **Decode**
- 数値 ID を再び人間可読なテキストに戻します。

```python
decoded_text = tokenizer.decode(encoded["input_ids"][0])
```

#### 分詞器の高階機能

#####  特殊トークンと語彙情報

- **[CLS]**: 分類タスク用の特別トークン
- **[SEP]**: 文の区切りを示すトークン

```python
print(f"CLS标记: {tokenizer.cls_token}")      # [CLS]
print(f"SEP标记: {tokenizer.sep_token}")      # [SEP]
print(f"词表大小: {len(tokenizer)}")          # 語彙数
print(f"特殊标记映射: {tokenizer.special_tokens_map}")
```

##### バッチ処理と長文対応

複数文を一度に処理し、パディングと切り詰めも自動化できます。

```python
batch_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

長いテキストを最大長で自動的に切り詰めて処理します。

```python
truncated = tokenizer(long_text, max_length=128, truncation=True)
```

#### 実装例

```python
from transformers import AutoTokenizer

def tokenizer_basics():
    # 分詞器ロード
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # テキスト入力
    text = "これはテスト"

    # 1. 分詞
    tokens = tokenizer.tokenize(text)
    print(f"Result: {tokens}")

    # 2. Token IDs へ変換
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")

    # 3. 編碼（特殊タグ含む）
    encoded = tokenizer(text, return_tensors="pt")
    print(f"Encoding result: {encoded}")

    # 4. 解码
    decoded = tokenizer.decode(encoded["input_ids"][0])
    print(f"Decoding result: {decoded}")

# 特殊トークン情報
def tokenizer_special_tokens():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(f"CLS map: {tokenizer.cls_token}")
    print(f"SEP map: {tokenizer.sep_token}")
    print(f"tokenizer length: {len(tokenizer)}")
    print(f"spcial token map: {tokenizer.special_tokens_map}")

# バッチ・長文処理
def batch_and_long_text_processing():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    texts = ["this is the first text", "this is the second text"]
    batch_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    print(f"batch processing result: {batch_encoding}")

    long_text = "this is a " + "very very very very very very very very very very " * 50 + "long text。"""
    truncated = tokenizer(long_text, max_length=128, truncation=True)
    print(f"Result: {truncated['input_ids']}")

# 実行例
if __name__ == "__main__":
    tokenizer_basics()
    tokenizer_special_tokens()
    batch_and_long_text_processing()
```


### 実用的な設定と最適化技法

#### モデルロード時の最適化
以下の方法でメモリ効率と推論速度を向上させることができます：

```python
from transformers import AutoModel
import torch

def setup_optimization():
    """モデルの最適化設定"""
    model = AutoModel.from_pretrained(
        "bert-base-chinese",
        device_map="auto",         # 自動的にGPU/CPUに割り当て
        torch_dtype=torch.float16, # 半精度を使用してメモリ削減
        low_cpu_mem_usage=True     # CPUメモリ消費を抑える
    )
    model.eval()  # 推論モードへ切り替え
    return model
```

#### バッチ処理の最適化
大規模なテキストデータを扱う際には、バッチ処理により処理速度が向上します。長文の分割も自動化しています。

```python
from typing import List

def batch_process(texts: List[str], batch_size: int, max_length: int) -> List[List[str]]:
    """長文を分割し、指定バッチサイズで処理する"""
    processed_texts = []
    for text in texts:
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            processed_texts.extend(chunks)
        else:
            processed_texts.append(text)
    
    return [processed_texts[i:i+batch_size] for i in range(0, len(processed_texts), batch_size)]
```



### 参考
- Hubからのファイルダウンロード詳細については [Transformers official doc](https://huggingface.co/docs/transformers) を参照。
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/transformers/tokenizer_summary)