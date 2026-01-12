---
title: LangChainの基本使い方
date: 2025-12-7 21:30:04
categories: [AI]
tags: [Deep Learning, 機械学習, AI, 人工知能, 深層学習, machine learning, LLM, 大規模言語モデル]
lang: ja　
description: 。。。
---

# LangChain

LLM は、人間のようにテキストを解釈して生成できる強力な AI ツールです。各タスクに専門的なトレーニングを必要とせずに、コンテンツの作成、言語の翻訳、要約、質問への回答を行うのに十分な多用途性があります。 -- [LLM Model](https://docs.langchain.com/oss/python/langchain/models)

## インストール

```bash
pip install langchain
```

## 基本構成要素

### Message

メッセージは、LangChain のモデルのコンテキストの基本単位です。

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")
ai_msg = AIMessage("I'm doing well, thanks! How can I help you today?")
```

### Models

モデルは、LLM をラップし、コンテキストを管理し、メッセージを処理するためのメソッドを提供します。

支持するモデルインタフェス：　[https://docs.langchain.com/oss/python/integrations/chat](https://docs.langchain.com/oss/python/integrations/chat)

#### インストール

例：`pip install -qU langchain-ollama`で、`ollama` 支持をインストール

#### 構成方法
