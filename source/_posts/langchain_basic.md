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

#### メッセージメタデータ

```python
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)
```

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-3.5")

response = model.invoke("Hello!")
response.usage_metadata
"""
{'input_tokens': 8,
 'output_tokens': 304,
 'total_tokens': 312,
 'input_token_details': {'audio': 0, 'cache_read': 0},
 'output_token_details': {'audio': 0, 'reasoning': 256}}
"""
```

#### ツールメッセージ
ツールメッセージは、Agentがツールを呼び出してから、ツールよりかえすメッセージです。

```python
from langchain.messages import ToolMessage

ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # Must match the call ID
)
```

### Models

モデルは、LLM をラップし、コンテキストを管理し、メッセージを処理するためのメソッドを提供します。

支持するモデルインタフェス：　[https://docs.langchain.com/oss/python/integrations/chat](https://docs.langchain.com/oss/python/integrations/chat)

#### インストール

例：`pip install -qU langchain-ollama`で、`ollama` 支持をインストール

#### 構成方法

1. モデルクラスによる

```python
from langchain_ollama import OllamaLLM

model = OllamaLLM(
    model="gpt-oss:20b",
    # temperature=0.3,  # Lower temperature for more focused responses
    # top_p=0.9,  # Limit token selection
    # top_k=40,  # Limit vocabulary consideration
    # num_predict=100,  # Limit response length
    # repeat_penalty=1.2,  # Penalize repetition
)
```

2. init_chat_model方法による

```python
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "..."

model = init_chat_model("gpt-3.5")
```

#### 呼び出す

1. 文字による

```python
response = model.invoke("Why do parrots have colorful feathers?")
```

2. 複数のdictによる

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation)
```

3. メッセージオブジェクトによる

```python
conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
```

#### stream / astream

すべでの`Runnable`クラスは、`stream`メソッドと`astream`メソッドをサポートしています。

```python
for chunk in model.stream("What color is the sky?"):
    # ここで、chunkはAIMessageChunkオブジェクトです
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"Reasoning: {reasoning}")
        elif block["type"] == "tool_call_chunk":
            print(f"Tool call chunk: {block}")
        elif block["type"] == "text":
            print(block["text"])
        else:
            ...
```

```python
async for chunk in model.astream("What color is the sky?"):
    ...
```

```python
async for event in model.astream_events("Hello"):

    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")

    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")

    elif event["event"] == "on_chat_model_end":
        print(f"Full message: {event['data']['output'].text}")

    else:
        pass
```

#### 構造化された出力

```python
class TrackInfo(BaseModel):
    name: str
    year: int
    mediaType: str
    genre: str
    unitPrice: float

model_with_structure = model.with_structured_output(TrackInfo, include_raw=True)
```

- `include_raw` は、構造化された出力に加えて、元のAIMessage含めるかどうかを指定します。

```python
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "year": {"type": "integer"},
        "mediaType": {"type": "string"},
        "genre": {"type": "string"},
        "unitPrice": {"type": "number"},
    },
}
model_with_json_schema = model.with_json_schema(json_schema, method="json_schema")
```

### ツール

```python
from langchain.tools import tool


@tool("baidu_search")  # Custom name
def get_search_results(keyword, num_results=3):
    """seach baidu for keyword"""
    results = search(keyword, num_results=num_results)
    assert results
    documents = WebBaseLoader(get_urls(results)).load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=0
    ).split_documents(documents)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in documents
    )
    return serialized
```

### bind_tools

定義したツールをモデルで使用できるようにするには、を使用してツールをバインドする必要があります bind_tools。その後の呼び出しでは、モデルは必要に応じてバインドされたツールのいずれかを呼び出すことを選択できます。

```python
model_with_tools = model.bind_tools([baidu_search])  

response = model_with_tools.invoke("...?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

### ToolRuntime

ToolRuntime は、状態、コンテキスト、ストア、ストリーミング、構成、およびツール呼び出し ID へのツール アクセスを提供する統合パラメータ。

```python
# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```