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

model = init_chat_model("gpt-3")

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

model = init_chat_model("gpt-3")
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

#### bind_tools

定義したツールをモデルで使用できるようにするには、を使用してツールをバインドする必要があります bind_tools。その後の呼び出しでは、モデルは必要に応じてバインドされたツールのいずれかを呼び出すことを選択できます。

```python
model_with_tools = model.bind_tools([baidu_search])  

response = model_with_tools.invoke("...?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

#### ToolRuntime

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

### エージェント

エージェントはモデルとツールを組み合わせて推論し、どのツールを使用するかを決定し、ソリューションに向けて繰り返し作業できるシステムです。

```python
from langchain.agents import create_agent

agent_1 = create_agent("openai:gpt-3", tools=tools)

model = ChatOllama(
    model=MODEL,
    base_url="http://127.0.0.1:11434",
)
agent_2 = create_agent(
    model=model, # ChatOllama instance
    tools=tools, # Tool list
    verbose=True, # Print intermediate steps
    max_iterations=10, # Maximum number of iterations
    early_stopping_method="generate", # Early stopping method
    return_intermediate_steps=True, # Return intermediate steps
)
```

#### dynamic model agent

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ...
advanced_model = ...

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

#### use tool and middleware

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


@tool
def search(query: str) -> str:
    """Search for information."""
    return ...

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return ...

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


agent = create_agent(
    model,
    tools=[search, get_weather],
    iddleware=[handle_tool_errors],
)
```

#### system prompt

```python
agent = create_agent(
    model,
    tools=tools,
    system_prompt="You are a helpful assistant, ...",
)
```

```python
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

class Context(TypedDict):
    user_role: str

agent = create_agent(
    ...,
    middleware=[user_role_prompt],
    context_schema=Context,
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
```

#### ToolStrategy

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-3",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

#### ProviderStrategy

ProviderStrategyはモデルにあるストラクチャ化方法をもちいるが、支持するモデルしか利用できない。

```python
agent = create_agent(
    model="gpt-3",
    response_format=ProviderStrategy(ContactInfo)
)
```

#### メモリ

エージェントはメッセージ状態を通じて会話履歴を自動的に維持します。会話中に追加情報を記憶するためにカスタム状態スキーマを使用するようにエージェントを構成することもできます。

```python
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

### ストリームモード

ストリームモードは、完全な応答の準備が整う前であっても出力を段階的に表示することで、ストリーミングは、特に LLM の遅延に対処する場合、ユーザー エクスペリエンス（UX）を大幅に向上させます。

```python
from langchain.agents import create_agent


agent = create_agent(
    model="gpt-3",
)

for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")

"""
step: model
content: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_OW2NYNsNSKhRZpjW0wm2Aszd'}]

step: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]

step: model
content: [{'type': 'text', 'text': 'It's always sunny in San Francisco!'}]
"""
```

<table>
    <thead>
        <tr>
            <th>モード</th>
            <th>説明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>updates</code></td>
            <td>各エージェント ステップ後に状態の更新をストリーミングします。同じステップで複数の更新が行われる場合（たとえば、複数のノードが実行される場合）、それらの更新は個別にストリーミングされます。</td>
        </tr>
        <tr>
            <td><code>messages</code></td>
            <td>ストリームタプル <code>(token, metadata)</code> LLM が呼び出される任意のグラフ ノードから。</td>
        </tr>
        <tr>
            <td><code>custom</code></td>
            <td>ストリーム ライターを使用して、グラフ ノード内からカスタム データをストリーミングします。</td>
        </tr>
    </tbody>
</table>

```python
for stream_mode, chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")
"""
stream_mode: updates
content: {'model': {'messages': [AIMessage(content='', response_metadata={'token_usage': {'completion_tokens': 280, 'prompt_tokens': 132, 'total_tokens': 412, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 256, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-nano-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-C9tlgBzGEbedGYxZ0rTCz5F7OXpL7', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--480c07cb-e405-4411-aa7f-0520fddeed66-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_KTNQIftMrl9vgNwEfAJMVu7r', 'type': 'tool_call'}], usage_metadata={'input_tokens': 132, 'output_tokens': 280, 'total_tokens': 412, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 256}})]}}


stream_mode: custom
content: Looking up data for city: San Francisco


stream_mode: custom
content: Acquired data for city: San Francisco


stream_mode: updates
content: {'tools': {'messages': [ToolMessage(content="It's always sunny in San Francisco!", name='get_weather', tool_call_id='call_KTNQIftMrl9vgNwEfAJMVu7r')]}}


stream_mode: updates
content: {'model': {'messages': [AIMessage(content='San Francisco weather: It's always sunny in San Francisco!\n\n', response_metadata={'token_usage': {'completion_tokens': 764, 'prompt_tokens': 168, 'total_tokens': 932, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 704, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-nano-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-C9tljDFVki1e1haCyikBptAuXuHYG', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--acbc740a-18fe-4a14-8619-da92a0d0ee90-0', usage_metadata={'input_tokens': 168, 'output_tokens': 764, 'total_tokens': 932, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 704}})]}}
"""
```

