---
title: LangChainの基本使い方
date: 2025-8-30 21:30:04
categories: [AI]
tags: [Deep Learning, 機械学習, AI, 人工知能, 深層学習, machine learning, LLM, 大規模言語モデル]
lang: ja　
description: LangChainの基本的な使い方について解説しており、LLM（大規模言語モデル）が人間のようにテキストを解釈・生成できる強力なAIツールであることを紹介し、コンテンツ作成、言語翻訳、要約、質問応答など多様なタスクに活用できることを述べています。記事では、メッセージ（HumanMessage、AIMessage、SystemMessage）、モデル（OllamaLLM、init_chat_modelなど）、ツール（@toolデコレーターを使ったカスタムツール定義）、エージェント（create_agentによる作成）、ストリームモード（updates、messages、customモード）、出力構造化（ToolStrategy、ProviderStrategy）、短期記憶（checkpointerによる会話履歴の保存）など、LangChainの主要な構成要素について詳細なコード例を交えて説明しています。
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

# システムメッセージ：AIに役割や指示を与える
system_msg = SystemMessage("You are a helpful assistant.")

# ヒューマンメッセージ：ユーザーからの入力を表す
human_msg = HumanMessage("Hello, how are you?")

# AIメッセージ：AIからの返答を表す
ai_msg = AIMessage("I'm doing well, thanks! How can I help you today?")
```

#### メッセージメタデータ

```python
# メッセージに追加情報を付与する例
human_msg = HumanMessage(
    content="Hello!",                    # 実際のメッセージ内容
    name="alice",                        # オプション：異なるユーザーを識別するための名前
    id="msg_123",                        # オプション：トレース用の一意な識別子
)
```

```python
from langchain.chat_models import init_chat_model

# チャットモデルの初期化
model = init_chat_model("gpt-3")

# モデルにメッセージを送信し、使用状況メタデータを取得
response = model.invoke("Hello!")
response.usage_metadata
"""
{'input_tokens': 8,                     # 入力トークン数
 'output_tokens': 304,                  # 出力トークン数
 'total_tokens': 312,                   # 合計トークン数
 'input_token_details': {'audio': 0, 'cache_read': 0},      # 入力トークンの詳細
 'output_token_details': {'audio': 0, 'reasoning': 256}}     # 出力トークンの詳細
"""
```

#### ツールメッセージ

ツールメッセージは、Agentがツールを呼び出してから、ツールよりかえすメッセージです。

```python
from langchain.messages import ToolMessage

# ツールの実行結果をAIに返すメッセージ
ToolMessage(
    content=weather_result,              # ツールの実行結果
    tool_call_id="call_123"              # ツール呼び出しID（一致させる必要がある）
)
```

### Models

モデルは、LLM をラップし、コンテキストを管理し、メッセージを処理するためのメソッドを提供します。

支持するモデルインタフェース：　[https://docs.langchain.com/oss/python/integrations/chat](https://docs.langchain.com/oss/python/integrations/chat)

#### インストール

例：`pip install -qU langchain-ollama`で、`ollama` 支持をインストール

#### 構成方法

1. モデルクラスによる

```python
from langchain_ollama import OllamaLLM

# Ollamaモデルの初期化
model = OllamaLLM(
    model="gpt-oss:20b",                 # 使用するモデル名
    # temperature=0.3,                   # 応答の多様性を制御（低いほど集中）
    # top_p=0.9,                        # トークン選択の範囲を制限
    # top_k=40,                         # 考慮する語彙の制限
    # num_predict=100,                  # 応答長さの制限
    # repeat_penalty=1.2,               # 繰り返しペナルティ
)
```

2. init_chat_model方法による

```python
from langchain.chat_models import init_chat_model

# APIキーの設定（OpenAIなどの場合）
os.environ["OPENAI_API_KEY"] = "..."

# モデルの初期化
model = init_chat_model("gpt-3")
```

#### 呼び出す

1. 文字による

```python
# 単純な文字列での呼び出し
response = model.invoke("Why do parrots have colorful feathers?")
```

2. 複数のdictによる

```python
# 会話履歴を含む辞書形式での呼び出し
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},  # システムプロンプト
    {"role": "user", "content": "Translate: I love programming."},                                  # ユーザー入力
    {"role": "assistant", "content": "J'adore la programmation."},                                 # AI応答
    {"role": "user", "content": "Translate: I love building applications."}                        # ユーザー入力
]

response = model.invoke(conversation)
```

3. メッセージオブジェクトによる

```python
# メッセージオブジェクトを使用した会話履歴
conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),  # システムメッセージ
    HumanMessage("Translate: I love programming."),                                   # ヒューマンメッセージ
    AIMessage("J'adore la programmation."),                                          # AIメッセージ
    HumanMessage("Translate: I love building applications.")                         # ヒューマンメッセージ
]

response = model.invoke(conversation)
```

#### stream / astream

すべでの`Runnable`クラスは、`stream`メソッドと`astream`メソッドをサポートしています。

```python
# モデルからの応答をチャンク単位で受け取る（同期）
for chunk in model.stream("What color is the sky?"):
    # ここで、chunkはAIMessageChunkオブジェクトです
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"Reasoning: {reasoning}")         # 推論過程の表示
        elif block["type"] == "tool_call_chunk":
            print(f"Tool call chunk: {block}")       # ツール呼び出しのチャンク
        elif block["type"] == "text":
            print(block["text"])                     # テキストチャンク
        else:
            ...
```

```python
# 非同期ストリーミング
async for chunk in model.astream("What color is the sky?"):
    ...
```

```python
# イベントベースのストリーミング（開始、途中、終了イベントを取得）
async for event in model.astream_events("Hello"):

    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")           # 入力イベント

    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")      # トークン受信イベント

    elif event["event"] == "on_chat_model_end":
        print(f"Full message: {event['data']['output'].text}")  # 完了イベント

    else:
        pass
```

#### 構造化された出力

```python
from pydantic import BaseModel

# 構造化された出力の型定義
class TrackInfo(BaseModel):
    name: str                           # 曲名
    year: int                           # 年
    mediaType: str                      # メディアタイプ
    genre: str                          # ジャンル
    unitPrice: float                    # 単価
```

```python
# 構造化出力用のモデルを作成
model_with_structure = model.with_structured_output(TrackInfo, include_raw=True)
```

- `include_raw` は、構造化された出力に加えて、元のAIMessage含めるかどうかを指定します。

```python
# JSONスキーマを使用した構造化出力
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},             # 曲名のプロパティ
        "year": {"type": "integer"},            # 年のプロパティ
        "mediaType": {"type": "string"},        # メディアタイプのプロパティ
        "genre": {"type": "string"},            # ジャンルのプロパティ
        "unitPrice": {"type": "number"},        # 単価のプロパティ
    },
}

# JSONスキーマに基づくモデル
model_with_json_schema = model.with_json_schema(json_schema, method="json_schema")
```

### ツール

```python
from langchain.tools import tool


# カスタムツールの定義
@tool("baidu_search")                    # カスタム名
def get_search_results(keyword, num_results=3):
    """baiduでキーワードを検索する"""
    # 検索結果を取得
    results = search(keyword, num_results=num_results)
    assert results
    
    # 結果からWebページをロード
    documents = WebBaseLoader(get_urls(results)).load()
    
    # ドキュメントをチャンクに分割
    documents = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=0
    ).split_documents(documents)
    
    # 分割されたドキュメントをシリアライズ
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in documents
    )
    return serialized
```

#### bind_tools

定義したツールをモデルで使用できるようにするには、bind_toolsを使用してツールをバインドする必要があります。その後の呼び出しでは、モデルは必要に応じてバインドされたツールのいずれかを呼び出すことを選択できます。

```python
# モデルにツールをバインド
model_with_tools = model.bind_tools([baidu_search])  

# モデルにリクエストを送信
response = model_with_tools.invoke("...?")

# モデルによって行われたツール呼び出しを確認
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")       # ツール名
    print(f"Args: {tool_call['args']}")       # 引数
```

#### ToolRuntime

ToolRuntime は、状態、コンテキスト、ストア、ストリーミング、構成、およびツール呼び出し ID へのツール アクセスを提供する統合パラメータ。

```python
# カスタム状態フィールドにアクセス
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime                 # ToolRuntimeパラメータはモデルには見えない
) -> str:
    """ユーザー設定値を取得する"""
    # ユーザー設定を状態から取得
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

### エージェント

エージェントはモデルとツールを組み合わせて推論し、どのツールを使用するかを決定し、ソリューションに向けて繰り返し作業できるシステムです。

```python
from langchain.agents import create_agent

# OpenAIモデルを使用したエージェントの作成
agent_1 = create_agent("openai:gpt-3", tools=tools)

# Ollamaモデルの初期化
model = ChatOllama(
    model=MODEL,
    base_url="http://127.0.0.1:11434",
)

# Ollamaモデルを使用したエージェントの作成
agent_2 = create_agent(
    model=model,                          # ChatOllamaインスタンス
    tools=tools,                          # ツールリスト
    verbose=True,                         # 中間ステップを印刷
    max_iterations=10,                    # 最大反復回数
    early_stopping_method="generate",     # 早期停止方法
    return_intermediate_steps=True,       # 中間ステップを返す
)
```

#### dynamic model agent

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# 基本モデルと高度モデルの定義
basic_model = ...
advanced_model = ...

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """会話の複雑さに基づいてモデルを選択する"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # 長い会話には高度なモデルを使用
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

# モデル選択ミドルウェアを持つエージェント
agent = create_agent(
    model=basic_model,                   # デフォルトモデル
    tools=tools,                         # ツール一覧
    middleware=[dynamic_model_selection] # モデル選択ミドルウェア
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
    """情報検索ツール"""
    return ...

@tool
def get_weather(location: str) -> str:
    """場所の天気情報を取得するツール"""
    return ...

@wrap_tool_call
def handle_tool_errors(request, handler):
    """ツール実行エラーをカスタムメッセージで処理"""
    try:
        return handler(request)
    except Exception as e:
        # モデルにカスタムエラーメッセージを返す
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# ツールエラーハンドリングミドルウェアを持つエージェント
agent = create_agent(
    model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors],      # ツールエラーハンドリングミドルウェア
)
```

#### system prompt

```python
# システムプロンプトを設定したエージェント
agent = create_agent(
    model,
    tools=tools,
    system_prompt="You are a helpful assistant, ...",  # システムプロンプト
)
```

```python
from langchain.agents.dynamic import dynamic_prompt
from typing import TypedDict

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """ユーザーロールに基づいてシステムプロンプトを生成"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."    # 専門家向け
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon." # 初心者向け

    return base_prompt

class Context(TypedDict):
    user_role: str

# ダイナミックプロンプトミドルウェアを持つエージェント
agent = create_agent(
    ...,                                    # モデル
    middleware=[user_role_prompt],          # ダイナミックプロンプトミドルウェア
    context_schema=Context,                 # コンテキストスキーマ
)

# コンテキストに基づいてシステムプロンプトが動的に設定される
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}         # エキスパート向けにコンテキストを設定
)
```

#### ToolStrategy

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str                              # 名前
    email: str                             # メールアドレス
    phone: str                             # 電話番号

# ToolStrategyを使用した構造化出力のエージェント
agent = create_agent(
    model="gpt-3",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)  # ツール戦略で応答フォーマットを指定
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

#### ProviderStrategy

ProviderStrategyはモデルにあるネイティブ構造化方法をもちいるが、サポートしているモデルのみ利用可能。

```python
# プロバイダー固有の構造化出力を使用したエージェント
agent = create_agent(
    model="gpt-3",
    response_format=ProviderStrategy(ContactInfo)  # プロバイダー戦略を使用
)
```

#### メモリ

エージェントはメッセージ状態を通じて会話履歴を自動的に維持します。会話中に追加情報を記憶するためにカスタム状態スキーマを使用するようにエージェントを構成することもできます。

```python
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict                 # ユーザー設定を保持

# カスタム状態スキーマを持つエージェント
agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)

# エージェントはメッセージ以外の追加状態を追跡できる
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],      # メッセージ
    "user_preferences": {"style": "technical", "verbosity": "detailed"},             # ユーザー設定
})
```

### ストリームモード

ストリームモードは、完全な応答の準備が整う前であっても出力を段階的に表示することで、ストリーミングは、特に LLM の遅延に対処する場合、ユーザー エクスペリエンス（UX）を大幅に向上させます。

```python
from langchain.agents import create_agent


# エージェントの作成
agent = create_agent(
    model="gpt-3",
)

# 更新ストリーミングモード
for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},    # メッセージ
    stream_mode="updates",                                                       # 更新モード
):
    for step, data in chunk.items():
        print(f"step: {step}")                                                   # ステップ名
        print(f"content: {data['messages'][-1].content_blocks}")                 # 内容

"""
step: model
content: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_OW2NYNsNSKhRZpjW0wm2Aszd'}]

step: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]

step: model
content: [{'type': 'text', 'text': 'It\'s always sunny in San Francisco!'}]
"""
```

| モード | 説明 |
|--------|------|
| `updates` | 各エージェント ステップ後に状態の更新をストリーミングします。同じステップで複数の更新が行われる場合（たとえば、複数のノードが実行される場合）、それらの更新は個別にストリーミングされます。 |
| `messages` | ストリームタプル `(token, metadata)` LLM が呼び出される任意のグラフ ノードから。 |
| `custom` | ストリーム ライターを使用して、グラフ ノード内からカスタム データをストリーミングします。 |

```python
# 複数のストリーミングモードを同時に使用
for stream_mode, chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]                                           # 更新とカスタムモード
):
    print(f"stream_mode: {stream_mode}")                                        # ストリーミングモード
    print(f"content: {chunk}")                                                  # コンテンツ
    print("\n")
"""
stream_mode: updates
content: {'model': {'messages': [AIMessage(content='', response_metadata={'token_usage': {'completion_tokens': 280, 'prompt_tokens': 132, 'total_tokens': 412, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 256, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3-2025-08-07', 'system_fingerprint': None, 'id': 'lc_run--480c07cb-e405-4411-aa7f-0520fddeed66-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_KTNQIftMrl9vgNwEfAJMVu7r', 'type': 'tool_call'}], usage_metadata={'input_tokens': 132, 'output_tokens': 280, 'total_tokens': 412, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 256}})]}}


stream_mode: custom
content: Looking up data for city: San Francisco


stream_mode: custom
content: Acquired data for city: San Francisco


stream_mode: updates
content: {'tools': {'messages': [ToolMessage(content="It's always sunny in San Francisco!", name='get_weather', tool_call_id='call_KTNQIftMrl9vgNwEfAJMVu7r')]}}


stream_mode: updates
content: {'model': {'messages': [AIMessage(content='San Francisco weather: It\'s always sunny in San Francisco!\n\n', response_metadata={'token_usage': {'completion_tokens': 764, 'prompt_tokens': 168, 'total_tokens': 932, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 704, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-C9tljDFVki1e1haCyikBptAuXuHYG', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--acbc740a-18fe-4a14-8619-da92a0d0ee90-0', usage_metadata={'input_tokens': 168, 'output_tokens': 764, 'total_tokens': 932, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 704}})]}}
"""
```

### 出力構造化

出力構造化により、エージェントは特定の予測可能な形式でデータを返すことができます。

```python
def create_agent(
    ...
    # 応答フォーマットの型定義
    response_format: Union[
        ToolStrategy[StructuredResponseT],     # ツール呼び出し戦略
        ProviderStrategy[StructuredResponseT], # プロバイダー固有戦略
        type[StructuredResponseT],             # 型ヒント
        None,                                  # 構造化なし
    ]
```

- ToolStrategy[StructuredResponseT]: 構造化出力を呼び出すツールを使用します
- ProviderStrategy[StructuredResponseT]: プロバイダーネイティブの構造化出力を使用します
- type[StructuredResponseT]: スキーマタイプ - モデルの機能に基づいて最適な戦略を自動的に選択します
- None: 構造化出力は明示的に要求されていません

ToolStrategy は、ネイティブの構造化出力をサポートしていないモデルの場合、LangChain はツール呼び出しを使用して同じ結果を達成します。

### 短期記憶

エージェントは、短期記憶を介して過去の会話を記憶し、それを現在の会話に反映させることができます。

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  


# メモリーセーバー付きエージェントの作成
agent = create_agent(
    "gpt-3",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),          # メモリーセーバーの設定
)

# スレッドIDを使って会話を実行
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  # スレッドIDの指定
)
```

生産環境では、データベースに記憶を保存する。

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver  


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()                   # PostgreSQLテーブルの自動作成
    agent = create_agent(
        "gpt-3",
        tools=[get_user_info],
        checkpointer=checkpointer,         # PostgreSQLチェックポイント
    )
```

#### state_schema

Agent は、特定偏向状態を記憶するために、state_schema を使用します。

```python
from typing import Dict, Any

class CustomAgentState(AgentState):  
    user_id: str                          # ユーザーID
    preferences: dict                     # 設定

# カスタム状態スキーマを持つエージェント
agent = create_agent(
    "gpt-3",
    tools=[get_user_info],
    state_schema=CustomAgentState,        # カスタム状態スキーマ
    checkpointer=InMemorySaver(),
)

# カスタム状態をinvokeで渡す
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",            # ユーザーID
        "preferences": {"theme": "dark"}  # 設定
    },
    {"configurable": {"thread_id": "1"}}   # スレッドID
)
```

state_schemaのアクセス

```python
@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """ユーザー情報を検索"""
    user_id = runtime.state["user_id"]     # 状態からユーザーIDを取得
    return ...
```

#### class AgentState

write a **middleware** method to receive the state before model call.

```python
from langchain.agents import before_model
from langchain.messages import RemoveMessage

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """コンテキストウィンドウに合うように最後のいくつかのメッセージだけを保持"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 変更の必要なし
    
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
```

update agent state

```python
from langchain.agents import Command

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """ユーザー情報を検索して更新"""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={  
        "user_name": name,
        # メッセージ履歴を更新
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })
```