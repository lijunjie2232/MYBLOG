---
title: DeepAgents ソースコード解析 Part 1：create_deep_agent の全体組み立て・DeepAgentState と Middleware スタック順序
date: 2026-09-20 02:34:59
categories: [AI]
tags: [Python, LLM, AI Agent, DeepAgents, LangGraph, ソースコード解析]
lang: ja
description: DeepAgents のエントリポイント create_deep_agent（graph.py）を徹底解剖。DeltaChannel による O(N) チェックポイント最適化（DeepAgentState）、HarnessProfile の適用、デフォルト Middleware スタックの厳密な順序設計、およびカスタム Middleware のマージ・除外機構を実際のソースコードから解説する。
---

# DeepAgents ソースコード解析 Part 1：`create_deep_agent` の全体組み立て・`DeepAgentState` と Middleware スタック順序

LangChain / LangGraph エコシステム上で「長期タスクを自律的に計画・実行するエージェント（Deep Agent）」を構築するフレームワーク **DeepAgents**。本連載では、その実装の中核となるソースコードを 6 回に分けて徹底的に読み解きます。

第 1 回では、すべての起点となるエントリポイント関数 [`create_deep_agent`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py) を中心に、以下の 4 つの設計ポイントをソースコードレベルで解剖します。

1. **`DeepAgentState` と `DeltaChannel`**：メッセージ履歴のチェックポイント肥大化を $O(N^2)$ から $O(N)$ に抑える状態管理
2. **`create_deep_agent` の全体アセンブリフロー**：モデル解決・`HarnessProfile`・Backend バインド・サブエージェント構築
3. **デフォルト Middleware スタックの厳密な順序**：Base Stack / User Slot / Tail Stack がなぜこの順序で並んでいるのか
4. **`_apply_custom_middleware` と `_excluded_middleware`**：名前ベースのインプレース置換と保護スキャフォールディング検証

---

## 1. `DeepAgentState`：なぜ `DeltaChannel` を使うのか

LangGraph で長期実行エージェントを動かすと、1 回の実行で数十〜数百ステップに及ぶツール呼び出しが発生します。標準的な `AgentState`（`add_messages` リデューサ）では、ステップごとに `messages` リスト全体がチェックポイントにスナップショット保存されるため、ステップ数 $N$ に対してチェックポイント総容量が $O(N^2)$ で膨張する問題があります。

DeepAgents は [`graph.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L77-L80) で `AgentState` を継承した `DeepAgentState` を定義し、`messages` フィールドに `DeltaChannel` を差し込んでいます。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py
class DeepAgentState(AgentState):
    """AgentState with `DeltaChannel` on messages to reduce checkpoint growth from O(N²) to O(N)."""

    # DeltaChannel を messages チャネルに適用する
    # - _messages_delta_reducer: 全メッセージ配列ではなく「差分（追加・更新・削除）」のみを記録するリデューサ
    # - snapshot_frequency=50: 50 ステップごとにフルスナップショットを出力し、復元時のリプレイコストを一定に保つ
    messages: Required[
        Annotated[
            list[AnyMessage],
            DeltaChannel(_messages_delta_reducer, snapshot_frequency=50),
        ]
    ]
```

### 設計意図のポイント

- **チェックポイント容量の線形化**：各ステップでは直前ステップからの増分メッセージだけを永続化するため、100 ステップを超えるディープリサーチや大規模コーディングでも DB 容量とシリアライズ遅延を抑えられます。
- **復元コストとのバランス**：`snapshot_frequency=50` を指定しているため、チェックポイントから再開する際も最大 49 ステップ分の差分リプレイで済みます。
- **カスタム `state_schema` の制約**：ユーザーが独自の `state_schema` を渡す場合も、`DeepAgentState` のサブクラス（`TypedDict`）にすることが推奨されています。なお `TypedDict` は Python ランタイムで `issubclass()` をサポートしないため、`graph.py` 内のコメントにもある通り、この制約は型チェッカ側で担保されています。

---

## 2. `create_deep_agent` の全体組み立てフロー

[`create_deep_agent`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L296-L1046) は、ユーザーから受け取ったモデル・ツール・プロンプト・Backend・SubAgent・Skills・Memory・Permissions・Middleware を 1 つの `CompiledStateGraph` に統合するファクトリ関数です。

まずは全体のアセンブリフローを Mermaid 図で俯瞰しましょう。

{% mermaid %}
flowchart TD
    A["create_deep_agent(...) 呼び出し"] --> B["モデル解決: resolve_model(model)"]
    B --> C["HarnessProfile 取得 & 除外設定の事前検証<br/>_harness_profile_for_model / _validate_excluded_middleware_config"]
    C --> D["ツール説明の上書き: _apply_tool_description_overrides"]
    D --> E["Backend 初期化 (デフォルト StateBackend)<br/>& カスタム Middleware へ bind_backend"]
    E --> F["System Prompt 組み立て<br/>USER + BASE(profile) + SUFFIX(profile)"]
    F --> G["SubAgents の分類・前処理<br/>AsyncSubAgent / CompiledSubAgent / Declarative SubAgent"]
    G --> H{"general-purpose<br/>SubAgent を自動追加?"}
    H -- "有効 & 未上書き" --> I["GP SubAgent 構築 (先頭 index 0 に挿入)"]
    H -- "無効 or 上書き済" --> J["メイン Agent の Middleware スタック構築"]
    I --> J
    J --> K["Base Stack -> Tail Stack 構築<br/>& _apply_excluded_middleware / _apply_custom_middleware"]
    K --> L["private_state_keys 収集<br/>-> SubAgentMiddleware に伝播"]
    L --> M["_verify_excluded_middleware_coverage<br/>未使用の除外設定をエラー検知"]
    M --> N["create_agent(...) + with_config(recursion_limit=9999)"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style C fill:#fff3e0,stroke:#ef6c00,color:#e65100
    style F fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    style K fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style N fill:#fce4ec,stroke:#c2185b,color:#880e4f
{% endmermaid %}

### 2.1 モデル解決・`HarnessProfile`・プロンプト合成

関数の前半（[`graph.py` L641-L697](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L641-L697)）では、モデルごとの差異を吸収する `HarnessProfile` を解決し、プロンプトと Backend を準備します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py
_model_spec: str | None = model if isinstance(model, str) else None

if model is None:
    # model=None は 0.5.3 で非推奨（1.0.0 で削除予定）。フォールバックは claude-sonnet-4-6
    warn_deprecated(...)
    model = _build_default_model()
else:
    # "openai:gpt-5.5" のような文字列指定や BaseChatModel を正規化
    model = resolve_model(model)

# モデル種別に応じた HarnessProfile（ツール説明上書き・追加/除外ミドルウェア等）を取得
_profile = _harness_profile_for_model(model, _model_spec)

# 保護対象の必須ミドルウェア（FilesystemMiddleware / SubAgentMiddleware）が除外されていないか検証
_validate_excluded_middleware_config(
    _profile,
    required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
    required_names=_REQUIRED_MIDDLEWARE_NAMES,
)

# Backend が未指定なら LangGraph State ベースの StateBackend を既定にする
backend = backend if backend is not None else StateBackend()
# BackendConsumer プロトコルを実装するユーザー Middleware に同一 backend を注入
_bind_backend(middleware or (), backend)

# System Prompt は USER -> BASE -> SUFFIX の順で組み立てる
base_prompt = _apply_profile_prompt(_profile, "")
if system_prompt is None:
    final_system_prompt: str | SystemMessage = base_prompt
elif isinstance(system_prompt, SystemMessage):
    # SystemMessage が渡された場合は既存の cache_control ブロックを破壊せず末尾にテキストブロック追加
    if base_prompt:
        final_system_prompt = SystemMessage(
            content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{base_prompt}"}]
        )
    else:
        final_system_prompt = system_prompt
else:
    final_system_prompt = system_prompt + (f"\n\n{base_prompt}" if base_prompt else "")
```

ここで注目すべき細部は以下の 2 点です。

- **`SystemMessage` の `content_blocks` 保持**：ユーザーが Anthropic のプロンプトキャッシュ用に `cache_control` 付きの `SystemMessage` を渡した場合、単純な文字列結合をするとキャッシュブレークポイントが失われます。そのため `content_blocks` のリストを維持したまま末尾に新規ブロックを追加する実装になっています。
- **`BackendConsumer` プロトコルによる DI**：`_bind_backend` は `isinstance(m, BackendConsumer)` を判定し、`m.bind_backend(backend)` を呼び出します。これにより、ユーザーが渡したカスタム Middleware もエージェント本体とまったく同じストレージ空間（`StateBackend` や `CompositeBackend`）を透過的に共有できます。

### 2.2 最後の仕上げ：`private_state_keys` と `recursion_limit=9_999`

[`graph.py` L1005-L1046](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L1005-L1046) の末尾では、全 Middleware の `state_schema` からプライベート状態キー（サブエージェントへ漏洩・逆流させてはいけない内部チャネル）を収集し、`SubAgentMiddleware` に伝播したうえで LangChain の `create_agent()` に委譲します。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py
# ユーザー定義 state_schema + 各 Middleware が宣言した state_schema からプライベートキーを抽出
state_schemas = [state_schema] if state_schema is not None else []
state_schemas.extend(
    mw.state_schema for mw in deepagent_middleware if getattr(mw, "state_schema", None) is not None
)
private_state_keys = private_state_field_names(*state_schemas)

# SubAgentMiddleware がカスタム置換された後でも確実に反映されるよう、組み立て完了後にセットする
for middleware_instance in deepagent_middleware:
    if isinstance(middleware_instance, SubAgentMiddleware):
        middleware_instance.private_state_keys = (
            middleware_instance.private_state_keys | private_state_keys
        )

return create_agent(
    model,
    system_prompt=final_system_prompt,
    tools=_tools,
    middleware=deepagent_middleware,
    response_format=response_format,
    context_schema=context_schema,
    checkpointer=checkpointer,
    store=store,
    debug=debug,
    name=name,
    cache=cache,
    state_schema=state_schema if state_schema is not None else DeepAgentState,
).with_config(
    {
        # LangGraph 標準の recursion_limit=25 では長期タスクが即座に打ち切られるため 9,999 に拡張
        "recursion_limit": 9_999,
        "metadata": {
            "ls_integration": "deepagents",
            "lc_versions": {"deepagents": _lc_version()},
            "lc_agent_name": name,
        },
