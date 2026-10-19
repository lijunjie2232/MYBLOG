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
    }
)
```

---

## 3. デフォルト Middleware スタックの厳密な順序設計

DeepAgents の真骨頂は、**Middleware スタックの層構造と並び順**にあります。[`graph.py` L920-L1004](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L920-L1004) で組み立てられるメインエージェントのスタックは、大きく **Base Stack（コア層）**、**User Middleware 挿入スロット**、**Tail Stack（プロンプト最適化・ガード層）** の 3 ブロックに分かれています。

{% mermaid %}
flowchart LR
    subgraph BASE["1. Base Stack (_main_core_names)"]
        direction TB
        M1["1. FilesystemMiddleware<br/>(ファイル操作・権限・巨大結果退避)"]
        M2["2. SubAgentMiddleware<br/>(task ツール・同期サブエージェント)"]
        M3["3. _ParentNotifierMiddleware<br/>(親通知フック)"]
        M4["4. SummarizationMiddleware<br/>(コンテキスト圧縮パイプライン)"]
        M5["5. PatchToolCallsMiddleware<br/>(宙ぶらりん tool_call 修復)"]
        M6["6. AsyncSubAgentMiddleware<br/>(非同期リモートサブエージェント / 条件付)"]
        M1 --> M2 --> M3 --> M4 --> M5 --> M6
    end

    subgraph USER["2. User Middleware Slot"]
        direction TB
        U1["新規カスタム Middleware を<br/>ここにスプライス挿入<br/>(既存名一致はインプレース置換)"]
    end

    subgraph TAIL["3. Tail Stack (プロンプト・キャッシュ・安全層)"]
        direction TB
        T1["7. Profile extra_middleware"]
        T2["8. SkillsMiddleware (条件付)"]
        T3["9. PromptCachingMiddleware<br/>(Anthropic / Bedrock / Fireworks)"]
        T4["10. MemoryMiddleware (条件付)<br/>(add_cache_control=True)"]
        T5["11. HumanInTheLoopMiddleware (条件付)"]
        T6["12. UnsupportedContentMiddleware"]
        T7["13. ModelRefusalMiddleware"]
        T8["14. _ToolExclusionMiddleware (条件付)"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8
    end

    BASE --> USER --> TAIL

    style BASE fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style USER fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    style TAIL fill:#fff3e0,stroke:#ef6c00,color:#e65100
{% endmermaid %}

### なぜこの順序でなければならないのか？

ソースコード内のコメントには、各層の順序に関する極めて実践的な設計理由が明記されています。

1. **`SummarizationMiddleware` が `SkillsMiddleware` やキャッシュ層より前にある理由**
   - `SkillsMiddleware` は「圧縮（compaction）後の会話履歴」と「ルーティング・フォールバック後の実際に呼び出されるモデル」を見て、必要なスキルツールだけを段階的に開示（Progressive Disclosure）する必要があります。そのため、ユーザー Middleware や `SummarizationMiddleware` より**内側（後段）**に配置されています。
2. **`PromptCachingMiddleware` と `MemoryMiddleware` の位置関係**
   - `MemoryMiddleware`（`AGENTS.md` などの動的メモリ）はシステムプロンプトの内容を更新し得るため、静的なプレフィックスをキャッシュする `PromptCachingMiddleware` の**後**に配置し、メモリ更新で Anthropic Prompt Cache のプレフィックス全体が無効化されるのを防いでいます（`MemoryMiddleware` 自身も `add_cache_control=True` で独自のキャッシュブレークポイントを付与します）。
3. **`ModelRefusalMiddleware` が末尾にある理由**
   - ユーザーが `ModelFallbackMiddleware` などのフォールバック用 Middleware を渡した場合、ユーザー層が先にモデルの拒否応答（refusal）を捕捉して別モデルへ切り替えられるようにするため、`ModelRefusalMiddleware` は Tail Stack の終端近くに置かれています。
4. **`_ToolExclusionMiddleware` が絶対最後尾にある理由**
   - `HarnessProfile.excluded_tools` で禁止されたツール名を、すべてのツール注入 Middleware やユーザーカスタム Middleware の `wrap_model_call` が終わった後に一括で剥がすことで、カスタム Middleware による意図しない復活を防ぎます。

---

## 4. カスタム Middleware のマージと除外の仕組み

### 4.1 `_apply_custom_middleware`：名前一致のインプレース置換と Permissions 保護

ユーザーが `create_deep_agent(..., middleware=[...])` で渡した Middleware は、単純に末尾へ追加されるのではなく、[`_apply_custom_middleware`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L223-L263) によって賢くマージされます。

```python
# https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py
def _apply_custom_middleware(
    base: list[AgentMiddleware[Any, Any, Any]],
    custom: Sequence[AgentMiddleware[Any, Any, Any]],
    *,
    core_names: set[str] | None = None,
) -> list[AgentMiddleware[Any, Any, Any]]:
    if not custom:
        return list(base)
    current_names = {m.name for m in base}
    replacements: dict[str, AgentMiddleware[Any, Any, Any]] = {}
    to_append: list[AgentMiddleware[Any, Any, Any]] = []
    for m in custom:
        if m.name in current_names:
            # 既存スタックと同じ .name を持つ場合は「インプレース置換」候補へ
            replacements[m.name] = m
        else:
            # 新規の Middleware は挿入候補へ
            to_append.append(m)

    result = list(base)
    for i, m in enumerate(result):
        if m.name in replacements:
            replacement = replacements[m.name]
            # FilesystemMiddleware を置換する際、エージェントレベルの permissions を誤って剥がさないためのガード
            if isinstance(m, FilesystemMiddleware) and m._permissions and isinstance(replacement, FilesystemMiddleware):
                if replacement._permissions and replacement._permissions != m._permissions:
                    msg = "FilesystemMiddleware permissions conflict with the agent permissions; use the same rules or omit middleware permissions."
                    raise ValueError(msg)
                # 置換側に permissions が未設定ならエージェントのルールを自動継承
                replacement = replacement._with_permissions(m._permissions)
            result[i] = replacement

    if to_append and core_names is not None:
        # 新規 Middleware は「最後の core_names の直後（= Tail Stack の手前）」にスプライス挿入する
        pos = max((i for i, m in enumerate(result) if m.name in core_names), default=len(result) - 1) + 1
        result[pos:pos] = to_append
    else:
        result.extend(to_append)
    return result
```

この設計のおかげで、例えばカスタムの `FilesystemMiddleware` や `SkillsMiddleware` を渡してもスタック内の相対位置（順序）が崩れず、同時にセキュリティ境界である `permissions` ルールの抜け漏れも防止されます。

### 4.2 必須スキャフォールディング保護と 2 パス除外（`_excluded_middleware.py`）

`HarnessProfile.excluded_middleware` を使うと、モデル特性に合わせて特定の Middleware をスタックから除外できます。しかし、コア機能を支える Middleware まで消してしまうとエージェントがサイレントに壊れます。

そこで [`graph.py` L266-L293](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py#L266-L293) と [`_excluded_middleware.py`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/_excluded_middleware.py) では以下の 3 段階の防御を敷いています。

1. **必須スキャフォールディングの保護（`_REQUIRED_MIDDLEWARE`）**：`FilesystemMiddleware` と `SubAgentMiddleware` は除外禁止として `_validate_excluded_middleware_config` で即座に `ValueError` を送出します。
2. **2 パスの `_apply_excluded_middleware`**：
   - 1 回目は `_apply_custom_middleware` の**前**に実行し、デフォルトスタックから不要なものを落とします。
   - 2 回目は `_apply_custom_middleware` の**後**に実行します。クラス指定の除外は `isinstance` ではなく **厳密な型一致（`type(mw)`）** で判定されるため、ユーザーがサブクラスで差し替えた場合は温存される設計です。
3. **カバレッジ検証（`_verify_excluded_middleware_coverage`）**：メインエージェントと General-Purpose サブエージェントの全スタックを処理し終えた後、**一度もマッチしなかった除外指定**があればタイポや古いプロファイルとみなして `ValueError` で明示的に失敗させます。

---

## まとめ

Part 1 では、[`create_deep_agent`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py) が単なるラッパーではなく、長期自律エージェントを破綻させないための精密な「組み立て工場」であることを見てきました。

- **`DeepAgentState`** は `DeltaChannel` によりチェックポイント肥大化を $O(N)$ に抑制。
- **Middleware スタック**は「ファイル/サブエージェント/要約 → ユーザー拡張 → プロファイル/スキル/キャッシュ/メモリ/HITL/安全ガード」という因果関係に基づいた順序で厳密に構成。
- **`_apply_custom_middleware`** と **`_excluded_middleware`** により、拡張性とセキュリティ不変条件（`permissions` や必須 Middleware）を両立。

次回 Part 2 では、DeepAgents の全ファイル操作とシェル実行を支えるストレージ抽象化層 **`BackendProtocol` と各種 Backend 実装**（`StateBackend`, `FilesystemBackend`, `StoreBackend`, `CompositeBackend`）の内部構造を詳しく解説します。
