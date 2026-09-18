---
title: "LangChain v1 ソースコード解析 Part 2：Middleware 実行エンジン（middleware/types.py, middleware/_execution.py）のフック合成とラッピング構造"
date: 2026-08-15 02:20:37
categories: [AI]
tags: [Python, LLM, AI Agent, LangChain, Middleware, ソースコード解析]
lang: ja
description: "LangChain v1 の Middleware 実行エンジン（middleware/types.py, factory.py, middleware/_execution.py）を精読し、6大フック、デコレータによる動的サブクラス生成、オニオン型ラッパー合成、およびシェル実行ポリシーの内部構造を解説する。"
---

Part 1 では `create_agent()` がミドルウェアをグラフノードとエッジに配線する全体像を確認しました。Part 2 では、Middleware 実行エンジンの型定義・デコレータ実装を担う [`agents/middleware/types.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py)、ラッパー合成を担う [`agents/factory.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py) の合成関数群、そして永続シェルミドルウェアの隔離実行エンジンを担う [`agents/middleware/_execution.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/_execution.py) を掘り下げます。

---

## 1. `AgentMiddleware` の 6 大フックとデータモデル

[`agents/middleware/types.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py#L387-L826) に定義された `AgentMiddleware[StateT, ContextT, ResponseT]` は、エージェントのライフサイクルを **ノード型フック（4 種）** と **ラッパー型フック（2 種）** の 2 カテゴリ・計 6 フック（同期 / 非同期ペアで 12 メソッド）に分類しています。

| カテゴリ | フック（同期 / 非同期） | 実行単位 | 制御できること |
| :--- | :--- | :--- | :--- |
| ノード型 | `before_agent` / `abefore_agent` | 実行開始時に 1 回 | 初期状態更新、`jump_to` |
| ノード型 | `before_model` / `abefore_model` | 各モデル呼び出し前 | メッセージ圧縮・検査、`jump_to` |
| ノード型 | `after_model` / `aafter_model` | 各モデル呼び出し後 | 出力検証・HITL 中断、`jump_to` |
| ノード型 | `after_agent` / `aafter_agent` | 実行終了時に 1 回 | 最終出力の事後処理・クリーンアップ |
| ラッパー型 | `wrap_model_call` / `awrap_model_call` | モデル呼び出しを包む | リクエスト改変、リトライ、フォールバック、キャッシュ短絡 |
| ラッパー型 | `wrap_tool_call` / `awrap_tool_call` | 各ツール実行を包む | 引数改変、動的ツール実行、リトライ、エラー変換 |

### 1.1 `ModelRequest.override()` のイミュータブル設計

`wrap_model_call` に渡される `ModelRequest`（[`types.py#L89-L272`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py#L89-L272)）は、複数ミドルウェア間でリクエストを安全に受け渡すため **イミュータブルな更新（`override()`）** を強制しています。

```python
@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel
    messages: list[AnyMessage]  # system_message を除いたメッセージ履歴
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        # 直接代入は DeprecationWarning を出し、override() の使用を促す
        if name == "system_prompt":
            warnings.warn("Direct attribute assignment to ModelRequest.system_prompt is deprecated...", DeprecationWarning, stacklevel=2)
            object.__setattr__(self, "system_message", None if value is None else SystemMessage(content=value))
            return
        warnings.warn(f"Direct attribute assignment to ModelRequest.{name} is deprecated...", DeprecationWarning, stacklevel=2)
        object.__setattr__(self, name, value)

    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
        # dataclasses.replace() で元のインスタンスを破壊せずに新しい ModelRequest を返す
        if "system_prompt" in overrides and "system_message" in overrides:
            raise ValueError("Cannot specify both system_prompt and system_message")
        if "system_prompt" in overrides:
            system_prompt = cast("str | None", overrides.pop("system_prompt"))
            overrides["system_message"] = None if system_prompt is None else SystemMessage(content=system_prompt)
        return replace(self, **overrides)
```

なぜ `override()` が重要なのかというと、外側のミドルウェアがリトライやフォールバックで `handler(request)` を複数回呼び出す際、内側のミドルウェアが `request` を破壊的に書き換えてしまうと、2 回目のリトライ時に意図しない変更が残ってしまうためです。

### 1.2 `ExtendedModelResponse` と `AgentState` のメタデータ

`wrap_model_call` の戻り値型 `ModelCallResult`（[`types.py#L317`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py#L317)）は、`ModelResponse` / `AIMessage` / `ExtendedModelResponse` の 3 種類を受け入れます。特に `ExtendedModelResponse(model_response=..., command=Command(update=...))` を使うと、モデル呼び出しラッパー内から LangGraph のステート更新コマンドを追加発行できます。

また、`AgentState`（[`types.py#L351-L356`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py#L351-L356)）では `jump_to` フィールドに LangGraph の `EphemeralValue` チャネルと `PrivateStateAttr` を併用しています。

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    # EphemeralValue により 1 ステップ終了後に自動クリアされ、PrivateStateAttr で入出力スキーマから除外
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

`EphemeralValue` を使うことで、あるノードが `{"jump_to": "end"}` を返した直後のエッジ評価でのみ有効になり、次のステップに値が残留して無限ジャンプを引き起こす事故を構造的に防いでいます。

---

## 2. デコレータによる動的ミドルウェアクラス生成

[`types.py#L881-L2203`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py#L881-L2203) には、クラスを明示的に継承せず関数だけでミドルウェアを作れる 7 つのデコレータが用意されています（`@hook_config`, `@before_model`, `@after_model`, `@before_agent`, `@after_agent`, `@dynamic_prompt`, `@wrap_model_call`, `@wrap_tool_call`）。

これらの実装は非常に一貫しており、Python の組み込み `type(name, bases, dict)` を使って **実行時に `AgentMiddleware` のサブクラスを動的生成して即座にインスタンス化** します。

```python
def decorator(func: _CallableWithStateAndRuntime[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]:
    # 関数が async def かどうかを検査し、abefore_model と before_model のどちらを定義するか分岐
    is_async = iscoroutinefunction(func)
    # デコレータ引数 or @hook_config(can_jump_to=...) のメタデータを引き継ぐ
    func_can_jump_to = (
        can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
    )

    if is_async:
        async def async_wrapped(_self, state, runtime):
            return await func(state, runtime)
        if func_can_jump_to:
            async_wrapped.__can_jump_to__ = func_can_jump_to
        middleware_name = name or getattr(func, "__name__", "BeforeModelMiddleware")
        # 動的サブクラス生成 & インスタンス化
        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "abefore_model": async_wrapped,
            },
        )()
```

特に巧妙なのが `@dynamic_prompt`（[`types.py#L1706-L1845`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py#L1706-L1845)）です。同期関数をデコレートした場合、`wrap_model_call` だけでなく **`awrap_model_call`（`async_wrapped_from_sync`）も同時に生成** します。これにより、同期の `@dynamic_prompt` を定義したエージェントを `ainvoke()` / `astream()` で非同期実行しても `NotImplementedError` にならず透過的に動作します。

---

## 3. ラッパー合成エンジン：`_chain_model_call_handlers` のオニオン構造

複数のミドルウェアが `wrap_model_call` を実装している場合、`factory.py` の `_chain_model_call_handlers()`（[`factory.py#L284-L373`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L284-L373)）がそれらを 1 つの合成関数へと畳み込みます。

{% mermaid %}
sequenceDiagram
    participant MN as "model_node"
    participant M1 as "Outer: M1.wrap_model_call"
    participant M2 as "Inner: M2.wrap_model_call"
    participant EX as "_execute_model_sync"

    MN->>M1: "composed(request, _execute_model_sync)"
    Note over M1: "accumulated_commands = []"
    M1->>M2: "inner_handler(req1)"
    Note over M2: "accumulated_commands.clear() (リトライ安全)"
    M2->>EX: "handler(req2)"
    EX-->>M2: "ModelResponse"
    M2-->>M1: "ExtendedModelResponse(resp, cmd_M2)<br/>※ cmd_M2 はクロージャに退避し ModelResponse のみ返却"
    M1-->>MN: "_ComposedExtendedModelResponse(resp, [cmd_M2, cmd_M1])"
{% endmermaid %}

この実装には **2 つの重要な設計上の工夫** が隠れています。

```python
def compose_two(outer, inner):
    def composed(request, handler) -> _ComposedExtendedModelResponse:
        # 内側ミドルウェアが返した Command を退避するクロージャ変数
        accumulated_commands: list[Command[Any]] = []

        def inner_handler(req: ModelRequest[ContextT]) -> ModelResponse:
            # ★ 工夫1: 外側ミドルウェア（例: ModelRetryMiddleware）が inner_handler を
            #    複数回リトライ呼び出しした場合、失敗した試行の Command が重複蓄積しないよう毎回クリアする
            accumulated_commands.clear()
            inner_result = inner(req, handler)
            if isinstance(inner_result, _ComposedExtendedModelResponse):
                accumulated_commands.extend(inner_result.commands)
                return inner_result.model_response
            if isinstance(inner_result, ExtendedModelResponse):
                if inner_result.command is not None:
                    accumulated_commands.append(inner_result.command)
                return inner_result.model_response
            return _normalize_to_model_response(inner_result)

        # ★ 工夫2: outer には常に正規化された ModelResponse だけを返し、
        #    戻り値構築時に inner-first -> outer の順序で Command を連結する
        outer_result = outer(request, inner_handler)
        return _to_composed_result(
            outer_result,
            extra_commands=accumulated_commands or None,
        )

    return composed

# 右から左へ畳み込むことで handlers[0] が最外層になる
composed_handler = compose_two(handlers[-2], handlers[-1])
for h in reversed(handlers[:-2]):
    composed_handler = compose_two(h, composed_handler)
```

1. **リトライ安全性の保証**：`inner_handler` の先頭で `accumulated_commands.clear()` を呼ぶことで、外側のリトライミドルウェアが `handler(request)` を再実行した際に、前回の失敗試行で生成された `Command` が混入するバグを防いでいます。
2. **インターフェースの単純化と順序保証**：内側のミドルウェアが `ExtendedModelResponse` を返しても、`inner_handler` が `Command` をクロージャに吸収して `ModelResponse` だけを `outer` に渡します。結果として、`outer` は `ExtendedModelResponse` のアンラップを意識せずに済み、最終的に `commands` は `[inner_cmd, ..., outer_cmd]` の順（非リデューサフィールドでは外側が勝つ順）で LangGraph に適用されます。

なお、`wrap_tool_call` についても同様の右畳み込みを行う `_chain_tool_call_wrappers` / `_chain_async_tool_call_wrappers`（[`factory.py#L713-L822`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L713-L822)）が `ToolNode` に渡され、LangSmith トレーシング時には `_scrub_inputs()`（[`factory.py#L157-L166`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py#L157-L166)）でシリアライズ不能な `runtime` と `handler` が自動除去されます。

---

## 4. `middleware/_execution.py`：シェル実行ポリシーの隔離アーキテクチャ

もう 1 つ、`middleware/` 配下で「実行エンジン」を担う重要なモジュールが [`agents/middleware/_execution.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/_execution.py) です。これは `ShellToolMiddleware` が永続シェルプロセスを起動・隔離するための **Execution Policy 層** を抽象化しています。

{% mermaid %}
classDiagram
    class BaseExecutionPolicy {
        <<abstract>>
        +float command_timeout = 30.0
        +float startup_timeout = 30.0
        +float termination_timeout = 10.0
        +int max_output_lines = 100
        +int|None max_output_bytes
        +spawn(workspace, env, command)* Popen
    }
    class HostExecutionPolicy {
        +int|None cpu_time_seconds
        +int|None memory_bytes
        +bool create_process_group = True
        +spawn() Popen
        -_create_preexec_fn()
        -_apply_post_spawn_limits()
    }
    class CodexSandboxExecutionPolicy {
        +str binary = "codex"
        +Literal platform = "auto"
        +Mapping config_overrides
        +spawn() Popen
        -_build_command()
    }
    class DockerExecutionPolicy {
        +str image = "python:3.12-alpine3.19"
        +bool network_enabled = False
        +bool read_only_rootfs = False
        +str|None cpus
        +spawn() Popen
        -_build_command()
    }

    BaseExecutionPolicy <|-- HostExecutionPolicy
    BaseExecutionPolicy <|-- CodexSandboxExecutionPolicy
    BaseExecutionPolicy <|-- DockerExecutionPolicy
{% endmermaid %}

### 4.1 OS 差異を吸収するリソース制限（`HostExecutionPolicy`）

`HostExecutionPolicy`（[`_execution.py#L91-L188`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/_execution.py#L91-L188)）は、プラットフォームごとの `resource` モジュールの挙動差を吸収しています。

```python
def spawn(self, *, workspace: Path, env: Mapping[str, str], command: Sequence[str]) -> subprocess.Popen[str]:
    process = _launch_subprocess(
        list(command),
        env=env,
        cwd=workspace,
        # Linux で prlimit が使える場合はフォーク後の安全性を優先し preexec_fn を使わない
        # macOS など prlimit 非対応環境では exec 前の preexec_fn で setrlimit を適用
        preexec_fn=self._create_preexec_fn(),
        # タイムアウト時に子プロセスツリーごと終了できるよう独立プロセスグループを作成
        start_new_session=self.create_process_group,
    )
    # Linux 環境では起動後に PID 指定で resource.prlimit(RLIMIT_CPU / RLIMIT_AS) を適用
    self._apply_post_spawn_limits(process)
    return process
```

マルチスレッドな Python プロセスでの `preexec_fn` はデッドロックのリスクがあるため、Linux では `prlimit` を優先しつつ、macOS では `preexec_fn` にフォールバックする極めて実践的な実装になっています。

### 4.2 コンテナ・サンドボックス隔離（`DockerExecutionPolicy` / `CodexSandboxExecutionPolicy`）

- **`DockerExecutionPolicy`**（[`_execution.py#L266-L378`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/_execution.py#L266-L378)）：デフォルトで `--network none`（ネットワーク遮断）と `--rm` を付与し、`workspace.name.startswith("langchain-shell-")` な一時ディレクトリの場合はホストマウントを行わず `-w /` で起動することでホスト露出を最小化します。
- **`CodexSandboxExecutionPolicy`**（[`_execution.py#L190-L264`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/_execution.py#L190-L264)）：macOS の Seatbelt や Linux の Landlock/seccomp を使う `codex sandbox <platform> -c key=val -- <command>` を自動構築してサブプロセス起動します。

---

## まとめ

LangChain v1 の Middleware 実行エンジンは、以下の 3 つの柱で高い拡張性と安全性を両立しています。

1. **型とデコレータの軽量さ**：`ModelRequest.override()` による不変性、`EphemeralValue` による `jump_to` の自動リセット、`type()` による関数ベースの動的ミドルウェア生成。
2. **オニオン型ラッパー合成の堅牢さ**：`_chain_model_call_handlers` でのクロージャ退避と `accumulated_commands.clear()` により、リトライや状態コマンド発行が複数層で競合しない。
3. **実行ポリシーの分離**：`_execution.py` により、ホスト直実行から Seatbelt/Landlock サンドボックス、Docker コンテナまで同一インターフェースで切り替え可能。

次回 Part 3 では、[`agents/structured_output.py`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py) を中心に、`ToolStrategy` と `ProviderStrategy`、そして自動判定 `AutoStrategy` の内部機構を徹底解析します。
