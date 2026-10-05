---
title: "LangGraph ソースコード解析 Part 2：Channels 抽象と状態更新（LastValue・BinaryOperatorAggregate・Topic・DeltaChannel・add_messages）"
date: 2026-09-12 13:30:30
categories: [AI]
tags: [Python, LLM, AI Agent, LangGraph, ソースコード解析, Channels]
lang: ja
description: "LangGraph の状態管理とチェックポイント永続化を支える BaseChannel インターフェースと、LastValue / BinaryOperatorAggregate / Topic / DeltaChannel / add_messages の内部実装をソースコードから詳解します。"
---

Part 1 では、`StateGraph` が型ヒントやエッジをすべて **Channel** ベースの配線へとコンパイルすることを確認しました。LangGraph において「State」とは単なる辞書ではなく、**各キーごとに独立した更新セマンティクスと永続化戦略を持つ `BaseChannel` インスタンスの集合**です。

Part 2 では、以下の 6 つのコアファイルを読み込み、チャネル抽象のライフサイクルと代表的な実装、さらにチャット履歴管理の要である `add_messages` の仕組みを解き明かします。

- [`channels/base.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/base.py)：`BaseChannel` 抽象基底クラス
- [`channels/last_value.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/last_value.py)：`LastValue` と遅延発火用 `LastValueAfterFinish`
- [`channels/binop.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/binop.py)：`BinaryOperatorAggregate` と `Overwrite` 制御
- [`channels/topic.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/topic.py)：PubSub 型チャネル `Topic`
- [`channels/delta.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/delta.py)：リプレイ型最適化チャネル `DeltaChannel`
- [`graph/message.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py)：ID ベースのマージと削除を実現する `add_messages`

---

## 1. `BaseChannel` のライフサイクルと契約

[`channels/base.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/base.py) で定義されている `BaseChannel[Value, Update, Checkpoint]` は、Pregel の BSP（Bulk Synchronous Parallel）実行ループとチャネル間の唯一のインターフェースです。

```python
class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    """すべてのチャネルの基底クラス。__slots__ でメモリフットプリントを最小化している。"""
    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ
        self.key = key

    # --- 読み取り系 ---
    @abstractmethod
    def get(self) -> Value:
        """現在値を返す。未更新（空）の場合は EmptyChannelError を送出する。"""

    def is_available(self) -> bool:
        """チャネルが利用可能（空でない）かを返す。サブクラスは高速判定でオーバーライド推奨。"""

    # --- 書き込み・ライフサイクル系 ---
    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool:
        """
        各スーパーステップ末尾に Pregel から全チャネルに対して呼ばれる。
        values は同一ステップ内で複数ノードが書き込んだ値のシーケンス（順序は不定）。
        更新がなかったステップでも空シーケンス [] で呼ばれる点に注意。
        状態が変化した場合に True を返す。
        """

    def consume(self) -> bool:
        """このチャネルを購読しているタスクが実行された後に呼ばれる（既定は no-op）。"""
        return False

    def finish(self) -> bool:
        """実行すべきタスクが無くなりグラフ終了直前に呼ばれる（既定は no-op）。
        True を返すと終了を阻止し、遅延タスク（defer=True 等）を起動できる。"""
        return False

    # --- シリアライズ系 ---
    def checkpoint(self) -> Checkpoint | Any:
        try:
            return self.get()
        except EmptyChannelError:
            return MISSING

    @abstractmethod
    def from_checkpoint(self, checkpoint: Checkpoint | Any) -> Self: ...
```

注目すべきは、`update(values)` が **書き込みが無かったステップでも `[]` で呼ばれる** こと、そして `consume()` と `finish()` という 2 つのライフサイクルフックが存在することです。これにより、チャネル自身の実装だけで「1 ステップ限りの揮発性（`EphemeralValue`）」「消費されたら消える」「全タスク完了後まで待機する（`defer=True`）」といった振る舞いを表現できます。

{% mermaid %}
classDiagram
    class BaseChannel {
        <<abstract>>
        +key: str
        +typ: Any
        +get() Value
        +update(values) bool
        +consume() bool
        +finish() bool
        +checkpoint() Checkpoint
        +from_checkpoint(cp) Self
    }
    class LastValue {
        +value: Value
        +update() "1ステップ1値のみ許可"
    }
    class LastValueAfterFinish {
        +value: Value
        +finished: bool
        +finish() "finished=Trueで解放"
        +consume() "消費後にクリア"
    }
    class BinaryOperatorAggregate {
        +value: Value
        +operator: Callable
        +update() "Reducer畳み込み + Overwrite"
    }
    class Topic {
        +values: list
        +accumulate: bool
        +update() "平坦化して蓄積/毎ステップ初期化"
    }
    class DeltaChannel {
        +value: Value
        +reducer: Callable
        +snapshot_frequency: int
        +checkpoint() "常にMISSINGを返す"
        +replay_writes() "履歴リプレイ復元"
    }

    BaseChannel <|-- LastValue
    BaseChannel <|-- LastValueAfterFinish
    BaseChannel <|-- BinaryOperatorAggregate
    BaseChannel <|-- Topic
    BaseChannel <|-- DeltaChannel
{% endmermaid %}

---

## 2. `LastValue` と `LastValueAfterFinish`

[`channels/last_value.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/last_value.py) にある `LastValue` は、`Annotated` を付けない通常の State フィールドに割り当てられるデフォルトチャネルです。

```python
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value",)

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False
        # 同一スーパーステップ内で 2 つ以上のノードが同時に書き込むと衝突エラー
        if len(values) != 1:
            msg = create_error_message(
                message=f"At key '{self.key}': Can receive only one value per step. "
                        "Use an Annotated key to handle multiple values.",
                error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
            )
            raise InvalidUpdateError(msg)

        self.value = values[-1]
        return True
```

並列ブランチが同じ `LastValue` キーを更新すると `INVALID_CONCURRENT_GRAPH_UPDATE` が即座に送出されるのは、まさにこの `len(values) != 1` チェックによるものです。

一方、`LastValueAfterFinish` は Part 1 で見た `add_node(..., defer=True)` のトリガーチャネルとして使われます。

```python
class LastValueAfterFinish(Generic[Value], BaseChannel[Value, Value, tuple[Value, bool]]):
    __slots__ = ("value", "finished")

    def update(self, values: Sequence[Value | Any]) -> bool:
        if len(values) == 0:
            return False
        # 値を受け取っても finished=False の間は get() / is_available() が空扱いになる
        self.finished = False
        self.value = values[-1]
        return True

    def finish(self) -> bool:
        # 他に実行すべきタスクが尽きたタイミングで Pregel が finish() を呼ぶ
        # ここで finished=True にすることで、遅延ノードが次のステップで起動する
        if not self.finished and self.value is not MISSING:
            self.finished = True
            return True
        return False

    def consume(self) -> bool:
        # 遅延ノードが 1 回実行されたらクリアし、無限ループを防ぐ
        if self.finished:
            self.finished = False
            self.value = MISSING
            return True
        return False
```

---

## 3. `BinaryOperatorAggregate` と `Overwrite` バイパス機構

`Annotated[list[int], operator.add]` のように 2 引数 Reducer を指定したフィールドは、[`channels/binop.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/binop.py) の `BinaryOperatorAggregate` に変換されます。

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(typ)
        self.operator = operator
        # typing / collections.abc の抽象型（Sequence, Mapping 等）を具象型へ変換して初期値を生成
        typ = _strip_extras(typ)
        if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
            typ = list
        if typ in (collections.abc.Set, collections.abc.MutableSet):
            typ = set
        if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
            typ = dict
        try:
            self.value = typ()   # 例: list() -> []、int() -> 0
        except Exception:
            self.value = MISSING

    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False
        if self.value is MISSING:
            self.value = values[0]
            values = values[1:]
        seen_overwrite: bool = False
        for value in values:
            # Overwrite ラッパーまたは JSON 化された {"__overwrite__": ...} を検出
            is_overwrite, overwrite_value = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    # 同一ステップで複数の Overwrite は非決定的なため禁止
                    raise InvalidUpdateError(...)
                self.value = overwrite_value
                seen_overwrite = True
                continue
            if not seen_overwrite:
                # 通常の Reducer 畳み込み: acc = op(acc, val)
                self.value = self.operator(self.value, value)
        return True
```

### `Overwrite` の 3 形式認識（`_get_overwrite`）
`BinaryOperatorAggregate` は Reducer を経由せずに値を丸ごとリセットするための `Overwrite(value)` をサポートしています。`_get_overwrite()` を見ると、Python オブジェクトの `Overwrite` インスタンスだけでなく、API サーバー経由の JSON シリアライズで型情報が落ちた `{"__overwrite__": value}` および `{"type": "__overwrite__", "value": ...}` の 3 形式すべてを認識する堅牢な設計になっています。

---

## 4. `Topic`：PubSub 型メッセージチャネル

[`channels/topic.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/topic.py) の `Topic` は、単一値またはリストの書き込みを `_flatten()` で平坦化して蓄積する PubSub チャネルです。

```python
class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    __slots__ = ("values", "accumulate")

    def update(self, values: Sequence[Value | list[Value]]) -> bool:
        updated = False
        # accumulate=False の場合、毎スーパーステップ開始時に前回の値を破棄する
        if not self.accumulate:
            updated = bool(self.values)
            self.values = list[Value]()
        if flat_values := tuple(_flatten(values)):
            updated = True
            self.values.extend(flat_values)
        return updated
```

`accumulate=False`（デフォルト）では「前ステップで何らかのイベントが発行された場合のみ次ステップの購読ノードを起動し、書き込みが無いステップで空に戻す」挙動となり、内部的には `TASKS`（動的 `Send` パケット用チャネル）などのイベントキューとして活用されています。

---

## 5. `DeltaChannel`：巨大状態のチェックポイント肥大化を防ぐリプレイ型チャネル

長時間の会話や大量のドキュメントを保持するエージェントでは、毎ステップごとに State 全体をスナップショット保存する `BinaryOperatorAggregate` だと、ストレージ容量がステップ数の二乗 $O(N^2)$ で膨らみます。これを解決するために導入されたのが [`channels/delta.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/channels/delta.py) の **`DeltaChannel`** です。

```python
class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any]):
    __slots__ = ("value", "reducer", "snapshot_frequency")

    def checkpoint(self) -> Any:
        # 重要: 通常の checkpoint() では常に MISSING を返し、blob 保存をスキップする！
        # snapshot_frequency 回目（または最大スーパーステップ到達時）のみ
        # create_checkpoint() 側が _DeltaSnapshot(ch.get()) を明示的に保存する。
        return MISSING

    def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
        """
        チェックポイント復元時に、直近スナップショット以降の祖先 writes を
        古い順に一括で reducer に畳み込んで現在値を再構築する。
        """
        values = [v for _, _, v in writes]
        if not values:
            return
        base = self.value
        start = 0
        # 途中に Overwrite が含まれていれば、最後の Overwrite を新しい基点にする
        for i, v in enumerate(values):
            is_ow, ow_value = _get_overwrite(v)
            if is_ow:
                base = _copy.copy(ow_value) if ow_value is not None else self.typ()
                start = i + 1
        remaining = values[start:]
        # バッチ不変性: reducer(reducer(s, xs), ys) == reducer(s, xs + ys) を前提に一括適用
        self.value = self.reducer(base, remaining) if remaining else base
```

{% mermaid %}
flowchart LR
    subgraph Save["チェックポイント保存時 (create_checkpoint)"]
        U1["通常ステップ (k % freq != 0)"] --> M1["checkpoint() -> MISSING<br/>(差分 writes のみ記録)"]
        U2["スナップショット到達時<br/>(k >= snapshot_frequency)"] --> M2["_DeltaSnapshot(value)<br/>を blob に保存"]
    end

    subgraph Restore["チェックポイント復元時"]
        S0["直近の _DeltaSnapshot をロード"] --> H["get_delta_channel_history()<br/>で以降の差分 writes を取得"]
        H --> R["DeltaChannel.replay_writes()<br/>最後の Overwrite 以降を一括 fold"]
        R --> V["現在 State を完全に再構築"]
    end

    M2 -.-> S0
    M1 -.-> H

    style Save fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style Restore fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
{% endmermaid %}

`DeltaChannel` の Reducer は `(state, list[writes]) -> new_state` というバッチ形式を取り、**結合則（batching-invariance）** `reducer(reducer(state, xs), ys) == reducer(state, xs + ys)` を満たす必要があります。これにより、復元時に数十ステップ分の差分書き込みを 1 回の `reducer()` 呼び出しで高速にリプレイできる設計になっています。

---

## 6. `add_messages`：ID ベースのメッセージマージと削除ロジック

LangGraph のチャットエージェントで最も多用される Reducer が [`graph/message.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py) の `add_messages` です。単純な `list.__add__` とは異なり、以下の 4 つの機能を実装しています。

```python
@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    remove_all_idx = None
    # 単一メッセージが渡された場合もリストへ正規化
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    # BaseMessage への変換（高速パス: 全要素が既に BaseMessage で Chunk でなければスキップ）
    left = _coerce_to_messages(left)
    right = _coerce_to_messages(right)

    # ID 未設定のメッセージに UUID4 を自動付与 + REMOVE_ALL_MESSAGES の検出
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    # REMOVE_ALL_MESSAGES ("__remove_all__") が含まれていれば、それ以前を全消去
    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # O(N) で ID ベースのマージ・上書き・個別削除を実行
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)       # 既存メッセージの削除マーク
            else:
                ids_to_remove.discard(m.id)   # 同 ID で再挿入された場合は削除を取り消し
                merged[existing_idx] = m      # 同一 ID のメッセージをインプレース置換
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )
            merged_by_id[m.id] = len(merged)
            merged.append(m)

    if ids_to_remove:
        merged = [m for m in merged if m.id not in ids_to_remove]

    if format == "langchain-openai":
        merged = _format_messages(merged)
    return merged
```

### 実装の細かな工夫
1. **カリー化デコレータ（`_add_messages_wrapper`）**：`Annotated[list, add_messages]` のように関数そのままでも、`Annotated[list, add_messages(format="langchain-openai")]` のようにオプション付きの部分適用（`functools.partial`）でも使える二通りの呼び出しを 1 つのラッパーで実現しています。
2. **型チェックによる高速パス（`_coerce_to_messages`）**：長い会話履歴で毎ステップ全要素を `convert_to_messages()` に通すコストを避けるため、`set(map(type, messages))` でユニークな型集合だけを確認し、すべて通常の `BaseMessage`（非 `BaseMessageChunk`）であればそのまま返却します。
3. **削除の遅延フィルタリング**：ループ内で `list.pop()` するとインデックスがずれて $O(N^2)$ になるため、`ids_to_remove` セットに記録しておき、最後に 1 回のリスト内包表記でフィルタリングしています。

---

## 7. まとめ

Part 2 では、LangGraph の状態モデルの中核である Channels 層を解析しました。

- **`BaseChannel`** の `update` / `consume` / `finish` の 3 フックにより、揮発性・遅延実行・バリア同期といった多様な制御パターンがチャネル側へ綺麗にカプセル化されている
- **`LastValue`** は並列書き込みの衝突を即座に検出し、**`BinaryOperatorAggregate`** は Reducer 畳み込みと `Overwrite` リセットを両立する
- **`DeltaChannel`** はスナップショット＋差分リプレイにより、長寿命スレッドのチェックポイント肥大化を根本から抑える
- **`add_messages`** は UUID 自動採番・ID ベース置換・`RemoveMessage` / `REMOVE_ALL_MESSAGES` による履歴編集を $O(N)$ で実現する

次回の Part 3 では、これらのチャネルを実際に駆動する **Pregel BSP 実行ループ（`pregel/main.py`, `pregel/_loop.py`）** の Plan → Execution → Update サイクルを徹底解析します。
