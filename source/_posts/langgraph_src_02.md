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
