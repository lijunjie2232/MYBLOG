---
title: 遅延バインディング (late binding)
date: 2024-4-9 11:15:00
categories: [Python]
tags: [Python]
lang: ja
description: Pythonの遅延バインディングについて解説します。
---

## 目次
- [目次](#%E7%9B%AE%E6%AC%A1)
- [問題のコード](#%E5%95%8F%E9%A1%8C%E3%81%AE%E3%82%B3%E3%83%BC%E3%83%89)
  - [コード例1](#%E3%82%B3%E3%83%BC%E3%83%89%E4%BE%8B1)
  - [コード例2](#%E3%82%B3%E3%83%BC%E3%83%89%E4%BE%8B2)
- [解決方法1：デフォルト引数を使う](#%E8%A7%A3%E6%B1%BA%E6%96%B9%E6%B3%951%E3%83%87%E3%83%95%E3%82%A9%E3%83%AB%E3%83%88%E5%BC%95%E6%95%B0%E3%82%92%E4%BD%BF%E3%81%86)
- [解決方法2：paritial を使う](#%E8%A7%A3%E6%B1%BA%E6%96%B9%E6%B3%952paritial-%E3%82%92%E4%BD%BF%E3%81%86)


---



## 問題のコード

### コード例1
まず、問題のコードを見てみましょう。

```python
>>> my_ld = [lambda x: x * i for i in range(3)]
>>> my_list = [ld(2) for ld in my_ld]
>>> my_list
[4, 4, 4]
>>>
```

### コード例2
```python
>>> def my_func(x, a=1):
...     return x * a
... 
>>> my_func(1,3)
3
>>> my_func(5,3)
15
>>> my_ld = [lambda x: my_func(x, i) for i in range(3)]
>>> my_list = [ld(2) for ld in my_ld]
>>> my_list
[4, 4, 4]
```

- `lambda x: x * i`：つまり、$x^i$という意味。

- `ld(2)`：つまり、$2 \times i$という意味、`i`は`[0,1,2]`です。

- 見込む結果は[0, 2, 4]ですが、実際の結果は[4, 4, 4]でした。

このコードでは、リスト内包表記の中で `lambda x: x * i` を定義していますが、各 `lambda` は変数 [i] の「現在の値」ではなく、「参照（アドレス）」を保持しています。

ループが終了した時点で [i] は `2` になっているため、すべての `lambda` 関数が `x * 2` を実行することになり、結果として `[4, 4, 4]` になります。

## 解決方法1：デフォルト引数を使う

Python の関数や `lambda` の**デフォルト引数は定義された時点で評価される**ため、これを使って「現在の値を固定」することができます。

```python
>>> my_ld = [lambda x, a=i: x * a for i in range(3)]
>>> my_list = [ld(2) for ld in my_ld]
>>> my_list
[0, 2, 4]
```

```python
>>> my_ld = [lambda x, a=i: my_func(x, a) for i in range(3)]
>>> my_list = [ld(2) for ld in my_ld]
>>> my_list
[0, 2, 4]
```

- `a=i` というデフォルト引数により、ループごとに `i` の値が固定されます。
- 各 `lambda` はそれぞれ `a=0`, `a=1`, `a=2` を持つようになるので、期待通りの結果を得られます。

## 解決方法2：paritial を使う

`functools.partial` は Python 標準ライブラリの functools モジュールに含まれる関数で、関数の一部の引数を固定して新しい関数を作成するための機能です。

```python
>>> my_ld = [partial(lambda x, i_val: x * i, i_val=i) for i in range(3)]
>>> my_list = [ld(2) for ld in my_ld]
>>> my_list
[0, 2, 4]
```

```python
>>> my_ld = [partial(my_func, a=i) for i in range(3)]
>>> my_list = [ld(2) for ld in my_ld]
>>> my_list
```
