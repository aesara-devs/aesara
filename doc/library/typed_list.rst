.. _libdoc_typed_list:

===============================
:mod:`typed_list` -- Typed List
===============================

.. note::

    This has been added in release 0.7.

.. note::

    This works, but is not well integrated with the rest of Aesara. If
    speed is important, it is probably better to pad to a dense
    tensor.

This is a type that represents a list in Aesara. All elements must have
the same Aesara type. Here is an example:

>>> import aesara.typed_list
>>> tl = aesara.typed_list.TypedListType(aesara.tensor.fvector)()
>>> v = aesara.tensor.fvector()
>>> o = aesara.typed_list.append(tl, v)
>>> f = aesara.function([tl, v], o)
>>> f([[1, 2, 3], [4, 5]], [2])
[array([ 1.,  2.,  3.], dtype=float32), array([ 4.,  5.], dtype=float32), array([ 2.], dtype=float32)]

A second example with Scan. Scan doesn't yet have direct support of
TypedList, so you can only use it as non_sequences (not in sequences or
as outputs):

>>> import aesara.typed_list
>>> a = aesara.typed_list.TypedListType(aesara.tensor.fvector)()
>>> l = aesara.typed_list.length(a)
>>> s, _ = aesara.scan(fn=lambda i, tl: tl[i].sum(),
...                    non_sequences=[a],
...                    sequences=[aesara.tensor.arange(l, dtype='int64')])
>>> f = aesara.function([a], s)
>>> f([[1, 2, 3], [4, 5]])
array([ 6.,  9.], dtype=float32)

.. automodule:: aesara.typed_list.basic
    :members:
