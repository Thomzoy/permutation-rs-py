"""Polars plugin for Feistel-based permutations.

Provides constant-space, constant-time random access permutations
over dense integer ranges, powered by Feistel Network ciphers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars_permutation.typing import IntoExprColumn

LIB = Path(__file__).parent


def feistel_permute(expr: IntoExprColumn, *, n: int, seed: int) -> pl.Expr:
    """Apply a Feistel permutation to each value in the column.

    Each input value ``i`` is mapped to ``perm.get(i)`` where ``perm``
    is a permutation over ``0..n`` constructed with the given ``seed``.

    Parameters
    ----------
    expr
        Column of non-negative integer indices (must be < ``n``).
    n
        Size of the permutation domain ``[0, n)``.
    seed
        Seed for the permutation.

    Returns
    -------
    pl.Expr
        A UInt64 column with the permuted indices.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_permutation as pp
    >>> df = pl.DataFrame({"idx": [0, 1, 2, 3, 4]})
    >>> df.with_columns(permuted=pp.feistel_permute("idx", n=5, seed=42))
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="feistel_permute",
        is_elementwise=True,
        kwargs={"n": n, "seed": seed},
    )


def feistel_shuffle(expr: IntoExprColumn, *, seed: int) -> pl.Expr:
    """Shuffle a column using a Feistel permutation.

    Creates a permutation over ``0..len(column)`` with the given ``seed``
    and reorders the column values accordingly. Each output position ``i``
    receives the value from input position ``perm.get(i)``.

    Parameters
    ----------
    expr
        Any column to shuffle.
    seed
        Seed for the permutation.

    Returns
    -------
    pl.Expr
        The shuffled column (same dtype as input).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_permutation as pp
    >>> df = pl.DataFrame({"values": [10, 20, 30, 40, 50]})
    >>> df.with_columns(shuffled=pp.feistel_shuffle("values", seed=42))
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="feistel_shuffle",
        is_elementwise=False,
        kwargs={"seed": seed},
    )
