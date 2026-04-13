import polars as pl
import polars_permutation as pp
from polars.testing import assert_frame_equal


def test_feistel_permute_basic():
    """Each index maps to a unique value in [0, n)."""
    n = 10
    seed = 42
    df = pl.DataFrame({"idx": list(range(n))}).cast({"idx": pl.UInt64})
    result = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=seed))

    # The permuted column should contain a permutation of 0..n
    permuted = result["permuted"].sort().to_list()
    assert permuted == list(range(n))


def test_feistel_permute_deterministic():
    """Same seed and n always produce the same permutation."""
    n = 100
    seed = 123
    df = pl.DataFrame({"idx": list(range(n))}).cast({"idx": pl.UInt64})
    r1 = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=seed))
    r2 = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=seed))
    assert_frame_equal(r1, r2)


def test_feistel_permute_different_seeds():
    """Different seeds produce different permutations."""
    n = 100
    df = pl.DataFrame({"idx": list(range(n))}).cast({"idx": pl.UInt64})
    r1 = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=1))
    r2 = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=2))
    # Very unlikely to be equal with different seeds
    assert r1["permuted"].to_list() != r2["permuted"].to_list()


def test_feistel_permute_handles_nulls():
    """Null values in the input column remain null in the output."""
    n = 10
    seed = 42
    df = pl.DataFrame({"idx": [0, None, 2, None, 4]}).cast({"idx": pl.UInt64})
    result = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=seed))

    # Null positions should be preserved
    assert result["permuted"].is_null().to_list() == [False, True, False, True, False]

    # Non-null values should be valid permuted indices
    non_null = result["permuted"].drop_nulls().to_list()
    assert all(0 <= v < n for v in non_null)


def test_feistel_shuffle_basic():
    """Shuffled column contains the same values as the original."""
    seed = 42
    df = pl.DataFrame({"values": [10, 20, 30, 40, 50]})
    result = df.with_columns(shuffled=pp.feistel_shuffle("values", seed=seed))

    # Same multiset of values
    assert sorted(result["shuffled"].to_list()) == sorted(df["values"].to_list())


def test_feistel_shuffle_deterministic():
    """Same seed always produces the same shuffle."""
    seed = 99
    df = pl.DataFrame({"values": list(range(50))})
    r1 = df.with_columns(shuffled=pp.feistel_shuffle("values", seed=seed))
    r2 = df.with_columns(shuffled=pp.feistel_shuffle("values", seed=seed))
    assert_frame_equal(r1, r2)


def test_feistel_shuffle_different_seeds():
    """Different seeds produce different shuffles."""
    df = pl.DataFrame({"values": list(range(50))})
    r1 = df.with_columns(shuffled=pp.feistel_shuffle("values", seed=1))
    r2 = df.with_columns(shuffled=pp.feistel_shuffle("values", seed=2))
    assert r1["shuffled"].to_list() != r2["shuffled"].to_list()


def test_feistel_shuffle_preserves_dtype():
    """Shuffle preserves the column data type."""
    seed = 42
    for dtype, values in [
        (pl.Int64, [1, 2, 3, 4, 5]),
        (pl.Float64, [1.1, 2.2, 3.3, 4.4, 5.5]),
        (pl.String, ["a", "b", "c", "d", "e"]),
    ]:
        df = pl.DataFrame({"col": values})
        result = df.with_columns(shuffled=pp.feistel_shuffle("col", seed=seed))
        assert result["shuffled"].dtype == dtype


def test_feistel_shuffle_empty():
    """Shuffling an empty column returns an empty column."""
    df = pl.DataFrame({"values": []}).cast({"values": pl.Int64})
    result = df.with_columns(shuffled=pp.feistel_shuffle("values", seed=42))
    assert result["shuffled"].len() == 0


def test_feistel_permute_large():
    """Verify permutation property on a larger dataset."""
    n = 10000
    seed = 7
    df = pl.DataFrame({"idx": list(range(n))}).cast({"idx": pl.UInt64})
    result = df.with_columns(permuted=pp.feistel_permute("idx", n=n, seed=seed))

    permuted = result["permuted"].sort().to_list()
    assert permuted == list(range(n))
