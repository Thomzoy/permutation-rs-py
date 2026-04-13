use feistel_permutation_rs::{DefaultBuildHasher, Permutation};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;
use serde::Deserialize;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[derive(Deserialize)]
struct PermuteKwargs {
    n: u64,
    seed: u64,
}

/// Apply the Feistel permutation to each value in the column.
///
/// Each input value `i` is mapped to `perm.get(i)` where `perm` is a
/// permutation over `0..n` constructed with the given `seed`.
///
/// Input values must be non-negative and less than `n`.
#[polars_expr(output_type=UInt64)]
fn feistel_permute(inputs: &[Series], kwargs: PermuteKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.cast(&DataType::UInt64)?;
    let ca = ca.u64()?;
    let perm = Permutation::new(kwargs.n, kwargs.seed, DefaultBuildHasher::new());
    let out: UInt64Chunked = ca.apply(|opt_v: Option<u64>| opt_v.map(|v: u64| perm.get(v)));
    Ok(out.into_series())
}

/// Shuffle a column using a Feistel permutation.
///
/// Creates a permutation over `0..len(column)` with the given `seed`,
/// then reorders the column values according to that permutation.
/// Each output position `i` receives the value from input position `perm.get(i)`.
#[derive(Deserialize)]
struct ShuffleKwargs {
    seed: u64,
}

#[polars_expr(output_type_func=same_output_type)]
fn feistel_shuffle(inputs: &[Series], kwargs: ShuffleKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let n = s.len() as u64;
    if n == 0 {
        return Ok(s.clone());
    }
    let perm = Permutation::new(n, kwargs.seed, DefaultBuildHasher::new());
    let indices: Vec<u32> = (0..n).map(|i| perm.get(i) as u32).collect();
    let idx = IdxCa::from_vec("".into(), indices);
    s.take(&idx)
}

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}
