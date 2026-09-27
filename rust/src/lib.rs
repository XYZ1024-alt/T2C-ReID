mod evaluation;
mod image_pipeline;
mod rerank;

use pyo3::prelude::*;

const NATIVE_ABI_VERSION: u32 = 2;

#[pyfunction]
fn native_version() -> (&'static str, u32) {
    (env!("CARGO_PKG_VERSION"), NATIVE_ABI_VERSION)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    module.add_function(wrap_pyfunction!(evaluation::evaluate_scores, module)?)?;
    module.add_function(wrap_pyfunction!(image_pipeline::load_image_batch, module)?)?;
    module.add_function(wrap_pyfunction!(rerank::select_topk_distances, module)?)?;
    module.add_function(wrap_pyfunction!(rerank::expanded_reciprocal_sets, module)?)?;
    module.add_class::<rerank::SparseRerankIndex>()?;
    module.add("NATIVE_ABI_VERSION", NATIVE_ABI_VERSION)?;
    Ok(())
}
