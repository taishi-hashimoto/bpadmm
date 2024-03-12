extern crate blas_src;
extern crate ndarray;

use num_complex::Complex64;
use ndarray::*;


pub fn soft_threshold<D>(
    x: &ArrayView<'_, Complex64, D>,
    threshold: f64
) -> Array<Complex64, D>
where
    D: Dimension,
{
    let a = x.mapv(|e| Complex64::new(f64::max(e.norm() - threshold, 0.), 0.));
    x * &a / (&a + threshold)
}
