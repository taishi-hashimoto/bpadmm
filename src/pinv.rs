extern crate blas_src;
extern crate ndarray;

use num_complex::Complex64;
use ndarray::*;
use ndarray_linalg::LeastSquaresSvd;


pub fn pinv(
    a: &ArrayView2<'_, Complex64>,
) -> Array2<Complex64>
{
    // Using lstsq, because implementation using svd is very slow.
    let a_h: ArrayBase<OwnedRepr<num_complex::Complex<f64>>, Dim<[usize; 2]>> = a.t().mapv(|x|x.conj());
    let result = a.dot(&a_h).t().least_squares(&a_h.t()).unwrap();
    result.solution.t().to_owned()
}
