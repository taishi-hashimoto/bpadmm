extern crate blas_src;
// #[macro_use]
extern crate ndarray;
// #[macro_use]
extern crate ndarray_linalg;

mod soft_threshold;
mod pinv;
mod basis_pursuit_admm;

use numpy::{Complex64, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use crate::soft_threshold::soft_threshold;
use crate::pinv::pinv;
use crate::basis_pursuit_admm::{basis_pursuit_admm, BPADMMInfo};

/// Basis Pursuit (BP) with Alternating Direction Method of Multipliers (ADMM)
#[pymodule]
#[pyo3(name="_impl")]
fn bpadmm<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {

    /// Soft thresholding function.
    ///
    /// Parameters
    /// ==========
    /// x: ndarray of complex
    ///     Input values.
    /// threshold: float
    ///     Soft threshold.
    ///     Values in x whose magnitudes are less than `threshold` are truncated to zero.
    #[pyfn(m)]
    fn _soft_threshold<'py>(
        py: Python<'py>,
        a: PyReadonlyArrayDyn<'py, Complex64>,
        threshold: f64
    ) -> PyResult<&'py PyArrayDyn<Complex64>> {
        Ok(soft_threshold(&a.as_array(), threshold).to_pyarray(py))
    }

    #[pyfn(m)]
    fn _pinv<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<&'py PyArray2<Complex64>> {
        Ok(pinv(&a.as_array()).to_pyarray(py))
    }

    #[pyfn(m)]
    fn _basis_pursuit_admm<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, Complex64>,
        y: PyReadonlyArray2<'py, Complex64>,
        threshold: f64,
        maxiter: i64,
        miniter: i64,
        tol: f64,
        njobs: usize,
        progress: bool,
    ) -> PyResult<(&'py PyArray2<Complex64>, BPADMMInfo)>{

        let (x, info) = basis_pursuit_admm(
            &a.as_array(),
            &y.as_array(),
            threshold,
            maxiter,
            miniter,
            tol,
            njobs,
            progress,
        );
        Ok((x.to_pyarray(py), info))
    }

    Ok(())
}