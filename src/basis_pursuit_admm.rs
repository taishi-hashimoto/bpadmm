extern crate blas_src;
extern crate ndarray;
extern crate dict_derive;

use rayon;
use crossbeam_channel;
use ndarray_linalg::Norm;
use num_complex::Complex64;
use ndarray::*;
use dict_derive::{FromPyObject, IntoPyObject};
use indicatif::{ProgressBar, ProgressStyle};
use crate::soft_threshold::soft_threshold;
use crate::pinv::pinv;

use std::time::Instant;


#[derive(FromPyObject, IntoPyObject)]
pub struct BPADMMInfo {
    success: Vec<bool>,
    niters: Vec<i64>,
    cond: Vec<f64>,
    elapsed: Vec<f64>
}

struct EachInfo {
    // Snapshot number.
    index: usize,
    success: bool,
    niters: i64,
    cond: f64,
    elapsed: f64,
}

fn bpadmm_many(
    indices: &[usize],
    tx_ch: &crossbeam_channel::Sender<(Array2<Complex64>, EachInfo)>,
    a: &ArrayView2<'_, Complex64>,
    y: &ArrayView2<'_, Complex64>,
    a1: &ArrayView2<'_, Complex64>,
    threshold: f64,
    maxiter: i64,
    miniter: i64,
    tol: f64,
) {
    match indices.len() {
        1 => { tx_ch.clone().send(
            bpadmm_one(indices[0], a, y, a1, threshold, maxiter, miniter, tol)).unwrap();
        },
        _ => {
            let icenter = indices.len() / 2;
            rayon::join(
            || bpadmm_many(&indices[..icenter], tx_ch, a, y, a1, threshold, maxiter, miniter, tol),
            || bpadmm_many(&indices[icenter..], tx_ch, a, y, a1, threshold, maxiter, miniter, tol));
        },
    };

}


fn bpadmm_one(
    index: usize,
    a: &ArrayView2<'_, Complex64>,
    y: &ArrayView2<'_, Complex64>,
    a1: &ArrayView2<'_, Complex64>,
    threshold: f64,
    maxiter: i64,
    miniter: i64,
    tol: f64,
) -> (Array2<Complex64>, EachInfo) {
    let n = a.shape()[0];
    let p = a.shape()[1];
    let mut x = Array2::<Complex64>::zeros((p, 1));
    let mut z = Array2::<Complex64>::zeros((p, 1));
    let mut u = Array2::<Complex64>::zeros((p, 1));
    let mut info = EachInfo {
        index,
        success: false,
        niters: 0,
        cond: f64::INFINITY,
        elapsed: f64::INFINITY,
    };
    let a1y = a1.dot(&y.column(index).to_shape((n, 1)).unwrap());
    let t0 = Instant::now();
    let mut init = true;
    for i in 1..=maxiter {
        let x0 = x;
        let v = &z - &u;
        x = &v + &a1y - a1.dot(&a.dot(&v));
        let w = &x + u;
        z = soft_threshold(&w.view(), threshold);
        u = w - &z;
        info.cond = (x0.norm_l1() - x.norm_l1()).abs();
        info.niters = i;
        if info.cond < tol {
            if init {
                continue
            }
            info.success = true;
            break
        } else {
            if i < miniter {
                continue
            }
            init = false;
        }
    }
    info.elapsed = (Instant::now() - t0).as_secs_f64();
    (x, info)
}


pub fn basis_pursuit_admm(
    a: &ArrayView2<'_, Complex64>,
    y: &ArrayView2<'_, Complex64>,
    threshold: f64,
    maxiter: i64,
    miniter: i64,
    tol: f64,
    njobs: usize,
    progress: bool
) -> (Array2<Complex64>, BPADMMInfo)
{
    let shape = a.shape();
    let p = shape[1];
    let shape2 = y.shape();
    assert_eq!(shape[0], shape2[0]);
    let n = shape2[1];

    let bar = ProgressBar::new(n as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[ETA {eta_precise}] {bar:40.cyan/blue} {percent:>3}% {pos:>7}/{len:7} [{elapsed_precise}] {msg}").unwrap()
    );

    // Initialization.
    let a1 = pinv(a);
    let mut x = Array2::<Complex64>::zeros((p, n));
    let mut info = BPADMMInfo{
        success: vec![false; n],
        niters: vec![0; n],
        cond: vec![f64::INFINITY; n],
        elapsed: vec![f64::INFINITY; n],
    };

    let (tx_rt, rx_rt) = crossbeam_channel::bounded::<(Array2<Complex64>, EachInfo)>(njobs);
    rayon::join(
        // Sender.
        || bpadmm_many( Vec::from_iter(0..n).as_slice(), &tx_rt, a, y, &a1.view(), threshold, maxiter, miniter, tol),
        // Receiver.
        || {
            for _ in 0..n {
                let (x1, info1) = rx_rt.recv().unwrap();
                let i = info1.index;
                x.column_mut(i).assign(&x1.column(0));
                info.niters[i] = info1.niters;
                info.success[i] = info1.success;
                info.cond[i] = info1.cond;
                info.elapsed[i] = info1.elapsed;
                if progress {
                    bar.inc(1);
                }
            }
        }
    );
    if progress {
        bar.finish();
    }
    (x, info)
}
