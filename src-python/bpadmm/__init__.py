import numpy as np
from typing import Tuple
from bpadmm._impl import _basis_pursuit_admm


def basis_pursuit_admm(
    A: np.ndarray,
    y: np.ndarray,
    threshold: float,
    maxiter: int = 1000,
    miniter: int = 10,
    tol: float = 1e-4,
    njobs: int = -1,
    info: bool = False,
    progress: bool = False,
) -> Tuple[np.ndarray, dict]:
    """ADMM for basis pursuit problem.

    Basis pursuit problem is defined as following sparse modeling:

    minimize |x|_1
        x
    subject to y = Ax

    ADMM solves this problem by the following iterations:

    A1 = A^T(A A^T)^(-1)  # pseudo inverse
    x = z - u + A1 @ (y - A @ (z - u))
    z = soft_threshold(x + u, threshold)
    u += x - z


    Parameters
    ==========
    A: np.ndarray
        Observation matrix of the size (n, p).
        n is the number of observation and p is the number of parameters.
        Usually `p > n`.
    y: np.ndarray
        Observed values of the size (n, k).
        k is the number of snapshots.
        If k > 0 they will be processed using multiple cpu cores.
    threshold: float
        Soft threshold, or the step size of update.
    maxiter: int
        Maximum number of iterations.
    miniter: int
        Minimum number of iterations.
    tol: float
        Tolerance about the absolute difference of L_1 norm of x.
        Stop the iteration when absolute difference of `|x|_1` between
        ierations became less than this.
    njobs: int
        The number of jobs to be run to process multiple snapshots in `y`
        simultaneously.
    info: bool
        Set True to return some information about convergence.
        Default is False.
    progress: bool
        Set True to show progress.
        Default is False.
    
    Returns
    =======
    x: ndarray
        The basis.
    info: dict, optional
        Returned when `info = True`.
        Some informations about convergence.
    """
    if njobs == -1:
        import multiprocessing
        njobs = multiprocessing.cpu_count()

    x, info = _basis_pursuit_admm(
         A, y, threshold, maxiter, miniter, tol, njobs, progress)
    if info:
        return x, info
    else:
        return x


def sparse_dft(
    y: np.ndarray,
    factor: int = 1,
    threshold: float = None,
    maxiter: int = 1000,
    miniter: int = 10,
    tol: float = 1e-4,
    axis: int = -1,
    info: bool = False,
    njobs: int = -1,
):
    s = np.shape(y)
    y = np.atleast_2d(y)
    if axis < 0:
        axis += len(s)
    is_1d = len(s) == 1
    is_t = is_1d or axis == 1
    if is_t:
        y = y.T
    # The number of time samples.
    nt = np.shape(y)[0]
    # The number of frequencies to be decomposed.
    nf = nt * factor
    # Phase matrix for inverse Fourier transform.
    w = np.arange(nt)[:, None] * np.arange(nf)[None, :] * 2 * np.pi / nf
    # Inverse Fourier transform matrix.
    A = np.exp(1j * w)

    if factor == 1:  # DFT.
        x = A.conj().dot(y)
    else:
        if threshold is None:
            threshold = 1e-3 * np.linalg.norm(A)
        x, _info = basis_pursuit_admm(
                A, y, threshold, maxiter, miniter, tol, njobs)
    if is_t:
        x = x.T
    if is_1d:
        x = x.ravel()

    if info:
        return x, _info
    else:
        return x
