"Reference inplementation in pure Python."
import numpy as np
from tqdm.notebook import tqdm


def soft_threshold(x: np.ndarray, threshold: float):
    """Soft thresholding function.

    This also works for complex inputs.
    """
    a = np.fmax(np.abs(x) - threshold, 0)
    return a / (a + threshold) * x


def basis_pursuit_admm(
    A: np.ndarray,
    y: np.ndarray,
    threshold: float,
    maxiter: int = 1000,
    miniter: int = 10,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    trace: bool = False,
    progress: bool = False,
):
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
        Observed values of the size (n, 1).
    threshold: float
        Soft threshold, or the step size of update.
    maxiter: int
        Maximum number of iterations.
    miniter: int
        Minimum number of iterations.
    xtol: float
        Tolerance about the absolute difference of x.
        Stop the iteration when absolute difference of x between interations
        became less than this.
    ftol: float
        Tolerance about the absolute difference of L_1 norm of x.
        Stop the iteration when absolute difference of `|x|_1` between
        ierations became less than this.
    trace: bool
        Set True to keep track of all variables x, z, and u in ADMM.
        Default is False.
    progress: bool
        Set True to show progress.
        Default is False.
    """
    n, p = A.shape
    assert y.shape == (n, 1)

    A1 = np.linalg.pinv(A)

    # Initialization.
    x = np.zeros((p, 1), dtype=complex)
    z = x
    u = np.zeros((p, 1), dtype=complex)

    if trace:
        trace = {
            "x": np.zeros((p, maxiter), dtype=complex),
            "z": np.zeros((p, maxiter), dtype=complex),
            "u": np.zeros((p, maxiter), dtype=complex),
            "i": 0,
            "success": False,
        }

    init = True
    # ADMM for basis pursuit problem
    for i in tqdm(range(maxiter)) if progress else range(maxiter):
        x0 = x
        # Basic form of BP-ADMM
        v = z - u
        x = v + A1 @ (y - A @ v)
        z = soft_threshold(x + u, threshold)
        u += x - z
        # Simplified, equivalent form of BP-ADMM
        # x = soft_threshold(z, threshold)
        # z = x + A1 @ (y - A @ (2 * x - z))
        if trace:
            trace["x"][:, i] = x.ravel()
            trace["z"][:, i] = z.ravel()
            trace["u"][:, i] = u.ravel()
            trace["i"] = i
        # Check terminate conditions
        cond_x = np.linalg.norm(x0 - x)
        cond_f = np.abs(np.sum(np.abs(x0)) - np.sum(np.abs(x)))
        if cond_x < xtol or cond_f < ftol:
            if init:
                continue
            if trace:
                trace["x"] = trace["x"][:, :i+1]
                trace["z"] = trace["z"][:, :i+1]
                trace["u"] = trace["u"][:, :i+1]
                trace["cond_x"] = cond_x
                trace["cond_f"] = cond_f
                trace["success"] = True
            break
        else:
            if i < miniter:
                continue
            init = False
    if trace:
        return trace
    else:
        return x


def sparse_dft(
    y: np.ndarray,
    factor: int = 1,
    maxiter: int = 1000,
    threshold: float = None,
    trace: bool = False,
):
    y = np.reshape(y, (-1, 1))
    # The number of time samples.
    nt = y.size
    # The number of frequencies to be decomposed.
    nf = nt * factor
    # Phase matrix for inverse Fourier transform.
    w = np.arange(nt)[:, None] * np.arange(nf)[None, :] * 2 * np.pi / nf
    # Inverse Fourier transform matrix.
    A = np.exp(1j * w)

    if factor == 1:  # DFT.
        return A.conj().dot(y).ravel()
    if threshold is None:
        threshold = 1e-3 * np.linalg.norm(A)

    x = basis_pursuit_admm(
        A, y, threshold=threshold, maxiter=maxiter, trace=trace)

    if trace:
        return x
    else:
        return x.ravel()
