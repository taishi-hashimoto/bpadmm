
import numpy as np
from typing import Any
from . import basis_pursuit_admm, cosine_decay_schedule

def ocft_matrix(
    nt: int,
    dt: float = 1,
    factor: int = 1,
    decimation: int = 1,
    rng: np.random.RandomState = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse DFT matrix.

    Parameters
    ==========
    nt : int
        Number of time samples.
    dt : float
        Sampling interval.
        Default is 1, which means the sampling frequency is 1 Hz.
    factor : int
        Factor for the number of frequencies to be decomposed.
    decimation : int
        Decimation factor for the number of frequencies to be decomposed.
    seed : int
        Random seed for decimation.
        Default is None, which does not set the seed.
        If decimation > 1, the frequencies are randomly decimated.

    Returns
    =======
    A: np.ndarray
        Sparse DFT matrix.
    f: np.ndarray
        Frequencies for the DFT.
    """
    # The number of frequencies to be decomposed.
    nf = nt * factor
    # Index for picking frequencies.
    index = np.arange(nf)
    if decimation > 1:
        if rng is None:
            rng = np.random.default_rng()
        index = rng.choice(index, size=nf // decimation, replace=False)
    # Phase matrix for inverse Fourier transform.
    w = np.arange(nt)[:, None] * index[None, :] * 2 * np.pi / nf
    # Inverse Fourier transform matrix.
    freq = np.fft.fftfreq(nf, dt)
    return np.exp(1j * w) / np.sqrt(nt), freq[index]


def ocft(
    y: np.ndarray,
    factor: int = 1,
    decimation: int = 1,
    threshold: float = None,
    maxiter: int = 1000,
    stepiter: int = 10,
    patience: int = 10,
    axis: int = -1,
    info: bool = False,
    A: np.ndarray = None,
    f: np.ndarray = None,
    Ai: np.ndarray = None,
    rng: np.random.RandomState = None,
    device_kind: str = None,
    dt: float = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Overcomplete Fourier transform.
    
    Parameters
    ==========
    y: np.ndarray
        Input signal.
        Can be 1-d or 2-d array.
    factor: int
        Factor multiplied to the number of output frequencies.
    decimation: int
        Decimation factor for the number of frequencies.
    threshold: float
        Soft thresholding.
    axis: int
        Axis in which the transform is performed.
        Default is -1, which means the last axis.
    dt: float
        Sampling interval.
        Default is 1, which means the sampling frequency is 1 Hz.
    """
    s = np.shape(y)
    y = np.reshape(y, (-1, s[axis]))
    s2 = y.shape
    if axis < 0:
        axis += len(s2)

    is_1d = len(s) == 1
    is_t = axis == 0

    if is_t:
        y = y.T

    if A is None:
        # The number of time samples.
        nt = np.shape(y)[axis]
        A, f = ocft_matrix(nt, dt=dt, factor=factor, decimation=decimation, rng=rng)

    if Ai is None:
        Ai = np.linalg.pinv(A)

    if factor == 1:  # DFT.
        x = A.conj().dot(y)
    else:
        if threshold is None:
            threshold = 1e-3
        elif isinstance(threshold, tuple):
            thr_min, thr_max = threshold
            threshold = cosine_decay_schedule(
                maxiter * stepiter, thr_beg=thr_max, thr_end=thr_min)
        result = basis_pursuit_admm(
            A, y, threshold,
            maxiter=maxiter, stepiter=stepiter, patience=patience,
            Ai=Ai, device_kind=device_kind)
        x = result.x

    if is_t:
        x = x.T
    if is_1d:
        x = x.ravel()

    if info:
        return x, f, result
    else:
        return x, f
