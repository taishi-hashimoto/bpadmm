"Various utility functions for the BP-ADMM algorithm."

import numpy as np
from numpy.typing import ArrayLike


def initial_threshold_guess(
    A: ArrayLike,
    y: ArrayLike,
) -> np.ndarray:
    """Initial guess for a suitable soft threshold for the basis pursuit problem.
    
    Parameters
    ==========
    A : ArrayLike
        Observation matrix of shape (n, p), where n is the number of measurements and p is the number of parameters.
    y : ArrayLike
        Observed values of shape (b, n), where b is the batch size.
    
    Notes
    =====
    
    The dual problem of the original basis pursuit problem is given by:
    
    ```
    maximize y^T u
        u
    subject to ||A^T u||_∞ ≤ 1
    ```
    
    and one of the feasible solution to this problem is:
    
    ```
    u_0 = y / ||A^T y||_∞ .
    ```
    
    Then, this function returns `||u_0||_2 = ||y||_2 / ||A^T y||_∞` as a reasonable guess
    for the initial soft threshold.
    
    You can use it with `cosine_decay_schedule`:
    
    ```Python
    
    thr_beg = initial_threshold_guess(A, y)
    thr_end = thr_beg * 0.01  # or any other small value
    threshold = cosine_decay_schedule(total_steps, thr_beg, thr_end)  # -> (batch_size, total_steps)
    ```
    """
    return np.linalg.norm(y, axis=1) / np.linalg.norm(np.conjugate(A).T @ np.transpose(y), np.inf, axis=0)


def cosine_decay_schedule(
    total_steps: int,
    thr_beg: float | ArrayLike = 1.0,
    thr_end: float | ArrayLike = 0.0
) -> np.ndarray:
    """Cosine decay soft threshold schedule.
    
    Parameters
    ----------
    total_steps : int
        Total number of steps (iterations).
    thr_beg : float | ArrayLike
        Initial soft threshold (maximum).
        If an array is given, the length must be (batch_size,).
    thr_end : float | ArrayLike
        Final soft threshold (minimum).
        If an array is given, the length must be (batch_size,).

    Returns
    -------
    np.ndarray
        Soft threshold schedule of shape (total_steps,), or (batch_size, total_steps) if thr_beg and thr_end are arrays.
    """
    thr_beg = np.atleast_1d(thr_beg)
    thr_end = np.atleast_1d(thr_end)
    if thr_beg.size > 1 or thr_end.size > 1:
        # Reshape input for batch processing.
        thr_beg = np.reshape(thr_beg, (-1, 1))
        thr_end = np.reshape(thr_end, (-1, 1))
    steps = np.arange(total_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * steps / total_steps))
    learning_rates = thr_end + (thr_beg - thr_end) * cosine_decay
    return learning_rates
