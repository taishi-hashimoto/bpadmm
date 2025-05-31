"JAX implementation."

import re
from typing import Any
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


def soft_threshold(x: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """Soft thresholding function for JAX.

    This also works for complex inputs.
    """
    a = jnp.fmax(jnp.abs(x) - threshold, 0)
    return a / (a + threshold) * x


def basis_pursuit_admm(
    A: np.ndarray,
    y: np.ndarray,
    threshold: float,
    maxiter: int = 1000,
    stepiter: int = 100,
    patience: int = 10,
    Ai: np.ndarray = None,
    device_kind: str = None,
    info: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, Any]]:
    """ADMM for basis pursuit problem.

    Basis pursuit problem is defined as following sparse modeling:

    ```
    minimize |x|_1
        x
    subject to y = Ax
    ```

    ADMM solves this problem by the following iterations:

    ```Python
    A1 = A^T(A A^T)^(-1)  # pseudo inverse
    x = z - u + A1 @ (y - A @ (z - u))
    z = soft_threshold(x + u, threshold)
    u += x - z
    ```

    This function is a JAX implementation of the above algorithm.

    Parameters
    ==========
    A: np.ndarray
        Observation matrix of the size `(n, p)`.
        `n` is the number of observation and `p` is the number of parameters.
        Usually `p > n`.
    y: np.ndarray
        Observed values of the size `(b, n)`.
        `b` is the number of batches which is processed sequentially.
    threshold: float
        Soft threshold, or the step size for update in ADMM.
    maxiter: int
        Maximum number of iterations.
    patience: int
        Number of iterations to wait before stopping if no improvement in MSE found.
    Ai: np.ndarray
        Pseudo-inverse matrix of `A`, if given.
    device_kind: str
        Regex pattern for target devices.
        Default is None, which will use all visible GPUs by default.
    """
    n, p = A.shape
    ndims = np.ndim(y)
    y = np.atleast_2d(y)
    nbatches, n_ = y.shape
    assert n_ == n, f"A.shape = {A.shape}, y.shape = {y.shape}, y.shape[1] = {n_} != A.shape[0] = {n}"

    # Initialization.
    A = jnp.array(A)
    y = jnp.array(y)
    x = jnp.zeros((nbatches, p), dtype=complex)
    z = jnp.zeros((nbatches, p), dtype=complex)
    u = jnp.zeros((nbatches, p), dtype=complex)

    if Ai is None:
        A1 = jnp.linalg.pinv(A)
    else:
        A1 = jnp.array(Ai)

    A = jax.device_put(A)
    A1 = jax.device_put(A1)
    y = jax.device_put(y)
    x = jax.device_put(x)
    z = jax.device_put(z)
    u = jax.device_put(u)

    # State of the loop.
    @jax.tree_util.register_pytree_node_class
    @dataclass
    class State:
        "State of the loop."
        i: int
        x: jnp.ndarray
        z: jnp.ndarray
        u: jnp.ndarray
        bad_count: int
        best_loss: float
        best_x: jnp.ndarray
        diff_x: jnp.ndarray
        l1_norm: jnp.ndarray

        def tree_flatten(self):
            """Flatten the state for JAX tree utilities."""
            children = (
                self.i, self.x, self.z, self.u,
                self.bad_count, self.best_loss,
                self.best_x, self.diff_x, self.l1_norm)
            aux_data = None
            return children, aux_data

        @classmethod
        def tree_unflatten(cls, _, children):
            """Unflatten the state from JAX tree utilities."""
            return cls(*children)

    def loop(y, state: State):
        """Main loop of the ADMM algorithm."""

        def cond(state: State):
            "Stopping condition."
            # Stop if maximum iterations or no improvement in MSE for `patience` iterations.
            return jnp.logical_and(state.i < maxiter, state.bad_count < patience)

        def step(state: State):
            "Each step of the loop."

            def admm(_, val):
                "ADMM iteration."
                x, z, u = val
                v = z - u
                x = v + A1 @ (y - A @ v)
                z = soft_threshold(x + u, threshold)
                u = u + x - z
                return x, z, u

            # Run ADMM iterations for stepiter.
            x, z, u = jax.lax.fori_loop(0, stepiter, admm, (state.x, state.z, state.u))

            # diff of x
            diff_x = jnp.linalg.norm(x - state.x)
            
            # L1 norm of x
            l1_norm = jnp.sum(jnp.abs(x))

            # Check if the new L1 norm is better than the best L1 norm so far.
            is_improving = diff_x < state.best_loss
            best_loss = jnp.where(is_improving, diff_x, state.best_loss)
            best_x = jnp.where(is_improving, x, state.best_x)
            bad_count = jnp.where(is_improving, 0, state.bad_count + 1)
            # Update state
            return State(
                i=state.i + 1,
                x=x,
                z=z,
                u=u,
                bad_count=bad_count,
                best_loss=best_loss,
                best_x=best_x,
                diff_x=state.diff_x.at[state.i].set(diff_x),
                l1_norm=state.l1_norm.at[state.i].set(l1_norm),
            )

        # Run the whole loop.
        return jax.lax.while_loop(cond, step, state)

    devices = jax.devices()
    if device_kind is not None:
        devices = [d for d in devices if re.match(device_kind, d.device_kind, re.IGNORECASE)]    
    ndevices = len(devices)

    # Initialize state

    state = State(
        i=jnp.zeros(nbatches, dtype=int),
        x=x,  # shape (n_sample, p)
        z=z,
        u=u,
        bad_count=jnp.zeros(nbatches, dtype=int),
        best_loss=jnp.full((nbatches,), np.inf),
        best_x=x,
        diff_x=jnp.full((nbatches, maxiter), np.inf),
        l1_norm=jnp.full((nbatches, maxiter), np.inf),
    )

    # Run the loop.
    if ndevices == 1 or nbatches == 1:  # Use vmap
        state = jax.vmap(loop, in_axes=(0, 0))(y, state)
    else:  # Use pmap
        batches_per_device = nbatches // ndevices
        rem = nbatches - batches_per_device * ndevices
        nadd = 0
        if rem != 0:
            # Add small number of dummy tasks to make it divisible by n_devices
            nadd = ndevices - rem
            y = _extend(y, nadd)
            state = jax.tree_map(lambda a: _extend(a, nadd), state)
        # Split batches into devices.
        y = _split(y, ndevices)
        state = jax.tree_map(lambda a: _split(a, ndevices), state)
        # Distribute tasks to devices.
        state = jax.pmap(
            jax.vmap(loop, in_axes=(0, 0)),
            axis_name="batch",
            devices=devices
        )(y, state)
        # Collect result from devices.
        state = jax.tree_map(_merge, state)
        # Discard unnecessary part.
        state = jax.tree_map(lambda a: a[:nbatches], state)

    x = state.x
    
    if ndims == 1:
        x = x.ravel()

    # Return results.
    if info:
        return x, {
            "i": state.i - 1 - state.bad_count,
            "n": state.i,
            "diff_x": state.diff_x,
            "l1_norm": state.l1_norm,
        }
    else:
        return x


def _split(arr: jax.Array, ndevices: int):
    """Reshape arr so that its leading axis becomes [ndevices, batches_per_device, …]."""
    b, *rest = arr.shape
    if b % ndevices != 0:
        raise ValueError(f"Batch size {b} not divisible by #devices {ndevices}")
    batches_per_device = b // ndevices
    return arr.reshape(ndevices, batches_per_device, *rest)


def _merge(arr: jax.Array):
    """Inverse of _split()."""
    ndevices, batches_per_device, *rest = arr.shape
    return arr.reshape(ndevices * batches_per_device, *rest)


def _extend(arr: jax.Array, nadd: int):
    """Extend rows by nadd"""
    nsamples, *rest = arr.shape
    return jnp.resize(arr, (nsamples + nadd, *rest))


def make_ocft_matrix(
    nt: int,
    dt: float = 1,
    factor: int = 1,
    decimation: int = 1,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse DFT matrix.

    Parameters
    ==========
    nt : int
        Number of time samples.
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
        if seed is not None:
            np.random.seed(seed)
        index = np.random.choice(index, size=nf // decimation, replace=False)
    # Phase matrix for inverse Fourier transform.
    w = np.arange(nt)[:, None] * index[None, :] * 2 * np.pi / nf
    # Inverse Fourier transform matrix.
    freq = np.fft.fftfreq(nf, dt)
    return np.exp(1j * w), freq[index]
    


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
    seed: int = None,
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
        A, f = make_ocft_matrix(nt, dt=dt, factor=factor, decimation=decimation, seed=seed)

    if Ai is None:
        Ai = np.linalg.pinv(A)

    _info = None
    if factor == 1:  # DFT.
        x = A.conj().dot(y)
    else:
        if threshold is None:
            threshold = 1e-3 * np.linalg.norm(A)
        x, _info = basis_pursuit_admm(
            A, y, threshold,
            maxiter=maxiter, stepiter=stepiter, patience=patience,
            Ai=Ai, info=info, device_kind=device_kind)

    if is_t:
        x = x.T
    if is_1d:
        x = x.ravel()

    if info:
        return x, f, _info
    else:
        return x, f
