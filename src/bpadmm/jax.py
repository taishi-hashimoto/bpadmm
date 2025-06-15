"JAX implementation of Basis Pursuit by ADMM."
import re
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


def basis_pursuit_admm(
    A: np.ndarray,
    y: np.ndarray,
    threshold: float | np.ndarray,
    maxiter: int = 1000,
    stepiter: int = 100,
    patience: int = 10,
    Ai: np.ndarray = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    init_x: np.ndarray | bool | None = None,
    device_kind: str = None,
) -> 'BpADMMResult':
    """ADMM for basis pursuit problem.

    Basis pursuit problem is defined as following sparse modeling:

    ```
    minimize |x|_1
        x
    subject to y = Ax
    ```

    ADMM solves this problem by following iteration procedure:

    ```Python
    A1 = A^T(A A^T)^(-1)  # pseudo inverse
    while not converged:
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
        If an array is given, each element is used for each step of ADMM.
        In that case, it should have the same size as `maxiter * stepiter`.
    maxiter: int
        Maximum number of evaluation of the terminate condition.
    stepiter: int
        Number of iterations in each step of ADMM.
        The total number of iterations is `maxiter * stepiter`.
    patience: int
        Number of iterations to wait before stopping if no improvement found.
    Ai: np.ndarray
        Pseudo-inverse matrix of `A`, if given.
        If not specified, it will be computed by `jnp.linalg.pinv(A)`.
    atol: float
        Absolute tolerance for convergence.
        Default is 1e-4.
    rtol: float
        Relative tolerance for convergence.
        Default is 1e-3.
    init_x: np.ndarray | bool
        Initial value for `x`.
        If `True`, L2 guess, i.e., `A1 @ y` is used.
        If `False` or `None`, zero vector is used.
        Default is `None`.
    device_kind: str
        Regex pattern for target devices.
        Default is None, which will use all visible GPUs by default.

    Returns
    =======
    BpADMMResult
        Result of the ADMM algorithm, containing the solution `x`, status, messages, and other metrics.
        See `BpADMMResult` for details.
    """
    n, p = A.shape
    ndims = np.ndim(y)
    y = np.atleast_2d(y)
    nbatches, n_ = y.shape
    assert n_ == n, f"A.shape = {A.shape}, y.shape = {y.shape}, y.shape[1] = {n_} != A.shape[0] = {n}"

    if isinstance(threshold, float):
        # If threshold is a float, create an array of the same size as the number of iterations.
        threshold = jnp.full((maxiter * stepiter,), threshold, dtype=jnp.float32)
    else:
        # If threshold is an array, it should have the same size as the number of iterations.
        threshold = jnp.array(threshold, dtype=jnp.float32)
        assert threshold.shape == (maxiter * stepiter,), f"Threshold {threshold.shape} does not match the total iteration number {(maxiter * stepiter,)}."

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

    if init_x is not None:
        if isinstance(init_x, bool) and init_x:
            # If init_x is True, use L2 guess.
            x = jnp.array([A1 @ y1.reshape(-1, 1) for y1 in y]).squeeze(axis=-1)
        else:
            # If init_x is given, use it as the initial value.
            x = jnp.array(init_x)
        try:
            x = x.reshape(nbatches, p).astype(complex)
        except Exception as e:
            raise ValueError(f"init_x shape {x.shape} does not match (nbatches, p) = {(nbatches, p)}.") from e

    def loop(y, state: BpADMMState):
        """Main loop of the ADMM algorithm."""

        def cond(state: BpADMMState):
            "Stopping condition."
            # Stop if maximum iterations or no improvement in MSE for `patience` iterations.
            return (
                (state.i < patience) |
                (
                    (state.i <= maxiter) &
                    (state.bad_count <= patience) &
                    jnp.logical_not(state.eval_subopt)
                )
            )

        def body(state: BpADMMState):
            "Each step of the loop."

            def admm(i, val):
                "ADMM single step."
                j = state.i * stepiter + i
                x, z, u = val
                v = z - u
                x = v + A1 @ (y - A @ v)
                z = soft_threshold(x + u, threshold[j])
                u = u + x - z
                return x, z, u

            # Run ADMM iterations for stepiter.
            x, z, u = jax.lax.fori_loop(0, stepiter, admm, (state.x, state.z, state.u))

            # Evaluates convergence and residuals.

            # Suboptimality.
            subopt = jnp.linalg.norm(state.x - x)

            # L1 norm (objective function value).
            l1_norm = jnp.linalg.norm(x, ord=1)

            # Primal residual.
            # NOTE: In compressed sensing, `Ax - y` is by definition zero, so it cannot be used.
            primal_residual = jnp.linalg.norm(x - z)

            # Dual residual.
            dual_residual = jnp.linalg.norm(state.z - z)

            # Current loss.
            curr_loss = jnp.array([subopt, l1_norm, primal_residual, dual_residual])
            eval_subopt = subopt < atol + rtol*jnp.maximum(jnp.linalg.norm(state.x), jnp.linalg.norm(x))

            # Check if the new losses are better than the best losses so far.

            is_improving_each = curr_loss < state.best_loss
            is_improving_any = jnp.any(is_improving_each)
            best_loss = jnp.where(is_improving_each, curr_loss, state.best_loss)
            best_x = jnp.where(is_improving_any, x, state.best_x)
            bad_count = jnp.where(is_improving_any, 0, state.bad_count + 1)

            # Update state
            return BpADMMState(
                i=state.i + 1,
                x=x,
                z=z,
                u=u,
                bad_count=bad_count,
                best_loss=best_loss,
                best_x=best_x,
                diff_x=state.diff_x.at[state.i].set(subopt),
                l1_norm=state.l1_norm.at[state.i].set(l1_norm),
                res_prim=state.res_prim.at[state.i].set(primal_residual),
                res_dual=state.res_dual.at[state.i].set(dual_residual),
                eval_subopt=eval_subopt,
            )

        # Run the whole loop.
        return jax.lax.while_loop(cond, body, state)

    devices = jax.devices()
    if device_kind is not None:
        devices = [d for d in devices if re.match(device_kind, d.device_kind, re.IGNORECASE)]    
    ndevices = len(devices)

    # Initialize state
    state = BpADMMState(
        i=jnp.zeros(nbatches, dtype=int),
        x=x,  # shape (nbatches, p)
        z=z,
        u=u,
        bad_count=jnp.zeros(nbatches, dtype=int),
        best_loss=jnp.full((nbatches, 4), np.inf),
        best_x=x,
        diff_x=jnp.full((nbatches, maxiter), np.inf),
        l1_norm=jnp.full((nbatches, maxiter), np.inf),
        res_prim=jnp.full((nbatches, maxiter), np.inf),
        res_dual=jnp.full((nbatches, maxiter), np.inf),
        eval_subopt=jnp.full((nbatches,), False),
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
            state = jax.tree_util.tree_map(lambda a: _extend(a, nadd), state)
        # Split batches into devices.
        y = _split(y, ndevices)
        state = jax.tree_util.tree_map(lambda a: _split(a, ndevices), state)
        # Distribute tasks to devices.
        state = jax.pmap(
            jax.vmap(loop, in_axes=(0, 0)),
            axis_name="batch",
            devices=devices
        )(y, state)
        # Collect result from devices.
        state = jax.tree_util.tree_map(_merge, state)
        # Discard unnecessary part.
        state = jax.tree_util.tree_map(lambda a: a[:nbatches], state)

    # Restore the best result.
    x = state.best_x
    
    if ndims == 1:
        x = x.ravel()

    # Status of the method.
    messages_dict = {
        0: "Maximum iterations reached.",
        1: "Suboptimality reached.",
        2: "Early stopping due to no improvement."
    }
    status = np.zeros(nbatches, dtype=int)
    status[state.eval_subopt] = 1
    status[state.bad_count >= patience] = 2
    messages = np.array([messages_dict[s] for s in status])

    # Return results.
    return BpADMMResult(
        x=x,
        status=status,
        messages=messages,
        nit=state.i - 1,
        state=state
    )


def soft_threshold(x: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """Soft thresholding function for JAX.

    This also works for complex inputs.
    """
    a = jnp.fmax(jnp.abs(x) - threshold, 0)
    return a / (a + threshold) * x


def cosine_decay_schedule(total_steps, thr_beg=1.0, thr_end=0.0):
    """Cosine decay soft threshold schedule.
    
    Parameters
    ----------
    total_steps : int
        Total number of steps (iterations).
    thr_beg : float
        Initial soft threshold (maximum).
    thr_end : float
        Final soft threshold (minimum).
        
    Returns
    -------
    jnp.ndarray
        Soft threshold schedule of shape (total_steps,).
    """
    steps = jnp.arange(total_steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * steps / total_steps))
    learning_rates = thr_end + (thr_beg - thr_end) * cosine_decay
    return learning_rates


@dataclass
class BpADMMResult:
    x: np.ndarray
    """Solution vector of the basis pursuit ADMM algorithm."""

    status: np.ndarray
    """Status of the algorithm for each batch, indicating convergence or failure.

    0: Exceeded maximum iterations.
    1: suboptimality reached.
    2: Early stopping due to no improvement
    """
    
    messages: np.ndarray
    "Description of the cause of termination for each batch."

    nit: int
    """Number of iterations performed for each batch."""
    
    state: 'BpADMMState'
    """Raw state of the ADMM loop, containing various metrics and results.
    
    This is JAX's pytree.
    """
    
    def __str__(self):
        return (
            f"BpADMMResult(nit={self.nit}, "
            f"status={self.status}, "
            f"messages={self.messages})"
        )

    def __repr__(self):
        width_batch = len(str(self.status.shape[0]))
        width_nit = len(str(max(self.nit)))
        lines = []
        for ib, (status, nit, message) in enumerate(zip(self.status, self.nit, self.messages)):
            lines.append(f"{ib:{width_batch}d} [{status:2d}] {nit:{width_nit}d}it: {message}")
        return "\n".join(lines)


@jax.tree_util.register_pytree_node_class
@dataclass
class BpADMMState:
    "State of ADMM loop."
    i: int
    "Current iteration index."
    x: jnp.ndarray
    "Current solution vector."
    z: jnp.ndarray
    "Current primal variable."
    u: jnp.ndarray
    "Current dual variable."
    bad_count: int
    "Number of consecutive iterations without improvement."
    best_loss: jnp.ndarray
    "Best loss so far for each batch."
    best_x: jnp.ndarray
    "Best solution vector so far for each batch."
    diff_x: jnp.ndarray
    "Difference of the solution vector from the previous iteration for each batch."
    l1_norm: jnp.ndarray
    "L1 norm of the solution vector for each batch."
    res_prim: jnp.ndarray
    "Primal residual for each batch."
    res_dual: jnp.ndarray
    "Dual residual for each batch."
    eval_subopt: jnp.ndarray
    "Whether the suboptimality condition is met for each batch."

    def tree_flatten(self):
        """Flatten the state for JAX tree utilities."""
        children = (
            self.i, self.x, self.z, self.u,
            self.bad_count, self.best_loss,
            self.best_x, self.diff_x, self.l1_norm, self.res_prim, self.res_dual,
            self.eval_subopt
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _, children):
        """Unflatten the state from JAX tree utilities."""
        return cls(*children)


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
