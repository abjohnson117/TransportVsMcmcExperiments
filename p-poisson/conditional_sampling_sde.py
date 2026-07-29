from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray
from tqdm.auto import tqdm
from functools import partial
from diffrax import (
    ControlTerm,
    Heun,
    Dopri5,
    MultiTerm,
    ODETerm,
    PIDController,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
import lineax

from triangular_transport.flows.dataloaders import (
    dataloader,
    get_data,
)
from triangular_transport.typing import Numeric
from abc import ABC, abstractmethod

@eqx.filter_jit
def sample_trajectory_SDE(
    velocity,  # maps [x, t] -> drift b_t(x)
    score,  # maps [x, t] -> η_z(t, x)
    X0,  # shape (dim,)
    key,
    yu_dimension,
    eps=0.1,  # global scale for ε(t)
    solver=Heun(),  # Itô SDE solver
    dt0=1e-2,
    max_steps=20000,
    tol=1e-6,
    saveat="dense",
    D_mask=None,  # shape (dim,), 0 for y-dims, 1 for u-dims
    stepsize_controller=None,
):
    """Implements dX_t = (b_t - (ε/γ) D η_z) dt + sqrt(2 ε) D dW_t"""

    dim = X0.shape[0]
    if D_mask is None:
        # Fallback: assume first half is y (no noise), second half is u (noise) TODO: change this to yu_dimension
        D_mask = jnp.concatenate(
            [jnp.zeros(yu_dimension[0]), jnp.ones(yu_dimension[1])]
        )

    D_mask = jnp.asarray(D_mask)
    # assert D_mask.shape == (dim,)

    def kappa(t):
        return eps * (1.0 - t) ** 2

    def epsilon(t):
        # your schedule; stays >= 0 on [0, 1]
        return (t * (1.0 - jnp.sqrt(t))) * eps
        # return kappa(t) * gamma(t)

    # σ(t, x): R^dim -> R^{dim x dim}; diagonal with zeros in y-block, ones in u-block
    def diffusion(t, x, args):
        diag = jnp.sqrt(2.0 * epsilon(t)) * D_mask
        # Return a LinearOperator for efficiency
        return lineax.DiagonalLinearOperator(diag)

    # Drift: b_t(x) - (ε/γ) D η_z(t, x)
    @eqx.filter_jit
    def drift(t, x, args):
        t_col = jnp.full((dim, 1), t)
        # print(f"This is the shape of t_col: {t_col.shape}")
        # print(f"This is the shape of x: {x.shape}")
        xt = jnp.hstack([t_col, x])
        b = velocity(xt)  # shape (dim,)
        s = score(t_col, x)  # shape (dim,)
        s_proj = D_mask * s  # apply D
        return b + epsilon(t) * s_proj

    t0, t1 = 0.0, 1.0
    brownian = VirtualBrownianTree(t0, t1, tol=tol, shape=X0.shape, key=key)

    terms = MultiTerm(
        ODETerm(drift),
        ControlTerm(diffusion, brownian),
    )

    if saveat == "dense":
        if stepsize_controller is None:
            sa = SaveAt(dense=True)
            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt0,
                y0=X0,
                saveat=sa,
                max_steps=max_steps,
            )
            return sol.evaluate(1.0), sol
        else:
            sa = SaveAt(dense=True)
            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt0,
                y0=X0,
                saveat=sa,
                max_steps=max_steps,
                stepsize_controller=stepsize_controller,
            )
            return sol.evaluate(1.0), sol
    elif saveat == "t1":
        if stepsize_controller is None:
            sa = SaveAt(t1=True)
            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt0,
                y0=X0,
                saveat=sa,
                max_steps=max_steps,
            )
            return sol.ys[-1]
        else:
            sa = SaveAt(t1=True)
            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt0,
                y0=X0,
                saveat=sa,
                max_steps=max_steps,
                stepsize_controller=stepsize_controller,
            )
            return sol.ys[-1]
    else:
        raise ValueError("saveat must be 'dense' or 't1'")

@partial(jax.jit, static_argnames=("nsamples", "y_dim", "u_dim"))
def _get_ref_cond_samples_core(key, y_row, *, nsamples: int, y_dim: int, u_dim: int, u0):
    y = jnp.broadcast_to(y_row, (nsamples, y_dim))
    return jnp.hstack([y, u0])

def _canonicalize_cond_value(cond_value, y_dim: int):
    cv = jnp.asarray(cond_value)

    # Convert to a 1D array y_row of length y_dim
    if cv.ndim == 0:
        # Avoid .item() — use broadcast
        y_row = jnp.broadcast_to(cv, (y_dim,))
    elif cv.ndim == 1:
        if cv.size == 1:
            y_row = jnp.broadcast_to(cv[0], (y_dim,))
        elif cv.size == y_dim:
            y_row = cv
        else:
            raise ValueError(f"cond_value must be scalar or length {y_dim}, got shape {cv.shape}")
    else:
        raise ValueError(f"cond_value must have ndim 0 or 1, got ndim {cv.ndim}")

    return y_row

def _get_reference_conditional_samples(
    *, key, cond_value, nsamples: int, yu_dimension, reference_sampler, reference_sampler_args=None, u0_cond=None
):
    y_dim, u_dim = yu_dimension
    y_row = _canonicalize_cond_value(cond_value, y_dim)

    if u0_cond is None:
        args = reference_sampler_args or {}
        u0_cond = reference_sampler(key=key, shape=(nsamples, u_dim), **args)

    return _get_ref_cond_samples_core(
        key, y_row, nsamples=nsamples, y_dim=y_dim, u_dim=u_dim, u0=u0_cond
    )

def conditional_sample(
    velocity,
    score,
    cond_values: list | jax.Array | Numeric,
    yu_dimension: tuple,
    reference_sampler: Callable,
    reference_sampler_args: dict | None = None,
    nsamples: int = 1000,
    u0_cond: NDArray | None = None,
    solver_args: dict | None = None,
    key=None,
):
    solver_args = solver_args or {}
    if key is None:
        key = jax.random.key(0)

    if isinstance(cond_values, list):
        conditional_list = []
        for cond_value in tqdm(cond_values):
            if u0_cond is not None:
                key, kperm = jax.random.split(key)
                u0_cond = jax.random.permutation(kperm, u0_cond, axis=0)
            key, kref, ksolve = jax.random.split(key, 3)
            x0_cond = _get_reference_conditional_samples(
                key=kref,
                cond_value=cond_value,
                nsamples=nsamples,
                yu_dimension=yu_dimension,
                reference_sampler=reference_sampler,
                reference_sampler_args=reference_sampler_args,
                u0_cond=u0_cond,
            )
            x1_pred = sample_trajectory_SDE(
                velocity,
                score,
                X0=x0_cond,
                key=ksolve,
                yu_dimension=yu_dimension,
                **solver_args,
            )
            conditional_list.append(x1_pred)
    else:
        if not isinstance(cond_values, (jax.Array, Numeric)):
            raise TypeError(
                f"cond_values must be of type jaxlib.xla_extension.ArrayImpl or float or int "
                f"got {type(cond_values)}"
            )
        x0_cond = _get_reference_conditional_samples(
            key=key,
            cond_value=cond_values,
            nsamples=nsamples,
            yu_dimension=yu_dimension,
            reference_sampler=reference_sampler,
            reference_sampler_args=reference_sampler_args,
            u0_cond=u0_cond,
        )
        x1_pred = sample_trajectory_SDE(
            velocity,
            score,
            X0=x0_cond,
            key=key,
            yu_dimension=yu_dimension,
            **solver_args,
        )
        conditional_list = x1_pred
    return conditional_list
