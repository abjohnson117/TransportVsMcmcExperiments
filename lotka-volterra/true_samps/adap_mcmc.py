import time
from triangular_transport.mcmc.mcmc_samplers import MCMC
from typing import Callable, List
from triangular_transport.typing import Numeric
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from seaborn import kdeplot

from jax import random, jit
import jax.numpy as jnp
from jax.numpy.linalg import cholesky

from tqdm.auto import tqdm


class AdaptiveMCMC(MCMC):
    def __init__(
        self,
        target_density: Callable,
        alpha_function: Callable,
        seed: Numeric,
        train_dim: Numeric | NDArray,
        steps: Numeric,
        name: str,
        min_iter_adapt: int = 100,
        iter_step_adapt: int = 3,
        std_err: Numeric = 0.1,
        burn_in: Numeric = 5000,
        cond_no: Numeric | None = None,
        alpha_kwargs: dict | None = None,
        multimodal: bool = False,
        x0: Numeric | NDArray | None = None,
        log_density: Callable | None = None,
    ):
        super().__init__(
            target_density=target_density,
            seed=seed,
            train_dim=train_dim,
            steps=steps,
            burn_in=burn_in,
            cond_no=cond_no,
            name=name,
            multimodal=multimodal,
            x0=x0,
        )
        if log_density is not None:
            self.log_density = jit(log_density)
        self.iter_step_adapt = iter_step_adapt
        self.min_iter_adapt = min_iter_adapt
        if isinstance(self.train_dim, tuple):
            self.dim = int(self.train_dim[1])
        else:
            self.dim = self.train_dim
        self.iden = jnp.eye(self.dim)
        self.std_err = std_err
        self.cov = (self.std_err**2) * self.iden
        self.alpha_function = alpha_function
        self.alpha_kwargs = alpha_kwargs or {}

    def forward(
        self,
        x: NDArray,
        C: NDArray,
        idx: int,
        mean0: NDArray,
        keys: List,
        log_update: bool,
    ):
        subkey1, subkey2 = random.split(key=keys[1], num=2)
        xi_t = random.normal(key=subkey1, shape=self.dim)
        if log_update:
            x = jnp.log(x)
        L = cholesky(C)
        w = x + jnp.dot(L, xi_t)
        if log_update:
            w = jnp.exp(w)
            # x = jnp.exp(x)
        mean1 = _running_mean(x=x, mean=mean0, idx=idx)
        if (idx > self.min_iter_adapt) and (idx % self.iter_step_adapt == 0):
            C = _cov_estimation(
                x=x,
                covariance=C,
                previous_mean=mean0,
                current_mean=mean1,
                idx=idx,
                iden_matrix=self.iden,
            )
        if log_update:
            x = jnp.exp(x)
        accept_prob = self.alpha_function(
            x=x,
            w=w,
            log_density=self.log_density,
            **self.alpha_kwargs,
        )
        u = jnp.log(random.uniform(key=subkey2))
        if u < accept_prob:
            accepted = True
            x = w
        else:
            accepted = False
        return (x, C, mean1, accepted)

    def fit(self, print_every: int | None = None, log_update: bool = False):
        x = self.x0
        C = self.cov
        mean0 = _running_mean(x=x, mean=jnp.mean(x), idx=0)
        max_iter = self.burn_in + self.steps
        subkey1 = self.key2
        accept_count = 0
        start_time = time.time()
        for j in tqdm(range(max_iter)):
            subkey1, subkey2 = random.split(key=subkey1, num=2)
            keys = random.split(key=subkey2, num=3)
            x, C, mean0, accepted = self.forward(
                x=x,
                C=C,
                idx=j,
                mean0=mean0,
                keys=keys,
                log_update=log_update,
            )
            if j % self.iter_step_adapt == 0:
                print(f"This is C: {C}")
            if accepted:
                accept_count += 1
            if self.multimodal:
                if j % self.jump_every == 0:
                    if random.uniform(key=keys[2]) < 0.1:
                        x = global_jump(x, keys)
            if j >= self.burn_in:
                idx = j - self.burn_in
                if isinstance(self.train_dim, Numeric):
                    self.samples = self.samples.at[idx].set(x)
                else:
                    self.samples = self.samples.at[
                        idx, 0 : self.train_dim[1]
                    ].set(x[0])
            if print_every is not None:
                if (j % print_every == 0) and (j != 0):
                    print(f"The acceptance rate is: {accept_count / j}")
        self.cov = C
        self.end_time = time.time() - start_time


@jit
def _running_mean(x: NDArray, mean: NDArray, idx: int) -> NDArray:
    return (1 / (idx + 1)) * (x + idx * mean)


@jit
def _cov_estimation(
    x: NDArray,
    covariance: NDArray,
    previous_mean: NDArray,
    current_mean: NDArray,
    idx: Numeric,
    iden_matrix: NDArray,
) -> NDArray:
    stability = 1e-6
    first_part = covariance * ((idx - 1) / idx)
    second_part = (1 / idx) * (
        idx * jnp.outer(previous_mean, previous_mean)
        - (idx + 1) * jnp.outer(current_mean, current_mean)
        + jnp.outer(x, x)
        + stability * iden_matrix
    )
    return first_part + second_part


# Phase this out. We want a better multimodal region explorer anyways.
@jit
def global_jump(x, keys, scale=1.5):
    subkey = keys[0]
    key1, _ = random.split(key=subkey, num=2)
    return x + random.normal(key=key1, shape=x.shape) * scale

from abc import ABC, abstractmethod


class MCMC(ABC):
    def __init__(
        self,
        target_density: Callable,
        seed: Numeric,
        train_dim: Numeric | NDArray,
        steps: Numeric,
        name: str,
        burn_in: Numeric = 5000,
        cond_no: Numeric | None = None,
        multimodal: bool = False,
        x0: Numeric | NDArray | None = None,
    ):
        def log_density(x):
            return jnp.log(target_density(x))

        self.target_density = target_density
        self.log_density = jit(log_density)
        self.train_dim = train_dim
        if isinstance(self.train_dim, Numeric):
            samples_shape = self.train_dim
        else:
            samples_shape = self.train_dim[1]
        self.steps = steps
        self.burn_in = burn_in
        self.name = name

        self.seed = seed
        key = random.PRNGKey(seed=self.seed)
        key1, key2 = random.split(key=key, num=2)
        self.key1, self.key2 = key1, key2
        self.samples = jnp.zeros(shape=(self.steps, samples_shape))
        self.end_time = 0

        self.cond_no = cond_no
        self.multimodal = multimodal
        if self.multimodal:
            self.jump_every = 1

        self.x0 = x0
        if self.x0 is None:
            self.x0 = random.normal(key=self.key1, shape=self.train_dim)

    def fit(self, print_every: int | None = None):
        x = self.x0
        max_iter = self.burn_in + self.steps
        subkey1 = self.key2
        accept_count = 0
        start_time = time.time()
        for j in tqdm(range(max_iter)):
            subkey1, subkey2 = random.split(key=subkey1, num=2)
            keys = random.split(key=subkey2, num=3)
            x, accepted = self.forward(x=x, keys=keys)
            if accepted:
                accept_count += 1
            if self.multimodal:
                if j % self.jump_every == 0:
                    if random.uniform(key=keys[2]) < 0.1:
                        x = global_jump(x, keys)
            if j >= self.burn_in:
                idx = j - self.burn_in
                if isinstance(self.train_dim, Numeric):
                    self.samples = self.samples.at[idx].set(x)
                else:
                    self.samples = self.samples.at[
                        idx, 0 : self.train_dim[1]
                    ].set(x[0])
            if print_every is not None:
                if (j % print_every == 0) and (j != 0):
                    print(f"The acceptance rate is: {accept_count / j}")
        self.end_time = time.time() - start_time

    @abstractmethod
    def forward(self):
        pass

    def density_plot(self):
        if self.cond_no is None:
            plt.hist2d(self.samples[:, 0], self.samples[:, 1], bins=100)  # noqa
            plt.xlabel(r"$y$")
            plt.ylabel(r"$u$")
            plt.title(f"2D Histogram for {self.name} Sampling")
        else:
            kdeplot(
                self.samples[:, 1],
                bw_adjust=0.9,
                label="Approximation",
                c="olive",
            )
            plt.xlabel(r"$u$")
            plt.title(
                r"$y$="
                + str(self.cond_no)
                + ": "
                + str(self.name)
                + " Sampling"
            )
            plt.legend(fontsize="x-small")


@jit
def global_jump(x, keys, scale=1.5):
    subkey = keys[0]
    key1, _ = random.split(key=subkey, num=2)
    return x + random.normal(key=key1, shape=x.shape) * scale