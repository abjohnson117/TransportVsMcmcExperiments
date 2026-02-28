from typing import Callable
from numpy.typing import NDArray
from jax import random, vmap, jit
import jax.numpy as jnp
from diffrax import (Dopri5, ODETerm, SaveAt, diffeqsolve)
import equinox as eqx
from functools import partial

from triangular_transport.flows.methods.utils import UnitGaussianNormalizer



class LV:
    def __init__(
        self,
        seed: int,
        no_samples: int,
        prior_sampler: Callable,
        likelihood_sampler: Callable,
        sigma: float = 0.01,
        u_true: NDArray = jnp.array([0.92, 0.05, 1.50, 0.02]),
        t0: int = 0,
        t1: int = 20,
        normalize: bool = False,
        solver: Callable = Dopri5(),
    ):
        key = random.key(seed)
        self.keys = random.split(key=key, num=3)
        self.u_dim = 4
        self.u_true = u_true

        mu_ones = jnp.ones(shape=(no_samples, self.u_dim))
        self.mu_base = jnp.array([-0.125, -3.0, -0.125, -3.0])
        self.mu_prior = self.mu_base * mu_ones
        self.std_prior = 1 / jnp.sqrt(2)

        self.sigma = sigma

        self.no_samples = no_samples
        self.prior_sampler = prior_sampler
        self.likelihood_sampler = likelihood_sampler

        self.t0 = t0
        self.t1 = t1
        self.tt = jnp.linspace(self.t0, self.t1, num=1000)
        y0_ones = jnp.ones(shape=(no_samples, 2))
        self.y0_start = jnp.array([30, 1])
        self.y0 = self.y0_start * y0_ones
        self.dt0 = 0.5 # Tested this with Hsu... should be enough to get good solutions.
        self.true = u_true

        self.normalize = normalize
        if self.normalize:
            self.ys_normalizer = None
            self.us_normalizer = None

        self.solver = solver

    def sample_prior(self):
        key, _ = random.split(key=self.keys[0], num=2)
        u = self.prior_sampler(
            key=key,
            shape=(self.no_samples, self.u_dim),
            mu=self.mu_prior,
            sigma=self.std_prior,
        )
        return u
    
    # @partial(jit, static_argnums=0)
    # def prior_pdf(self, x):
    #     if len(x.shape) > 1:
    #         x = x.squeeze()
    #     var_prior = self.std_prior ** 2
    #     cov_mat = jnp.diag(jnp.full(self.u_dim, var_prior))
    #     inv_cov_mat = jnp.diag(jnp.full(self.u_dim, 1 / var_prior))
    #     norm_factor = (1 / (jnp.prod(x))) * (1 / (2 * jnp.pi)) ** (int(self.u_dim) / 2) * (1 / jnp.prod(jnp.diag(cov_mat))) ** (0.5)
    #     exp_part = jnp.exp((-1 / 2) * jnp.sum((jnp.log(x) - self.mu_base) * (inv_cov_mat @ (jnp.log(x) - self.mu_base))))
    #     return norm_factor * exp_part
    
    # def log_prior_pdf(self, x):
    #     prior = self.prior_pdf(x)
    #     return jnp.log(prior)
    
    @partial(jit, static_argnums=0)
    def log_prior_pdf(self, x):
        x = jnp.squeeze(x)
        z = jnp.log(x)
        d = x.shape[0]

        var = self.std_prior ** 2
        resid = z - self.mu_base
        quad = 0.5 * (jnp.sum(resid * resid) / var)

        log_prior = -0.5 * d * jnp.log(2 * jnp.pi) - d * jnp.log(self.std_prior) - quad - jnp.sum(z)
        return log_prior
    
    @partial(jit, static_argnums=0)
    def prior_pdf(self, x):
        return jnp.exp(self.log_prior_pdf(x))


    @partial(jit, static_argnums=0)
    def log_likelihood_pdf(self, y, x):
        x = jnp.squeeze(x)
        y = jnp.ravel(y)

        # Lognormal requires y > 0
        bad = jnp.any(y <= 0.0)

        xt = self.solve_lv(self.y0_start, x)
        xt = jnp.ravel(jnp.abs(xt))  # consider removing abs later

        d = y.shape[0]
        var = self.sigma**2

        resid = jnp.log(y) - xt
        quad = jnp.sum(resid * resid) / var

        # log(1/prod(y)) = -sum(log(y))
        # Normal constant: -(d/2)log(2π var)
        log_like = -jnp.sum(jnp.log(y)) - 0.5 * d * jnp.log(2 * jnp.pi * var) - 0.5 * quad

        return log_like

    @partial(jit, static_argnums=0)
    def likelihood_pdf(self, y, x):
        return jnp.exp(self.log_likelihood_pdf(y, x))
    
    @partial(jit, static_argnums=0)
    def posterior(self, y, x): # At this stage, y should be the variable we condition on.
        if len(x.shape) > 1:
            x = x.squeeze()
        if len(y.shape) > 1:
            y = y.ravel()
        return self.likelihood_pdf(y, x) * self.prior_pdf(x)
    
    @partial(jit, static_argnums=0)
    def log_posterior(self, y, x):
        if len(x.shape) > 1:
            x = x.squeeze()
        if len(y.shape) > 1:
            y = y.ravel()
        return self.log_likelihood_pdf(y, x) + self.log_prior_pdf(x)


    @eqx.filter_jit
    def lv_ode(self, t, x, args):
        p1, p2 = x
        alpha, beta, gamma, delta = args
        dp1 = alpha * p1 - beta * p1 * p2
        dp2 = -gamma * p2 + delta * p1 * p2
        return jnp.array([dp1, dp2])

    @partial(jit, static_argnums=0)
    def solve_lv(self, y0, u):
        term = ODETerm(self.lv_ode)
        args = u
        ts = jnp.arange(self.t0 + 2, self.t1, step=2) # This takes care of the 1: indexing.
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            self.solver,
            self.t0,
            self.t1,
            self.dt0,
            y0,
            args=args,
            saveat=saveat,
        )
        ys = sol.ys
        return ys

    def solve_lv_large_t(self, u):
        term = ODETerm(self.lv_ode)
        args = u
        saveat = SaveAt(ts=self.tt)
        sol = diffeqsolve(
            term,
            self.solver,
            self.t0,
            self.t1,
            self.dt0,
            self.y0_start,
            args=args,
            saveat=saveat,
        )
        ys = sol.ys
        return ys

    def get_target_samples(self):
        key = self.keys[1]
        key1, _ = random.split(key=key, num=2)
        u = self.sample_prior()

        vmap_solve_lv = vmap(self.solve_lv, in_axes=(0,0))

        ys = vmap_solve_lv(self.y0, u)

        ys_noisy = self.likelihood_sampler(
            key=key1, shape=ys.shape, mu=jnp.log(ys), sigma=self.sigma
        )

        ys = ys_noisy.reshape(
            ys_noisy.shape[0], -1
        )  # This reshapes the tensored batches to be of size no_samples x 18

        # IMPORTANT: Getting rid of Nan values from solving ODE system
        mask = ~jnp.any(jnp.isnan(ys), axis=1)
        ys = ys[mask, :]
        u = u[mask, :]

        target_samples = jnp.hstack([ys, u])
        if self.normalize:
            self.ys_normalizer = UnitGaussianNormalizer(ys)
            self.u_normalizer = UnitGaussianNormalizer(u)
            ys_normalized = self.ys_normalizer.encode()
            u_normalized = self.u_normalizer.encode()
            target_samples_normalized = jnp.hstack([ys_normalized, u_normalized])
            return target_samples_normalized
        return target_samples
    
    def get_y_true(self, obs_noise: float = 0.35):
        subkey1, subkey2 = random.split(key=self.keys[2], num=2)
        vmap_solve_lv = vmap(self.solve_lv, in_axes=(0,0))
        ys = vmap_solve_lv(self.y0_start.reshape(-1,1).T, self.u_true.reshape(-1,1).T)
        y_true = ys.squeeze()
        y_true_noisy = self.prior_sampler(
            key=subkey1, shape=y_true.shape, mu=jnp.log(y_true), sigma=self.sigma,
        )
        y_true = y_true_noisy.reshape(-1,1)

        y_obs = self.prior_sampler(
            key=subkey2, shape=y_true.shape, mu=jnp.log(y_true), sigma=obs_noise,
        )

        if self.normalize:
            y_true_normalized = self.ys_normalizer.encode(y_true.T)
            y_obs_normalized = self.ys_normalizer.encode(y_obs.T)
            return (y_true_normalized, y_obs_normalized)
        return (y_true, y_obs)