from typing import Callable
from numpy.typing import NDArray
from jax import random, vmap, jit, scipy
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
        dt0: float = 1e-2,
        normalize: bool = False,
        solver: Callable = Dopri5(),
        log_y_mean: float | NDArray | None = None,
        log_y_std: float | NDArray | None = None,
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
        # self.dt0 = 0.1 # Tested this with Hsu... should be enough to get good solutions.
        # self.dt0 = 1e-2
        self.dt0 = dt0
        self.true = u_true

        self.normalize = normalize
        if self.normalize:
            self.ys_normalizer = None
            self.us_normalizer = None

        self.solver = solver
        self.log_y_mean = log_y_mean
        self.log_y_std = log_y_std

    def sample_prior(self):
        key, _ = random.split(key=self.keys[0], num=2)
        u = self.prior_sampler(
            key=key,
            shape=(self.no_samples, self.u_dim),
            mu=self.mu_prior,
            sigma=self.std_prior,
        )
        return u
    
    @partial(jit, static_argnums=0)
    def log_prior_pdf(self, x):
        r"""
        Here, x must be in log-space!!! In other words, x $\sim$ N(mu_base, 0.5^2 I_4) so that we're working with just Gaussians.
        """
        d = x.shape[0]
        x = jnp.squeeze(x)
        mean = self.mu_base
        cov = self.std_prior ** 2 * jnp.eye(d)

        return scipy.stats.multivariate_normal.logpdf(x=x, mean=mean, cov=cov)
    
    @partial(jit, static_argnums=0)
    def prior_pdf(self, x):
        return jnp.exp(self.log_prior_pdf(x))

    @partial(jit, static_argnums=0)
    def _normalize_log_observation(self, z):
        """
        z is already in log-data space, shape (obs_dim,).
        """
        z = jnp.ravel(z)
        eps = 1e-12
        return (z - self.log_y_mean) / (self.log_y_std + eps)


    @partial(jit, static_argnums=0)
    def _forward_log_observation(self, x):
        """
        Map log-parameters x to logged ODE observations.
        """
        x = jnp.ravel(x)
        xt = self.solve_lv(self.y0_start, jnp.exp(x))
        xt = jnp.ravel(xt)

        # Safer than abs: preserves monotonicity better and avoids reflecting negatives.
        xt = jnp.clip(xt, a_min=1e-12)
        xt = jnp.log(xt)
        return xt


    @partial(jit, static_argnums=0)
    def _normalized_forward_log_observation(self, x):
        xt_log = self._forward_log_observation(x)
        return self._normalize_log_observation(xt_log)


    @partial(jit, static_argnums=0)
    def log_likelihood_pdf(self, y, x):
        """
        y should already be in log-data space.
        x is in log-parameter space.
        """
        x = jnp.ravel(x)
        y = jnp.ravel(y)

        y_norm = self._normalize_log_observation(y)
        xt_norm = self._normalized_forward_log_observation(x)

        # If original model is:
        #   log Y | x ~ N(log G(exp(x)), sigma^2 I),
        # then after dividing each coordinate by log_y_std,
        # covariance becomes sigma^2 * diag(1 / log_y_std^2).
        eps = 1e-12
        var_diag = (self.sigma ** 2) / ((self.log_y_std + eps) ** 2)
        cov = jnp.diag(var_diag)

        return scipy.stats.multivariate_normal.logpdf(x=y_norm, mean=xt_norm, cov=cov)


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
        if self.log_y_mean is None:
            self.log_y_mean = jnp.mean(jnp.log(ys), axis=0).ravel()
        if self.log_y_std is None:
            self.log_y_std = jnp.std(jnp.log(ys), axis=0).ravel()

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