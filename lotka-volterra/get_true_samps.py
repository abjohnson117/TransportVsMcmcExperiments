import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from tqdm.auto import tqdm
import json
import time
import jax
from jax import jit, random
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from triangular_transport.mcmc.adaptive_mcmc import AdaptiveMCMC
from triangular_transport.flows.dataloaders import log_normal_reference_sampler
from lv import LV
jax.config.update("jax_enable_x64", True)

# @partial(jit, static_argnums=2)
def alpha(x, w, log_density):
    log_w = log_density(w) + jnp.sum(jnp.log(w))
    log_x = log_density(x) + jnp.sum(jnp.log(x))
    print(f"This is log w: {log_w}")
    print(f"This is log x: {log_x}")
    return jnp.minimum(0.0, log_w - log_x)

seed = np.random.choice(100000)
print(seed)
no_samples = 1
u_true = jnp.array([0.83194674, 0.04134147, 1.0823151, 0.03991483]) # TODO: Need to figure out how to get u_true here from a y_obs.
sigma = jnp.sqrt(0.1).item()
lv_sampler = LV(
    seed=seed,
    no_samples=no_samples,
    prior_sampler=log_normal_reference_sampler,
    likelihood_sampler=log_normal_reference_sampler,
    normalize=False,
    u_true=u_true,
    sigma=sigma,
)
ys = lv_sampler.solve_lv(lv_sampler.y0_start, u_true)      # shape (len(ts), 2)
xt = jnp.abs(ys).ravel()     
key = random.key(123) 
yobs = jnp.log(xt) + sigma * random.normal(key, shape=xt.shape)
# yobs = np.load("y_obs.npy")
pi = lv_sampler.posterior
def post(x):
    return pi(yobs, x)
def log_post(x):
    return lv_sampler.log_posterior(yobs, x)

# neg_post = lambda x : -pi(yobs, x)
def neg_post(x):
    return -pi(yobs, x)
xmap = minimize(neg_post, u_true, method="BFGS")

x0 = xmap.x
nfev = xmap.nfev.item()
nsteps = 500


adapt_mcmc = AdaptiveMCMC(
    target_density=post,
    log_density=log_post,
    alpha_function=alpha,
    seed=np.random.choice(1000000),
    train_dim=4,
    steps=nsteps,
    name="Adaptive - Conditional",
    std_err=0.6,
    iter_step_adapt=1,
    cond_no=yobs,
    burn_in=100,
    x0=x0,
)
adapt_mcmc.fit(print_every=100, log_update=True)

us = adapt_mcmc.samples
np.save("true_us_obs.npy", us)
