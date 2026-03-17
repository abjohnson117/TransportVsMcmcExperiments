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
    log_w = log_density(w) #w is in log space
    log_x = log_density(x) #x is in log space
    # Don't need proposal densities here bc they end up cancelling out.
    return jnp.minimum(0.0, log_w - log_x)

seed = np.random.choice(100000)
no_samples = 1
u_true = jnp.array([0.83194674, 0.04134147, 1.0823151, 0.03991483]) # TODO: Need to figure out how to get u_true here from a y_obs.
sigma = 1.0
mean = np.load("../log_y_mean.npy")
std = np.load("../log_y_std.npy")
lv_sampler = LV(
    seed=0,
    no_samples=no_samples,
    prior_sampler=log_normal_reference_sampler,
    likelihood_sampler=log_normal_reference_sampler,
    normalize=False,
    u_true=u_true,
    sigma=sigma,
    dt0=0.1,
    log_y_mean=mean,
    log_y_std=std
)
yobs = np.log(np.load("y_rare.npy"))
# rng = np.random.RandomState(42)
# yobs = yobs + 0.001 * rng.randn(yobs.shape[0],)

pi = lv_sampler.posterior
log_pi = lv_sampler.log_posterior
def post(x):
    return pi(yobs, x)
def log_post(x):
    return log_pi(yobs, x)

def neg_post(x):
    return -log_pi(yobs, x)
x_inits = []
n_restarts = 25
u_dim = 4
init_key = random.key(seed=9)
for i in range(n_restarts):
    init_key, subkey = random.split(init_key)
    x_inits.append(
        random.normal(subkey, shape=(u_dim,)) * lv_sampler.std_prior + lv_sampler.mu_base
    )

best_val = jnp.inf
best_result = None
for i, x_init in enumerate(tqdm(x_inits)):
    result = minimize(neg_post, x_init, method="BFGS", tol=1e-8)
    val = neg_post(result.x)
    print(f"Init {i}: neg_post = {val:.4f}, exp(x) = {jnp.exp(result.x)}")
    if val < best_val:
        best_val = val
        best_result = result

x0 = best_result.x
print(f"\nBest x0: {jnp.exp(x0)}  (neg_post = {best_val:.4f})")
nsteps = 500000


adapt_mcmc = AdaptiveMCMC(
    target_density=post,
    log_density=log_post,
    alpha_function=alpha,
    seed=np.random.choice(1000000),
    train_dim=4,
    steps=nsteps,
    name="Adaptive - Conditional",
    std_err=0.05,
    iter_step_adapt=100,
    cond_no=yobs,
    burn_in=3000,
    x0=x0,
)
adapt_mcmc.fit(print_every=100, log_update=False)

us = adapt_mcmc.samples
np.save("true_us_rare_obs_3.npy", us)
