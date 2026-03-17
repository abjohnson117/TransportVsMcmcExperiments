import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
sigma = jnp.sqrt(0.1).item()
lv_sampler = LV(
    seed=seed,
    no_samples=no_samples,
    prior_sampler=log_normal_reference_sampler,
    likelihood_sampler=log_normal_reference_sampler,
    normalize=False,
    u_true=u_true,
    sigma=sigma,
    dt0=0.25,
)
yobs = np.log(np.load("y_rare.npy")) # This was solved with many time steps -- same as what will be used for transport.
rng = np.random.RandomState(42)
yobs = yobs + 0.001 * rng.randn(yobs.shape[0],) 

pi = lv_sampler.posterior
log_pi = lv_sampler.log_posterior
def post(x):
    return pi(yobs, x)
def log_post(x):
    return log_pi(yobs, x)

def neg_post(x):
    return -log_pi(yobs, x)
init_key = random.key(seed=2)
x_init = jnp.log(log_normal_reference_sampler(
    key=init_key,
    shape=u_true.shape,
    mu=lv_sampler.mu_base,
    sigma=lv_sampler.std_prior,
))
print(f"This is x_init: {x_init}")
print(f"This is the neg posterior val: {neg_post(x_init)}")
xmap = minimize(neg_post, x_init, method="BFGS")


x0 = xmap.x
p_key = random.key(seed=4)
x0 = x0 + 0.15 * random.normal(key=p_key, shape=x0.shape)
print(f"This is x0: {jnp.exp(x0)}")
nfev = xmap.nfev.item()
print(f"This is the number of function evals: {nfev}")
nsteps = 500000


adapt_mcmc = AdaptiveMCMC(
    target_density=post,
    log_density=log_post,
    alpha_function=alpha,
    seed=np.random.choice(1000000),
    train_dim=4,
    steps=nsteps,
    name="Adaptive - Conditional",
    std_err=0.005,
    iter_step_adapt=700,
    cond_no=yobs,
    burn_in=3000,
    x0=x0,
)
adapt_mcmc.fit(print_every=10000, log_update=False)

us = adapt_mcmc.samples
np.save("true_us_rare_obs_4.npy", us)
