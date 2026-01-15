import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import jit
import jax
from ksd import compute_ksd_jax
from jax import vmap, grad
import matplotlib.pyplot as plt
import numpy as np
from ot.sliced import sliced_wasserstein_distance as swd
from tqdm.auto import tqdm
import json
import time
import argparse

from triangular_transport.mcmc.adaptive_mcmc import AdaptiveMCMC

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

output_dir = f"mcmc_results_{args.run_id}"
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
burn_in = 1
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-5, high=0.4, size=(budget, ))
conditioning_list = [4 ** i for i in range(9)]
chain_list = list(reversed(conditioning_list))

a = 2
b = 0.1
sigma_x = 1

def alpha(x, w, log_density):
    log_w = log_density(w)
    log_x = log_density(x)
    return jnp.minimum(0.0, log_w - log_x)

start = time.perf_counter()
for i, chain_length in enumerate(tqdm(chain_list)):
    num_cond_vars = conditioning_list[i]
    cond_vars = conditioning_ys[rng.choice(budget, size=num_cond_vars, replace=False)]
    mcmc_samps = np.zeros((chain_length, num_cond_vars))
    for k, cond_no in enumerate(cond_vars):
        @jit
        def density_V(u):
            y = cond_no
            scale = 1 / (2 * sigma_x**2)
            sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
            return jnp.squeeze(jnp.exp(-scale * sum_part))
        adapt_mcmc = AdaptiveMCMC(
            target_density=density_V,
            alpha_function=alpha,
            seed=np.random.choice(1000000),
            train_dim=1,
            steps=chain_length,
            name="Adaptive - Conditional",
            std_err=0.6,
            iter_step_adapt=1,
            cond_no=cond_no,
            burn_in=burn_in,
        )
        adapt_mcmc.fit(print_every=20000)
        mcmc_samps[:, k] = (adapt_mcmc.samples).reshape(-1)
    
    output_path = os.path.join(output_dir, f"mcmc_samps_{chain_length}_{num_cond_vars}.npy")
    np.save(output_path, mcmc_samps)
    output_path_cond_vars = os.path.join(output_dir, f"cond_vars_{num_cond_vars}.npy")
    np.save(output_path_cond_vars, cond_vars)
    print("Saved successfully!")
        


elapsed = time.perf_counter() - start
timings = {
    "mcmc_time": elapsed,
    "timestamp": time.time(),
}

with open(os.path.join(output_dir, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)