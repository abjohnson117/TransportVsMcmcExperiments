import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import jit
import jax
import numpy as np
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

output_dir = f"mcmc_results/mcmc_results_{args.run_id}"
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
burn_in = 1
scale = 0.3
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.0, high=3.0, size=(budget, ))
conditioning_list = [4 ** i for i in range(9)]
chain_list = list(reversed(conditioning_list))

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
        print(f"This is the conditioning variable: {cond_no}")
        @jit
        def tanh_density(u):
            shift = jnp.tanh(cond_no)
            base = (1.0 / scale) * jnp.exp(-(u - shift) / scale)
            return jnp.where(u < shift, 0.0, base)

        def log_density(u):
            return np.log(tanh_density(cond_no, u))
        if cond_no <= -1:
            minval = -0.2
            maxval = 0.0
        elif -1 < cond_no <= 0:
            minval = 0.0
            maxval = 0.1
        elif 0 < cond_no <= 1:
            minval = 0.95
            maxval = 1.05
        else:
            minval = 1.1
            maxval = 1.25
        x0 = jax.random.uniform(jax.random.key(k), shape=1, minval=minval, maxval=maxval)
        adapt_mcmc = AdaptiveMCMC(
            target_density=tanh_density,
            alpha_function=alpha,
            seed=np.random.choice(1000000),
            train_dim=1,
            steps=chain_length,
            name="Adaptive - Conditional",
            std_err=0.6,
            iter_step_adapt=1,
            cond_no=cond_no,
            burn_in=burn_in,
            x0=x0,
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