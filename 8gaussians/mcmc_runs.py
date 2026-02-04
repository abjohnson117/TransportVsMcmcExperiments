import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import jit
import jax
import numpy as np
from tqdm.auto import tqdm
import json
import time
import argparse
from emcee import EnsembleSampler
import emcee

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
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.1, high=3.1, size=(budget, ))
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
        def make_eight_gaussians_cond(y, scale=4.0):
            # Precompute centers as a JAX array (constants)
                base = jnp.array([
                    [ 1.0,  0.0],
                    [-1.0,  0.0],
                    [ 0.0,  1.0],
                    [ 0.0, -1.0],
                    [ 1.0/jnp.sqrt(2.0),  1.0/jnp.sqrt(2.0)],
                    [ 1.0/jnp.sqrt(2.0), -1.0/jnp.sqrt(2.0)],
                    [-1.0/jnp.sqrt(2.0),  1.0/jnp.sqrt(2.0)],
                    [-1.0/jnp.sqrt(2.0), -1.0/jnp.sqrt(2.0)],
                ])  # (8,2)

                centers = (scale / jnp.sqrt(2.0)) * base  # matches your construction

                factor = 1.0 / jnp.sqrt(jnp.pi)

                def eight_gaussians_cond(u):
                    # u can be scalar or shape (batch,)
                    u = jnp.asarray(u)
                    x = jnp.stack([jnp.full_like(u, y), u], axis=-1)   # (..., 2)

                    # squared distances to each center, broadcasted
                    # x[..., None, :] -> (..., 1, 2); centers[None, ...] -> (1, 8, 2)
                    diffs = x[..., None, :] - centers[None, :, :]      # (..., 8, 2)
                    norm_part = jnp.sum(diffs * diffs, axis=-1)        # (..., 8)

                    exp_part = jnp.exp(-scale * norm_part)             # (..., 8)
                    return factor * jnp.sum(exp_part, axis=-1)         # (...)

                def log_eight_gaussians_cond(u):
                    # if you ever hit underflow, you can add tiny eps:
                    return jnp.log(eight_gaussians_cond(u) + 1e-30)

                return eight_gaussians_cond, log_eight_gaussians_cond
            
        eight_gaussians_cond, log_eight_gaussians_cond = make_eight_gaussians_cond(y=cond_no)
        if chain_length <= 1:
            adapt_mcmc = AdaptiveMCMC(
                target_density=eight_gaussians_cond,
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
        else:
            # rng2 = np.random.RandomState(42) #TODO : Think about this some more. This may be creating the same chain every time. Nevermind, it didn't since I didn't use rng2 later down.
            if chain_length >= 16:
                initial = np.random.randn(16, 1)
            else:
                initial = np.random.randn(4, 1)
            nwalkers, ndim = initial.shape
            nsteps = chain_length // nwalkers

            sampler = EnsembleSampler(nwalkers, ndim, log_eight_gaussians_cond, moves=[
                (emcee.moves.DEMove(), 0.5),
                (emcee.moves.DESnookerMove(), 0.15),
                (emcee.moves.KDEMove(), 0.35),
            ])
            sampler.run_mcmc(initial, nsteps, progress=True);
    
            samps = np.vstack(sampler.chain)
            mcmc_samps[:, k] = samps.reshape(-1)
    
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