import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import jit
import jax
from jax import vmap, grad
import matplotlib.pyplot as plt
import numpy as np
from ot.lp import wasserstein_1d as wd
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

output_root = "budget_results"
output_dir = os.path.join(output_root, f"ode_{args.run_id}")
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-5, high=0.4, size=(budget, ))
conditioning_list = [4 ** i for i in range(9)]
gen_sample_list = list(reversed(conditioning_list))
rej_samples = np.load(f"rej_results/_{seed}/rej_samps.npy")
# nn_samples = np.load("nn_results_no_loop/ode/nn_samps.npy")
nn_samples = (np.load(f"nn_results/ode_{seed}/nn_samps.npy").T)[:, 1::2]

wd_nn_array = np.zeros(len(conditioning_list))
wd_mcmc_array = np.zeros(len(conditioning_list))
wd_nn_array_std = np.zeros(len(conditioning_list))
wd_mcmc_array_std = np.zeros(len(conditioning_list))
for i, cond_num in enumerate(tqdm(conditioning_list)):
    #TODO: from here we are going to read in all of the relevant files: rej_samples, cond_vars, nn_samps, and mcmc_samps. Can we index by i and -i? I think so, so only one for loop to read in files.
    # We need to get the arg index for the cond_vars values to then subsample from conditioning_ys. Then we choose the corresponding rej_samples and calculate W1d and KSD.
    nsamples = gen_sample_list[i]
    cond_vars = np.load(f"mcmc_results/mcmc_results_{seed}/cond_vars_{cond_num}.npy")
    # nn_samps = np.load(f"nn_results/ode/nn_samps_{nsamples}_{cond_num}.npy")
    mcmc_samps = np.load(f"mcmc_results/mcmc_results_{seed}/mcmc_samps_{nsamples}_{cond_num}.npy")
    lookup = {v: i for i, v in enumerate(conditioning_ys)}
    # print(lookup)
    idxs = np.array([lookup[v] for v in cond_vars])
    rej_samps = rej_samples[idxs, :].T
    nn_samps = nn_samples[:, idxs]

    bad_cols = np.where(np.isnan(nn_samps).any(axis=0))[0]
    keep_cols = np.setdiff1d(np.arange(nn_samps.shape[1]), bad_cols)
    nn_samps = nn_samps[:, keep_cols]
    mcmc_samps = mcmc_samps[:, keep_cols]
    rej_samps = rej_samps[:, keep_cols]
    cond_vars = cond_vars[keep_cols]
    cond_num = cond_vars.shape[0]

    nn_array = np.zeros(cond_num)
    mcmc_array = np.zeros(cond_num)
    for j in range(cond_num):
        nn_array[j] = wd(nn_samps[:, j].reshape(-1), rej_samps[:, j].reshape(-1))
        mcmc_array[j] = wd(mcmc_samps[:, j].reshape(-1), rej_samps[:, j].reshape(-1))
    wd_nn_array[i] = np.mean(nn_array)
    wd_mcmc_array[i] = np.mean(mcmc_array)
    wd_nn_array_std[i] = np.std(nn_array)
    wd_mcmc_array_std[i] = np.std(mcmc_array)

np.save(os.path.join(output_dir, "wd_nn_array_mean.npy"), wd_nn_array)
np.save(os.path.join(output_dir, "wd_nn_array_std.npy"), wd_nn_array_std)
np.save(os.path.join(output_root, f"wd_mcmc_array_mean_{seed}.npy"), wd_mcmc_array)
np.save(os.path.join(output_root, f"wd_mcmc_array_std_{seed}.npy"), wd_mcmc_array_std)


