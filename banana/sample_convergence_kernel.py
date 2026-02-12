import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, vmap, random
import optax
import diffrax
from seaborn import kdeplot
import wandb
from ot.lp import wasserstein_1d
import argparse
import pickle

from triangular_transport.flows.flow_trainer import (
    KernelTrainer,
)

from triangular_transport.flows.interpolants import (
    linear_interpolant,
    linear_interpolant_der,
)
from triangular_transport.flows.loss_functions import kernel_vec_field_loss
from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.methods.sampling import inf_train_gen
from triangular_transport.flows.dataloaders import (
    gaussian_reference_sampler
)

from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    vectorize_kfunc,
    get_sum_of_kernels,
    get_prod_gaussianRBF,
)
from ksd import compute_ksd_jax

plt.style.use("ggplot")

best_hyperparams = {
    'lam_value': 0.001784240471,
    'gamma1': 0.0996334585856,
    'gamma2': 0.243442877083,
    'gamma3': 1.2982789204244
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()
RANK = args.run_id

run = wandb.init(
    project="2D Convergence - Banana - Transport - kernel",
    name=f"run={RANK}-ode-converge",
)

# Kernel stuff for MMD

output_root0 = "converge_results_0_kernel"
output_dir0 = os.path.join(output_root0, f"run_{RANK:02d}")
os.makedirs(output_dir0, exist_ok=True)

output_root1 = "converge_results_1_kernel"
output_dir1 = os.path.join(output_root1, f"run_{RANK:02d}")
os.makedirs(output_dir1, exist_ok=True)

output_root4 = "converge_results_4_kernel"
output_dir4 = os.path.join(output_root4, f"run_{RANK:02d}")
os.makedirs(output_dir4, exist_ok=True)

nsamples = 100000
seed = 1
rng = np.random.RandomState(seed)
samps0 = np.load("rej_samples_0.npy")
samps1 = np.load("rej_samples_1.npy")
samps4 = np.load("rej_samples_4.npy")
us_base = rng.randn(nsamples, 1) * 2
print("About to calculate wd...")
base_wd0 = wasserstein_1d(
    us_base,
    samps0,
    p=2,
)
base_wd1 = wasserstein_1d(
    us_base,
    samps1,
    p=2,
)
base_wd4 = wasserstein_1d(
    us_base,
    samps4,
    p=2,
)
print(f"This is the base wd0: {base_wd0}")
print(f"This is the base wd1: {base_wd1}")
print(f"This is the base wd4: {base_wd4}")

a = 2
b = 0.1
sigma_x = 1
cond_no0 = 0.0
cond_no1 = -1.0
cond_no4 = -4.2
cond_vals = [cond_no0, cond_no1, cond_no4]

sample_no_list = [2**i for i in range(1, 15)]
sample_no_list.append(20000)
sample_no_list.append(30000)
sample_no_list.append(40000)
sample_no_list.append(50000)
sample_no_list.append(60000)
sample_no_list.append(70000)
sample_no_list.append(80000)
sample_no_list.append(90000)
sample_no_list.append(100000)

wd0_array = np.zeros(len(sample_no_list))
wd1_array = np.zeros(len(sample_no_list))
wd4_array = np.zeros(len(sample_no_list))
rng2 = np.random.RandomState(RANK)
x1_data = inf_train_gen(data="banana", rng=rng2, batch_size=100000)
solver_args = {"solver": diffrax.Dopri5(), "max_steps": 50000, "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)}
for i, sample_no in tqdm(enumerate(sample_no_list)):


    train_dim = sample_no
    num_batches = 3000
    key = random.key(seed=seed)
    key, key1 = random.split(key=key, num=2)
    batch_size = 1000
    batch_size = min(batch_size, train_dim)
    batch_size -= 1
    num_pivots = 1000
    num_pivots = min(num_pivots, train_dim)
    num_pivots -= 1
    lam = best_hyperparams["lam_value"]
    print_every = 5000
    yu_dimension = (1, 1)
    target_data = x1_data[:sample_no, :]
    dim = yu_dimension[0] + yu_dimension[1]

    def model():
        pass
    gammas = {v: best_hyperparams[v] for v in best_hyperparams if "gamma" in v}
    k = get_prod_gaussianRBF(**gammas)
    kvv = vectorize_kfunc(k)

    interpolant = linear_interpolant
    interpolant_der = linear_interpolant_der
    interpolant_args = {"t": None, "x1": None, "x0": None}
    reference_sampler_args = {"mu": 0, "sigma": 2.0}

    velocity = KernelTrainer(
        target_density=None,
        model=model,
        optimizer=optax.adam(1e-5),
        interpolant=interpolant,
        interpolant_der=interpolant_der,
        reference_sampler=gaussian_reference_sampler,
        loss=kernel_vec_field_loss,
        interpolant_args=interpolant_args,
        seed=seed+43,
        yu_dimension=yu_dimension,
        cond=True,
        reference_sampler_args=reference_sampler_args,
    )

    velocity.train(
        x1_data=target_data,
        train_dim=train_dim,
        batch_size=batch_size,
        num_batches=num_batches,
        num_pivots=num_pivots,
        x0_data=None,
        lam=lam,
        k=k,
        num_M_iters=1,

    );
    print("Calculating MMD and SWD and KSD...")
    cond_samples = velocity.conditional_sample(
        cond_values=cond_vals,
        nsamples=nsamples,
        u0_cond=None,
        solver_args=solver_args,
    )
    # mmd_iter_array = np.zeros(2)
    wd_iter_array = np.zeros(3)
    # ksd_iter_array = np.zeros(2)
    for k, cond_sample in enumerate(cond_samples):
        us_gen = cond_samples[k][:, 1:2]
        if k == 0:
            samps = samps0
            base_wd = base_wd0
        elif k == 1:
            samps = samps1
            base_wd = base_wd1
        elif k == 2:
            samps = samps4
            base_wd = base_wd4
        wd_iter_array[k] = (
            wasserstein_1d(np.array(us_gen), samps, p=2)
            / base_wd
        )
    wd0_array[i] = wd_iter_array[0]
    wd1_array[i] = wd_iter_array[1]
    wd4_array[i] = wd_iter_array[2]
    wandb.log(
        {
            "relative error (wd): 0": wd0_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (wd): -1": wd1_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (wd): -4.2": wd4_array[i]
        },
        step=sample_no,
    )
    print(f"This is the relative SWD error on 0.0: {wd0_array[i]}")
    print(f"This is the relative SWD error on -1.0: {wd1_array[i]}")
    print(f"This is the relative SWD error on -4.2: {wd4_array[i]}")

np.save(os.path.join(output_dir0, f"nn_conv_wd_{RANK}.npy"), wd0_array)
np.save(os.path.join(output_dir1, f"nn_conv_wd_{RANK}.npy"), wd1_array)
np.save(os.path.join(output_dir4, f"nn_conv_wd_{RANK}.npy"), wd4_array)
print("Successfully trained all models and now saving results!")