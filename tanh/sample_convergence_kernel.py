import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

plt.style.use("ggplot")

best_hyperparams = {
    'lam_value': 0.001,
    'gamma1': 0.12,
    'gamma2': 0.35,
    'gamma3': 1.4,
}

def exp_reference_sampler(
    key: random.PRNGKey, shape: tuple[int, int]
):
    samples = 0.3 * (random.exponential(key=key, shape=shape))
    return samples

@vmap
def sigmoid_interpolant(t: jnp.array, x1: jnp.array, x0: jnp.array):
    return (1 - sigmoid(t)) * x0 + sigmoid(t) * x1

def sigmoid(t: float) -> float:
    return jax.nn.sigmoid(27 * (t - 0.35)) # Changed this to 25

sigmoid_dot = vmap(grad(sigmoid))

@vmap
def sigmoid_interpolant_der(t: jnp.array, x1: jnp.array, x0: jnp.array):
    return sigmoid_dot(t) * (x1 - x0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()
RANK = args.run_id

run = wandb.init(
    project="2D Convergence - Tanh - Transport - kernel",
    name=f"run={RANK}-ode-converge",
)

# Kernel stuff for MMD

output_root0 = "converge_results_0_kernel"
output_dir0 = os.path.join(output_root0, f"run_{RANK:02d}")
os.makedirs(output_dir0, exist_ok=True)

output_root2 = "converge_results_2_kernel"
output_dir2 = os.path.join(output_root2, f"run_{RANK:02d}")
os.makedirs(output_dir2, exist_ok=True)

output_root3 = "converge_results_3_kernel"
output_dir3 = os.path.join(output_root2, f"run_{RANK:02d}")
os.makedirs(output_dir2, exist_ok=True)

nsamples = 100000
seed = 1
rng = np.random.RandomState(seed)
samps0 = np.load("samps_0.npy")
samps2 = np.load("samps_2.npy")
samps3 = np.load("samps_3.npy")
# us_base = rng.randn(nsamples, 1) * 2
key_base = random.key(seed)
us_base = np.array(exp_reference_sampler(key_base, (nsamples,)))

print("About to calculate wd...")
base_wd0 = wasserstein_1d(
    us_base,
    samps0.squeeze(),
    p=2,
)
base_wd2 = wasserstein_1d(
    us_base,
    samps2.squeeze(),
    p=3,
)
base_wd3 = wasserstein_1d(
    us_base,
    samps3.squeeze(),
    p=2,
)
print(f"This is the base wd0: {base_wd0}")
print(f"This is the base wd2: {base_wd2}")
print(f"This is the base wd3: {base_wd3}")

cond_no0 = 0.0
cond_no1 = 2.0
cond_no4 = -3.0
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
wd2_array = np.zeros(len(sample_no_list))
wd3_array = np.zeros(len(sample_no_list))
rng2 = np.random.RandomState(RANK)
ys = rng2.uniform(low=-3.0, high=3.0, size=(100000, 1))
us = np.tanh(ys) + rng2.gamma(shape=1.0, scale=0.3, size=(100000, 1))
x1_data = np.hstack([ys, us])
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

    interpolant = sigmoid_interpolant
    interpolant_der = sigmoid_interpolant_der
    interpolant_args = {"t": None, "x1": None, "x0": None}

    velocity = KernelTrainer(
        target_density=None,
        model=model,
        optimizer=optax.adam(1e-5),
        interpolant=interpolant,
        interpolant_der=interpolant_der,
        reference_sampler=exp_reference_sampler,
        loss=kernel_vec_field_loss,
        interpolant_args=interpolant_args,
        seed=seed+43,
        yu_dimension=yu_dimension,
        cond=True,
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
            samps = samps2
            base_wd = base_wd2
        elif k == 2:
            samps = samps3
            base_wd = base_wd3
        wd_iter_array[k] = (
            wasserstein_1d(np.array(us_gen), samps, p=2)
            / base_wd
        )
    wd0_array[i] = wd_iter_array[0]
    wd2_array[i] = wd_iter_array[1]
    wd3_array[i] = wd_iter_array[2]
    wandb.log(
        {
            "relative error (wd): 0": wd0_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (wd): 2": wd2_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (wd): -3": wd3_array[i]
        },
        step=sample_no,
    )
    print(f"This is the relative SWD error on 0.0: {wd0_array[i]}")
    print(f"This is the relative SWD error on 2.0: {wd2_array[i]}")
    print(f"This is the relative SWD error on -3.0: {wd3_array[i]}")

np.save(os.path.join(output_dir0, f"nn_conv_wd_{RANK}.npy"), wd0_array)
np.save(os.path.join(output_dir2, f"nn_conv_wd_{RANK}.npy"), wd2_array)
np.save(os.path.join(output_dir3, f"nn_conv_wd_{RANK}.npy"), wd3_array)
print("Successfully trained all models and now saving results!")