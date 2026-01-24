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
from seaborn import kdeplot
import wandb
from ot.lp import wasserstein_1d
import argparse
import pickle

from triangular_transport.flows.sde_flow_trainer import NNSDE
from triangular_transport.flows.interpolants import linear_interpolant_noise, linear_interpolant_der_noise
from triangular_transport.flows.loss_functions import vec_field_loss, denoiser_loss
from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.methods.sampling import inf_train_gen
from triangular_transport.flows.dataloaders import (
    standard_gaussian_reference_sampler,
)

from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    vectorize_kfunc,
    get_sum_of_kernels,
)
from ksd import compute_ksd_jax

plt.style.use("ggplot")


def median_heuristic_sigma_jax(X, Y=None, max_points=5000, seed=0):
    X = jnp.asarray(X).reshape(X.shape[0], -1)
    if Y is not None:
        Y = jnp.asarray(Y).reshape(Y.shape[0], -1)
        Z = jnp.concatenate([X, Y], axis=0)
    else:
        Z = X

    n = Z.shape[0]
    if n > max_points:
        idx = jax.random.choice(
            jax.random.PRNGKey(seed), n, (max_points,), replace=False
        )
        Z = Z[idx]

    a2 = jnp.sum(Z * Z, axis=1, keepdims=True)
    D2 = a2 + a2.T - 2.0 * (Z @ Z.T)
    D2 = jnp.triu(D2, k=1)  # zero elsewhere
    D = jnp.sqrt(jnp.clip(D2[D2 > 0], a_min=0.0))
    sigma = jnp.median(D)
    return float(sigma)

def gamma_fn(t):
    return 0.1 * jnp.sqrt(2 * (t - t**2) + 1e-8)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()
RANK = args.run_id

run = wandb.init(
    project="2D Convergence - Banana - Transport - sde (debug)",
    name=f"run={RANK}-sde-converge",
)

# Kernel stuff for MMD

output_root0 = "converge_results_0_sde"
output_dir0 = os.path.join(output_root0, f"run_{RANK:02d}")
os.makedirs(output_dir0, exist_ok=True)

output_root1 = "converge_results_1_sde"
output_dir1 = os.path.join(output_root1, f"run_{RANK:02d}")
os.makedirs(output_dir1, exist_ok=True)

output_root4 = "converge_results_4_sde"
output_dir4 = os.path.join(output_root4, f"run_{RANK:02d}")
os.makedirs(output_dir4, exist_ok=True)

nsamples = 20000
seed = 1
rng = np.random.RandomState(seed)
samps0 = np.load("rej_samples_0.npy")[
    rng.choice(100000, size=(nsamples,)), :
]
samps1 = np.load("rej_samples_1.npy")[
    rng.choice(100000, size=(nsamples,)), :
]
samps4 = np.load("rej_samples_4.npy")[
    rng.choice(100000, size=(nsamples,)), :
]
us_base = rng.randn(nsamples, 1)
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
print(f"This is the base wd0: {base_wd1}")
print(f"This is the base wd4: {base_wd4}")
gamma = median_heuristic_sigma_jax(us_base, samps4)
k1 = get_gaussianRBF(gamma)
k2 = get_gaussianRBF(gamma - 6.0)
k3 = get_gaussianRBF(gamma - 3.0)
k4 = get_gaussianRBF(gamma + 3.0)
k5 = get_gaussianRBF(gamma + 6.0)
c1 = [0.2] * 5
kernels = [k1, k2, k3, k4, k5]
ker = get_sum_of_kernels(kernels, c1)

ker = vectorize_kfunc(ker)

ker_jit = jax.jit(ker)


@jax.jit
def MMD(X, Y):
    x_mean_emb = jnp.mean(ker(X, X))
    y_mean_emb = jnp.mean(ker(Y, Y))
    xy_mean_emb = jnp.mean(ker(X, Y))
    return x_mean_emb + y_mean_emb - 2 * xy_mean_emb


@jax.jit
def get_kme(X, Y):
    return (MMD(X, Y)) ** 2 / (jnp.mean(ker(Y, Y))) ** 2

a = 2
b = 0.1
sigma_x = 1
cond_no0 = 0.0
cond_no1 = -1.0
cond_no4 = -4.2
cond_vals = [cond_no0, cond_no1, cond_no4]

def log_density_V0(u):
    y = cond_no0
    scale = 1 / (2 * sigma_x**2)
    sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
    return jnp.squeeze(-scale * sum_part)
def log_density_V1(u):
    y = cond_no1
    scale = 1 / (2 * sigma_x**2)
    sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
    return jnp.squeeze(-scale * sum_part)
def log_density_V4(u):
    y = cond_no4
    scale = 1 / (2 * sigma_x**2)
    sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
    return jnp.squeeze(-scale * sum_part)


score_V0 = vmap(vmap(grad(log_density_V0)))
score_V1 = vmap(vmap(grad(log_density_V0)))
score_V4 = vmap(vmap(grad(log_density_V4)))
bandwidth = median_heuristic_sigma_jax(samps4)
base_ksd0 = compute_ksd_jax(samps0, score_V0, bandwidth=bandwidth)
base_ksd1 = compute_ksd_jax(samps1, score_V1, bandwidth=bandwidth)
base_ksd4 = compute_ksd_jax(samps4, score_V4, bandwidth=bandwidth)
# base_ksd_no_jax = compute_ksd(samps, score_V)
# print(f"This is the base_ksd: {base_ksd}")
# print(f"This is the base ksd without jax: {base_ksd_no_jax}")

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
mmd0_array = np.zeros(len(sample_no_list))
ksd0_array = np.zeros(len(sample_no_list))
wd1_array = np.zeros(len(sample_no_list))
mmd1_array = np.zeros(len(sample_no_list))
ksd1_array = np.zeros(len(sample_no_list))
wd4_array = np.zeros(len(sample_no_list))
mmd4_array = np.zeros(len(sample_no_list))
ksd4_array = np.zeros(len(sample_no_list))
epochs = 7000
rng2 = np.random.RandomState(RANK)
x1_data = inf_train_gen(data="banana", rng=rng2, batch_size=100000)
solver_args = {"saveat": "t1", "eps": 5e-3}
for i, sample_no in tqdm(enumerate(sample_no_list)):

    key = random.PRNGKey(i + 1 + RANK)
    key, key1, key2, key3, key4 = random.split(key, 5)
    key, subkey1 = random.split(key, 2)
    train_dim = sample_no
    batch_size = 2048
    # batch_size = hyperparams["batch_size"]
    batch_size = min(batch_size, train_dim)
    batch_size -= 1
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    steps = steps_per_epoch * epochs
    print_every = 10000
    yu_dimension = (1, 1)
    dim = yu_dimension[0] + yu_dimension[1]
    # hidden_layer_list = [256] * 4 if train_dim < 8000 else [1024] * 8
    # hidden_layer_list = [512] * 6
    hidden_layer_list_vel = [256] * 3
    hidden_layer_list_score = [512] * 4
    # hidden_layer_list = [1024] * 8
    target_data = x1_data[:sample_no, :]
    velocity = MLP(
        key=key1,
        dim=dim,
        time_varying=True,
        w=hidden_layer_list_vel,
        num_layers=len(hidden_layer_list_vel) + 1,
        activation_fn=jax.nn.gelu,  # GeLU worked well
    )
    score = MLP(
        key=key2,
        dim=dim,
        time_varying=True,
        w=hidden_layer_list_score,
        num_layers=len(hidden_layer_list_score) + 1,
        activation_fn=jax.nn.gelu,  # GeLU worked well
    )
    v_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=2000,
        decay_steps=steps,
        end_value=1e-5,
    )
    s_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=2000,
        decay_steps=steps,
        end_value=1e-5,
    )

    v_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(v_schedule)
    )
    s_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(s_schedule)
    )

    interpolant = linear_interpolant_noise
    interpolant_der = linear_interpolant_der_noise
    interpolant_args = {"t": None, "x1": None, "x0": None, "z": None}

    trainer = NNSDE(
        target_density=None,
        velocity=velocity,
        score=score,
        v_optimizer=v_optimizer,
        s_optimizer=s_optimizer,
        interpolant=interpolant,
        interpolant_der=interpolant_der,
        reference_sampler=standard_gaussian_reference_sampler,
        v_loss=vec_field_loss,
        s_loss=denoiser_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
    )

    trainer.train(
        x1_data=target_data,
        train_dim=train_dim,
        batch_size=batch_size,
        steps=steps,
        x0_data=None,
        print_every=print_every,
    )
    print("Calculating MMD and SWD and KSD...")
    cond_samples = trainer.conditional_sample(
        cond_values=cond_vals,
        nsamples=nsamples,
        u0_cond=None,
        solver_args=solver_args,
        gamma=gamma_fn,
    )
    mmd_iter_array = np.zeros(3)
    wd_iter_array = np.zeros(3)
    ksd_iter_array = np.zeros(3)
    for k, cond_sample in enumerate(cond_samples):
        us_gen = cond_samples[k][:, 1:2]
        if k == 0:
            samps = samps0
            score_V = score_V0
            base_wd = base_wd0
            base_ksd = base_ksd0
        elif k == 1:
            samps = samps1
            score_V = score_V1
            base_wd = base_wd1
            base_ksd = base_ksd1
        elif k == 2:
            samps = samps4
            score_V = score_V4
            base_wd = base_wd4
            base_ksd = base_ksd4
        mmd_iter_array[k] = get_kme(us_gen, samps)
        wd_iter_array[k] = (
            wasserstein_1d(np.array(us_gen), samps, p=2)
            / base_wd
        )
        ksd_iter_array[k] = compute_ksd_jax(us_gen, score_V, bandwidth=bandwidth) / base_ksd
    mmd0_array[i] = mmd_iter_array[0]
    wd0_array[i] = wd_iter_array[0]
    ksd0_array[i] = ksd_iter_array[0]
    mmd1_array[i] = mmd_iter_array[1]
    wd1_array[i] = wd_iter_array[1]
    ksd1_array[i] = ksd_iter_array[1]
    mmd4_array[i] = mmd_iter_array[2]
    wd4_array[i] = wd_iter_array[2]
    ksd4_array[i] = ksd_iter_array[2]
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
    # wandb.log({"relative error (swd) - mcmc": swd_mcmc[i]}, step=sample_no)
    wandb.log({"relative error (mmd): 0": mmd0_array[i]}, step=sample_no)
    wandb.log({"relative error (mmd): -1": mmd1_array[i]}, step=sample_no)
    wandb.log({"relative error (mmd): -4.2": mmd4_array[i]}, step=sample_no)
    wandb.log({"relative error (ksd): 0": ksd0_array[i]}, step=sample_no)
    wandb.log({"relative error (ksd): -1": ksd1_array[i]}, step=sample_no)
    wandb.log({"relative error (ksd): -4.2": ksd4_array[i]}, step=sample_no)
    print(f"This is the relative MMD error on 0.0: {mmd0_array[i]}")
    print(f"This is the relative SWD error on 0.0: {wd0_array[i]}")
    print(f"This is the relative KSD error on 0.0: {ksd0_array[i]}")
    print(f"This is the relative MMD error on -1.0: {mmd1_array[i]}")
    print(f"This is the relative SWD error on -1.0: {wd1_array[i]}")
    print(f"This is the relative KSD error on -1.0: {ksd1_array[i]}")
    print(f"This is the relative MMD error on -4.2: {mmd4_array[i]}")
    print(f"This is the relative SWD error on -4.2: {wd4_array[i]}")
    print(f"This is the relative KSD error on -4.2: {ksd4_array[i]}")

print("Successfully trained all models and now saving results!")
np.save(os.path.join(output_dir0, f"nn_conv_mmd_{RANK}.npy"), mmd0_array)
np.save(os.path.join(output_dir0, f"nn_conv_wd_{RANK}.npy"), wd0_array)
np.save(os.path.join(output_dir0, f"nn_conv_ksd_{RANK}.npy"), ksd0_array)

np.save(os.path.join(output_dir1, f"nn_conv_mmd_{RANK}.npy"), mmd1_array)
np.save(os.path.join(output_dir1, f"nn_conv_wd_{RANK}.npy"), wd1_array)
np.save(os.path.join(output_dir1, f"nn_conv_ksd_{RANK}.npy"), ksd1_array)

np.save(os.path.join(output_dir4, f"nn_conv_mmd_{RANK}.npy"), mmd4_array)
np.save(os.path.join(output_dir4, f"nn_conv_wd_{RANK}.npy"), wd4_array)
np.save(os.path.join(output_dir4, f"nn_conv_ksd_{RANK}.npy"), ksd4_array)

