import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Callable, List


import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, random
import optax
from tqdm.auto import tqdm
from typing import Callable
import pickle
import equinox as eqx
from ot.sliced import sliced_wasserstein_distance as swd

import os

from triangular_transport.flows.flow_trainer import (
    NNTrainer,
)

from triangular_transport.flows.interpolants import (
    linear_interpolant,
    linear_interpolant_der,
    trig_interpolant,
    trig_interpolant_der,
    sigmoid_interpolant,
    sigmoid_interpolant_der,
)
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer

# from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.dataloaders import gaussian_reference_sampler
from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    vectorize_kfunc,
    get_sum_of_kernels,
)

import json
import h5py

import argparse
import wandb

# plt.style.use("ggplot")


class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]  # main hidden layers
    skips: List[
        eqx.nn.Linear | None
    ]  # projections for residuals (or None for identity)
    out: eqx.nn.Linear
    activation_fn: List[Callable]

    def __init__(
        self,
        key: jax.random.PRNGKey,
        dim: int,
        out_dim: int | None = None,
        num_layers: int = 4,
        activation_fn: List[Callable] | Callable = jax.nn.gelu,
        w: int | List[int] = 64,
        time_varying: bool = False,
    ):
        if out_dim is None:
            out_dim = dim

        # normalize activation list
        if isinstance(activation_fn, list):
            if len(activation_fn) == 1:
                activation_fn *= num_layers - 1
        else:
            activation_fn = [activation_fn] * (num_layers - 1)
        self.activation_fn = activation_fn

        # normalize widths
        if isinstance(w, list):
            if len(w) == 1:
                w *= num_layers - 1
            widths = w
        else:
            widths = [w] * (num_layers - 1)

        k = jax.random.split(key, 2 * num_layers)  # enough keys

        in_dim0 = dim + (1 if time_varying else 0)

        # build hidden layers + skip projections
        self.layers = []
        self.skips = []
        in_dim = in_dim0
        for i, width in enumerate(widths):
            self.layers.append(eqx.nn.Linear(in_dim, width, key=k[i]))
            # projection: identity if dims match, else linear map
            if in_dim == width:
                self.skips.append(None)  # treat as identity in __call__
            else:
                self.skips.append(eqx.nn.Linear(in_dim, width, key=k[i + num_layers]))
            in_dim = width

        # output layer
        self.out = eqx.nn.Linear(in_dim, out_dim, key=k[-1])

    def __call__(self, x):
        for layer, skip, act in zip(self.layers, self.skips, self.activation_fn):
            h = act(layer(x))
            s = x if skip is None else skip(x)
            x = h + s  # projected residual
        return self.out(x)


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_default_device", jax.devices()[1])

# RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
RANK = 0
SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))

configs = {
    "dataset": "darcy_flow_si_hmala",
    "hidden_layer": 623,
    "interpolant": sigmoid_interpolant,
    "interpolant_der": sigmoid_interpolant_der,
    "activation_fn": jax.nn.gelu,
    "batch_size": 379,
    "num_hidden_layers": 10,
    "optimizer": optax.adagrad,
    "peak_value": 0.0064240033048,
}

run = wandb.init(
    # set the wandb project where this run will be logged
    project="Poisson - SI v hMALA, no PCA",
    config=configs,
)


def read_data_h5(path="data.h5"):
    with h5py.File(path, "r") as f:
        targets = f["/target"][...]
        data = f["/data"][...]
    return targets, data


def get_pca_fns(us):
    mean_us = us.mean(axis=0)
    X = us - mean_us

    U, S, Vt = np.linalg.svd(
        X / (np.sqrt(train_dim - 1)), full_matrices=False
    )  # Changed from us to X
    V = Vt.T

    expl_var = (S**2) / (S**2).sum()
    k = np.searchsorted(np.cumsum(expl_var), 0.98) + 1

    V = V[:, :k]
    S = S[:k]
    V_res, S_res = V[:, k:], S[k:]

    def pca_encode(b):
        # whitened coeffs z
        return (b - mean_us) @ V / S

    def pca_decode(z):
        # undo whitening
        return mean_us + (z * S) @ V.T

    def sample_extra(n=1):
        eps = np.random.randn(n, S_res.shape[0])  # ~ N(0, I)
        coeffs = eps * S_res  # ~ N(0, diag(S_res^2))
        return coeffs @ V_res.T

    def extra_cov():
        return V_res @ np.diag(S_res**2) @ V_res.T

    return pca_encode, pca_decode, k, sample_extra, extra_cov


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


def gamma_from_sigma_jax(sigma):
    return 1.0 / (2.0 * (sigma**2))


sep = "\n" + "#" * 80 + "\n"
output_root = "convergence_results"
# output_dir = os.path.join(output_root, f"chain_{RANK:02d}")
# output_dir = os.path.join(output_root, f"chain_{RANK:02d}")
# output_dir = os.path.join(output_root, )
output_dir = os.path.join(output_root, f"run_{RANK:02d}")
os.makedirs(output_dir, exist_ok=True)

with open("poisson.yaml") as fid:
    inargs = yaml.full_load(fid)

utrue = np.load("training_dataset/true_param_grid.npy")
ytrue = np.load("training_dataset/true_state_grid.npy")
map_est = np.load("training_dataset/map_param_grid.npy")
targets, yobs = read_data_h5()

# Load h-MALA samples
nsamples = inargs["MCMC"]["nsamples"] - inargs["MCMC"]["burnin"]
nx = ny = 33
flat_length = nx * ny
hmala_root = "training_dataset"
chains = []
for i in tqdm(range(40)):
    hmala_dir = f"chain_{i:02d}"
    hmala_path = os.path.join(hmala_root, hmala_dir, "hmala_samples_grid.npy")
    hmala_samps_chain = np.load(hmala_path).reshape(nsamples, flat_length)
    thinned_samps = hmala_samps_chain[::40, :]
    chains.append(thinned_samps)
hmala_samps = np.vstack(
    chains
)  # This is a chain with 20,000 independently drawn samples

# Load training data
train_dim = 50000
ys = (np.load("training_dataset/solutions_grid_delta.npy"))[:train_dim]
us = (np.load("training_dataset/parameters_delta.npy"))[:train_dim, :].reshape(
    train_dim, flat_length
)

us_ref = (np.load("training_dataset/parameters_delta.npy"))[
    train_dim : train_dim * 2, :
].reshape(train_dim, flat_length)
np.random.shuffle(us_ref)

us_test = (np.load("training_dataset/parameters_delta.npy"))[
    2 * train_dim : 2 * train_dim + nsamples, :
].reshape(nsamples, flat_length)
np.random.shuffle(us_test)

ys_normalizer = UnitGaussianNormalizer(ys)
ys_normalized = ys_normalizer.encode()

# pca_encode_total, pca_decode_total, k = get_pca_fns(us)
# us_pca_total = pca_encode(us)
# us_ref_pca_total = pca_encode(us_ref)
# us_test_pca = pca_encode_total(us_test)
# hmala_pca = pca_encode_total(hmala_samps)

pca_encode, pca_decode, k, sample_extra, extra_cov = get_pca_fns(us_ref)
extra_samp = sample_extra(1).reshape(nx, ny)
extra_pca_cov = extra_cov()
extra_pca_var = np.diag(extra_pca_cov).reshape(nx, ny)

us_pca = pca_encode(us)
us_pca = jnp.asarray(us_pca)
us_ref_pca = pca_encode(us_ref)
us_ref_pca = jnp.asarray(us_ref_pca)
us_test_pca = pca_encode(us_test)
us_test_pca = jnp.asarray(us_test_pca)
hmala_pca = pca_encode(hmala_samps)
hmala_pca = jnp.asarray(hmala_pca)

gamma = median_heuristic_sigma_jax(us_ref_pca, hmala_pca)
# gamma = gamma_from_sigma_jax(sigma)
# gamma = 5.0
print(f"This is the median heuristic bandwidth: {gamma}")

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
def mean_emb(X):
    return jnp.mean(ker(X, X))


@jax.jit
def xy_mean_emb(X, Y):
    return jnp.mean(ker(X, Y))


@jax.jit
def get_kme(X, Y):
    return (MMD(X, Y)) ** 2 / (jnp.mean(ker(Y, Y))) ** 2


print(
    f"This is the MMD between the prior and hmala: {MMD(us_ref_pca[np.random.choice(len(us_ref_pca), (20000,)), :], hmala_pca)}"
)

seed = 42
n_projections = 4024
random_idxs = np.random.choice(len(us_ref_pca), (20000,))
base_swd = swd(
    us_ref[random_idxs, :], hmala_samps, n_projections=n_projections, seed=seed
)
print(f"This is the base swd: {base_swd}")

sample_no_list = [2**i for i in range(1, 15)]
sample_no_list.append(nsamples)
sample_no_list.append(30000)
sample_no_list.append(40000)
sample_no_list.append(train_dim)
# sample_no_list = [2, 8, 20000, train_dim]

mmd_array = np.zeros(len(sample_no_list))
rel_err_array = np.zeros(len(sample_no_list))
u_mean_array = np.zeros((len(sample_no_list), nx, ny))
u_var_array = np.zeros((len(sample_no_list), nx, ny))
mmd_array_no_pca = np.zeros(len(sample_no_list))
rel_error_no_pca = np.zeros(len(sample_no_list))
for i, sample_no in tqdm(enumerate(sample_no_list)):
    print("Starting training...")
    y = ys_normalized[1 : (sample_no + 1), :]
    u_pca = us_pca[1 : (sample_no + 1), :]
    u_ref_pca = us_ref_pca[1 : (sample_no + 1), :]

    # pca_encode, pca_decode, k = get_pca_fns(u)
    # us_pca = pca_encode(u)
    # us_ref_pca = pca_encode(u_ref)
    # us_test_pca = pca_encode(us_test)

    target_data = jnp.hstack([y, u_pca])
    ref_data = jnp.hstack([y, u_ref_pca])
    print(
        f"This is the size of target and ref data: {target_data.shape} and {ref_data.shape}"
    )
    with open(f"hyperparam_results/iteration_{i}/best_hyperparams.pkl", "rb") as f:
        hyperparams = pickle.load(f)

    key = random.PRNGKey(seed=np.random.choice(1000))
    key1, key2 = random.split(key=key, num=2)
    # if sample_no < 128:
    #     batch_size = sample_no
    # else:
    #     batch_size = configs["batch_size"]
    batch_size = hyperparams["batch_size"]
    # steps = 50000
    steps = 15000
    print_every = 5000
    yu_dimension = (100, k.item())
    dim = yu_dimension[0] + yu_dimension[1]
    hidden_layer_list = [hyperparams["hidden_layer"]] * hyperparams["num_hidden_layers"]
    if hyperparams["activation"] == "gelu":
        activation = jax.nn.gelu
    elif hyperparams["activation"] == "silu":
        activation = jax.nn.silu
    elif hyperparams["activation"] == "celu":
        activation = jax.nn.celu
    elif hyperparams["activation"] == "selu":
        activation = jax.nn.selu

    model = MLP(
        key=key2,
        dim=dim,
        time_varying=True,
        w=hidden_layer_list,
        num_layers=len(hidden_layer_list) + 1,
        activation_fn=activation,  # GeLU worked well
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=hyperparams["peak_value"],
        warmup_steps=2_000,
        decay_steps=steps,
        end_value=1e-5,
    )
    # lr = 1e-4
    # optimizer = optax.adamw(schedule)
    # optimizer = optax.adamw(lr)
    if hyperparams["optimizer"] == "adamw":
        opt = optax.adamw
    elif hyperparams["optimizer"] == "adam":
        opt = optax.adam
    elif hyperparams["optimizer"] == "adagrad":
        opt = optax.adagrad
    elif hyperparams["optimizer"] == "adamaxw":
        opt = optax.adagrad

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), opt(schedule))
    # interpolant = configs["interpolant"]
    # interpolant_der = configs["interpolant_der"]
    if hyperparams["interpolant"] == "linear_interpolant":
        interpolant = linear_interpolant
    elif hyperparams["interpolant"] == "trig_interpolant":
        interpolant = trig_interpolant
    elif hyperparams["interpolant"] == "sigmoid_interpolant":
        interpolant = sigmoid_interpolant

    if hyperparams["interpolant_der"] == "linear_interpolant_der":
        interpolant_der = linear_interpolant_der
    elif hyperparams["interpolant_der"] == "trig_interpolant_der":
        interpolant_der = trig_interpolant_der
    elif hyperparams["interpolant_der"] == "sigmoid_interpolant_der":
        interpolant_der = sigmoid_interpolant_der
    interpolant_args = {"t": None, "x1": None, "x0": None}

    trainer = NNTrainer(
        target_density=None,
        model=model,
        optimizer=optimizer,
        interpolant=interpolant,
        interpolant_der=interpolant_der,
        reference_sampler=gaussian_reference_sampler,
        loss=vec_field_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
    )
    trainer.train(
        train_data=target_data,
        train_dim=train_dim,
        batch_size=batch_size,
        steps=steps,
        x0_data=ref_data,
    )
    ytrue_flat = yobs.copy()
    ytrue_flat_normalized = ys_normalizer.encode(ytrue_flat)

    cond_values = ytrue_flat_normalized
    cond_samples = trainer.conditional_sample(
        cond_values=cond_values, u0_cond=us_test_pca, nsamples=20000
    )
    all_samples = cond_samples

    u_samples_gen = all_samples[:, yu_dimension[0] :]
    u_samples_gen = jnp.asarray(u_samples_gen)
    u_samples = pca_decode(u_samples_gen)
    # u_samples_pca = pca_encode_total(u_samples) # To make sure in exactly the correct PCA basis

    u_samples = u_samples.reshape(nsamples, nx, ny)
    u_samples = jnp.asarray(u_samples)
    u_samples += extra_samp
    u_mean = jnp.mean(u_samples, axis=0)
    u_var = jnp.var(u_samples, axis=0)
    u_var += extra_pca_var
    u_mean_array[i] = u_mean
    u_var_array[i] = u_var

    print("Calculating MMD...")
    mmd_array[i] = MMD(u_samples_gen, hmala_pca)
    # rel_err_array[i] = (
    #     swd(u_samples_gen, hmala_pca, n_projections=n_projections, seed=seed) / base_swd
    # )
    rel_err_array[i] = (
        swd(
            u_samples.reshape(nsamples, nx * ny),
            jnp.asarray(hmala_samps),
            n_projections=n_projections,
            seed=seed,
        )
        / base_swd
    )
    wandb.log({"relative error (swd)": rel_err_array[i]}, step=sample_no)
    # ref_indices = np.random.choice(20000)
    # rel_err_array[i] = (MMD(u_samples_gen, us_ref_pca[ref_indices, :]) ** 2) / (MMD(hmala_samps, us_ref_pca[ref_indices, :]) ** 2)
    mmd_array_no_pca[i] = MMD(u_samples.reshape(nsamples, nx * ny), hmala_samps)
    rel_error_no_pca[i] = get_kme(u_samples.reshape(nsamples, nx * ny), hmala_samps)
    wandb.log({"relative error (mmd)": rel_error_no_pca[i]}, step=sample_no)
    print(f"This is the MMD: {mmd_array[i]}")
    print(f"This is the calculated relative error: {rel_err_array[i]}")
    print(f"This is the MMD (no PCA): {mmd_array_no_pca[i]}")
    print(f"This is the calculated relative error (no PCA): {rel_error_no_pca[i]}")
    print(
        f"This is the sliced Wasserstein distance: {swd(u_samples_gen, hmala_pca, n_projections=100)}"
    )
    print(
        f"This is the relative sliced Wasserstein error: {(swd(u_samples_gen, hmala_pca, n_projections=100)) ** 2 / (np.var(hmala_samps)) ** 2}"
    )
    print(f"This is the X mean embedding: {mean_emb(u_samples_gen)}")
    print(f"This is the Y mean embedding: {mean_emb(hmala_pca)}")
    print(f"This is the XY mean embedding: {xy_mean_emb(u_samples_gen, hmala_pca)}")
    print(
        f"These are the kernel values for gen samples: {ker_jit(u_samples_gen, u_samples_gen)}"
    )
    print(
        f"These are the kernel values for hmala samples: {ker_jit(hmala_pca, hmala_pca)}"
    )
    np.save(os.path.join(output_dir, f"u_samps_{i}.npy"), u_samples_gen)

print("Successfully trained all models and now saving results!")
np.save(os.path.join(output_dir, "nn_sample_convergence.npy"), mmd_array)
np.save(os.path.join(output_dir, "nn_sample_convergence_rel_error.npy"), rel_err_array)

print("Making plots...")
fig, ax = plt.subplots(3, 6, figsize=(16, 16))
# fig, ax = plt.subplots(2, 2, figsize=(16, 16))
l = 0

for i in range(len(ax)):
    for j in range(len(ax[0, :])):
        im = ax[i][j].imshow(u_mean_array[l], origin="lower", interpolation="bilinear")
        ax[i][j].set_title(rf"Mean with $n = {sample_no_list[l]}$")
        fig.colorbar(im, ax=ax[i][j], fraction=0.046, pad=0.04)

        l += 1

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "param_field_means.png")
)  # TODO: Potentially change this to pdf later.
plt.close(fig)

fig, ax = plt.subplots(3, 6, figsize=(16, 16))
# fig, ax = plt.subplots(2, 2, figsize=(16, 16))
l = 0

for i in range(len(ax)):
    for j in range(len(ax[0, :])):
        im = ax[i][j].imshow(u_var_array[l], origin="lower", interpolation="bilinear")
        ax[i][j].set_title(rf"Var with $n = {sample_no_list[l]}$")
        fig.colorbar(im, ax=ax[i][j], fraction=0.046, pad=0.04)

        l += 1

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "param_field_variances.png")
)  # TODO: Potentially change this to pdf later.
plt.close(fig)

plt.plot(sample_no_list, rel_err_array)
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel(r"$N$")
plt.ylabel(
    r"$\frac{\mathrm{MMD}^2(\mu^N_{\mathrm{SI}}, \mu^N_{\mathrm{hMALA}})}{||\mu^N_{\mathrm{hMALA}}||^2_{\mathcal{H}}}$"
)
plt.title("Relative Error vs Sample Size")
plt.savefig(os.path.join(output_dir, "rel_error_plot.png"))
