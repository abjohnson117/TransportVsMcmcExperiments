import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, random
import optax
from tqdm.auto import tqdm
from typing import Callable
import pickle
import equinox as eqx

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
from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.flows.dataloaders import gaussian_reference_sampler
from triangular_transport.kernels.kernel_tools import get_gaussianRBF, vectorize_kfunc

import json
import h5py

import argparse

plt.style.use("ggplot")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_default_device", jax.devices()[1])

RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))


def read_data_h5(path="data.h5"):
    with h5py.File(path, "r") as f:
        targets = f["/target"][...]
        data = f["/data"][...]
    return targets, data

gamma = 10.0
rbf_kernel = get_gaussianRBF(gamma)

ker = vectorize_kfunc(rbf_kernel)


@jax.jit
def MMD(X, Y):
    return jnp.mean(ker(X, X)) + jnp.mean(ker(Y, Y)) - 2 * jnp.mean(ker(X, Y))


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
hmala_samps = np.load("training_dataset/chain_30/hmala_samples_grid_delta.npy").reshape(
    nsamples, 33, 33
)

# Load training data
train_dim = 50000
nx = ny = 33
flat_length = nx * ny
ys = (np.load("training_dataset/solutions_grid_delta.npy"))[:train_dim]
us = (np.load("training_dataset/parameters_delta.npy"))[:train_dim, :].reshape(
    train_dim, flat_length
)

us_ref = us.copy()
np.random.shuffle(us_ref)

us_test = (np.load("training_dataset/parameters_delta.npy"))[
    train_dim : train_dim + nsamples, :
].reshape(nsamples, flat_length)
np.random.shuffle(us_test)

ys_normalizer = UnitGaussianNormalizer(ys)
ys_normalized = ys_normalizer.encode()

mean_us = us.mean(axis=0)
X = us - mean_us

U, S, Vt = np.linalg.svd(us / (np.sqrt(train_dim - 1)), full_matrices=False)
V = Vt.T

expl_var = (S**2) / (S**2).sum()
k = np.searchsorted(np.cumsum(expl_var), 0.98) + 1
# k = 30

V = V[:, :k]
S = S[:k]

alpha = 0.1


def pca_encode(b):
    # whitened coeffs z
    return (b - mean_us) @ V / S


def pca_decode(z):
    # undo whitening
    return mean_us + (z * S) @ V.T


sample_no_list = [2**i for i in range(1, 15)]
sample_no_list.append(nsamples)
sample_no_list.append(30000)
sample_no_list.append(40000)
sample_no_list.append(train_dim)

mmd_array = jnp.zeros(len(sample_no_list))
for i, sample_no in tqdm(enumerate(sample_no_list)):
    print("Starting training...")
    y = ys_normalized[: (sample_no + 1), :]
    u = us[: (sample_no + 1), :]
    u_ref = us_ref[: (sample_no + 1), :]
    u_test = us_test[: (sample_no + 1), :]

    us_pca = pca_encode(us)
    us_ref_pca = pca_encode(us_ref)
    us_test_pca = pca_encode(us_test)

    target_data = jnp.hstack([ys_normalized, us_pca])
    ref_data = jnp.hstack([ys_normalized, us_ref_pca])

    key = random.PRNGKey(seed=np.random.choice(1000))
    key1, key2 = random.split(key=key, num=2)
    if sample_no < 128:
        batch_size = sample_no
    else:
        batch_size = 128
    steps = 50000
    print_every = 5000
    yu_dimension = (100, k.item())
    dim = yu_dimension[0] + yu_dimension[1]
    hidden_layer_list = [512] * 4
    model = MLP(
        key=key2,
        dim=dim,
        time_varying=True,
        w=hidden_layer_list,
        num_layers=len(hidden_layer_list) + 1,
        activation_fn=jax.nn.gelu,  # GeLU worked well
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=2_000,
        decay_steps=steps,
        end_value=1e-5,
    )
    optimizer = optax.adamw(schedule)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    interpolant = linear_interpolant
    interpolant_der = linear_interpolant_der
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
    ys_test_normalized = jnp.full(
        (nsamples, ytrue_flat_normalized.shape[0]), ytrue_flat_normalized
    )
    x0_test = jnp.hstack([ys_test_normalized, us_test_pca])

    cond_values = [tuple(ytrue_flat_normalized)]
    cond_samples = trainer.conditional_sample(
        cond_values=cond_values, x0_cond=x0_test, nsamples=20000
    )
    all_samples = cond_samples[0][0]

    u_samples = all_samples[:, yu_dimension[0] :]
    u_samples = pca_decode(u_samples)
    u_samples = u_samples.reshape(nsamples, nx, ny)

    print("Calculating MMD...")
    mmd_array.at[i].set(MMD(u_samples, hmala_samps))

print("Successfully trained all models and now saving results!")
np.save(os.path.join(output_root, "nn_sample_convergence.npy"))
