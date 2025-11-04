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

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from ConfigSpace.conditions import InCondition
from smac import HyperparameterOptimizationFacade, Scenario

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_default_device", jax.devices()[1])


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


class SiOdeSmac:
    def __init__(
        self,
        train_dim: int,
        steps: int,
        train_data: jax.Array,
        x0_data: jax.Array,
        yu_dimension: tuple,
        interpolant_args: dict,
    ):
        self.train_dim = train_dim
        self.steps = steps
        self.train_data = train_data
        self.x0_data = x0_data
        self.yu_dimension = yu_dimension
        self.interpolant_args = interpolant_args

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        interpolant = Categorical(
            "interpolant",
            ["linear_interpolant", "trig_interpolant", "sigmoid_interpolant"],
            default="linear_interpolant",
        )
        interpolant_der = Categorical(
            "interpolant_der",
            ["linear_interpolant_der", "trig_interpolant_der", "sigmoid_interpolant_der"],
            default="linear_interpolant_der",
        )
        activation = Categorical(
            "activation",
            ["gelu", "selu", "celu", "silu"],
            default="gelu",
        )
        optimizer = Categorical(
            "optimizer",
            ["adamw", "adam", "adagrad", "adamaxw"],
            default="adamw",
        )
        hidden_layer = Integer("hidden_layer", (100, 1000), default=512, log=True)
        num_hidden_layers = Integer("num_hidden_layers", (2, 10), default=4, log=True)
        batch_size = Integer("batch_size", (100, 2000), default=128, log=True)
        peak_value = Float("peak_value", (1e-4, 1e-2), default=3e-4, log=True)

        cs.add(
            [
                interpolant,
                interpolant_der,
                activation,
                hidden_layer,
                num_hidden_layers,
                batch_size,
                optimizer,
                peak_value,
            ]
        )
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = dict(config)

        if config_dict["interpolant"] == "linear_interpolant":
            interpolant = linear_interpolant
        elif config_dict["interpolant"] == "trig_interpolant":
            interpolant = trig_interpolant
        elif config_dict["interpolant"] == "sigmoid_interpolant":
            interpolant = sigmoid_interpolant
        
        if config_dict["interpolant_der"] == "linear_interpolant_der":
            interpolant_der = linear_interpolant_der
        elif config_dict["interpolant_der"] == "trig_interpolant_der":
            interpolant_der = trig_interpolant_der
        elif config_dict["interpolant_der"] == "sigmoid_interpolant_der":
            interpolant_der = sigmoid_interpolant_der

        if config_dict["activation"] == "gelu":
            activation = jax.nn.gelu
        elif config_dict["activation"] == "silu":
            activation = jax.nn.silu
        elif config_dict["activation"] == "celu":
            activation = jax.nn.celu
        elif config_dict["activation"] == "selu":
            activation = jax.nn.selu

        if config_dict["optimizer"] == "adamw":
            opt = optax.adamw
        elif config_dict["optimizer"] == "adam":
            opt = optax.adam
        elif config_dict["optimizer"] == "adagrad":
            opt = optax.adagrad
        elif config_dict["optimizer"] == "adamaxw":
            opt = optax.adagrad

        key = random.PRNGKey(seed=seed)
        key1, key2 = random.split(key=key, num=2)
        batch_size = config_dict["batch_size"]
        steps = self.steps
        yu_dimension = self.yu_dimension
        dim = yu_dimension[0] + yu_dimension[1]
        hidden_layer_list = [config_dict["hidden_layer"]] * (
            config_dict["num_hidden_layers"]
        )
        model = MLP(
            key=key2,
            dim=dim,
            time_varying=True,
            w=hidden_layer_list,
            num_layers=len(hidden_layer_list) + 1,
            activation_fn=activation,
        )
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config_dict["peak_value"],
            warmup_steps=2_000,
            decay_steps=steps,
            end_value=1e-5,
        )
        # opt = config_dict["optimizer"]
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), opt(schedule))
        # interpolant = config_dict["interpolant"]
        # interpolant_der = config_dict["interpolant_der"]

        trainer = NNTrainer(
            target_density=None,
            model=model,
            optimizer=optimizer,
            interpolant=interpolant,
            interpolant_der=interpolant_der,
            reference_sampler=gaussian_reference_sampler,
            loss=vec_field_loss,
            interpolant_args=self.interpolant_args,
            yu_dimension=yu_dimension,
        )
        trainer.train(
            train_data=self.train_data,
            train_dim=self.train_dim,
            batch_size=batch_size,
            steps=steps,
            x0_data=self.x0_data,
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
        print("Calculating SWD...")
        loss = (
            swd(
                u_samples.reshape(nsamples, nx * ny),
                jnp.asarray(hmala_samps),
                n_projections=n_projections,
                seed=SEED,
            )
            / base_swd
        )
        wandb.log({"relative error (swd)": loss})

        return loss


configs = {"dataset": "darcy_flow_si_hmala"}

run = wandb.init(
    # set the wandb project where this run will be logged
    project="Poisson - SI hyperparam tuning",
    config=configs,
)

sep = "\n" + "#" * 80 + "\n"
output_root = "hyperparam_results"
os.makedirs(output_root, exist_ok=True)

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

SEED = 42
n_projections = 4024
random_idxs = np.random.choice(len(us_ref_pca), (20000,))
base_swd = swd(
    us_ref[random_idxs, :], hmala_samps, n_projections=n_projections, seed=SEED
)
print(f"This is the base swd: {base_swd}")

interpolant_args = {"t": None, "x1": None, "x0": None}
steps = 10000
yu_dimension = (ys_normalized.shape[1], k.item())
train_data = jnp.hstack([ys_normalized, us_pca])
x0_data = jnp.hstack([ys_normalized, us_ref_pca])

regressor = SiOdeSmac(
    train_dim=train_dim,
    steps=steps,
    train_data=train_data,
    x0_data=x0_data,
    yu_dimension=yu_dimension,
    interpolant_args=interpolant_args,
)

scenario = Scenario(
    regressor.configspace,
    n_trials=500,
    deterministic=True,
)

initial_design = HyperparameterOptimizationFacade.get_initial_design(
    scenario, n_configs=5
)

print("Starting smac routine...")
smac = HyperparameterOptimizationFacade(
    scenario,
    regressor.train,
    initial_design=initial_design,
    overwrite=True,
)

incumbent = smac.optimize()

default_loss = smac.validate(regressor.configspace.get_default_configuration())
print(f"Default loss: {default_loss}")

incumbent_loss = smac.validate(incumbent)
print(f"Incumbent loss: {incumbent_loss}")

best_hyperparams = dict(incumbent)

print(f"These are the best hyperparameters selected: {best_hyperparams}")
save_path = os.path.join(output_root, "best_hyperparams.yaml")
with open(save_path, "w") as f:
    yaml.dump(best_hyperparams, f, default_flow_style=False)

print(f"Best hyperparameters saved to {save_path}")
