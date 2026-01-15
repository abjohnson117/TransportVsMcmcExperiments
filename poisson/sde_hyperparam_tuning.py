import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Callable, List
import gc

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60"

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, random
import optax
from tqdm.auto import tqdm
import pickle
import equinox as eqx
from ot.sliced import sliced_wasserstein_distance as swd

from triangular_transport.flows.sde_flow_trainer import NNSDE

from triangular_transport.flows.loss_functions import vec_field_loss, denoiser_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer

from triangular_transport.flows.dataloaders import gaussian_reference_sampler

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

    return pca_encode, pca_decode, k, sample_extra, extra_cov, S


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


def gamma_fn(t):
    return 0.1 * jnp.sqrt(2 * (t - t**2) + 1e-8)


gamma_vmap = vmap(gamma_fn)

gammadot = vmap(grad(gamma_fn))


@vmap
def linear_interpolant(t, x1, x0, z):
    return t * x1 + (1 - t) * x0 + gamma_vmap(t) * z


@vmap
def linear_interpolant_der(t, x1, x0, z):
    return x1 - x0 + gammadot(t) * z


@vmap
def trig_interpolant(t: jnp.array, x1: jnp.array, x0: jnp.array, z):
    return (
        jnp.cos((jnp.pi / 2) * t) * x0
        + jnp.sin((jnp.pi / 2) * t) * x1
        + gamma_vmap(t) * z
    )


@vmap
def trig_interpolant_der(t: jnp.array, x1: jnp.array, x0: jnp.array, z):
    return (jnp.pi / 2) * (
        -jnp.sin((jnp.pi / 2) * t) * x0 + jnp.cos((jnp.pi / 2) * t) * x1
    ) + gammadot(t) * z


@vmap
def sigmoid_interpolant(t: jnp.array, x1: jnp.array, x0: jnp.array, z):
    return (1 - sigmoid(t)) * x0 + sigmoid(t) * x1 + gamma_vmap(t) * z


def sigmoid(t: float) -> float:
    return jax.nn.sigmoid(10 * (t - 0.5))  # Changed this to 25


@vmap
def sigmoid_interpolant_der(t, x1, x0, z):
    return sigmoid_dot(t) * (x1 - x0) + gamma_vmap(t) + z


sigmoid_dot = vmap(grad(sigmoid))


class SiSdeSmac:
    def __init__(
        self,
        train_dim: int,
        steps: int,
        train_data: jax.Array,
        x0_data: jax.Array,
        yu_dimension: tuple,
        interpolant_args: dict,
        reference_sampler_args: dict,
    ):
        self.train_dim = train_dim
        self.steps = steps
        self.train_data = train_data
        self.x0_data = x0_data
        self.yu_dimension = yu_dimension
        self.interpolant_args = interpolant_args
        self.reference_sampler_args = reference_sampler_args

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
            [
                "linear_interpolant_der",
                "trig_interpolant_der",
                "sigmoid_interpolant_der",
            ],
            default="linear_interpolant_der",
        )
        v_activation = Categorical(
            "v_activation",
            ["gelu", "selu", "celu", "silu"],
            default="gelu",
        )
        s_activation = Categorical(
            "s_activation",
            ["gelu", "selu", "celu", "silu"],
            default="gelu",
        )
        v_optimizer = Categorical(
            "v_optimizer",
            ["adamw", "adam", "adagrad", "adamaxw"],
            default="adamw",
        )
        s_optimizer = Categorical(
            "s_optimizer",
            ["adamw", "adam", "adagrad", "adamaxw"],
            default="adamw",
        )
        batch_size = Integer("batch_size", (100, 2000), default=128, log=True)

        v_hidden_layer = Integer("v_hidden_layer", (100, 512), default=512, log=True)
        v_num_hidden_layers = Integer(
            "v_num_hidden_layers", (2, 6), default=4, log=True
        )
        v_peak_value = Float("v_peak_value", (1e-4, 1e-2), default=3e-4, log=True)
        s_hidden_layer = Integer("s_hidden_layer", (100, 512), default=512, log=True)
        s_num_hidden_layers = Integer(
            "s_num_hidden_layers", (2, 6), default=4, log=True
        )
        s_peak_value = Float("s_peak_value", (1e-4, 1e-2), default=3e-4, log=True)

        cs.add(
            [
                interpolant,
                interpolant_der,
                v_activation,
                s_activation,
                v_hidden_layer,
                s_hidden_layer,
                v_num_hidden_layers,
                s_num_hidden_layers,
                batch_size,
                v_optimizer,
                s_optimizer,
                v_peak_value,
                s_peak_value,
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

        if config_dict["v_activation"] == "gelu":
            v_activation = jax.nn.gelu
        elif config_dict["v_activation"] == "silu":
            v_activation = jax.nn.silu
        elif config_dict["v_activation"] == "celu":
            v_activation = jax.nn.celu
        elif config_dict["v_activation"] == "selu":
            v_activation = jax.nn.selu

        if config_dict["s_activation"] == "gelu":
            s_activation = jax.nn.gelu
        elif config_dict["s_activation"] == "silu":
            s_activation = jax.nn.silu
        elif config_dict["s_activation"] == "celu":
            s_activation = jax.nn.celu
        elif config_dict["s_activation"] == "selu":
            s_activation = jax.nn.selu

        if config_dict["v_optimizer"] == "adamw":
            v_opt = optax.adamw
        elif config_dict["v_optimizer"] == "adam":
            v_opt = optax.adam
        elif config_dict["v_optimizer"] == "adagrad":
            v_opt = optax.adagrad
        elif config_dict["v_optimizer"] == "adamaxw":
            v_opt = optax.adamaxw

        if config_dict["s_optimizer"] == "adamw":
            s_opt = optax.adamw
        elif config_dict["s_optimizer"] == "adam":
            s_opt = optax.adam
        elif config_dict["s_optimizer"] == "adagrad":
            s_opt = optax.adagrad
        elif config_dict["s_optimizer"] == "adamaxw":
            s_opt = optax.adamaxw

        key = random.PRNGKey(seed=seed)
        key1, key2 = random.split(key=key, num=2)
        batch_size = config_dict["batch_size"]
        steps = self.steps
        yu_dimension = self.yu_dimension
        dim = yu_dimension[0] + yu_dimension[1]
        v_hidden_layer_list = [config_dict["v_hidden_layer"]] * (
            config_dict["v_num_hidden_layers"]
        )
        s_hidden_layer_list = [config_dict["s_hidden_layer"]] * (
            config_dict["s_num_hidden_layers"]
        )
        velocity = MLP(
            key=key1,
            dim=dim,
            time_varying=True,
            w=v_hidden_layer_list,
            num_layers=len(v_hidden_layer_list) + 1,
            activation_fn=v_activation,
        )
        score = MLP(
            key=key2,
            dim=dim,
            time_varying=True,
            w=s_hidden_layer_list,
            num_layers=len(s_hidden_layer_list) + 1,
            activation_fn=s_activation,
        )
        v_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config_dict["v_peak_value"],
            warmup_steps=2_000,
            decay_steps=steps,
            end_value=1e-5,
        )
        s_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config_dict["s_peak_value"],
            warmup_steps=2_000,
            decay_steps=steps,
            end_value=1e-5,
        )
        # opt = config_dict["optimizer"]
        v_optimizer = optax.chain(optax.clip_by_global_norm(1.0), v_opt(v_schedule))
        s_optimizer = optax.chain(optax.clip_by_global_norm(1.0), s_opt(s_schedule))
        # interpolant = config_dict["interpolant"]
        # interpolant_der = config_dict["interpolant_der"]

        trainer = NNSDE(
            target_density=None,
            velocity=velocity,
            score=score,
            v_optimizer=v_optimizer,
            s_optimizer=s_optimizer,
            interpolant=interpolant,
            interpolant_der=interpolant_der,
            reference_sampler=gaussian_reference_sampler,
            v_loss=vec_field_loss,
            s_loss=denoiser_loss,
            interpolant_args=self.interpolant_args,
            reference_sampler_args=self.reference_sampler_args,
            yu_dimension=yu_dimension,
        )
        try:
            trainer.train(
                x1_data=self.train_data,
                train_dim=self.train_dim,
                batch_size=batch_size,
                steps=steps,
                x0_data=self.x0_data,
            )
            solver_args = {"saveat": "t1", "D_mask": D_mask}
            cond_samples = trainer.conditional_sample(
                cond_values=cond_values, u0_cond=us_test_pca, nsamples=20000, solver_args=solver_args, gamma=gamma_fn
            )
            print("Calculating SWD...")
            swd_list = np.zeros(3)
            for i, all_samples in enumerate(cond_samples):
                u_samples_gen = all_samples[:, yu_dimension[0] :]
                u_samples_gen = jnp.asarray(u_samples_gen)
                u_samples = pca_decode(u_samples_gen)
                # u_samples_pca = pca_encode_total(u_samples) # To make sure in exactly the correct PCA basis

                u_samples = u_samples.reshape(nsamples, nx, ny)
                u_samples = jnp.asarray(u_samples)
                u_samples += extra_samp
                swd_list[i] = swd(
                        u_samples.reshape(nsamples, nx * ny),
                        jnp.asarray(hmala_list[i]),
                        n_projections=n_projections,
                        seed=SEED,
                    ) / base_swd_list[i]
            swd_average = np.mean(swd_list)
            loss = swd_average.item()
            wandb.log({"relative error (swd)": loss})

            return loss
        finally:
            del trainer, velocity, score
            del cond_samples, u_samples_gen, u_samples
            del v_optimizer, s_optimizer
            jax.clear_caches()
            gc.collect()


configs = {"dataset": "darcy_flow_si_hmala"}

sep = "\n" + "#" * 80 + "\n"
output_root = "hyperparam_results_sde_mult"
os.makedirs(output_root, exist_ok=True)

with open("poisson.yaml") as fid:
    inargs = yaml.full_load(fid)

utrue = np.load("training_dataset/true_param_grid.npy")
ytrue = np.load("training_dataset/true_state_grid.npy")
map_est = np.load("training_dataset/map_param_grid.npy")
targets, yobs = read_data_h5()
yobs_med = np.load("data_50.npy")
yobs_98 = np.load("data_98.npy")

# Load h-MALA samples
# nsamples = inargs["MCMC"]["nsamples"] - inargs["MCMC"]["burnin"]
nsamples = 20000
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

med_root = "mcmc_median"
chains = []
for i in tqdm(range(20)):
    hmala_dir = f"chain_{i:02d}"
    hmala_path = os.path.join(med_root, hmala_dir, "hmala_samples.npy")
    hmala_samps_chain = np.load(hmala_path).reshape(47000, flat_length)[7000:, :]
    thinned_samps = hmala_samps_chain[::40, :]
    chains.append(thinned_samps)
hmala_med = np.vstack(chains) # 20,000 independent samples at median
print(f"This is the shape of hmala_med: {hmala_med.shape}")

med_root = "mcmc_98"
chains = []
for i in tqdm(range(20)):
    hmala_dir = f"chain_{i:02d}"
    hmala_path = os.path.join(med_root, hmala_dir, "hmala_samples.npy")
    hmala_samps_chain = np.load(hmala_path).reshape(47000, flat_length)[7000:, :]
    thinned_samps = hmala_samps_chain[::40, :]
    chains.append(thinned_samps)
hmala_98 = np.vstack(chains) # 20,000 independent samples at median
print(f"This is the shape of hmala_98: {hmala_98.shape}")
hmala_list = [hmala_samps, hmala_med, hmala_98]

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
yobs_normalized = ys_normalizer.encode(yobs)
ymed_normalized = ys_normalizer.encode(yobs_med)
y98_normalized = ys_normalizer.encode(yobs_98)
cond_values = [yobs_normalized, ymed_normalized, y98_normalized]

pca_encode, pca_decode, k, sample_extra, extra_cov, S = get_pca_fns(us_ref)
extra_samp = sample_extra(1).reshape(nx, ny)
extra_pca_cov = extra_cov()
extra_pca_var = np.diag(extra_pca_cov).reshape(nx, ny)

us_pca = pca_encode(us)
us_pca = jnp.asarray(us_pca)
us_ref_pca = pca_encode(us_ref)
us_ref_pca = jnp.asarray(us_ref_pca)
us_test_pca = pca_encode(us_test)
us_test_pca = jnp.asarray(us_test_pca)

SEED = 42
n_projections = 4024
random_idxs = np.random.choice(len(us_ref_pca), (20000,))
base_swd = swd(
    us_ref[random_idxs, :], hmala_samps, n_projections=n_projections, seed=SEED
)
swd_med = swd(
    us_ref[random_idxs, :], hmala_med, n_projections=n_projections, seed=SEED
)
swd_98 = swd(
    us_ref[random_idxs, :], hmala_98, n_projections=n_projections, seed=SEED
)
print(f"This is the base swd: {base_swd}")
base_swd_list = [base_swd, swd_med, swd_98]
D_mask = jnp.concatenate([jnp.zeros(ys_normalized.shape[1]), jnp.sqrt(S)])

interpolant_args = {"t": None, "x1": None, "x0": None, "z": None}
reference_sampler_args = {"D_mask": D_mask}
steps = 10000
yu_dimension = (ys_normalized.shape[1], k.item())
train_data = jnp.hstack([ys_normalized, us_pca])
x0_data = jnp.hstack([ys_normalized, us_ref_pca])
sample_no_list = [2**i for i in range(1, 15)]
sample_no_list.append(nsamples)
sample_no_list.append(30000)
sample_no_list.append(40000)
sample_no_list.append(train_dim)
rel_error_array = np.zeros(len(sample_no_list))

for i, sample_no in enumerate(sample_no_list):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="Poisson - SI hyperparams - SDE multiple conditioning vals",
        name=f"iter={i}_n={sample_no}",
        group="sweep",
        reinit=True,
        config={"iter": i, "n": sample_no, **configs},
        settings=wandb.Settings(start_method="thread"),
    )
    train_data_iter = train_data[1 : (sample_no + 1), :]
    x0_data_iter = x0_data[1 : (sample_no + 1), :]

    regressor = SiSdeSmac(
        train_dim=train_dim,
        steps=steps,
        train_data=train_data_iter,
        x0_data=x0_data_iter,
        yu_dimension=yu_dimension,
        interpolant_args=interpolant_args,
        reference_sampler_args=reference_sampler_args,
    )

    scenario = Scenario(
        regressor.configspace,
        n_trials=150,
        deterministic=True,
        n_workers=1,
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario, n_configs=7
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
    rel_error_array[i] = incumbent_loss

    best_hyperparams = dict(incumbent)

    print(f"These are the best hyperparameters selected: {best_hyperparams}")
    iter_folder = os.path.join(output_root, f"iteration_{i}")
    os.makedirs(iter_folder, exist_ok=True)
    save_path = os.path.join(iter_folder, "best_hyperparams.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(best_hyperparams, f)

    print(f"Best hyperparameters saved to {save_path}")

np.save(os.path.join(output_root, "incumbent_loss.npy"), rel_error_array)
print("Code terminated")
