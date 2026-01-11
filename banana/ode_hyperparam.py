import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Callable, List
import gc

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from ksd import median_heuristic_sigma_jax

from triangular_transport.flows.flow_trainer import (
    NNTrainer,
)
from triangular_transport.flows.methods.sampling import inf_train_gen

from triangular_transport.flows.interpolants import (
    linear_interpolant,
    linear_interpolant_der,
    trig_interpolant,
    trig_interpolant_der,
    sigmoid_interpolant,
    sigmoid_interpolant_der,
)
from triangular_transport.flows.loss_functions import vec_field_loss

# from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.dataloaders import standard_gaussian_reference_sampler
from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    vectorize_kfunc,
    get_sum_of_kernels,
)
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

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# jax.config.update("jax_default_device", jax.devices()[1])


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
        epochs: int,
        train_data: jax.Array,
        x0_data: jax.Array,
        yu_dimension: tuple,
        interpolant_args: dict,
    ):
        self.train_dim = train_dim
        self.epochs = epochs
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
        hidden_layer = Integer("hidden_layer", (850, 1100), default=1000, log=True)
        num_hidden_layers = Integer("num_hidden_layers", (7, 10), default=8, log=True)
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
        batch_size = min(batch_size, self.train_dim) - 1
        epochs = self.epochs
        steps_per_epoch = int(np.ceil(train_dim / batch_size))
        steps = steps_per_epoch * epochs
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
            reference_sampler=standard_gaussian_reference_sampler,
            loss=vec_field_loss,
            interpolant_args=self.interpolant_args,
            yu_dimension=yu_dimension,
        )
        try:
            trainer.train(
                train_data=self.train_data,
                train_dim=self.train_dim,
                batch_size=batch_size,
                steps=steps,
                x0_data=self.x0_data,
            )
            
            cond_samples = trainer.conditional_sample(
                cond_values=cond_values, u0_cond=None, nsamples=nsamples
            )
            print("Calculating SWD...")
            swd_list = np.zeros(3)
            mmd_list = np.zeros(3)
            for i, all_samples in enumerate(cond_samples):
                u_samples_gen = all_samples[:, yu_dimension[0] :]
                us_gen = np.array(u_samples_gen)
                swd_list[i] = swd(
                        us_gen,
                        samps_list[i],
                        n_projections=n_projections,
                        seed=SEED,
                    ) / base_swd_list[i]
                mmd_list[i] = get_kme(us_gen, samps_list[i])
            swd_average = np.mean(swd_list)
            mmd_average = np.mean(mmd_list)
            loss = swd_average.item() + mmd_average.item()
            wandb.log({"relative error (swd + mmd)": loss})
        finally:
            del trainer, model
            del cond_samples, u_samples_gen, us_gen
            del optimizer
            jax.clear_caches()
            gc.collect()

        return loss


configs = {"dataset": "banana"}

sep = "\n" + "#" * 80 + "\n"
output_root = "hyperparam_results_big_network"
os.makedirs(output_root, exist_ok=True)


nsamples = 20000
cond_values = [0.0, -1.0, -4.2]
gen_seed = 1
rng = np.random.RandomState(gen_seed)
samps0 = np.load("rej_samples_0.npy")[np.random.choice(100000, size=(nsamples, )), :]
samps1 = np.load("rej_samples_1.npy")[np.random.choice(100000, size=(nsamples, )), :]
samps4 = np.load("rej_samples_4.npy")[np.random.choice(100000, size=(nsamples, )), :]
samps_list = [samps0, samps1, samps4]
us_base = rng.randn(nsamples, 1)

gamma = median_heuristic_sigma_jax(us_base, samps1)
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

SEED = 42
n_projections = 2048
base_swd0 = swd(
    us_base, samps0, n_projections=n_projections, seed=SEED
)
base_swd1 = swd(
    us_base, samps1, n_projections=n_projections, seed=SEED
)
base_swd4 = swd(
    us_base, samps4, n_projections=n_projections, seed=SEED
)
print(f"This is the base swd (0): {base_swd0}")
print(f"This is the base swd (-1): {base_swd1}")
print(f"This is the base swd (-4.2): {base_swd4}")
base_swd_list = [base_swd0, base_swd1, base_swd4]

interpolant_args = {"t": None, "x1": None, "x0": None}
epochs = 2000
yu_dimension = (1,1)
x0_data = None
sample_no_list = [2**i for i in range(1, 15)]
sample_no_list.append(nsamples)
sample_no_list.append(30000)
sample_no_list.append(40000)
sample_no_list.append(50000)
sample_no_list.append(60000)
sample_no_list.append(70000)
sample_no_list.append(80000)
sample_no_list.append(90000)
sample_no_list.append(100000)

rel_error_array = np.zeros(len(sample_no_list))

# print(f"This is the sample_no_list: {sample_no_list}")
for i, sample_no in enumerate(sample_no_list):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="Banana - SI hyperparams - multiple conditioning values - big network size",
        name=f"iter={i}_n={sample_no}",
        group="sweep",
        reinit=True,
        config={"iter": i, "n":sample_no, **configs},
        settings=wandb.Settings(start_method="thread")
    )
    train_dim = sample_no
    train_data = inf_train_gen(data="banana", rng=rng, batch_size=train_dim)

    regressor = SiOdeSmac(
        train_dim=train_dim,
        epochs=epochs,
        train_data=train_data,
        x0_data=x0_data,
        yu_dimension=yu_dimension,
        interpolant_args=interpolant_args,
    )

    scenario = Scenario(
        regressor.configspace,
        n_trials=150,
        deterministic=True,
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
