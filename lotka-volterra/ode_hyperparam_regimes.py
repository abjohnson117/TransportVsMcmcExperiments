"""
Hyperparameter sweep across three data regimes (low=64, mid=2048, high=70000).

Interpolant is fixed to linear_interpolant. The key hypothesis being tested is:
  - smaller models may be better in the low-data regime (and vice-versa)
  - early stopping based on validation loss can prevent overfitting in low-data regimes

One independent SMAC MultiFidelityFacade sweep is run per regime.
The config space allows models ranging from very small (32 units, 1 layer) to
large (512 units, 13 layers), letting SMAC find the optimal size per regime.

Results are saved to  hyperparam_results/best_hyperparams_<regime>.pkl
"""

import gc
import os
import pickle
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import copy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import diffrax
import wandb
from jax import random
from ot.sliced import sliced_wasserstein_distance as swd
from tqdm.auto import tqdm

from ConfigSpace import ConfigurationSpace, Float, Integer
from smac import MultiFidelityFacade, Scenario
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue

from ksd import median_heuristic_sigma_jax
from triangular_transport.flows.dataloaders import (
    dataloader,
    log_normal_reference_sampler,
)
from triangular_transport.flows.flow_trainer import NNTrainer
from triangular_transport.flows.interpolants import (
    trig_interpolant,
    trig_interpolant_der,
)
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    get_sum_of_kernels,
    vectorize_kfunc,
)
from triangular_transport.networks.flow_networks import MLP

# ---------------------------------------------------------------------------
# Data regimes to sweep over
# ---------------------------------------------------------------------------
REGIMES = [
    {
        "name": "low",
        "n_train": 128,
        "min_budget": 20,
        "max_budget": 300,
        "n_trials": 80,
    },
    {
        "name": "mid",
        "n_train": 2048,
        "min_budget": 50,
        "max_budget": 500,
        "n_trials": 80,
    },
    {
        "name": "high",
        "n_train": 70000,
        "min_budget": 100,
        "max_budget": 650,
        "n_trials": 100,
    },
]

# Early-stopping parameters (applied inside each SMAC trial)
ES_EVAL_EVERY_EPOCHS = 10   # evaluate SWD every this many epochs
ES_PATIENCE = 3             # stop if no improvement for this many eval periods


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def gaussian_reference_sampler(key, shape, mu, sigma, normalizer):
    samples = mu + random.normal(key=key, shape=shape) * sigma
    if normalizer is not None:
        samples = normalizer.encode(samples)
    return samples


class TimingCallback(Callback):
    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self.completed = 0
        self.start_time = None

    def on_tell_end(self, smac, info: TrialInfo, value: TrialValue):
        if self.start_time is None:
            self.start_time = time.time()
        self.completed += 1
        elapsed = time.time() - self.start_time
        avg = elapsed / self.completed
        remaining = avg * (self.n_trials - self.completed)
        print(
            f"Trial {self.completed}/{self.n_trials} | "
            f"budget={info.budget:.0f} | "
            f"cost={value.cost:.4f} | "
            f"elapsed={elapsed/60:.1f}m | "
            f"ETA={remaining/60:.1f}m"
        )


def _batch_size_for(train_dim: int) -> int:
    """Return a batch size that is strictly < train_dim (required by dataloader)."""
    return min(2048, train_dim) - 1


def _steps_for(train_dim: int, batch_size: int, epochs: int) -> int:
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    return steps_per_epoch * epochs


# ---------------------------------------------------------------------------
# Custom training loop with validation-loss early stopping
# ---------------------------------------------------------------------------
def train_with_early_stopping(
    trainer: NNTrainer,
    train_data,
    train_dim: int,
    batch_size: int,
    max_epochs: int,
    eval_every_epochs: int,
    patience: int,
    key: jax.Array,
    eval_fn,
):
    """Train using trainer.make_step; stop early when eval_fn stalls.

    eval_fn(model) -> float  –  lower is better (e.g. relative SWD).
    Called every eval_every_epochs epochs; training halts when no improvement
    exceeds 1 % of the best value for `patience` consecutive evaluations.

    The optimizer state is kept alive across the full run so the learning-rate
    schedule is not reset mid-training.  trainer.model is updated in place with
    the best checkpoint found.

    Returns the number of gradient steps actually taken.
    """
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    max_steps = steps_per_epoch * max_epochs
    eval_every_steps = steps_per_epoch * eval_every_epochs

    # Initialise optimiser state once
    opt_state = trainer.optimizer.init(eqx.filter(trainer.model, eqx.is_array))

    key, loader_key = jax.random.split(key)
    train_loader = dataloader(
        data=train_data,
        batch_size=batch_size,
        train_dim=train_dim,
        key=loader_key,
    )

    prepare_batch = trainer._prepare_batch_jit()

    model = trainer.model
    best_metric = float("inf")
    best_model = model
    no_improve = 0
    steps_done = 0

    for step, x1 in zip(tqdm(range(max_steps)), train_loader):
        key, subkey = jax.random.split(key)
        batch = prepare_batch(key=subkey, x1=x1, x0=None, batch_size=batch_size)
        model, opt_state, _ = trainer.make_step(
            model, opt_state, batch["t"], batch["Ivals"], batch["It_vals"]
        )
        steps_done += 1

        if steps_done % eval_every_steps == 0:
            metric = eval_fn(model)
            if metric < best_metric * (1.0 - 0.01):   # must improve by ≥1 %
                best_metric = metric
                best_model = model
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"  Early stop at step {steps_done}/{max_steps} "
                        f"(epoch ~{steps_done // steps_per_epoch}); "
                        f"best metric = {best_metric:.4f}"
                    )
                    break

    # Check if the final model is better than the saved checkpoint
    final_metric = eval_fn(model)
    if final_metric < best_metric:
        best_model = model

    trainer.model = best_model
    trainer.train_dim = train_dim
    return steps_done


# ---------------------------------------------------------------------------
# SMAC objective class
# ---------------------------------------------------------------------------
class SiOdeSmacRegimes:
    """SMAC objective for a single data regime."""

    def __init__(
        self,
        train_dim: int,
        train_data,      # normalised x1 for training
        yu_dimension: tuple,
        interpolant_args: dict,
        reference_sampler_args: dict,
        # evaluation artefacts (shared across all regimes)
        ncond_vals: list,
        samps_list: list,
        base_swd_list: list,
        nsamples: int,
        n_projections: int,
        swd_seed: int,
        us_normalizer,
        wandb_run,
    ):
        self.train_dim = train_dim
        self.train_data = train_data
        self.yu_dimension = yu_dimension
        self.interpolant_args = interpolant_args
        self.reference_sampler_args = reference_sampler_args
        self.ncond_vals = ncond_vals
        self.samps_list = samps_list
        self.base_swd_list = base_swd_list
        self.nsamples = nsamples
        self.n_projections = n_projections
        self.swd_seed = swd_seed
        self.us_normalizer = us_normalizer
        self.wandb_run = wandb_run

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        # Allow small models (for low-data) up to large models (for high-data)
        cs.add(Integer("hidden_layer", (32, 512), default=256, log=False))
        cs.add(Integer("num_hidden_layers", (1, 13), default=6, log=False))
        cs.add(Float("peak_value", (1e-4, 1e-2), default=3e-4, log=True))
        cs.add(Float("weight_decay", (1e-6, 1e-2), default=1e-4, log=True))
        return cs

    def train(self, config, seed: int = 0, budget: float = 100.0) -> float:
        cfg = dict(config)
        train_dim = self.train_dim
        batch_size = _batch_size_for(train_dim)
        max_epochs = int(budget)

        key = random.PRNGKey(seed)
        key, model_key = random.split(key)

        yu_dimension = self.yu_dimension
        dim = yu_dimension[0] + yu_dimension[1]

        hidden_layer_list = [cfg["hidden_layer"]] * cfg["num_hidden_layers"]
        model = MLP(
            key=model_key,
            dim=dim,
            time_varying=True,
            w=hidden_layer_list,
            num_layers=len(hidden_layer_list) + 1,
            activation_fn=jax.nn.gelu,
        )

        total_steps = _steps_for(train_dim, batch_size, max_epochs)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg["peak_value"],
            warmup_steps=max(50, total_steps // 20),
            decay_steps=total_steps,
            end_value=1e-5,
        )
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg["weight_decay"]))

        trainer = NNTrainer(
            target_density=None,
            model=model,
            optimizer=optimizer,
            interpolant=trig_interpolant,
            interpolant_der=trig_interpolant_der,
            reference_sampler=gaussian_reference_sampler,
            reference_sampler_args=self.reference_sampler_args,
            loss=vec_field_loss,
            interpolant_args=self.interpolant_args,
            yu_dimension=yu_dimension,
        )

        solver_args = {
            "solver": diffrax.Dopri5(),
            "max_steps": 50_000,
            "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6),
        }
        # Use fewer samples for early-stopping SWD to keep each eval fast.
        es_nsamples = min(500, self.nsamples)

        def swd_eval_fn(model):
            """Relative SWD evaluated cheaply for early stopping."""
            trainer.model = model
            cs = trainer.conditional_sample(
                cond_values=self.ncond_vals,
                u0_cond=None,
                nsamples=es_nsamples,
                solver_args=solver_args,
            )
            vals = np.zeros(3)
            for i, samples in enumerate(cs):
                u_gen = samples[:, yu_dimension[0]:]
                u_gen = self.us_normalizer.decode(u_gen)
                u_gen = np.exp(np.array(u_gen))
                vals[i] = (
                    swd(
                        u_gen,
                        self.samps_list[i],
                        a=np.ones(len(u_gen)) / len(u_gen),
                        b=np.ones(len(self.samps_list[i])) / len(self.samps_list[i]),
                        n_projections=self.n_projections,
                        seed=self.swd_seed,
                        p=2,
                    )
                    / self.base_swd_list[i]
                )
            return float(np.mean(vals))

        try:
            train_with_early_stopping(
                trainer=trainer,
                train_data=self.train_data,
                train_dim=train_dim,
                batch_size=batch_size,
                max_epochs=max_epochs,
                eval_every_epochs=ES_EVAL_EVERY_EPOCHS,
                patience=ES_PATIENCE,
                key=key,
                eval_fn=swd_eval_fn,
            )

            cond_samples = trainer.conditional_sample(
                cond_values=self.ncond_vals,
                u0_cond=None,
                nsamples=self.nsamples,
                solver_args=solver_args,
            )

            swd_list = np.zeros(3)
            for i, samples in enumerate(cond_samples):
                u_gen = samples[:, yu_dimension[0]:]
                u_gen = self.us_normalizer.decode(u_gen)
                u_gen = np.exp(np.array(u_gen))
                swd_list[i] = (
                    swd(
                        u_gen,
                        self.samps_list[i],
                        a=np.ones(len(u_gen)) / len(u_gen),
                        b=np.ones(len(self.samps_list[i])) / len(self.samps_list[i]),
                        n_projections=self.n_projections,
                        seed=self.swd_seed,
                        p=2,
                    )
                    / self.base_swd_list[i]
                )

            loss = float(np.mean(swd_list))
            if self.wandb_run is not None:
                wandb.log({"relative_swd": loss})

        finally:
            del trainer, model, optimizer
            if "cond_samples" in dir():
                del cond_samples
            jax.clear_caches()
            gc.collect()

        return loss


# ---------------------------------------------------------------------------
# Global data loading (done once, shared across regimes)
# ---------------------------------------------------------------------------
configs = {"dataset": "lotka-volterra"}

output_root = "hyperparam_results"
os.makedirs(output_root, exist_ok=True)

input_root = "true_samps"
cond_val1 = np.load(os.path.join(input_root, "y_obs.npy"))
cond_val2 = np.log(np.load(os.path.join(input_root, "y_moderate.npy")))
cond_val3 = np.log(np.load(os.path.join(input_root, "y_rare.npy")))
cond_values_raw = [cond_val1, cond_val2, cond_val3]

nsamples = 5000
swd_seed = 42
n_projections = 512

samps1 = np.exp(np.load(os.path.join(input_root, "true_us_obs.npy"))[::20, :])
samps2 = np.exp(np.load(os.path.join(input_root, "true_us_moderate_obs_2.npy"))[::20, :])
samps3 = np.exp(np.load(os.path.join(input_root, "true_us_rare_obs_3.npy"))[::20, :])
samps_list = [samps1, samps2, samps3]

gen_key = random.key(1)
us_base = log_normal_reference_sampler(
    key=gen_key,
    shape=samps1.shape,
    mu=jnp.array([-0.125, -3.0, -0.125, -3.0]),
    sigma=1.0 / jnp.sqrt(2),
)

print("Computing base SWDs (prior vs. true posterior) ...")
base_swd_list = []
for samps in samps_list:
    b = swd(
        np.asarray(us_base),
        np.asarray(samps),
        a=np.ones(len(us_base)) / len(us_base),
        b=np.ones(len(samps)) / len(samps),
        seed=swd_seed,
        n_projections=n_projections,
        p=2,
    )
    base_swd_list.append(b)
print(f"Base SWDs: obs={base_swd_list[0]:.4f}  mod={base_swd_list[1]:.4f}  rare={base_swd_list[2]:.4f}")

# Load all training data once; subsets are taken per regime
all_train_data = np.load("training_data.npy")
# Normaliser is fit on the FULL dataset to keep feature scales consistent
ys_full, us_full = all_train_data[:, :18], all_train_data[:, 18:]
ys_log_full, us_log_full = np.log(ys_full), np.log(us_full)
ys_normalizer = UnitGaussianNormalizer(ys_log_full)
us_normalizer = UnitGaussianNormalizer(us_log_full)
ysn_full = ys_normalizer.encode()
usn_full = us_normalizer.encode()
x1_full = jnp.hstack([ysn_full, usn_full])

# Conditional observation values in normalised space (shared)
ncond_vals_full = [ys_normalizer.encode(yobs) for yobs in cond_values_raw]

yu_dimension = (18, 4)
interpolant_args = {"t": None, "x1": None, "x0": None}
reference_sampler_args = {
    "mu": jnp.array([-0.125, -3.0, -0.125, -3.0]),
    "sigma": 1.0 / jnp.sqrt(2),
    "normalizer": us_normalizer,
}


# ---------------------------------------------------------------------------
# Main loop: one SMAC sweep per regime
# ---------------------------------------------------------------------------
all_results = {}

for regime in REGIMES:
    regime_name = regime["name"]
    n_train = regime["n_train"]
    print(f"\n{'='*70}")
    print(f"  Regime: {regime_name}  (n_train={n_train})")
    print(f"{'='*70}\n")

    # Shuffle with a fixed seed so each regime uses the same ordering
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x1_full))[:n_train]
    train_data = x1_full[perm]

    print(f"  Train size: {n_train}")

    # ------------------------------------------------------------------
    # WandB run for this regime
    # ------------------------------------------------------------------
    run = wandb.init(
        project="LV - SI hyperparams - regimes",
        name=f"smac-{regime_name}-n{n_train}",
        group="regime-sweep",
        reinit=True,
        config={
            "regime": regime_name,
            "n_train": n_train,
            "interpolant": "trig_interpolant",
            **configs,
        },
        settings=wandb.Settings(start_method="thread"),
    )

    # ------------------------------------------------------------------
    # Build SMAC objective and run the sweep
    # ------------------------------------------------------------------
    objective = SiOdeSmacRegimes(
        train_dim=n_train,
        train_data=train_data,
        yu_dimension=yu_dimension,
        interpolant_args=interpolant_args,
        reference_sampler_args=reference_sampler_args,
        ncond_vals=ncond_vals_full,
        samps_list=samps_list,
        base_swd_list=base_swd_list,
        nsamples=nsamples,
        n_projections=n_projections,
        swd_seed=swd_seed,
        us_normalizer=us_normalizer,
        wandb_run=run,
    )

    scenario = Scenario(
        objective.configspace,
        n_trials=regime["n_trials"],
        deterministic=True,
        min_budget=regime["min_budget"],
        max_budget=regime["max_budget"],
    )

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=7)

    smac = MultiFidelityFacade(
        scenario,
        objective.train,
        initial_design=initial_design,
        callbacks=[TimingCallback(n_trials=regime["n_trials"])],
        overwrite=True,
    )

    print("Starting SMAC optimisation ...")
    incumbent = smac.optimize()

    default_loss = smac.validate(objective.configspace.get_default_configuration())
    incumbent_loss = smac.validate(incumbent)
    best_hp = dict(incumbent)
    # Always record the interpolant so downstream scripts are self-contained
    best_hp["interpolant"] = "trig_interpolant"

    print(f"\n  Default loss  : {default_loss:.4f}")
    print(f"  Incumbent loss: {incumbent_loss:.4f}")
    print(f"  Best HP       : {best_hp}")

    # Save
    save_path = os.path.join(output_root, f"best_hyperparams_{regime_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(best_hp, f)
    print(f"  Saved → {save_path}")

    all_results[regime_name] = {
        "best_hp": best_hp,
        "incumbent_loss": incumbent_loss,
        "default_loss": default_loss,
    }

    np.save(
        os.path.join(output_root, f"incumbent_loss_{regime_name}.npy"),
        np.array([incumbent_loss]),
    )
    wandb.finish()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, res in all_results.items():
    print(f"\n  {name:6s}  loss={res['incumbent_loss']:.4f}  HP={res['best_hp']}")
print("\nCode terminated.")
