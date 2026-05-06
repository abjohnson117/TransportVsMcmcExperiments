"""
Hyperparameter sweep across three data regimes for the p-Poisson problem.

Interpolant is fixed to trig_interpolant. The key hypothesis being tested is:
  - smaller models may be better in the low-data regime (and vice-versa)
  - early stopping based on validation SWD can prevent overfitting in low-data regimes

One independent SMAC MultiFidelityFacade sweep is run per regime.
PCA dimension reduction is applied to the z=0 surface of the parameter field;
residual variance is injected into each generated sample before SWD evaluation.

Training data layout
--------------------
  solutions.npy  : (250000, 100)   — noisy 10×10 pointwise observations (y)
  parameters.npy : (250000, 1323)  — 3-D parameter field (21×21×3) in Fortran
                   order; z=0 surface (21×21 = 441) extracted for training.

Results saved to hyperparam_results/best_hyperparams_<regime>.pkl
"""

import gc
import os
import pickle
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import diffrax
import wandb
from jax import random
from tqdm.auto import tqdm
from typing import Callable, List

from ConfigSpace import ConfigurationSpace, Float, Integer
from smac import MultiFidelityFacade, Scenario
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue

from triangular_transport.flows.dataloaders import dataloader
from triangular_transport.flows.flow_trainer import NNTrainer
from triangular_transport.flows.interpolants import linear_interpolant, linear_interpolant_der
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer

# ---------------------------------------------------------------------------
# Data regimes
# ---------------------------------------------------------------------------
REGIMES = [
    {"name": "low",  "n_train": 500,   "min_budget": 20,  "max_budget": 300, "n_trials": 80},
    {"name": "mid",  "n_train": 5000,  "min_budget": 50,  "max_budget": 500, "n_trials": 80},
    {"name": "high", "n_train": 75000, "min_budget": 100, "max_budget": 650, "n_trials": 100},
]

ES_EVAL_EVERY_EPOCHS = 10
ES_PATIENCE = 3

# ---------------------------------------------------------------------------
# JAX sliced Wasserstein distance
# ---------------------------------------------------------------------------
def sliced_wasserstein_jax(x, y, n_projections=512, seed=42, chunk_size=64, p=2):
    """
    Memory-efficient SWD in JAX. Processes projections in chunks.
    Supports unequal sample sizes via quantile interpolation.
    chunk_size=64 uses ~100 MB/kernel; increase for speed, decrease for memory.
    """
    n_x, d = x.shape
    n_y = y.shape[0]
    assert n_projections % chunk_size == 0

    t_x = (jnp.arange(n_x, dtype=jnp.float32) + 0.5) / n_x
    t_y = (jnp.arange(n_y, dtype=jnp.float32) + 0.5) / n_y
    t   = jnp.sort(jnp.concatenate([t_x, t_y]))

    @jax.jit
    def chunk_cost(key):
        theta = jax.random.normal(key, (chunk_size, d))
        theta = theta / jnp.linalg.norm(theta, axis=1, keepdims=True)
        xp = jnp.sort(x @ theta.T, axis=0).T
        yp = jnp.sort(y @ theta.T, axis=0).T

        def one_cost(xi, yi):
            return jnp.mean((jnp.interp(t, t_x, xi) - jnp.interp(t, t_y, yi)) ** p)

        return jnp.mean(jax.vmap(one_cost)(xp, yp))

    keys = jax.random.split(jax.random.PRNGKey(seed), n_projections // chunk_size)
    cost = jnp.mean(jnp.array([chunk_cost(k) for k in keys]))
    return cost ** (1.0 / p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gaussian_reference_sampler(key, shape, mu, sigma, normalizer):
    samples = mu + random.normal(key=key, shape=shape) * sigma
    if normalizer is not None:
        samples = normalizer.encode(samples)
    return samples


def get_pca_fns(us, explained_var_threshold=0.98):
    """Whitened PCA. Returns encode/decode callables and a residual variance sampler."""
    n = us.shape[0]
    mean_us = us.mean(axis=0)
    X = us - mean_us

    U, S, Vt = np.linalg.svd(X / np.sqrt(n - 1), full_matrices=False)
    V = Vt.T

    expl_var = (S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(np.cumsum(expl_var), explained_var_threshold) + 1)

    V_kept, S_kept = V[:, :k], S[:k]
    V_res,  S_res  = V[:, k:], S[k:]

    def pca_encode(b):
        return (b - mean_us) @ V_kept / S_kept

    def pca_decode(z):
        return mean_us + (z * S_kept) @ V_kept.T

    def sample_extra(n_samp=1):
        """Sample from the residual (unexplained) variance component."""
        eps = np.random.randn(n_samp, S_res.shape[0])
        return (eps * S_res) @ V_res.T   # (n_samp, flat_length)

    def extra_cov():
        return V_res @ np.diag(S_res ** 2) @ V_res.T

    return pca_encode, pca_decode, k, sample_extra, extra_cov


class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    skips: List[eqx.nn.Linear | None]
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

        if isinstance(activation_fn, list):
            if len(activation_fn) == 1:
                activation_fn *= num_layers - 1
        else:
            activation_fn = [activation_fn] * (num_layers - 1)
        self.activation_fn = activation_fn

        if isinstance(w, list):
            if len(w) == 1:
                w *= num_layers - 1
            widths = w
        else:
            widths = [w] * (num_layers - 1)

        k_keys = jax.random.split(key, 2 * num_layers)
        in_dim0 = dim + (1 if time_varying else 0)

        self.layers = []
        self.skips = []
        in_dim = in_dim0
        for i, width in enumerate(widths):
            self.layers.append(eqx.nn.Linear(in_dim, width, key=k_keys[i]))
            if in_dim == width:
                self.skips.append(None)
            else:
                self.skips.append(eqx.nn.Linear(in_dim, width, key=k_keys[i + num_layers]))
            in_dim = width

        self.out = eqx.nn.Linear(in_dim, out_dim, key=k_keys[-1])

    def __call__(self, x):
        for layer, skip, act in zip(self.layers, self.skips, self.activation_fn):
            h = act(layer(x))
            s = x if skip is None else skip(x)
            x = h + s
        return self.out(x)


class TimingCallback(Callback):
    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self.completed = 0
        self.start_time = None

    def on_tell_end(self, _smac, info: TrialInfo, value: TrialValue):
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
    """Batch size strictly less than train_dim (required by dataloader)."""
    return min(2048, train_dim) - 1


def _steps_for(train_dim: int, batch_size: int, epochs: int) -> int:
    return int(np.ceil(train_dim / batch_size)) * epochs


# ---------------------------------------------------------------------------
# Early-stopping training loop
# ---------------------------------------------------------------------------
def train_with_early_stopping(
    trainer: NNTrainer,
    train_data,
    x0_data,
    train_dim: int,
    batch_size: int,
    max_epochs: int,
    eval_every_epochs: int,
    patience: int,
    key: jax.Array,
    eval_fn,
):
    """Train using trainer.make_step; stop early when eval_fn stalls.

    eval_fn(model) -> float  –  lower is better (relative SWD).
    Called every eval_every_epochs epochs; halts when no ≥1% improvement
    for `patience` consecutive evaluations.
    """
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    max_steps = steps_per_epoch * max_epochs
    eval_every_steps = steps_per_epoch * eval_every_epochs

    opt_state = trainer.optimizer.init(eqx.filter(trainer.model, eqx.is_array))

    key, loader_key = jax.random.split(key)
    x1_loader = dataloader(
        data=train_data,
        batch_size=batch_size,
        train_dim=train_dim,
        key=loader_key,
    )
    x0_loader = dataloader(
        data=x0_data,
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

    for _step, x1, x0 in zip(tqdm(range(max_steps)), x1_loader, x0_loader):
        key, subkey = jax.random.split(key)
        batch = prepare_batch(key=subkey, x1=x1, x0=x0, batch_size=batch_size)
        model, opt_state, _ = trainer.make_step(
            model, opt_state, batch["t"], batch["Ivals"], batch["It_vals"]
        )
        steps_done += 1

        if steps_done % eval_every_steps == 0:
            metric = eval_fn(model)
            if metric < best_metric * (1.0 - 0.01):
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

    final_metric = eval_fn(model)
    if final_metric < best_metric:
        best_model = model

    trainer.model = best_model
    trainer.train_dim = train_dim
    return steps_done


# ---------------------------------------------------------------------------
# SMAC objective
# ---------------------------------------------------------------------------
class SiOdeSmacRegimes:
    """SMAC objective for a single data regime."""

    def __init__(
        self,
        train_dim: int,
        train_data,
        x0_data,
        yu_dimension: tuple,
        interpolant_args: dict,
        reference_sampler_args: dict,
        ncond_vals: list,
        hmala_list: list,
        base_swd_list: list,
        nsamples: int,
        n_projections: int,
        swd_seed: int,
        pca_decode,
        sample_extra,
        u0_cond,
        wandb_run,
    ):
        self.train_dim = train_dim
        self.train_data = train_data
        self.x0_data = x0_data
        self.yu_dimension = yu_dimension
        self.interpolant_args = interpolant_args
        self.reference_sampler_args = reference_sampler_args
        self.ncond_vals = ncond_vals
        self.hmala_list = hmala_list
        self.base_swd_list = base_swd_list
        self.nsamples = nsamples
        self.n_projections = n_projections
        self.swd_seed = swd_seed
        self.pca_decode = pca_decode
        self.sample_extra = sample_extra
        self.u0_cond = u0_cond
        self.wandb_run = wandb_run

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        cs.add(Integer("hidden_layer",      (256, 1024),  default=512,  log=False))
        cs.add(Integer("num_hidden_layers", (4, 13),    default=6,    log=False))
        cs.add(Float("peak_value",          (1e-4, 1e-2), default=3e-4, log=True))
        cs.add(Float("weight_decay",        (1e-6, 1e-2), default=1e-4, log=True))
        return cs

    def _decode_samples(self, u_pca, n):
        """PCA decode and inject per-sample residual variance."""
        u_flat = np.array(self.pca_decode(u_pca))   # (n, flat_length)
        extra  = self.sample_extra(n)                 # (n, flat_length)
        return u_flat + extra

    def train(self, config, seed: int = 0, budget: float = 100.0) -> float:
        cfg = dict(config)
        train_dim  = self.train_dim
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
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=cfg["weight_decay"]),
        )

        trainer = NNTrainer(
            target_density=None,
            model=model,
            optimizer=optimizer,
            interpolant=linear_interpolant,
            interpolant_der=linear_interpolant_der,
            reference_sampler=gaussian_reference_sampler,
            loss=vec_field_loss,
            interpolant_args=self.interpolant_args,
            yu_dimension=yu_dimension,
        )

        solver_args = {
            "solver": diffrax.Dopri5(),
            "max_steps": 50_000,
            "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6),
        }

        # Cheaper SWD for early-stopping checks
        es_nsamples = min(500, self.nsamples)

        def swd_eval_fn(model):
            trainer.model = model
            cs = trainer.conditional_sample(
                cond_values=self.ncond_vals,
                u0_cond=self.u0_cond[:es_nsamples],
                nsamples=es_nsamples,
                solver_args=solver_args,
            )
            vals = np.zeros(len(cs))
            for i, samples in enumerate(cs):
                u_pca  = samples[:, yu_dimension[0]:]
                u_flat = self._decode_samples(u_pca, es_nsamples)
                vals[i] = float(
                    sliced_wasserstein_jax(
                        jnp.asarray(u_flat),
                        jnp.asarray(self.hmala_list[i][:es_nsamples]),
                        n_projections=self.n_projections,
                        seed=self.swd_seed,
                    ) / self.base_swd_list[i]
                )
            return float(np.mean(vals))

        try:
            train_with_early_stopping(
                trainer=trainer,
                train_data=self.train_data,
                x0_data=self.x0_data,
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
                u0_cond=self.u0_cond,
                nsamples=self.nsamples,
                solver_args=solver_args,
            )

            swd_list = np.zeros(len(cond_samples))
            for i, samples in enumerate(cond_samples):
                u_pca  = samples[:, yu_dimension[0]:]
                u_flat = self._decode_samples(u_pca, self.nsamples)
                swd_list[i] = float(
                    sliced_wasserstein_jax(
                        jnp.asarray(u_flat),
                        jnp.asarray(self.hmala_list[i]),
                        n_projections=self.n_projections,
                        seed=self.swd_seed,
                    ) / self.base_swd_list[i]
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
# Data loading (done once, shared across all regimes)
# ---------------------------------------------------------------------------
output_root = "hyperparam_results"
os.makedirs(output_root, exist_ok=True)

nsamples    = 50_000
nx = ny     = 20
flat_length = (nx + 1) * (ny + 1)  # 21 * 21 = 441  (z=0 surface only)

RESHAPE_NO      = 300_000
THIN_FACTOR     = 20
CHAIN_ITERS     = 3
AUX_RESHAPE_NO  = 300_000
AUX_THIN_FACTOR = 20
AUX_CHAIN_ITERS = 3

# MCMC chains — main observation
chains = []
for i in tqdm(range(CHAIN_ITERS), desc="Loading hmala chains (obs)"):
    hmala_path = os.path.join("mcmc_main", f"chain_{i:03d}", "hmala_samples.npy")
    chain = (np.load(hmala_path)
               .reshape(RESHAPE_NO, 21, 21, 3, order="F")[::THIN_FACTOR, :, :, 0]
               .reshape(-1, flat_length))
    chains.append(chain)
hmala_samps = np.vstack(chains)
print(f"h_main shape: {hmala_samps.shape}")

# MCMC chains — median conditioning value
chains = []
for i in tqdm(range(AUX_CHAIN_ITERS), desc="Loading hmala chains (med)"):
    hmala_path = os.path.join("mcmc_median", f"chain_{i:03d}", "hmala_samples.npy")
    chain = (np.load(hmala_path)
               .reshape(AUX_RESHAPE_NO, 21, 21, 3, order="F")[::AUX_THIN_FACTOR, :, :, 0]
               .reshape(-1, flat_length))
    chains.append(chain)
hmala_med = np.vstack(chains)
print(f"hmala_med shape: {hmala_med.shape}")

# MCMC chains — 98th-percentile conditioning value
chains = []
for i in tqdm(range(AUX_CHAIN_ITERS), desc="Loading hmala chains (98)"):
    hmala_path = os.path.join("mcmc_98", f"chain_{i:03d}", "hmala_samples.npy")
    chain = (np.load(hmala_path)
               .reshape(AUX_RESHAPE_NO, 21, 21, 3, order="F")[::AUX_THIN_FACTOR, :, :, 0]
               .reshape(-1, flat_length))
    chains.append(chain)
hmala_98 = np.vstack(chains)
print(f"hmala_98 shape: {hmala_98.shape}")
hmala_list = [hmala_samps, hmala_med, hmala_98]

# Training data (full dataset loaded once; subsets taken per regime)
total_dim = 100_000
ys_all = np.load("training_dataset/solutions.npy")                              # (250000, 100)
us_all = (np.load("training_dataset/parameters.npy")
          .reshape(-1, 21, 21, 3, order="F")[:, :, :, 0]
          .reshape(-1, flat_length))                                            # (250000, 441)

ys_full = ys_all[:total_dim]
us_full = us_all[:total_dim]

# Reference parameters used to fit PCA (held out from training)
us_ref = us_all[total_dim : total_dim * 2].copy()
np.random.shuffle(us_ref)

# Normalise observations
yobs     = np.load("data_obs.npy")
yobs_med = np.load("data_50.npy")
yobs_98  = np.load("data_98.npy")

ys_normalizer       = UnitGaussianNormalizer(ys_full)
ys_normalized_full  = ys_normalizer.encode()
yobs_normalized     = ys_normalizer.encode(yobs)
ymed_normalized     = ys_normalizer.encode(yobs_med)
y98_normalized      = ys_normalizer.encode(yobs_98)
cond_values         = [yobs_normalized, ymed_normalized, y98_normalized]

# PCA on reference parameters
pca_encode, pca_decode, k, sample_extra, extra_cov = get_pca_fns(us_ref)
print(f"PCA retains {k} components")

us_pca_full = pca_encode(us_full)
us_ref_pca  = pca_encode(us_ref)
x1_full     = np.hstack([np.asarray(ys_normalized_full), us_pca_full])
x0_full     = np.hstack([np.asarray(ys_normalized_full), us_ref_pca])

us_test     = us_all[total_dim * 2 :].copy()
us_test_pca = pca_encode(us_test)

yu_dimension          = (ys_normalized_full.shape[1], k)
interpolant_args      = {"t": None, "x1": None, "x0": None}
reference_sampler_args = {
    "mu":         jnp.zeros(k),
    "sigma":      1.0,
    "normalizer": None,
}

# Base SWDs: prior (reference parameters) vs. true posterior (MCMC)
SEED          = 42
n_projections = 512
rng_base      = np.random.default_rng(SEED)
ref_idxs      = rng_base.choice(len(us_ref), size=nsamples, replace=False)

print("Computing base SWDs (prior vs. true posterior) ...")
base_swd = float(sliced_wasserstein_jax(
    jnp.asarray(us_ref[ref_idxs]),
    jnp.asarray(hmala_samps),
    n_projections=n_projections,
    seed=SEED,
))
swd_med = float(sliced_wasserstein_jax(
    jnp.asarray(us_ref[ref_idxs]),
    jnp.asarray(hmala_med),
    n_projections=n_projections,
    seed=SEED,
))
swd_98 = float(sliced_wasserstein_jax(
    jnp.asarray(us_ref[ref_idxs]),
    jnp.asarray(hmala_98),
    n_projections=n_projections,
    seed=SEED,
))
print(f"Base SWDs: obs={base_swd:.4f}  med={swd_med:.4f}  98th={swd_98:.4f}")
base_swd_list = [base_swd, swd_med, swd_98]


# ---------------------------------------------------------------------------
# Main loop: one SMAC sweep per regime
# ---------------------------------------------------------------------------
all_results = {}

for regime in REGIMES:
    regime_name = regime["name"]
    n_train     = regime["n_train"]
    print(f"\n{'='*70}")
    print(f"  Regime: {regime_name}  (n_train={n_train})")
    print(f"{'='*70}\n")

    # Fixed-seed shuffle so each regime uses the same data ordering
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(x1_full))[:n_train]
    train_data = x1_full[perm]
    x0_data = x0_full[perm]

    run = wandb.init(
        project="pPoisson - SI hyperparams - regimes",
        name=f"smac-{regime_name}-n{n_train}",
        group="regime-sweep",
        reinit=True,
        config={
            "regime":       regime_name,
            "n_train":      n_train,
            "interpolant":  "trig_interpolant",
            "dataset":      "p_poisson_hmala",
        },
        settings=wandb.Settings(start_method="thread"),
    )

    objective = SiOdeSmacRegimes(
        train_dim=n_train,
        train_data=train_data,
        x0_data=x0_data,
        yu_dimension=yu_dimension,
        interpolant_args=interpolant_args,
        reference_sampler_args=reference_sampler_args,
        ncond_vals=cond_values,
        hmala_list=hmala_list,
        base_swd_list=base_swd_list,
        nsamples=nsamples,
        n_projections=n_projections,
        swd_seed=SEED,
        pca_decode=pca_decode,
        sample_extra=sample_extra,
        u0_cond=us_test_pca,
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

    default_loss   = smac.validate(objective.configspace.get_default_configuration())
    incumbent_loss = smac.validate(incumbent)
    best_hp        = dict(incumbent)
    best_hp["interpolant"] = "linear_interpolant"

    print(f"\n  Default loss  : {default_loss:.4f}")
    print(f"  Incumbent loss: {incumbent_loss:.4f}")
    print(f"  Best HP       : {best_hp}")

    save_path = os.path.join(output_root, f"best_hyperparams_{regime_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(best_hp, f)
    print(f"  Saved → {save_path}")

    all_results[regime_name] = {
        "best_hp":        best_hp,
        "incumbent_loss": incumbent_loss,
        "default_loss":   default_loss,
    }

    np.save(
        os.path.join(output_root, f"incumbent_loss_{regime_name}.npy"),
        np.array([incumbent_loss]),
    )
    wandb.finish()

    del smac, objective, scenario, initial_design, incumbent
    del train_data, x0_data, perm
    jax.clear_caches()
    gc.collect()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, res in all_results.items():
    print(f"\n  {name:6s}  loss={res['incumbent_loss']:.4f}  HP={res['best_hp']}")
print("\nCode terminated.")
