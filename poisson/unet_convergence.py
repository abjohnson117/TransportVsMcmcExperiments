"""
Convergence study for UNet-based triangular flow on the Darcy-flow (Poisson)
problem.

Mirrors sample_convergence.py but uses FlatDarcyUNet (from unet.py) instead
of the MLP baseline, and operates directly in the original 33×33 parameter-
field space — no PCA is used.

Key differences from the MLP baseline:
  - u_dim = 1089 (33×33 flat field) instead of 200 PCA components.
  - The field is globally normalised (zero mean, unit std computed from the
    training reference split) so the reference distribution ≈ N(0, I).
  - x0_data uses normalised prior samples from the reference split, not
    whitened PCA vectors.
  - u0_cond at test time is normalised us_test (prior samples), not PCA codes.
  - Generated samples are denormalised before SWD comparison with MCMC chains.
  - No pca_decode / residual-variance injection step needed.

Run with:
    CUDA_VISIBLE_DEVICES=2 python unet_convergence.py [--run_id N]

Results saved to  convergence_results_unet/run_NN/
"""

import gc
import os
import pickle
import time
import argparse
from typing import Callable, List

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import diffrax
import h5py
import wandb
from jax import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from triangular_transport.flows.flow_trainer import NNTrainer
from triangular_transport.flows.interpolants import (
    trig_interpolant,
    trig_interpolant_der,
    linear_interpolant,
    linear_interpolant_der,
)
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.flows.dataloaders import gaussian_reference_sampler

from darcy_unet import FlatDarcyUNet


# ---------------------------------------------------------------------------
# JAX sliced Wasserstein distance (same as sample_convergence.py)
# ---------------------------------------------------------------------------
def sliced_wasserstein_jax(x, y, n_projections=512, seed=42, chunk_size=64, p=2):
    n_x, d = x.shape
    n_y    = y.shape[0]
    assert n_projections % chunk_size == 0

    t_x = (jnp.arange(n_x, dtype=jnp.float32) + 0.5) / n_x
    t_y = (jnp.arange(n_y, dtype=jnp.float32) + 0.5) / n_y
    t   = jnp.sort(jnp.concatenate([t_x, t_y]))

    @jax.jit
    def chunk_cost(key):
        theta = jax.random.normal(key, (chunk_size, d))
        theta = theta / jnp.linalg.norm(theta, axis=1, keepdims=True)
        xp    = jnp.sort(x @ theta.T, axis=0).T
        yp    = jnp.sort(y @ theta.T, axis=0).T

        def one_cost(xi, yi):
            return jnp.mean((jnp.interp(t, t_x, xi) - jnp.interp(t, t_y, yi)) ** p)

        return jnp.mean(jax.vmap(one_cost)(xp, yp))

    keys = jax.random.split(jax.random.PRNGKey(seed), n_projections // chunk_size)
    cost = jnp.mean(jnp.array([chunk_cost(k) for k in keys]))
    return cost ** (1.0 / p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_data_h5(path="data.h5"):
    with h5py.File(path, "r") as f:
        targets = f["/target"][...]
        data    = f["/data"][...]
    return targets, data


# ---------------------------------------------------------------------------
# Arguments and output directory
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--gpu",    type=int, default=2,
                    help="Which CUDA device to use (default: 2)")
args = parser.parse_args()
RANK = args.run_id

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

output_root = "convergence_results_unet"
output_dir  = os.path.join(output_root, f"run_{RANK:02d}")
os.makedirs(output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Constants  (same field / chain sizes as sample_convergence.py)
# ---------------------------------------------------------------------------
train_dim_full = 100_000
nsamples       = 40_000
nx = ny        = 33
flat_length    = nx * ny           # 1089
n_projections  = 2048
swd_seed       = 42
EPOCHS         = 650

RESHAPE_NO      = 200_000
THIN_FACTOR     = 10
CHAIN_ITERS     = 2
AUX_RESHAPE_NO  = 200_000
AUX_THIN_FACTOR = 10
AUX_CHAIN_ITERS = 2

COND_LABELS = ["obs", "med", "98"]

# UNet architecture hyperparameters
UNET_BASE_CH        = 48
UNET_COND_EMBED_DIM = 32
UNET_ENC_HIDDEN     = 128
UNET_T_EMBED_DIM    = 8
UNET_T_MLP_HIDDEN   = 96

# Optimizer hyperparameters  (same schedule as MLP baseline in sample_convergence.py)
PEAK_LR      = 2e-3
WEIGHT_DECAY = 2e-4
CLIP_NORM    = 5.0


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
run = wandb.init(
    project="Poisson - UNet convergence",
    config={
        "dataset":    "darcy_flow_si_hmala_noise",
        "model":      "FlatDarcyUNet",
        "base_ch":    UNET_BASE_CH,
        "train_dim":  train_dim_full,
        "nsamples":   nsamples,
        "interpolant": "linear_interpolant",
        "peak_lr":    PEAK_LR,
    },
    name=f"run={RANK}_unet_convergence",
)


# ---------------------------------------------------------------------------
# Load h-MALA chains
# ---------------------------------------------------------------------------
print("Loading h-MALA chains ...")
for label, folder, no, thin, iters in [
    ("obs", "mcmc_main_ref", RESHAPE_NO,     THIN_FACTOR,     CHAIN_ITERS),
    ("med", "mcmc_med_ref",  AUX_RESHAPE_NO, AUX_THIN_FACTOR, AUX_CHAIN_ITERS),
    ("98",  "mcmc_98_ref",   AUX_RESHAPE_NO, AUX_THIN_FACTOR, AUX_CHAIN_ITERS),
]:
    chains = []
    for i in tqdm(range(iters), desc=f"chains ({label})"):
        path  = os.path.join(folder, f"chain_{i:03d}", "hmala_samples.npz")
        chain = np.load(path)["arr"].reshape(no, flat_length)
        chains.append(chain[::thin])
    arr = np.vstack(chains)
    print(f"h-MALA ({label}) shape: {arr.shape}")
    if label == "obs":
        hmala_obs = arr
    elif label == "med":
        hmala_med = arr
    else:
        hmala_98  = arr

hmala_list = [hmala_obs, hmala_med, hmala_98]


# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
print("Loading training data ...")
ys_all = np.load("training_dataset/solutions_grid_noise.npy")
us_all = np.load("training_dataset/parameters_noise.npy").reshape(-1, flat_length)
print(f"Full dataset: ys={ys_all.shape}  us={us_all.shape}")

ys  = ys_all[:train_dim_full]
us  = us_all[:train_dim_full]
us_ref  = us_all[train_dim_full : train_dim_full * 2].copy()
us_test = us_all[train_dim_full * 2 :].copy()
np.random.shuffle(us_ref)


# ---------------------------------------------------------------------------
# Normalise observations y  (same as MLP baseline)
# ---------------------------------------------------------------------------
targets, yobs = read_data_h5()
yobs     = np.load("data_obs.npy")
yobs_med = np.load("data_50.npy")
yobs_98  = np.load("data_98.npy")

ys_normalizer   = UnitGaussianNormalizer(ys)
ys_norm         = ys_normalizer.encode()
yobs_norm       = ys_normalizer.encode(yobs)
ymed_norm       = ys_normalizer.encode(yobs_med)
y98_norm        = ys_normalizer.encode(yobs_98)
ncond_vals      = [yobs_norm, ymed_norm, y98_norm]

y_dim = ys_norm.shape[1]   # 100
u_dim = flat_length        # 1089


# ---------------------------------------------------------------------------
# Normalise the u field
#
# We compute global statistics (scalar mean + std) from the reference split so
# that the normalised field values are centred and have unit overall variance.
# This makes the reference distribution N(0, ~1) per dimension (approximately),
# consistent with using gaussian_reference_sampler with mu=0, sigma=1 in
# NNTrainer — the same approximation that whitened PCA provides in the MLP
# baseline.
# ---------------------------------------------------------------------------
u_mean = float(us_ref.mean())
u_std  = float(us_ref.std())
print(f"u field stats (from reference split):  mean={u_mean:.4f}  std={u_std:.4f}")

def normalise_u(u):
    return (u - u_mean) / u_std

def denormalise_u(u_n):
    return u_n * u_std + u_mean

us_norm     = normalise_u(us)
us_ref_norm = normalise_u(us_ref)
us_test_norm = normalise_u(us_test)


# ---------------------------------------------------------------------------
# Joint training arrays
#   x1 = [y_norm, u_norm]  — target  (joint samples from p(y, u))
#   x0 = [y_norm, u_ref_norm]  — reference  (same y, different independent u)
# The x0 reference distribution for u is normalised prior samples.
# ---------------------------------------------------------------------------
x1_full = np.hstack([np.asarray(ys_norm),     us_norm])
x0_full = np.hstack([np.asarray(ys_norm),     us_ref_norm])

yu_dimension     = (y_dim, u_dim)   # (100, 1089)
interpolant_args = {"t": None, "x1": None, "x0": None}

# ODE initial conditions at test time: normalised prior samples
u0_cond = jnp.asarray(us_test_norm)   # (n_test, 1089)

solver_args = {
    "solver":             diffrax.Dopri5(),
    "max_steps":          50_000,
    "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6),
}

# gaussian_reference_sampler args: N(0, 1) is a good approximation of the
# normalised prior (see comment above).
reference_sampler_args = {
    "mu":         jnp.zeros(u_dim),
    "sigma":      1.0,
    "normalizer": None,
}


# ---------------------------------------------------------------------------
# Base SWDs: prior (reference samples in original space) vs. each h-MALA set
# ---------------------------------------------------------------------------
print("Computing base SWDs (prior vs. h-MALA in original field space) ...")
rng_base  = np.random.default_rng(swd_seed)
ref_idxs  = rng_base.choice(len(us_ref), size=nsamples, replace=False)
us_ref_sub = us_ref[ref_idxs]   # original (not normalised) for SWD in field space

base_swd_list = []
for label, h_flat in zip(COND_LABELS, hmala_list):
    bswd = float(sliced_wasserstein_jax(
        jnp.asarray(us_ref_sub),
        jnp.asarray(h_flat),
        n_projections=n_projections,
        seed=swd_seed,
    ))
    base_swd_list.append(bswd)
    print(f"  Base SWD ({label}): {bswd:.6f}")

np.save(os.path.join(output_dir, "base_swd_list.npy"), np.array(base_swd_list))


# ---------------------------------------------------------------------------
# Sample-size list  (same grid as sample_convergence.py)
# ---------------------------------------------------------------------------
sample_no_list = [2 ** i for i in range(1, 15)]
sample_no_list += [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
sample_no_list  = sorted(set(sample_no_list))
n_sizes         = len(sample_no_list)

n_cond        = len(COND_LABELS)
swd_array     = np.zeros((n_sizes, n_cond))
rel_swd_array = np.zeros((n_sizes, n_cond))
u_mean_arrays = [np.zeros((n_sizes, nx, ny)) for _ in COND_LABELS]
u_var_arrays  = [np.zeros((n_sizes, nx, ny)) for _ in COND_LABELS]


# ---------------------------------------------------------------------------
# Convergence loop
# ---------------------------------------------------------------------------
t_start = time.time()

for i, sample_no in tqdm(enumerate(sample_no_list), total=n_sizes, desc="sample sizes"):
    print(f"\n{'='*60}")
    print(f"  Training UNet with {sample_no:,} samples  ({i+1}/{n_sizes})")
    print(f"{'='*60}")

    key = random.PRNGKey(i + 1 + RANK)
    key, model_key = random.split(key)

    batch_size      = max(1, min(4096, sample_no) - 1)
    steps_per_epoch = int(np.ceil(sample_no / batch_size))
    steps           = steps_per_epoch * EPOCHS

    model = FlatDarcyUNet(
        y_dim=y_dim,
        u_dim=u_dim,
        true_hw=nx,
        base_ch=UNET_BASE_CH,
        cond_embed_dim=UNET_COND_EMBED_DIM,
        enc_hidden=UNET_ENC_HIDDEN,
        t_embed_dim=UNET_T_EMBED_DIM,
        t_mlp_hidden=UNET_T_MLP_HIDDEN,
        key=model_key,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=PEAK_LR,
        warmup_steps=max(50, steps // 20),
        decay_steps=steps,
        end_value=1e-4,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )

    trainer = NNTrainer(
        target_density=None,
        model=model,
        optimizer=optimizer,
        interpolant=linear_interpolant,
        interpolant_der=linear_interpolant_der,
        reference_sampler=gaussian_reference_sampler,
        loss=vec_field_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
        reference_sampler_args=reference_sampler_args,
    )

    trainer.train(
        train_data=x1_full[:sample_no],
        train_dim=sample_no,
        batch_size=batch_size,
        steps=steps,
        x0_data=x0_full[:sample_no],
        print_every=10_000,
    )

    # -- Generate posterior samples -------------------------------------------
    print("  Generating samples (3 conditions) ...")
    cond_samples_list = trainer.conditional_sample(
        cond_values=ncond_vals,
        u0_cond=u0_cond,
        nsamples=len(u0_cond),
        solver_args=solver_args,
    )

    # -- Per-condition metrics -------------------------------------------------
    log_dict = {}
    for j, (label, cond_samples, h_flat, base_swd) in enumerate(zip(
        COND_LABELS, cond_samples_list, hmala_list, base_swd_list
    )):
        # Extract u from joint output and denormalise to original field space
        u_norm_gen = np.array(cond_samples[:, y_dim:])          # (nsamples, 1089)
        u_flat_gen = denormalise_u(u_norm_gen)                   # (nsamples, 1089)

        u_grid              = u_flat_gen.reshape(nsamples, nx, ny)
        u_mean_arrays[j][i] = np.mean(u_grid, axis=0)
        u_var_arrays[j][i]  = np.var(u_grid,  axis=0)

        print(f"  Computing SWD ({label}, original field space) ...")
        swd_val = float(sliced_wasserstein_jax(
            jnp.asarray(u_flat_gen),
            jnp.asarray(h_flat),
            n_projections=n_projections,
            seed=swd_seed,
        ))
        swd_array[i, j]     = swd_val
        rel_swd_array[i, j] = swd_val / base_swd

        log_dict[f"relative_swd_{label}"] = rel_swd_array[i, j]
        log_dict[f"swd_{label}"]          = swd_array[i, j]
        print(f"  [{label}] SWD={swd_val:.6f}  rel={rel_swd_array[i,j]:.4f}")

    wandb.log(log_dict, step=sample_no)

    del model, trainer, optimizer, cond_samples_list
    jax.clear_caches()
    gc.collect()

elapsed = time.time() - t_start
print(f"\nTotal elapsed time: {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.save(os.path.join(output_dir, "sample_no_list.npy"), np.array(sample_no_list))
np.save(os.path.join(output_dir, "swd_array.npy"),      swd_array)
np.save(os.path.join(output_dir, "rel_swd_array.npy"),  rel_swd_array)
for j, label in enumerate(COND_LABELS):
    np.save(os.path.join(output_dir, f"u_mean_{label}.npy"), u_mean_arrays[j])
    np.save(os.path.join(output_dir, f"u_var_{label}.npy"),  u_var_arrays[j])
np.save(os.path.join(output_dir, "elapsed_time.npy"),   np.array([elapsed]))
np.save(os.path.join(output_dir, "u_norm_stats.npy"),   np.array([u_mean, u_std]))
print("Results saved.")


# ---------------------------------------------------------------------------
# Plots: posterior mean and variance fields
# ---------------------------------------------------------------------------
n_cols = 6
n_rows = int(np.ceil(n_sizes / n_cols))

for j, label in enumerate(COND_LABELS):
    for stat, arrays, fname in [
        ("Mean",     u_mean_arrays[j], f"param_field_means_{label}.png"),
        ("Variance", u_var_arrays[j],  f"param_field_variances_{label}.png"),
    ]:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        ax = ax.flatten()
        for l, sample_no in enumerate(sample_no_list):
            im = ax[l].imshow(arrays[l], origin="lower", interpolation="bilinear")
            ax[l].set_title(rf"{stat} $n={sample_no:,}$", fontsize=7)
            fig.colorbar(im, ax=ax[l], fraction=0.046, pad=0.04)
        for l in range(n_sizes, len(ax)):
            ax[l].set_visible(False)
        fig.suptitle(f"UNet {stat} fields — condition: {label}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convergence curves
# ---------------------------------------------------------------------------
colors  = ["C0", "C1", "C2"]
markers = ["o", "s", "^"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for j, (label, color, marker) in enumerate(zip(COND_LABELS, colors, markers)):
    axes[0].plot(sample_no_list, swd_array[:, j],
                 marker=marker, color=color, label=label)
    axes[1].plot(sample_no_list, rel_swd_array[:, j],
                 marker=marker, color=color, label=label)

for ax, ylabel, title in zip(
    axes,
    ["SWD (field space)", "Relative SWD"],
    ["SWD vs. Sample Size (UNet)", "Relative SWD vs. Sample Size (UNet)"],
):
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergence_curves.png"), dpi=150)
plt.close(fig)

print("Plots saved. Done.")
wandb.finish()
