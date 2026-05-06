import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx",     type=int, default=0,            help="Index of GPU to use")
parser.add_argument("--run_id",      type=int, default=0,            help="Run index — controls seed and output folder")
parser.add_argument("--ref_root",    type=str, default="../mcmc_256", help="Root directory containing reference chain_* folders")
parser.add_argument("--chain_start", type=int, default=0,            help="First chain index to load y_obs from (inclusive)")
parser.add_argument("--chain_end",   type=int, default=865,          help="Last chain index to load y_obs from (exclusive)")
args = parser.parse_args()
GPU_IDX = args.gpu_idx
run_id  = args.run_id

import os
os.environ["CUDA_VISIBLE_DEVICES"]          = f"{GPU_IDX}"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"]             = "cuda_malloc_async"

import gc
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import diffrax
from jax import random
from tqdm.auto import tqdm
from typing import Callable, List
import equinox as eqx

from triangular_transport.flows.flow_trainer import NNTrainer
from triangular_transport.flows.interpolants import linear_interpolant, linear_interpolant_der
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.flows.dataloaders import gaussian_reference_sampler


# ---------------------------------------------------------------------------
# Architecture — identical to sample_convergence.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# PCA helpers — identical to sample_convergence.py (threshold 0.99)
# ---------------------------------------------------------------------------
def get_pca_fns(us, explained_var_threshold=0.99):
    n = us.shape[0]
    mean_us = us.mean(axis=0)
    X = us - mean_us

    _, S, Vt = np.linalg.svd(X / np.sqrt(n - 1), full_matrices=False)
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
        eps = np.random.randn(n_samp, S_res.shape[0])
        return (eps * S_res) @ V_res.T

    def extra_cov():
        return V_res @ np.diag(S_res ** 2) @ V_res.T

    return pca_encode, pca_decode, k, sample_extra, extra_cov


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Hyperparameters from sample_convergence.py
    best_hyperparams = {
        "hidden_layer":      256,
        "num_hidden_layers": 3,
        "peak_value":        0.002,
        "weight_decay":      0.0002,
    }
    EPOCHS    = 1300       # 650 * 2, matching sample_convergence.py
    train_dim = 16384      # 2^14 — equal to the max MCMC budget (8064 steps × 2 + 256 MAP)
    nsamples  = 40000      # posterior samples generated per conditioning value

    nx = ny     = 20
    flat_length = (nx + 1) * (ny + 1)  # 21 * 21 = 441  (z=0 surface)
    n_obs       = 100

    chain_start = args.chain_start
    chain_end   = args.chain_end
    N_CHAINS    = chain_end - chain_start

    output_root = "budget_transport_samps"
    output_dir  = os.path.join(output_root, f"run_{run_id:02d}")
    os.makedirs(output_dir, exist_ok=True)

    # -- Training data -------------------------------------------------------
    print("Loading training data...")
    _here = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(_here, "..", "training_dataset")

    ys_all = np.load(os.path.join(data_root, "solutions_delta.npy"), mmap_mode="r")    # (250000, 100)
    us_all = (np.load(os.path.join(data_root, "parameters_delta.npy"), mmap_mode="r")
              .reshape(-1, 21, 21, 3, order="F")[:, :, :, 0]
              .reshape(-1, flat_length))                                                  # (250000, 441)
    print(f"Full dataset: ys={ys_all.shape}, us={us_all.shape}")

    ys = np.array(ys_all[:train_dim])
    us = np.array(us_all[:train_dim])
    us_ref  = np.array(us_all[train_dim: train_dim * 2]).copy()
    np.random.shuffle(us_ref)
    us_test = np.array(us_all[train_dim * 2:]).copy()

    # -- Observation normalizer ----------------------------------------------
    ys_normalizer = UnitGaussianNormalizer(ys)
    ys_normalized = ys_normalizer.encode()

    # -- PCA on reference parameters -----------------------------------------
    print("Computing PCA on reference parameters...")
    pca_encode, pca_decode, k_pca, sample_extra, _ = get_pca_fns(us_ref)
    print(f"PCA retains {k_pca} components (99% variance explained)")

    us_pca     = jnp.asarray(pca_encode(us))
    us_ref_pca = jnp.asarray(pca_encode(us_ref))
    u0_cond    = jnp.asarray(pca_encode(us_test[:nsamples]))

    x1_data = jnp.hstack([jnp.asarray(ys_normalized), us_pca])
    x0_data = jnp.hstack([jnp.asarray(ys_normalized), us_ref_pca])

    yu_dimension     = (ys_normalized.shape[1], k_pca)  # (100, k_pca)
    dim              = yu_dimension[0] + yu_dimension[1]
    interpolant_args = {"t": None, "x1": None, "x0": None}
    solver_args = {
        "solver":              diffrax.Dopri5(),
        "max_steps":           500_000,
        "stepsize_controller": diffrax.PIDController(rtol=1e-3, atol=1e-5),
    }

    # -- Load y_obs from reference chains ------------------------------------
    print(f"Loading y_obs from reference chains ({args.ref_root}), "
          f"chains {chain_start}–{chain_end - 1}...")
    all_y_obs = []
    abs_indices = list(range(chain_start, chain_end))
    for idx in tqdm(abs_indices, desc="chains"):
        path = os.path.join(args.ref_root, f"chain_{idx:03d}", "y_obs.npy")
        all_y_obs.append(np.load(path))

    # -- Build and train the model -------------------------------------------
    batch_size      = max(1, min(4096, train_dim) - 1)
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    steps           = steps_per_epoch * EPOCHS

    key, model_key = random.split(random.PRNGKey(run_id + 1))

    model = MLP(
        key=model_key,
        dim=dim,
        w=best_hyperparams["hidden_layer"],
        num_layers=best_hyperparams["num_hidden_layers"],
        activation_fn=jax.nn.gelu,
        time_varying=True,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=best_hyperparams["peak_value"],
        warmup_steps=max(50, steps // 20),
        decay_steps=steps,
        end_value=1e-4,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(schedule, weight_decay=best_hyperparams["weight_decay"]),
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
    )

    print(f"Training with {train_dim:,} samples, {steps:,} steps ({EPOCHS} epochs)...")
    t_train = time.perf_counter()
    trainer.train(
        train_data=x1_data,
        train_dim=train_dim,
        batch_size=batch_size,
        steps=steps,
        x0_data=x0_data,
        print_every=10_000,
    )
    elapsed_train = time.perf_counter() - t_train
    print(f"Training done in {elapsed_train / 60:.1f} min.")

    # -- Generate posterior samples for all chains ---------------------------
    ncond_vals = [ys_normalizer.encode(y_obs) for y_obs in all_y_obs]

    print(f"\nGenerating {nsamples:,} samples for each of {N_CHAINS} posteriors...")
    t_sample = time.perf_counter()
    cond_samples_list = trainer.conditional_sample(
        cond_values=ncond_vals,
        u0_cond=u0_cond,
        nsamples=nsamples,
        solver_args=solver_args,
    )

    for local_idx, (abs_idx, cond_samples) in enumerate(
        tqdm(zip(abs_indices, cond_samples_list), total=N_CHAINS, desc="decoding")
    ):
        u_pca_gen   = np.array(cond_samples[:, yu_dimension[0]:])  # (nsamples, k_pca)
        u_flat_gen  = np.array(pca_decode(u_pca_gen))              # (nsamples, flat_length)
        extra_noise = sample_extra(nsamples)                        # (nsamples, flat_length)
        u_flat      = u_flat_gen + extra_noise

        out_path = os.path.join(output_dir, f"si_samps_{abs_idx:03d}.npy")
        np.save(out_path, u_flat)

    elapsed_sample = time.perf_counter() - t_sample
    print(f"\nSampling done in {elapsed_sample / 60:.1f} min.")
    print(f"Total: {(elapsed_train + elapsed_sample) / 60:.1f} min.")


if __name__ == "__main__":
    main()
