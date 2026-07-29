import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import grad, vmap, random
import optax
import diffrax
from seaborn import kdeplot
import wandb
from ot.sliced import sliced_wasserstein_distance as swd
import argparse
import pickle
import time

from ksd import median_heuristic_sigma_jax

from triangular_transport.flows.flow_trainer import NNTrainer

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
# from ode_hyperparam import MLP
from triangular_transport.flows.dataloaders import log_normal_reference_sampler
from triangular_transport.flows.methods.utils import UnitGaussianNormalizer
from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    vectorize_kfunc,
    get_sum_of_kernels,
)
from conditional_sampling_sde import conditional_sample

def gaussian_reference_sampler(
    key: random.PRNGKey, shape: tuple[int, int], mu, sigma: float, normalizer,
):
    samples = mu + random.normal(key=key, shape=shape) * sigma
    if normalizer is not None:
        samples = normalizer.encode(samples)
    return samples

plt.style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()
RANK = args.run_id

run = wandb.init(
    project="LV Convergence - ODE - Gauss - Trig",
    name=f"run={RANK}-ode-converge-trig",
)

# Defining hyperparams
best_hyperparams = {
    "hidden_layer": 288,
    "num_hidden_layers": 5,
    "peak_value": 0.0025,
    "weight_decay": 1e-5,
    "interpolant": "trig_interpolant",
}
interpolant = trig_interpolant
interpolant_der = trig_interpolant_der
alpha_t = lambda t: jnp.cos(jnp.pi * t * 0.5)
beta_t = lambda t: jnp.sin(jnp.pi * t * 0.5)
alpha_dot = vmap(grad(alpha_t))
beta_dot = vmap(grad(beta_t))
alpha_t_vmap = vmap(alpha_t)
beta_t_vmap = vmap(beta_t)

def drift_to_score(drift):
    @eqx.filter_jit
    def score(t, x):
        # t: (n, 1), x: (n, d)
        # Use Wronskian W = alpha*beta_dot - alpha_dot*beta to avoid dividing
        # by alpha_t or beta_t individually (both hit 0 at t=0 or t=1).
        # For trig interpolant W = pi/2 everywhere — constant, never zero.
        # Derivation: multiply eq (2.4) through by alpha_t*beta_t, giving
        #   score = (beta_t * b_t(x) - beta_dot * x) / W
        t_sq = t.squeeze(-1)                        # (n,)
        a_t = alpha_t_vmap(t_sq)[:, None]           # (n, 1)
        b_t = beta_t_vmap(t_sq)[:, None]            # (n, 1)
        ad_t = alpha_dot(t_sq)[:, None]             # (n, 1)
        bd_t = beta_dot(t_sq)[:, None]              # (n, 1)
        W = a_t * bd_t - ad_t * b_t                # (n, 1), = pi/2 for trig
        num = b_t * drift(jnp.hstack([t, x])) - bd_t * x  # (n, d)
        return num / (a_t * W)                     # (n, d)
    return score

activation = jax.nn.gelu
peak_value = best_hyperparams["peak_value"]
weight_decay = best_hyperparams["weight_decay"]
hidden_layer_list = [best_hyperparams["hidden_layer"]] * best_hyperparams["num_hidden_layers"]

output_root1 = "converge_results_trig_1_regimes"
output_dir1 = os.path.join(output_root1, f"run_{RANK:02d}")
os.makedirs(output_dir1, exist_ok=True)

output_root2 = "converge_results_trig_2_regimes"
output_dir2 = os.path.join(output_root2, f"run_{RANK:02d}")
os.makedirs(output_dir2, exist_ok=True)

output_root3 = "converge_results_trig_3_regimes"
output_dir3 = os.path.join(output_root3, f"run_{RANK:02d}")
os.makedirs(output_dir3, exist_ok=True)

input_root = "true_samps"
cond_val1 = np.load(os.path.join(input_root, "y_obs.npy"))
cond_val2 = np.log(np.load(os.path.join(input_root, "y_moderate.npy")))
cond_val3 = np.log(np.load(os.path.join(input_root, "y_rare.npy")))
nsamples = 25000
cond_values = [cond_val1, cond_val2, cond_val3]
samps1 = np.exp(np.load(os.path.join(input_root, "true_us_obs.npy"))[::20, :])
samps2 = np.exp(np.load(os.path.join(input_root, "true_us_moderate_obs_2.npy"))[::20, :])
samps3 = np.exp(np.load(os.path.join(input_root, "true_us_rare_obs_3.npy"))[::20, :])
samps_list = [samps1, samps2, samps3]
gen_seed = 1
gen_key = random.key(gen_seed)
# Taking prior samples to be the base
us_base = log_normal_reference_sampler(
    key=gen_key,
    shape=samps1.shape,
    mu=jnp.array([-0.125, -3.0, -0.125, -3.0]),
    sigma=1 / jnp.sqrt(2),
)
print("About to calculate wd...")
swd_seed = 42
n_projections = 2048
base_swd1 = swd(
    np.asarray(us_base),
    np.asarray(samps1),
    a=np.ones(len(us_base)) / len(us_base),
    b=np.ones(len(samps1)) / len(samps1),
    seed=swd_seed,
    n_projections=n_projections,
    p=2,
)
base_swd2 = swd(
    np.asarray(us_base),
    np.asarray(samps2),
    a=np.ones(len(us_base)) / len(us_base),
    b=np.ones(len(samps2)) / len(samps2),
    seed=swd_seed,
    n_projections=n_projections,
    p=2,
)
base_swd3 = swd(
    np.asarray(us_base),
    np.asarray(samps3),
    a=np.ones(len(us_base)) / len(us_base),
    b=np.ones(len(samps3)) / len(samps3),
    seed=swd_seed,
    n_projections=n_projections,
    p=2,
)
print(f"This is the base swd (obs): {base_swd1}")
print(f"This is the base swd (moderate): {base_swd2}")
print(f"This is the base swd (rare): {base_swd3}")

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
wd2_array = np.zeros(len(sample_no_list))
wd3_array = np.zeros(len(sample_no_list))
mmd1_array = np.zeros(len(sample_no_list))
mmd2_array = np.zeros(len(sample_no_list))
mmd3_array = np.zeros(len(sample_no_list))
epochs = 300
train_data = np.load("training_data.npy")
ys, us = train_data[:, :18], train_data[:, 18:]
ys_log, us_log = np.log(ys), np.log(us)
ys_normalizer, us_normalizer = UnitGaussianNormalizer(ys_log), UnitGaussianNormalizer(us_log)
ysn, usn = ys_normalizer.encode(), us_normalizer.encode()
ncond_vals = [ys_normalizer.encode(yobs) for yobs in cond_values]
x1_data = jnp.hstack([ysn, usn])

solver_args = {"solver": diffrax.Dopri5(),
               "max_steps": 50000,
               "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)}
reference_sampler_args = {
    "mu": jnp.array([-0.125, -3.0, -0.125, -3.0]),
    "sigma": 1 / jnp.sqrt(2),
    "normalizer": us_normalizer,
}
interpolant_args = {"t": None, "x1": None, "x0": None}
shuffle_key = random.PRNGKey(RANK)
x1_data = x1_data[random.permutation(shuffle_key, len(x1_data)), :]
t_start = time.time()
for i, sample_no in tqdm(enumerate(sample_no_list)):

    key = random.PRNGKey(i + 1 + RANK)
    key, key1, key2, key3, key4 = random.split(key, 5)
    key, subkey1 = random.split(key, 2)
    train_dim = sample_no
    batch_size = 2048
    batch_size = min(batch_size, train_dim)
    batch_size -= 1
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    steps = steps_per_epoch * epochs
    print_every = 10000
    yu_dimension = (18, 4)
    dim = yu_dimension[0] + yu_dimension[1]
    target_data = x1_data[:sample_no, :]
    model = MLP(
        key=key1,
        dim=dim,
        time_varying=True,
        w=hidden_layer_list,
        num_layers=len(hidden_layer_list) + 1,
        activation_fn=activation,  # GeLU worked well
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_value,
        warmup_steps=400,
        decay_steps=steps,
        end_value=1e-5,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay)
    )

    trainer = NNTrainer(
        target_density=None,
        model=model,
        optimizer=optimizer,
        interpolant=interpolant,
        interpolant_der=interpolant_der,
        reference_sampler=gaussian_reference_sampler,
        reference_sampler_args=reference_sampler_args,
        loss=vec_field_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
        t_sampler=None,
    )

    trainer.train(
        train_data=target_data,
        train_dim=train_dim,
        batch_size=batch_size,
        steps=steps,
        x0_data=None,
        print_every=print_every,
    );
    velocity = jax.vmap(trainer.model)
    score = drift_to_score(velocity)
    print("Calculating MMD and SWD and KSD...")
    sde_solver_args = {
        "solver": diffrax.Heun(),
        "saveat": "t1",
        "max_steps": 50000,
    }
    cond_samples = conditional_sample(
        velocity=velocity,
        score=score,
        cond_values=ncond_vals,
        yu_dimension=yu_dimension,
        reference_sampler=gaussian_reference_sampler,
        reference_sampler_args=reference_sampler_args,
        nsamples=nsamples,
        solver_args=sde_solver_args,
        key=key2,
    )
    wd_iter_array = np.zeros(3)
    mmd_iter_array = np.zeros(3)
    for k, cond_sample in enumerate(cond_samples):
        us_gen = cond_samples[k][:, yu_dimension[0] :]
        us_gen = us_normalizer.decode(us_gen)
        us_gen = np.exp(np.array(us_gen))
        if k == 0:
            samps = samps1
            base_swd = base_swd1
        elif k == 1:
            samps = samps2
            base_swd = base_swd2
        elif k == 2:
            samps = samps3
            base_swd = base_swd3
        wd_iter_array[k] = (
            swd(
                np.asarray(us_gen),
                samps,
                a=np.ones(len(us_gen)) / len(us_gen),
                b=np.ones(len(samps.squeeze())) / len(samps.squeeze()),
                p=2,
                n_projections=n_projections,
                seed=swd_seed,
            )
            / base_swd
        )
        mmd_iter_array[k] = get_kme(us_gen, samps)
    wd0_array[i] = wd_iter_array[0]
    wd2_array[i] = wd_iter_array[1]
    wd3_array[i] = wd_iter_array[2]
    mmd1_array[i] = mmd_iter_array[0]
    mmd2_array[i] = mmd_iter_array[1]
    mmd3_array[i] = mmd_iter_array[2]
    wandb.log(
        {
            "relative error (swd): norm": wd0_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (swd): mod": wd2_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (swd): rare": wd3_array[i]
        },
        step=sample_no,
    )
    print(f"This is the relative SWD error on the normal ex: {wd0_array[i]}")
    print(f"This is the relative SWD error on the mod ex: {wd2_array[i]}")
    print(f"This is the relative SWD error on the rare ex: {wd3_array[i]}")
    print(f"This is the relative MMD error on the normal ex: {mmd1_array[i]}")
    print(f"This is the relative MMD error on the mod ex: {mmd2_array[i]}")
    print(f"This is the relative MMD error on the rare ex: {mmd3_array[i]}")

np.save(os.path.join(output_dir1, f"nn_conv_wd_{RANK}.npy"), wd0_array)

np.save(os.path.join(output_dir2, f"nn_conv_wd_{RANK}.npy"), wd2_array)

np.save(os.path.join(output_dir3, f"nn_conv_wd_{RANK}.npy"), wd3_array)

np.save(os.path.join(output_dir1, f"nn_conv_mmd_{RANK}.npy"), mmd1_array)

np.save(os.path.join(output_dir2, f"nn_conv_mmd_{RANK}.npy"), mmd2_array)

np.save(os.path.join(output_dir3, f"nn_conv_mmd_{RANK}.npy"), mmd3_array)

elapsed = time.time() - t_start
np.save(os.path.join(output_dir1, f"elapsed_time_{RANK}.npy"), np.array(elapsed))
print(f"Total elapsed time: {elapsed:.2f}s")
print("Successfully trained all models and now saving results!")
