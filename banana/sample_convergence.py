import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, vmap, random
import optax
from seaborn import kdeplot
import wandb
from ot.sliced import sliced_wasserstein_distance as swd
import argparse

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
from triangular_transport.flows.methods.sampling import inf_train_gen
from triangular_transport.flows.dataloaders import (
    standard_gaussian_reference_sampler,
)

from triangular_transport.kernels.kernel_tools import (
    get_gaussianRBF,
    vectorize_kfunc,
    get_sum_of_kernels,
)
from ksd import compute_ksd

plt.style.use("ggplot")


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()
RANK = args.run_id

run = wandb.init(
    project="2D Convergence - Banana - with ksd and mcmc",
    name=f"run={RANK}-ode-converge",
)

# Kernel stuff for MMD

output_root = "converge_results_banana"
output_dir = os.path.join(output_root, f"run_{RANK:02d}")
os.makedirs(output_dir, exist_ok=True)

nsamples = 20000
seed = 1
n_projections = 2048
rng = np.random.RandomState(seed)
# base_data = inf_train_gen(data="banana", rng=rng, batch_size=nsamples)
# us_base = base_data[:, 1:2]
samps = np.load("rej_samples_1.npy")[
    np.random.choice(100000, size=(nsamples,)), :
]
swd_mcmc = np.load("swd_array_mcmc.npy")
us_base = rng.randn(nsamples, 1)
print("About to calculate swd...")
base_swd = swd(
    us_base,
    samps,
    n_projections=n_projections,
    seed=seed,
)
print(f"This is the base swd: {base_swd}")
gamma = median_heuristic_sigma_jax(us_base, samps)
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


print(f"This is the base MMD: {get_kme(us_base, samps)}")

a = 2
b = 0.1
sigma_x = 1
cond_no = -1.0


def log_density_V(u):
    y = cond_no
    scale = 1 / (2 * sigma_x**2)
    sum_part = (a * (y + b * (u**2 + a**2))) ** 2 + (u**2) / (a**2)
    return jnp.squeeze(-scale * sum_part)


score_V = vmap(vmap(grad(log_density_V)))
base_ksd = compute_ksd(samps, score_V)

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
# sample_no_list = [30000, 40000]
loss_iter = 1
# hidden_layer_list = [[256] * 2, [256] * 3, [256] * 4, [512] * 3, [512] * 4, [512] * 6, [1024] * 8]
swd_array = np.zeros(len(sample_no_list))
mmd_array = np.zeros(len(sample_no_list))
ksd_array = np.zeros(len(sample_no_list))
epochs = 8000
for i, sample_no in tqdm(enumerate(sample_no_list)):
    key = random.PRNGKey(i + 1)
    key, key1, key2, key3, key4 = random.split(key, 5)
    key, subkey1 = random.split(key, 2)
    train_dim = sample_no
    batch_size = 2048
    batch_size = min(batch_size, train_dim)
    batch_size -= 1
    steps_per_epoch = int(np.ceil(train_dim / batch_size))
    steps = steps_per_epoch * epochs
    print_every = 5000
    yu_dimension = (1, 1)
    target_data = inf_train_gen(data="banana", rng=rng, batch_size=train_dim)
    dim = yu_dimension[0] + yu_dimension[1]
    # hidden_layer_list = [256] * 4 if train_dim < 8000 else [1024] * 8
    # hidden_layer_list = [512] * 6
    hidden_layer_list = [256] * 3
    # hidden_layer_list = [1024] * 8
    model = MLP(
        key=key1,
        dim=dim,
        time_varying=True,
        w=hidden_layer_list,
        num_layers=len(hidden_layer_list) + 1,
        activation_fn=jax.nn.gelu,  # GeLU worked well
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=2_000,
        decay_steps=steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule)
    )

    interpolant = linear_interpolant
    interpolant_der = linear_interpolant_der
    interpolant_args = {"t": None, "x1": None, "x0": None}

    trainer = NNTrainer(
        target_density=None,
        model=model,
        optimizer=optimizer,
        interpolant=interpolant,
        interpolant_der=interpolant_der,
        reference_sampler=standard_gaussian_reference_sampler,
        loss=vec_field_loss,
        interpolant_args=interpolant_args,
        yu_dimension=yu_dimension,
    )

    trainer.train(
        train_data=target_data,
        train_dim=train_dim,
        batch_size=batch_size,
        steps=steps,
        x0_data=None,
        print_every=print_every,
    )
    print("Calculating MMD and SWD and KSD...")
    swd_iter_array = np.zeros(loss_iter)
    mmd_iter_array = np.zeros(loss_iter)
    ksd_iter_array = np.zeros(loss_iter)
    for j in range(loss_iter):
        cond_samples = trainer.conditional_sample(
            cond_values=cond_no,
            nsamples=nsamples,
            u0_cond=None,
        )

        us_gen = cond_samples[:, 1:2]
        mmd_iter_array[j] = MMD(us_gen, samps)
        swd_iter_array[j] = (
            swd(np.array(us_gen), samps, n_projections=n_projections, seed=seed)
            / base_swd
        )
        ksd_iter_array[j] = compute_ksd(us_gen, score_V) / base_ksd
    mmd_array[i] = np.mean(mmd_iter_array)
    swd_array[i] = np.mean(swd_iter_array)
    ksd_array[i] = np.mean(ksd_iter_array)
    print(f"This is the variance for the SWD: {np.var(swd_iter_array)}")
    wandb.log(
        {
            "relative error (swd)": swd_array[i],
            "relative error (swd) - mcmc": swd_mcmc[i],
        },
        step=sample_no,
    )
    # wandb.log({"relative error (swd) - mcmc": swd_mcmc[i]}, step=sample_no)
    wandb.log({"relative error (mmd)": mmd_array[i]}, step=sample_no)
    wandb.log({"relative error (ksd)": ksd_array[i]}, step=sample_no)
    print(f"This is the relative MMD error: {mmd_array[i]}")
    print(f"This is the relative SWD error: {swd_array[i]}")

print("Successfully trained all models and now saving results!")
np.save(os.path.join(output_dir, "nn_conv_mmd.npy"), mmd_array)
np.save(os.path.join(output_dir, "nn_conv_swd.npy"), swd_array)
np.save(os.path.join(output_dir, "nn_conv_ksd.npy"), ksd_array)
