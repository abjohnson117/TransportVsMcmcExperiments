import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, vmap, random
import optax
import diffrax
from seaborn import kdeplot
import wandb
from ot.lp import wasserstein_1d
import argparse
import pickle

from triangular_transport.flows.flow_trainer import NNTrainer

from triangular_transport.flows.interpolants import linear_interpolant, linear_interpolant_der

from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.dataloaders import standard_gaussian_reference_sampler

plt.style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()
RANK = args.run_id

run = wandb.init(
    project="2D Convergence - tanh - Transport - ode",
    name=f"run={RANK}-ode-converge",
)

# Kernel stuff for MMD

output_root0 = "converge_results_0"
output_dir0 = os.path.join(output_root0, f"run_{RANK:02d}")
os.makedirs(output_dir0, exist_ok=True)

output_root1 = "converge_results_2"
output_dir1 = os.path.join(output_root1, f"run_{RANK:02d}")
os.makedirs(output_dir1, exist_ok=True)

output_root4 = "converge_results_3"
output_dir4 = os.path.join(output_root4, f"run_{RANK:02d}")
os.makedirs(output_dir4, exist_ok=True)

nsamples = 100000
seed = 1
rng = np.random.RandomState(seed)
samps0 = np.load("samps_0.npy")
samps2 = np.load("samps_2.npy")
samps3 = np.load("samps_3.npy")
us_base = rng.randn(nsamples,)
print("About to calculate wd...")
base_wd0 = wasserstein_1d(
    us_base,
    samps0.squeeze(),
    p=2,
)
base_wd2 = wasserstein_1d(
    us_base,
    samps2.squeeze(),
    p=2,
)
base_wd3 = wasserstein_1d(
    us_base,
    samps3.squeeze(),
    p=2,
)
print(f"This is the base wd0: {base_wd0}")
print(f"This is the base wd1: {base_wd2}")
print(f"This is the base wd4: {base_wd3}")

cond_no0 = 0.0
cond_no1 = 2.0
cond_no4 = -3.0
cond_vals = [cond_no0, cond_no1, cond_no4]

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
epochs = 400
rng2 = np.random.RandomState(RANK)
ys = np.random.uniform(low=-3.0, high=3.0, size=(100000, 1))
us = np.tanh(ys) + np.random.gamma(shape=1.0, scale=0.3, size=(100000, 1))
x1_data = np.hstack([ys, us])
solver_args = {"solver": diffrax.Dopri5(), "max_steps": 50000, "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)}
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
    yu_dimension = (1, 1)
    dim = yu_dimension[0] + yu_dimension[1]
    hidden_layer_list = [512] * 6 #TODO: This needs to be tuned to match performance on MCMC a bit better
    target_data = x1_data[:sample_no, :]
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
        warmup_steps=400,
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
    cond_samples = trainer.conditional_sample(
        cond_values=cond_vals,
        nsamples=nsamples,
        u0_cond=None,
    )
    wd_iter_array = np.zeros(3)
    for k, cond_sample in enumerate(cond_samples):
        us_gen = cond_samples[k][:, 1]
        if k == 0:
            samps = samps0
            base_wd = base_wd0
        elif k == 1:
            samps = samps2
            base_wd = base_wd2
        elif k == 2:
            samps = samps3
            base_wd = base_wd3
        wd_iter_array[k] = (
            wasserstein_1d(np.array(us_gen), samps.squeeze(), p=2)
            / base_wd
        )
    wd0_array[i] = wd_iter_array[0]
    wd2_array[i] = wd_iter_array[1]
    wd3_array[i] = wd_iter_array[2]
    wandb.log(
        {
            "relative error (wd): 0": wd0_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (wd): 2.45": wd2_array[i]
        },
        step=sample_no,
    )
    wandb.log(
        {
            "relative error (wd): -2.95": wd3_array[i]
        },
        step=sample_no,
    )
    print(f"This is the relative WD error on 0.0: {wd0_array[i]}")
    print(f"This is the relative WD error on 2.45: {wd2_array[i]}")
    print(f"This is the relative WD error on -2.95: {wd3_array[i]}")

np.save(os.path.join(output_dir0, f"nn_conv_wd_{RANK}.npy"), wd0_array)

np.save(os.path.join(output_dir1, f"nn_conv_wd_{RANK}.npy"), wd2_array)

np.save(os.path.join(output_dir4, f"nn_conv_wd_{RANK}.npy"), wd3_array)

print("Successfully trained all models and now saving results!")