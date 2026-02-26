import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, vmap, random
import optax
import diffrax
import time
import json
import argparse

from triangular_transport.flows.flow_trainer import (
    NNTrainer,
)

from triangular_transport.flows.interpolants import (
    linear_interpolant,
    linear_interpolant_der,
)
from triangular_transport.flows.loss_functions import vec_field_loss
from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.methods.sampling import inf_train_gen
from triangular_transport.flows.dataloaders import (
    standard_gaussian_reference_sampler,
)

def exp_reference_sampler(
    key: random.PRNGKey, shape: tuple[int, int]
):
    samples = 0.3 * (random.exponential(key=key, shape=shape))
    return samples

@vmap
def sigmoid_interpolant(t: jnp.array, x1: jnp.array, x0: jnp.array):
    return (1 - sigmoid(t)) * x0 + sigmoid(t) * x1

def sigmoid(t: float) -> float:
    return jax.nn.sigmoid(27 * (t - 0.35)) # Changed this to 25
sigmoid_dot = vmap(grad(sigmoid))

@vmap
def sigmoid_interpolant_der(t: jnp.array, x1: jnp.array, x0: jnp.array):
    return sigmoid_dot(t) * (x1 - x0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

plt.style.use("ggplot")

output_root = "nn_results"
output_dir = os.path.join(output_root, f"ode_{args.run_id}")
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
n_train_samps = budget # TODO: Can change this. But the idea of these plots is to use the max budget. NN evals don't require any more forward evals, so we can max out the budget here.
epochs = 400
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.0, high=3.0, size=(budget, ))
conditioning_list = [4 ** i for i in range(9)]
gen_sample_list = list(reversed(conditioning_list))
solver_args = {"solver": diffrax.Dopri5(), "max_steps": 50000, "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)}
# Train the one model
train_dim = n_train_samps
key = random.key(seed=seed)
key, key1 = random.split(key=key, num=2)
batch_size = 2048
batch_size = min(batch_size, train_dim)
batch_size -= 1
steps_per_epoch = int(np.ceil(train_dim / batch_size))
steps = steps_per_epoch * epochs
print_every = 10000
yu_dimension = (1, 1)
# target_data = inf_train_gen(data="banana", rng=None, batch_size=train_dim)
ys = np.random.uniform(low=-3.0, high=3.0, size=(budget, 1))
us = np.tanh(ys) + np.random.gamma(shape=1.0, scale=0.3, size=(budget, 1))
target_data = np.hstack([ys, us])
dim = yu_dimension[0] + yu_dimension[1]
# hidden_layer_list = [512] * 6
hidden_layer_list = [256] * 4
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
    warmup_steps=2000,
    decay_steps=steps,
    end_value=1e-5,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(schedule)
)

interpolant = linear_interpolant
interpolant_der = linear_interpolant_der
interpolant_args = {"t": None, "x1": None, "x0": None}

velocity = NNTrainer(
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

start_train = time.perf_counter()
velocity.train(
    train_data=target_data,
    train_dim=train_dim,
    batch_size=batch_size,
    steps=steps,
    x0_data=None,
    print_every=print_every,
);
elapsed_train = time.perf_counter() - start_train

# Start conditioning
start_sample = time.perf_counter()
nsamples = 5000
nn_samps = np.zeros((budget, nsamples))
cond_vars = conditioning_ys.tolist()
print("About to start drawing samples...")
cond_samples = velocity.conditional_sample(
    cond_values=cond_vars,
    nsamples=nsamples,
    u0_cond=None,
    solver_args=solver_args,
)
nn_samps = (jnp.hstack(cond_samples))[:, 1::2]
elapsed_sample = time.perf_counter() - start_sample

np.save(os.path.join(output_dir, "nn_samps.npy"), nn_samps)
timings = {
    "nn_time_ode": elapsed_train,
    "sample_ode_time": elapsed_sample,
    "timestamp": time.time(),
}
with open(os.path.join(output_dir, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)
print("Saved successfully!")