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

from triangular_transport.flows.sde_flow_trainer import NNSDE

from triangular_transport.flows.loss_functions import vec_field_loss, denoiser_loss
from triangular_transport.networks.flow_networks import MLP
from triangular_transport.flows.methods.sampling import inf_train_gen
from triangular_transport.flows.dataloaders import gaussian_reference_sampler
from triangular_transport.flows.interpolants import linear_interpolant_noise, linear_interpolant_der_noise

def gamma_fn(t):
    return 0.1 * jnp.sqrt(2 * (t - t**2) + 1e-8)


gamma_vmap = vmap(gamma_fn)

gammadot = vmap(grad(gamma_fn))


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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

plt.style.use("ggplot")

output_root = "nn_sde_results"
output_dir = os.path.join(output_root, f"ode_{args.run_id}")
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
n_train_samps = budget
epochs = 500
mean = 0.016955564
std = 2.029254
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.1, high=3.1, size=(budget, ))
conditioning_list = [4 ** i for i in range(9)]
gen_sample_list = list(reversed(conditioning_list))

# Train the one model
train_dim = n_train_samps
key = random.key(seed=seed)
key, key1, key2 = random.split(key=key, num=3)
batch_size = 2048
batch_size = min(batch_size, train_dim)
batch_size -= 1
steps_per_epoch = int(np.ceil(train_dim / batch_size))
steps = steps_per_epoch * epochs
print_every = 10000
yu_dimension = (1, 1)
target_data = inf_train_gen(data="8gaussians", rng=None, batch_size=train_dim)
dim = yu_dimension[0] + yu_dimension[1]
hidden_layer_list_vel = [256] * 3
hidden_layer_list_score = [512] * 4
# hidden_layer_list = [256] * 3
# hidden_layer_list = [1024] * 8
velocity = MLP(
    key=key1,
    dim=dim,
    time_varying=True,
    w=hidden_layer_list_vel,
    num_layers=len(hidden_layer_list_vel) + 1,
    activation_fn=jax.nn.gelu,  # GeLU worked well
)
score = MLP(
    key=key2,
    dim=dim,
    time_varying=True,
    w=hidden_layer_list_score,
    num_layers=len(hidden_layer_list_score) + 1,
    activation_fn=jax.nn.gelu,  # GeLU worked well
)
v_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=400,
    decay_steps=steps,
    end_value=1e-5,
)
s_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=400,
    decay_steps=steps,
    end_value=1e-5,
)
v_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(v_schedule)
)
s_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adam(s_schedule)
)

interpolant = linear_interpolant_noise
interpolant_der = linear_interpolant_der_noise
interpolant_args = {"t": None, "x1": None, "x0": None, "z": None}
reference_sampler_args = {"mu": mean, "sigma": std}

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
    interpolant_args=interpolant_args,
    yu_dimension=yu_dimension,
    reference_sampler_args=reference_sampler_args,
)

start_train = time.perf_counter()
trainer.train(
    x1_data=target_data,
    train_dim=train_dim,
    batch_size=batch_size,
    steps=steps,
    x0_data=None,
    print_every=print_every,
);
elapsed_train = time.perf_counter() - start_train

# Start conditioning
# , "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)
solver_args = {"saveat": "t1", "eps": 5e-3}
start_sample = time.perf_counter()
nsamples = 5000
nn_samps = np.zeros((budget, nsamples))
cond_vars = conditioning_ys.tolist()
print("About to start drawing samples...")
cond_samples = trainer.conditional_sample(
    cond_values=cond_vars,
    nsamples=nsamples,
    u0_cond=None,
    gamma=gamma_fn,
    solver_args=solver_args,
)
nn_samps = (jnp.hstack(cond_samples))[:, 1::2]
elapsed_sample = time.perf_counter() - start_sample

np.save(os.path.join(output_dir, "nn_samps.npy"), nn_samps)
timings = {
    "nn_time_sde": elapsed_train,
    "sample_sde_time": elapsed_sample,
    "timestamp": time.time(),
}
with open(os.path.join(output_dir, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)
print("Saved successfully!")