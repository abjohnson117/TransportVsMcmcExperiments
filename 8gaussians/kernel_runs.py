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
import diffrax
import time
import json
import argparse

from triangular_transport.flows.flow_trainer import (
    KernelTrainer,
)

from triangular_transport.flows.interpolants import (
    linear_interpolant,
    linear_interpolant_der,
)
from triangular_transport.flows.loss_functions import kernel_vec_field_loss
from triangular_transport.kernels.kernel_tools import get_prod_gaussianRBF, vectorize_kfunc
from triangular_transport.flows.methods.sampling import inf_train_gen
from triangular_transport.flows.dataloaders import (
    gaussian_reference_sampler,
)

best_hyperparams = {
    'lam_value': 0.001784240471,
    'gamma1': 0.0996334585856,
    'gamma2': 0.243442877083,
    'gamma3': 1.2982789204244
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

plt.style.use("ggplot")

output_root = "kernel_results"
output_dir = os.path.join(output_root, f"ode_{args.run_id}")
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
n_train_samps = budget # TODO: Can change this. But the idea of these plots is to use the max budget. NN evals don't require any more forward evals, so we can max out the budget here.
epochs = 1000
mean = 0.016955564
std = 2.029254
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.1, high=3.1, size=(budget, ))
conditioning_list = [4 ** i for i in range(9)]
gen_sample_list = list(reversed(conditioning_list))

# Train the one model
train_dim = n_train_samps
num_batches = 3000
key = random.key(seed=seed)
key, key1 = random.split(key=key, num=2)
batch_size = 1000
num_pivots = 1000
lam = best_hyperparams["lam_value"]
print_every = 5000
yu_dimension = (1, 1)
target_data = inf_train_gen(data="8gaussians", rng=None, batch_size=train_dim)
solver_args = {"solver": diffrax.Dopri5(), "max_steps": 50000, "stepsize_controller": diffrax.PIDController(rtol=1e-4, atol=1e-6)}
dim = yu_dimension[0] + yu_dimension[1]
def model():
    pass

gammas = {v: best_hyperparams[v] for v in best_hyperparams if "gamma" in v}
k = get_prod_gaussianRBF(**gammas)
kvv = vectorize_kfunc(k)

interpolant = linear_interpolant
interpolant_der = linear_interpolant_der
interpolant_args = {"t": None, "x1": None, "x0": None}
reference_sampler_args = {"mu": mean, "sigma": std}

velocity = KernelTrainer(
    target_density=None,
    model=model,
    optimizer=optax.adam(1e-5),
    interpolant=interpolant,
    interpolant_der=interpolant_der,
    reference_sampler=gaussian_reference_sampler,
    loss=kernel_vec_field_loss,
    interpolant_args=interpolant_args,
    seed=seed+43,
    yu_dimension=yu_dimension,
    cond=True,
    reference_sampler_args=reference_sampler_args,
)

start_train = time.perf_counter()
velocity.train(
    x1_data=target_data,
    train_dim=train_dim,
    batch_size=batch_size,
    num_batches=num_batches,
    num_pivots=num_pivots,
    x0_data=None,
    lam=lam,
    k=k,
    num_M_iters=1,

);
elapsed_train = time.perf_counter() - start_train

# Start conditioning
start_sample = time.perf_counter()
nsamples = 5000
k_samps = np.zeros((budget, nsamples))
cond_vars = conditioning_ys.tolist()
print("About to start drawing samples...")
cond_samples = velocity.conditional_sample(
    cond_values=cond_vars,
    nsamples=nsamples,
    u0_cond=None,
    solver_args=solver_args,
)
cond_sample_list = [cond_sample[:, yu_dimension[0]:] for cond_sample in cond_samples]
k_samps = jnp.hstack(cond_samples).T
elapsed_sample = time.perf_counter() - start_sample

np.save(os.path.join(output_dir, "k_samps.npy"), k_samps)
timings = {
    "k_time_ode": elapsed_train,
    "sample_ode_time": elapsed_sample,
    "timestamp": time.time(),
}
with open(os.path.join(output_dir, "timings.json"), "w") as f:
    json.dump(timings, f, indent=2)
print("Saved successfully!")