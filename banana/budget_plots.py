import numpy as np
import matplotlib.pyplot as plt
from ot.lp import wasserstein_1d
from ksd import compute_ksd_jax
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, random
import os

output_root = "nn_results"
output_dir = os.path.join(output_root, "ode")
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)