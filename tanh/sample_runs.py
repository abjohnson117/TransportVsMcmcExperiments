import os
import numpy as np
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id", type=int, default=0, help="Run index or ID for output folder"
)
args = parser.parse_args()

output_root = "samp_results"
output_dir = os.path.join(output_root, f"_{args.run_id}")
os.makedirs(output_dir, exist_ok=True)

budget = 4 ** 8
nsamples = 10000
seed = args.run_id
rng = np.random.RandomState(seed)
conditioning_ys = rng.uniform(low=-3.0, high=3.0, size=(budget, ))

rej_samps = np.zeros((budget, nsamples))

for i, y in enumerate(tqdm(conditioning_ys)):
    ys = np.random.uniform(low=-3.0, high=3.0, size=(nsamples, 1))
    us = np.tanh(ys) + np.random.gamma(shape=1.0, scale=0.3, size=(nsamples, 1))
    rej_samps[i, :] = us.squeeze()

print("Saving results...")
np.save(os.path.join(output_dir, "conditioning_ys.npy"), conditioning_ys)
np.save(os.path.join(output_dir, "rej_samps.npy"), rej_samps)
print("Results saved!")