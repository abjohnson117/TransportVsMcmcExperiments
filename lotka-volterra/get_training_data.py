import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from lv import LV
import numpy as np
import jax.numpy as jnp

from triangular_transport.flows.dataloaders import log_normal_reference_sampler

def main():
    nsamples = 100000
    target_samples = []
    gen_sample_no = nsamples
    while len(target_samples) < nsamples:
        seed = 0
        
        u_true = jnp.array([0.83194674, 0.04134147, 1.0823151, 0.03991483])
        # obs_noise = jnp.sqrt(0.9).item()
        obs_noise = 1.0
        
        lv_sampler = LV(
            seed=seed,
            no_samples=gen_sample_no,
            prior_sampler=log_normal_reference_sampler,
            likelihood_sampler=log_normal_reference_sampler,
            normalize=False,
            u_true=u_true,
            sigma=obs_noise,
            dt0=0.1,
        )

        target_samples = lv_sampler.get_target_samples()
        gen_sample_no *= 2

    target_samples = target_samples[np.random.choice(len(target_samples), size=(nsamples, )), :]
    np.save("training_data.npy", target_samples)

    

if __name__ == "__main__":
    main()