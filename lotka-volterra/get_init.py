import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
from tqdm.auto import tqdm
import jax.numpy as jnp
from jax import jit, random
from triangular_transport.flows.dataloaders import log_normal_reference_sampler
from ot.sliced import sliced_wasserstein_distance as swd
from jax.scipy.optimize import minimize
from lv import LV

def get_neg_post(lv_sampler, yobs):
    log_pi = lv_sampler.log_posterior

    def neg_post(x):
        return -log_pi(yobs, x)
    
    return neg_post


def get_inits(seed, lv_sampler, n_restarts=25, u_dim=4):
    init_key = random.key(seed=seed)
    x_inits = []
    for _ in range(n_restarts):
        init_key, subkey = random.split(init_key)
        x_inits.append(
            random.normal(subkey, shape=(u_dim,)) * lv_sampler.std_prior + lv_sampler.mu_base
        )
    return x_inits

def get_map(x_inits, neg_post):
    best_val = jnp.inf
    best_result = None
    opt_idx = None
    for i, x_init in enumerate(tqdm(x_inits, desc="MAP restarts")):
        result = minimize(neg_post, x_init, method="BFGS", tol=1e-8)
        val = neg_post(result.x)
        print(f"Init {i}: neg_post = {val:.4f}, exp(x) = {jnp.exp(result.x)}, nfev = {int(result.nfev)}")
        if val < best_val:
            best_val = val
            best_result = result
            opt_idx = i

    return best_result, opt_idx

def get_base_swd(us_base, save_path, swd_seed=42, n_projections=42):
    true_samps = np.exp(np.load(f"true_samps/{save_path}.npy")[::20, :])
    base_swd = swd(
        np.asarray(us_base),
        np.asarray(true_samps),
        a=np.ones(len(us_base)) / len(us_base),
        b=np.ones(len(true_samps)) / len(true_samps),
        seed=swd_seed,
        n_projections=n_projections,
        p=2,
    )
    return (base_swd, true_samps)

def main():
    no_samples = 1
    u_true = jnp.array([0.83194674, 0.04134147, 1.0823151, 0.03991483])
    sigma = 1.0
    mean = np.load("log_y_mean.npy")
    std = np.load("log_y_std.npy")

    output_root = "init_data"
    os.makedirs(output_root, exist_ok=True)

    lv_sampler = LV(
        seed=0,
        no_samples=no_samples,
        prior_sampler=log_normal_reference_sampler,
        likelihood_sampler=log_normal_reference_sampler,
        normalize=False,
        u_true=u_true,
        sigma=sigma,
        dt0=0.1,
        log_y_mean=mean,
        log_y_std=std,
    )

    norm_inits = get_inits(11, lv_sampler)
    mod_inits = get_inits(10, lv_sampler)
    rare_inits = get_inits(10, lv_sampler)

    norm_post = get_neg_post(lv_sampler, np.load("true_samps/y_obs.npy"))
    mod_post = get_neg_post(lv_sampler, np.log(np.load("true_samps/y_moderate.npy")))
    rare_post = get_neg_post(lv_sampler, np.log(np.load("true_samps/y_rare.npy")))

    print("Constructing MAP estimators...")
    br_norm, norm_idx = get_map(norm_inits, norm_post)
    br_mod, mod_idx = get_map(mod_inits, mod_post)
    br_rare, rare_idx = get_map(rare_inits, rare_post)

    print("Saving data...")
    x_init_norm = norm_inits[norm_idx]
    x_init_mod = mod_inits[mod_idx]
    x_init_rare = rare_inits[rare_idx]
    x_inits = [x_init_norm, x_init_mod, x_init_rare]
    np.save(os.path.join(output_root, "x_init_norm.npy"), np.exp(x_init_norm))
    np.save(os.path.join(output_root, "x_init_mod.npy"), np.exp(x_init_mod))
    np.save(os.path.join(output_root, "x_init_rare.npy"), np.exp(x_init_rare))

    print("Calculating SWDs...")
    gen_seed = 1
    gen_key = random.key(gen_seed)
    us_base = log_normal_reference_sampler(
        key=gen_key,
        shape=(25000, 4), # Number of samples that we compare against, always
        mu=jnp.array([-0.125, -3.0, -0.125, -3.0]),
        sigma=1 / jnp.sqrt(2),
    )
    base_swd_norm, true_norm = get_base_swd(us_base, save_path="true_us_obs")
    base_swd_mod, true_mod = get_base_swd(us_base, save_path="true_us_moderate_obs_2")
    base_swd_rare, true_rare = get_base_swd(us_base, save_path="true_us_rare_obs_3")
    base_swd_list = [base_swd_norm, base_swd_mod, base_swd_rare]
    true_samps_list = [true_norm, true_mod, true_rare]
    swd_list = []
    for i, x_init in enumerate(tqdm(x_inits)):
        true_samps = true_samps_list[i]
        base_swd = base_swd_list[i]
        swd_init = swd(
            np.asarray(x_init).reshape(1,-1),
            np.asarray(true_samps),
            a=np.ones(len(x_init)) / len(x_init),
            b=np.ones(len(true_samps)) / len(true_samps),
            seed=42,
            n_projections=2048,
            p=2,
        ) / base_swd
        swd_list.append(swd_init)
    np.save(os.path.join(output_root, "swd_list.npy"), np.array(swd_list))

    print("Calculating number of func evals...")
    norm_func_evals = br_norm.nfev + br_norm.njev
    mod_func_evals = br_mod.nfev + br_mod.njev
    rare_func_evals = br_rare.nfev + br_rare.njev

    print("Saving eval information...")
    eval_array = np.array([norm_func_evals, mod_func_evals, rare_func_evals])
    np.save(os.path.join(output_root, "eval_no_array.npy"), eval_array)

if __name__ == "__main__":
    main()



