# -*- coding: utf-8 -*-
"""
Budget analysis for Poisson h-MALA chains.

For each budget point (chain_length, num_cond_vars), randomly selects
num_cond_vars observations from mcmc_256/chain_*/y_obs.npy and runs
h-MALA chains of length chain_length.

Each h-MALA step costs 2 PDE solves (forward + adjoint), so chain_list
values are divided by 2. nfevs = 256 globally.

Usage (analogous to lotka-volterra/budget_mcmc/mcmc_runs.py):
    python budget_mcmc_runs.py --run_id 0 --output_root mcmc_256
"""

import os
import sys

_mpi_rank_early = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
os.environ["DIJITSO_CACHE_DIR"] = f"/tmp/.cache/dijitso_rank_{_mpi_rank_early}"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import json
import pickle
import yaml
import h5py
import numpy as np
import time
from tqdm.auto import tqdm

import dolfin as dl
import hippylib as hp
import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import hippylib2muq as hm
import argparse

MPI_RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
MPI_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))

parser = argparse.ArgumentParser(description="Poisson h-MALA budget analysis")
parser.add_argument("--run_id", type=int, default=0, help="Run ID for seeding and output naming")
parser.add_argument(
    "--output_root",
    type=str,
    default="mcmc_256",
    help="Root directory for budget_results output",
)
parser.add_argument(
    "--ref_root",
    type=str,
    default=None,
    help="Root directory containing reference chain_* folders (defaults to output_root)",
)
args = parser.parse_args()

run_id   = args.run_id
ref_root = args.ref_root if args.ref_root is not None else args.output_root

# ---------------------------------------------------------------------------
# Budget parameters — matching lotka-volterra/budget_mcmc/mcmc_runs.py
# ---------------------------------------------------------------------------
BURNIN    = 1
STEP_SIZE = 0.13
N_CHAINS  = 256  # total available observations

nfevs = 256  # MAP cost in PDE solves (global)
conditioning_list_full = [4 ** i for i in range(5)]  # [1, 4, 16, 64, 256]
# Each h-MALA step = 2 PDE solves (forward + adjoint) → divide by 2
raw_chain_list   = np.array(list(reversed([4 ** i for i in range(3, 8)]))) - nfevs
chain_list_full  = raw_chain_list // 2

# Non-positive entries (budget exhausted by MAP) get exactly 1 h-MALA step
chain_list        = np.maximum(chain_list_full, 1)
conditioning_list = conditioning_list_full

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def u_boundary(x, on_boundary):
    return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], 1.0))


def load_targets():
    with h5py.File("data.h5", "r") as f:
        targets = f["/target"][...]
    return targets


def load_all_y_obs(n_chains, chain_root):
    """Load y_obs.npy from every chain_* sub-directory."""
    y_obs_list = []
    for idx in range(n_chains):
        path = os.path.join(chain_root, f"chain_{idx:03d}", "y_obs.npz")
        y_obs_list.append(np.load(path)["arr"])
    return y_obs_list


def draw_x0(nu, model, obs_idx, base_seed=12344):
    np.random.seed(base_seed + obs_idx)
    noise = dl.Vector()
    nu.init_vector(noise, "noise")
    noise_local = np.random.normal(0.0, 1.0, size=noise.local_size())
    noise.set_local(noise_local)
    noise.apply("")
    pr_s   = model.generate_vector(hp.PARAMETER)
    post_s = model.generate_vector(hp.PARAMETER)
    nu.sample(noise, pr_s, post_s, add_mean=True)
    return hm.dlVector2npArray(post_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open("poisson.yaml") as fid:
        inargs = yaml.full_load(fid)

    sep = "\n" + "#" * 80 + "\n"
    output_dir = os.path.join(args.output_root, f"budget_results_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[run {run_id}] output → {output_dir}")
    print(f"chain_list (steps after burn-in): {chain_list.tolist()}")
    print(f"conditioning_list:                {conditioning_list}")

    # -----------------------------------------------------------------------
    # Mesh and function spaces (shared across all observations)
    # -----------------------------------------------------------------------
    nx = ny = inargs["nelement"]
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh2  = dl.FunctionSpace(mesh, "Lagrange", 2)
    Vh1  = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh   = [Vh2, Vh1, Vh2]

    print(
        "Number of dofs: STATE={}, PARAMETER={}, ADJOINT={}".format(
            Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()
        )
    )

    # -----------------------------------------------------------------------
    # Forward problem (shared)
    # -----------------------------------------------------------------------
    u_bdr  = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc  = dl.DirichletBC(Vh[hp.STATE], u_bdr,  u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)
    f   = dl.Constant(1.0)

    def pde_varf(u, m, p):
        return (
            dl.exp(m) * dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx
            - f * p * dl.dx
        )

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # -----------------------------------------------------------------------
    # Prior (shared)
    # -----------------------------------------------------------------------
    gamma     = 1.0
    delta     = 9.0
    anis_diff = dl.Identity(2)
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)
    print(
        "Prior: (delta_x - gamma*Laplacian)^2  delta={}, gamma={}".format(delta, gamma)
    )

    # -----------------------------------------------------------------------
    # Load targets and all y_obs
    # -----------------------------------------------------------------------
    targets      = load_targets()
    all_y_obs    = load_all_y_obs(N_CHAINS, ref_root)
    num_vertices = mesh.num_vertices()

    noise_variance = 1e-6
    noise_std_dev  = 1e-3

    rng = np.random.RandomState(run_id)

    # -----------------------------------------------------------------------
    # Budget loop
    # -----------------------------------------------------------------------
    start = time.perf_counter()

    for i, chain_length in enumerate(tqdm(chain_list)):
        num_cond_vars = conditioning_list[i]
        chain_length  = int(chain_length)

        random_idxs    = rng.choice(N_CHAINS, size=min(num_cond_vars, N_CHAINS), replace=False)
        selected_y_obs = [all_y_obs[idx] for idx in random_idxs]

        all_vertex_samples = []

        for j, (obs_idx, y_obs_data) in enumerate(zip(random_idxs, selected_y_obs)):
            print(
                f"\n[budget point {i}, obs {j + 1}/{num_cond_vars}] "
                f"chain_idx={obs_idx}, steps={chain_length}"
            )

            # Misfit — unique per observation
            misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
            misfit.d.set_local(y_obs_data)
            misfit.noise_variance = noise_variance

            model = hp.Model(pde, prior, misfit)

            # MAP estimation
            m      = prior.mean.copy()
            solver = hp.ReducedSpaceNewtonCG(model)
            solver.parameters["rel_tolerance"]  = 1e-6
            solver.parameters["abs_tolerance"]  = 1e-12
            solver.parameters["max_iter"]       = 25
            solver.parameters["GN_iter"]        = 5
            solver.parameters["globalization"]  = "LS"
            solver.parameters["LS"]["c_armijo"] = 1e-4

            x = solver.solve([None, m, None])

            if solver.converged:
                print(f"  MAP converged in {solver.it} iterations.")
            else:
                print("  MAP did not converge.")

            # Laplace posterior approximation (for h-MALA proposal)
            print(sep, "Building Laplace posterior for hMALA proposal", sep)
            model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
            Hmisfit = hp.ReducedHessian(model, misfit_only=True)
            k, p    = 100, 20
            Omega   = hp.MultiVector(x[hp.PARAMETER], k + p)
            hp.parRandom.normal(1.0, Omega)
            lam, V  = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

            nu      = hp.GaussianLRPosterior(prior, lam, V)
            nu.mean = x[hp.PARAMETER]

            # Work-graph / sampling problem
            idparam          = mm.IdentityOperator(Vh[hp.PARAMETER].dim())
            gaussprior       = hm.BiLaplaceGaussian(prior)
            log_gaussprior   = gaussprior.AsDensity()
            param2likelihood = hm.Param2LogLikelihood(model)
            log_target       = mm.DensityProduct(2)

            workgraph = mm.WorkGraph()
            workgraph.AddNode(idparam,          "Identity")
            workgraph.AddNode(log_gaussprior,   "Prior")
            workgraph.AddNode(param2likelihood, "Likelihood")
            workgraph.AddNode(log_target,       "Target")
            workgraph.AddEdge("Identity",   0, "Prior",      0)
            workgraph.AddEdge("Prior",      0, "Target",     0)
            workgraph.AddEdge("Identity",   0, "Likelihood", 0)
            workgraph.AddEdge("Likelihood", 0, "Target",     1)

            log_gaussprior.EnableCache()
            param2likelihood.EnableCache()

            postDens = workgraph.CreateModPiece("Target")
            problem  = ms.SamplingProblem(postDens)

            opts = {
                "NumSamples": chain_length + BURNIN,
                "BurnIn":     BURNIN,
                "PrintLevel": 0,
                "StepSize":   STEP_SIZE,
            }

            gauss_hmala = hm.LAPosteriorGaussian(nu, use_zero_mean=True)
            prop        = ms.MALAProposal(opts, problem, gauss_hmala)
            kern        = ms.MHKernel(opts, problem, prop)
            sampler     = ms.SingleChainMCMC(opts, [kern])

            x0    = draw_x0(nu, model, obs_idx)
            samps = sampler.Run([x0])

            elapsed_chain = sampler.TotalTime()
            print(f"  h-MALA elapsed: {elapsed_chain:.1f}s")

            samples_param  = samps.AsMatrix().T    # (n_stored, param_dim)
            num_stored     = samples_param.shape[0]
            vertex_samples = np.zeros((num_stored, num_vertices))
            f_fn = dl.Function(Vh[hp.PARAMETER])
            for s in range(num_stored):
                f_fn.vector().set_local(samples_param[s])
                vertex_samples[s] = f_fn.compute_vertex_values(mesh)

            all_vertex_samples.append(vertex_samples)

        # Save all samples for this budget point
        # Shape: (num_cond_vars, chain_length, num_vertices)
        mcmc_samps_arr = np.array(all_vertex_samples)
        out_path = os.path.join(output_dir, f"mcmc_samps_{chain_length}_{num_cond_vars}.npy")
        np.save(out_path, mcmc_samps_arr)

        cond_path = os.path.join(output_dir, f"cond_vars_{num_cond_vars}.npy")
        np.save(cond_path, np.array(selected_y_obs))

        print(f"Saved: steps={chain_length}, n_obs={num_cond_vars} → {out_path}")

    elapsed = time.perf_counter() - start
    timings = {
        "mcmc_time":        elapsed,
        "timestamp":        time.time(),
        "nfevs":            nfevs,
        "chain_list":       chain_list.tolist(),
        "conditioning_list": conditioning_list,
    }
    with open(os.path.join(output_dir, "timings.json"), "w") as ff:
        json.dump(timings, ff, indent=2)

    print(f"\nDone. Total time: {elapsed:.1f}s")
