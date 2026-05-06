# -*- coding: utf-8 -*-
#  hIPPYlib-MUQ interface for large-scale Bayesian inverse problems
#  Copyright (c) 2019-2020, The University of Texas at Austin,
#  University of California--Merced, Washington University in St. Louis,
#  The United States Army Corps of Engineers, Massachusetts Institute of Technology

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
hMALA sampling for 256 Poisson posteriors per trial — one chain per test-set observation.

Each trial (--run_id N) draws from a fresh block of 256 test observations:
    run_id=0 → ys_all[200_000 : 200_256]   (already done)
    run_id=1 → ys_all[200_256 : 200_512]
    run_id=2 → ys_all[200_512 : 200_768]
    ...
where ys_all = training_dataset/solutions_grid_noise.npy and train_dim = 100_000.

Output is written to  mcmc_256_run{N}/chain_{RANK:03d}/  (mcmc_256/ for run_id=0).
Pass --output_root explicitly to override.

── Parallel workflow (batches of 25 workers) ───────────────────────────────────

  Single trial:
    for batch in $(seq 0 10); do
        mpirun -n 25 python poisson_mcmc_256.py \\
            --run_id 0 --start_idx $((batch * 25))
    done

  Multiple trials sequentially, each parallelised internally:
    for trial in 1 2 3 4; do
        for batch in $(seq 0 10); do
            mpirun -n 25 python poisson_mcmc_256.py \\
                --run_id $trial --start_idx $((batch * 25))
        done
    done

Workers with RANK ≥ 256 exit immediately (handles the last batch of 25 where
only chains 250–255 are needed).

── Serial fallback ─────────────────────────────────────────────────────────────

    for trial in 1 2 3 4; do
        for i in $(seq 0 255); do
            python poisson_mcmc_256.py --run_id $trial --start_idx $i
        done
    done
"""

import os
import sys

# Give each MPI worker its own dijitso JIT cache to prevent race conditions
# when all 25 workers try to compile the same dolfin C++ module simultaneously.
# Must be set before dolfin is imported.
_mpi_rank_early = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
os.environ["DIJITSO_CACHE_DIR"] = f"/tmp/.cache/dijitso_rank_{_mpi_rank_early}"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import json
import pickle
import yaml
import h5py
import numpy as np

import dolfin as dl
import hippylib as hp
import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import hippylib2muq as hm
import argparse

# ---------------------------------------------------------------------------
# MPI / trial index
# ---------------------------------------------------------------------------
MPI_RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
MPI_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))

parser = argparse.ArgumentParser(description="hMALA — 256 test-set posteriors per trial")
parser.add_argument(
    "--run_id",
    type=int,
    default=0,
    help="Trial index; test observations are offset by run_id * N_CHAINS",
)
parser.add_argument(
    "--start_idx",
    type=int,
    default=0,
    help="Global chain index offset (first RANK in this batch)",
)
parser.add_argument(
    "--output_root",
    type=str,
    default=None,
    help="Root directory for chain output folders (default: mcmc_256 for run_id=0, mcmc_256_run{N} otherwise)",
)
args = parser.parse_args()

N_CHAINS = 256          # total posteriors to sample per trial
RANK     = args.start_idx + MPI_RANK

# Derive output root from run_id unless explicitly overridden
if args.output_root is not None:
    OUTPUT_ROOT = args.output_root
elif args.run_id == 0:
    OUTPUT_ROOT = "mcmc_256"
else:
    OUTPUT_ROOT = f"mcmc_256_run{args.run_id}"

if RANK >= N_CHAINS:
    print(f"[worker {MPI_RANK}] RANK={RANK} >= {N_CHAINS}, nothing to do — exiting.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# MCMC parameters — identical to poisson_mcmc_main.py
# ---------------------------------------------------------------------------
N_SAMPLES = 100001
BURNIN    = 2000
STEP_SIZE = 0.13

# Training-dataset split (must match sample_convergence.py)
TRAIN_DIM = 100_000     # first split: training
# reference split: [TRAIN_DIM, 2*TRAIN_DIM)
# test split:      [2*TRAIN_DIM, ...)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def u_boundary(x, on_boundary):
    return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], 1.0))


def load_targets():
    with h5py.File("data.h5", "r") as f:
        targets = f["/target"][...]
    return targets


def load_test_observation(rank, run_id=0):
    """Return the rank-th test-set observation for the given trial.

    run_id offsets the test-set block: run_id=0 → rows 0-255, run_id=1 → 256-511, etc.
    """
    ys_all = np.load("training_dataset/solutions_grid_noise.npy")
    test_start = 2 * TRAIN_DIM + run_id * N_CHAINS
    y_obs = ys_all[test_start + rank]      # shape: (n_sensors,)
    return y_obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open("poisson.yaml") as fid:
        inargs = yaml.full_load(fid)

    sep = "\n" + "#" * 80 + "\n"
    output_dir = os.path.join(OUTPUT_ROOT, f"chain_{RANK:03d}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[chain {RANK}] output → {output_dir}")

    # -----------------------------------------------------------------------
    # Mesh and function spaces
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
    # Forward problem
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
    # Prior
    # -----------------------------------------------------------------------
    gamma    = 1.0
    delta    = 9.0
    anis_diff = dl.Identity(2)
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)
    print(
        "Prior: (delta_x - gamma*Laplacian)^2  delta={}, gamma={}".format(delta, gamma)
    )

    # -----------------------------------------------------------------------
    # Misfit — test-set observation for this chain
    # -----------------------------------------------------------------------
    noise_variance = 1e-6
    noise_std_dev  = 1e-3

    targets = load_targets()
    data    = load_test_observation(RANK, args.run_id)

    print(f"[chain {RANK}] test observation index {RANK} (run_id={args.run_id}, ntargets={targets.shape[0]})")
    print(f"noise_std = {noise_std_dev:.4e},  noise_variance = {noise_variance:.4e}")

    misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
    misfit.d.set_local(data)
    misfit.noise_variance = noise_variance

    model = hp.Model(pde, prior, misfit)

    # -----------------------------------------------------------------------
    # MAP estimation
    # -----------------------------------------------------------------------
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
        print(f"\nConverged in {solver.it} iterations.")
    else:
        print("\nNot Converged")
    print("Termination reason: ", solver.termination_reasons[solver.reason])
    print("Final gradient norm:", solver.final_grad_norm)
    print("Final cost:         ", solver.final_cost)

    total_cg = int(getattr(solver, "total_cg_iter", -1))
    solver_results = {
        "newton_iterations":   int(solver.it),
        "total_cg_iter":       total_cg,
        "approx_pde_solves_lower_bound": int(solver.it) * 2 + max(total_cg, 0) * 2,
        "converged":           bool(solver.converged),
        "termination_reason":  solver.termination_reasons[solver.reason],
        "final_grad_norm":     float(solver.final_grad_norm),
        "final_cost":          float(solver.final_cost),
    }
    with open(os.path.join(output_dir, "solver_stats.json"), "w") as ff:
        json.dump(solver_results, ff, indent=4)

    # -----------------------------------------------------------------------
    # Laplace posterior approximation (for hMALA proposal)
    # -----------------------------------------------------------------------
    print(sep, "Building Laplace posterior for hMALA proposal", sep)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)
    k, p    = 100, 20
    Omega   = hp.MultiVector(x[hp.PARAMETER], k + p)
    hp.parRandom.normal(1.0, Omega)
    lam, V  = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

    nu      = hp.GaussianLRPosterior(prior, lam, V)
    nu.mean = x[hp.PARAMETER]

    # -----------------------------------------------------------------------
    # Work-graph / sampling problem
    # -----------------------------------------------------------------------
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

    options = {
        "NumSamples": N_SAMPLES,
        "BurnIn":     BURNIN,
        "PrintLevel": 3,
    }

    opts = dict(options)
    opts["StepSize"] = STEP_SIZE
    gauss_hmala = hm.LAPosteriorGaussian(nu, use_zero_mean=True)
    prop        = ms.MALAProposal(opts, problem, gauss_hmala)
    kern        = ms.MHKernel(opts, problem, prop)
    sampler     = ms.SingleChainMCMC(opts, [kern])

    # -----------------------------------------------------------------------
    # Initial point (deterministic per chain index)
    # -----------------------------------------------------------------------
    def draw_x0(rank, base_seed=12344):
        np.random.seed(base_seed + rank)
        noise = dl.Vector()
        nu.init_vector(noise, "noise")
        noise_local = np.random.normal(0.0, 1.0, size=noise.local_size())
        noise.set_local(noise_local)
        noise.apply("")
        pr_s   = model.generate_vector(hp.PARAMETER)
        post_s = model.generate_vector(hp.PARAMETER)
        nu.sample(noise, pr_s, post_s, add_mean=True)
        return hm.dlVector2npArray(post_s)

    x0 = draw_x0(RANK)
    print(f"[chain {RANK}] x0[:5] = {x0[:5]}")

    # -----------------------------------------------------------------------
    # Run MCMC
    # -----------------------------------------------------------------------
    print(f"Starting hMALA chain (N={N_SAMPLES}, BurnIn={BURNIN}) ...")
    samps = sampler.Run([x0])

    elapsed     = sampler.TotalTime()
    accept_rate = None
    kernel      = sampler.Kernels()[0]
    if "AcceptanceRate" in dir(kernel):
        accept_rate = float(kernel.AcceptanceRate())
    elif "AcceptanceRates" in dir(kernel):
        accept_rate = float(kernel.AcceptanceRates())

    print(f"Elapsed: {elapsed:.1f}s  |  AcceptRate: {accept_rate}")

    # -----------------------------------------------------------------------
    # Convert and save samples
    # -----------------------------------------------------------------------
    samples_param = samps.AsMatrix().T    # (n_stored, param_dim)
    num_vertices  = mesh.num_vertices()
    num_stored    = samples_param.shape[0]

    vertex_samples = np.zeros((num_stored, num_vertices))
    f_fn = dl.Function(Vh[hp.PARAMETER])
    for i in range(num_stored):
        f_fn.vector().set_local(samples_param[i])
        vertex_samples[i] = f_fn.compute_vertex_values(mesh)

    np.savez_compressed(os.path.join(output_dir, "hmala_samples.npz"), arr=vertex_samples)
    print(f"Saved {num_stored} samples to {output_dir}/hmala_samples.npz")

    # Save the observation used for this chain so it's self-contained
    np.savez_compressed(os.path.join(output_dir, "y_obs.npz"), arr=data)

    mcmc_stats = {
        "n_samples_total":  N_SAMPLES,
        "burnin":           BURNIN,
        "n_samples_stored": num_stored,
        "step_size":        STEP_SIZE,
        "elapsed_seconds":  float(elapsed),
        "acceptance_rate":  accept_rate,
        "chain_rank":       RANK,
        "observation":      f"test-set index {RANK}",
    }
    with open(os.path.join(output_dir, "mcmc_stats.json"), "w") as ff:
        json.dump(mcmc_stats, ff, indent=4)

    method_summary = {
        "hMALA": {
            "Options":     opts,
            "Samples":     vertex_samples,
            "ElapsedTime": elapsed,
            "AcceptRate":  accept_rate,
        }
    }
    with open(os.path.join(output_dir, "method-list-hmala.pkl"), "wb") as ff:
        pickle.dump(method_summary, ff)
