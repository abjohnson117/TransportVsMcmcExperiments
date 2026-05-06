# -*- coding: utf-8 -*-
"""
hMALA sampling for the p-Poisson inverse problem — 98th-PERCENTILE observation (data_98.npy).

Observation values are read from data_98.npy; target locations are the fixed
10x10 grid on the top surface (z=Height) matching generate_training_dataset.py.

Runs a single hMALA chain for 50,000 total MCMC steps.
Trial index is controlled by --start_idx (plus MPI rank for parallel runs).

Usage (10 serial trials):
    for i in $(seq 0 9); do
        python ppoisson_mcmc_98.py --output_root mcmc_98 --start_idx $i
    done

Usage (10 parallel trials via MPI):
    mpirun -n 10 python ppoisson_mcmc_98.py --output_root mcmc_98
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import json
import pickle
import yaml
import numpy as np
import argparse

import dolfin as dl
import hippylib as hp
import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import hippylib2muq as hm

from nonlinearPPoissonProblem import (
    NonlinearPPossionForm,
    EnergyFunctionalPDEVariationalProblem,
)

# ---------------------------------------------------------------------------
# MPI / trial index
# ---------------------------------------------------------------------------
MPI_RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
MPI_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))

parser = argparse.ArgumentParser(description="hMALA chain — p-Poisson 98th-percentile observation")
parser.add_argument("--output_root", type=str, default="mcmc_98",
                    help="Root directory for chain output folders")
parser.add_argument("--start_idx", type=int, default=0,
                    help="Global chain index offset (for serial trials)")
args = parser.parse_args()
RANK = args.start_idx + MPI_RANK

# ---------------------------------------------------------------------------
# MCMC parameters
# ---------------------------------------------------------------------------
N_SAMPLES = 50000
BURNIN    = 0
# STEP_SIZE = 0.105
STEP_SIZE = 0.007

# ---------------------------------------------------------------------------
# Problem geometry (must match ppoisson_box.yaml and generate_training_dataset.py)
# ---------------------------------------------------------------------------
Length = 1.0
Width  = 1.0
Height = 0.05

NOISE_STD = 1e-3


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------
class BottomBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], 0)

class SideBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            dl.near(x[0], 0) or dl.near(x[0], Length)
            or dl.near(x[1], 0) or dl.near(x[1], Width)
        )

class TopBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], Height)


def build_targets():
    """Fixed 10×10 observation grid on the top surface (z=Height)."""
    ntargets = 100
    grid_1d  = np.linspace(0.05, 0.95, 10)
    X, Y     = np.meshgrid(grid_1d, grid_1d)
    Z        = np.full((ntargets, 1), Height)
    return np.concatenate([X.ravel()[:, None], Y.ravel()[:, None], Z], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open("ppoisson_box.yaml") as fid:
        inargs = yaml.full_load(fid)

    sep = "\n" + "#" * 80 + "\n"
    output_dir = os.path.join(args.output_root, f"chain_{RANK:03d}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[chain {RANK}] output → {output_dir}")

    # -----------------------------------------------------------------------
    # Mesh and function spaces
    # -----------------------------------------------------------------------
    nx = ny = inargs["nelement"][0]
    nz      = inargs["nelement"][1]

    mesh   = dl.BoxMesh(dl.Point(0, 0, 0), dl.Point(Length, Width, Height), nx, ny, nz)
    bottom = BottomBoundary()
    side   = SideBoundary()

    Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh  = [Vh2, Vh1, Vh2]

    print("Number of dofs: STATE={}, PARAMETER={}, ADJOINT={}".format(
        Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()))

    # -----------------------------------------------------------------------
    # Forward problem (p-Poisson, p=3)
    # -----------------------------------------------------------------------
    dl.parameters["form_compiler"]["quadrature_degree"] = 3

    bc = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), side)

    boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundary_markers.set_all(0)
    bottom.mark(boundary_markers, 1)
    side.mark(boundary_markers, 2)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    order_ppoisson = 3.0
    functional = NonlinearPPossionForm(order_ppoisson, None, ds(1))
    pde        = EnergyFunctionalPDEVariationalProblem(Vh, functional, bc, bc)

    pde.solver          = dl.PETScKrylovSolver("cg", "icc")
    pde.solver_fwd_inc  = dl.PETScKrylovSolver("cg", "icc")
    pde.solver_adj_inc  = dl.PETScKrylovSolver("cg", "icc")
    pde.fwd_solver.solver = dl.PETScKrylovSolver("cg", "icc")
    pde.fwd_solver.parameters["gdu_tolerance"] = 1e-16
    pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20

    # -----------------------------------------------------------------------
    # Prior
    # -----------------------------------------------------------------------
    gamma = 1.0
    delta = 9.0
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, robin_bc=True)
    print("Prior: (delta_x - gamma*Laplacian)^2  delta={}, gamma={}".format(delta, gamma))

    # -----------------------------------------------------------------------
    # Misfit — 98th-PERCENTILE observation (data_98.npy)
    # -----------------------------------------------------------------------
    targets = build_targets()
    data    = np.load("data_98.npy")
    noise_std = NOISE_STD
    noise_variance = noise_std ** 2

    print(f"Loaded data_98.npy (shape={data.shape})")
    print(f"noise_std={noise_std:.4e}, noise_var={noise_variance:.4e}")

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
        "newton_iterations":              int(solver.it),
        "total_cg_iter":                  total_cg,
        "approx_pde_solves_lower_bound":  int(solver.it) * 2 + max(total_cg, 0) * 2,
        "converged":                      bool(solver.converged),
        "termination_reason":             solver.termination_reasons[solver.reason],
        "final_grad_norm":                float(solver.final_grad_norm),
        "final_cost":                     float(solver.final_cost),
    }
    with open(os.path.join(output_dir, "solver_stats.json"), "w") as ff:
        json.dump(solver_results, ff, indent=4)

    # -----------------------------------------------------------------------
    # Laplace posterior approximation (for hMALA proposal)
    # -----------------------------------------------------------------------
    print(sep, "Building Laplace posterior for hMALA proposal", sep)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)
    # k       = inargs["MCMC"].get("lr_rank", 150)
    # p_over  = inargs["MCMC"].get("lr_oversample", 20)
    k = 20
    p_over = 10
    Omega   = hp.MultiVector(x[hp.PARAMETER], k + p_over)
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

    np.save(os.path.join(output_dir, "hmala_samples.npy"), vertex_samples)
    print(f"Saved {num_stored} samples to {output_dir}/hmala_samples.npy")

    mcmc_stats = {
        "n_samples_total":  N_SAMPLES,
        "burnin":           BURNIN,
        "n_samples_stored": num_stored,
        "step_size":        STEP_SIZE,
        "elapsed_seconds":  float(elapsed),
        "acceptance_rate":  accept_rate,
        "chain_rank":       RANK,
        "observation":      "98th percentile (data_98.npy)",
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
