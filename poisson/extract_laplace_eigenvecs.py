# -*- coding: utf-8 -*-
"""
Compute and save the Laplace posterior eigenvectors (V) for the Poisson
inverse problem — MAIN observation (data_obs.npy).

Runs MAP estimation, builds the low-rank Hessian eigendecomposition via
doublePassG, and saves:
    laplace_eigenvecs/lam.npy   — eigenvalues, shape (k,)
    laplace_eigenvecs/V.npy     — eigenvectors, shape (num_vertices, k)

Eigenvectors are stored as vertex values (via compute_vertex_values) to
match the ordering used by hmala_samples.npy and parameters_noise.npy.

No MCMC is run.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import json
import yaml
import h5py
import numpy as np

import dolfin as dl
import hippylib as hp


NOISE_VARIANCE = 1e-6


def u_boundary(x, on_boundary):
    return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], 1.0))


def load_targets():
    with h5py.File("data.h5", "r") as f:
        return f["/target"][...]


if __name__ == "__main__":
    with open("poisson.yaml") as fid:
        inargs = yaml.full_load(fid)

    output_dir = "laplace_eigenvecs"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Mesh and function spaces
    # -----------------------------------------------------------------------
    nx = ny = inargs["nelement"]
    mesh = dl.UnitSquareMesh(nx, ny)

    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh  = [Vh2, Vh1, Vh2]

    print("Number of dofs: STATE={}, PARAMETER={}, ADJOINT={}".format(
        Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()))

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
    gamma     = 1.0
    delta     = 9.0
    anis_diff = dl.Identity(2)
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)
    print("Prior: (delta_x - gamma*Laplacian)^2  delta={}, gamma={}".format(delta, gamma))

    # -----------------------------------------------------------------------
    # Misfit — MAIN observation
    # -----------------------------------------------------------------------
    targets = load_targets()
    data    = np.load("data_obs.npy")

    print(f"data shape={data.shape}, noise_variance={NOISE_VARIANCE:.4e}")

    misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
    misfit.d.set_local(data)
    misfit.noise_variance = NOISE_VARIANCE

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
        "newton_iterations":             int(solver.it),
        "total_cg_iter":                 total_cg,
        "approx_pde_solves_lower_bound": int(solver.it) * 2 + max(total_cg, 0) * 2,
        "converged":                     bool(solver.converged),
        "termination_reason":            solver.termination_reasons[solver.reason],
        "final_grad_norm":               float(solver.final_grad_norm),
        "final_cost":                    float(solver.final_cost),
    }
    with open(os.path.join(output_dir, "solver_stats.json"), "w") as ff:
        json.dump(solver_results, ff, indent=4)

    # -----------------------------------------------------------------------
    # Low-rank Hessian eigendecomposition (Laplace posterior)
    # -----------------------------------------------------------------------
    print("\nBuilding low-rank Hessian eigendecomposition...")
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)

    k      = inargs["MCMC"].get("lr_rank",       100)
    p_over = inargs["MCMC"].get("lr_oversample",  20)
    Omega  = hp.MultiVector(x[hp.PARAMETER], k + p_over)
    hp.parRandom.normal(1.0, Omega)
    lam, V = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

    # -----------------------------------------------------------------------
    # Extract V as vertex-value arrays to match hmala_samples.npy ordering.
    #
    # hmala_samples.npy stores samples via:
    #   f_fn.vector().set_local(dof_values)
    #   vertex_samples[i] = f_fn.compute_vertex_values(mesh)
    # We apply the same transform to each eigenvector so the spaces align.
    # -----------------------------------------------------------------------
    num_vertices = mesh.num_vertices()
    V_np = np.zeros((num_vertices, k))
    f_fn = dl.Function(Vh[hp.PARAMETER])
    for i in range(k):
        f_fn.vector().set_local(V[i].get_local())
        V_np[:, i] = f_fn.compute_vertex_values(mesh)

    np.save(os.path.join(output_dir, "lam.npy"), lam)
    np.save(os.path.join(output_dir, "V.npy"),   V_np)
    print(f"Saved lam.npy (shape={lam.shape}), V.npy (shape={V_np.shape}) to {output_dir}/")
