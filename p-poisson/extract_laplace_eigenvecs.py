# -*- coding: utf-8 -*-
"""
Compute and save the Laplace posterior eigenvectors (V) for the p-Poisson
inverse problem — MAIN observation (data_obs.npy / training_dataset).

Runs MAP estimation, builds the low-rank Hessian eigendecomposition via
doublePassG, and saves:
    laplace_eigenvecs/lam.npy   — eigenvalues, shape (k,)
    laplace_eigenvecs/V.npy     — eigenvectors, shape (param_dim, k)

No MCMC is run.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import json
import yaml
import numpy as np

import dolfin as dl
import hippylib as hp

from nonlinearPPoissonProblem import (
    NonlinearPPossionForm,
    EnergyFunctionalPDEVariationalProblem,
)

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
    """Fixed 10x10 observation grid on the top surface (z=Height)."""
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

    output_dir = "laplace_eigenvecs"
    os.makedirs(output_dir, exist_ok=True)

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
    # Misfit — MAIN observation
    # -----------------------------------------------------------------------
    targets = build_targets()
    ys   = np.load("training_dataset/solutions_delta.npy")
    data = ys[0]
    del ys
    noise_variance = NOISE_STD ** 2

    print(f"data shape={data.shape}, noise_std={NOISE_STD:.4e}")

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
    # Low-rank Hessian eigendecomposition (Laplace posterior)
    # -----------------------------------------------------------------------
    print("\nBuilding low-rank Hessian eigendecomposition...")
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)
    # k      = 20
    # p_over = 10
    k       = inargs["MCMC"].get("lr_rank", 150)
    p_over  = inargs["MCMC"].get("lr_oversample", 20)
    Omega  = hp.MultiVector(x[hp.PARAMETER], k + p_over)
    hp.parRandom.normal(1.0, Omega)
    lam, V = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

    # -----------------------------------------------------------------------
    # Extract V as numpy array, restrict to bottom layer (z=0), and save.
    #
    # Row ordering must match parameters.npy / hmala_samples.npy:
    #   get_data_3d builds A[ix, iy, iz] with sorted xs/ys/zs, then the
    #   z=0 slice is C-order flattened → flat index = ix*21 + iy.
    #   So the primary sort key is x (slow), secondary is y (fast).
    # -----------------------------------------------------------------------
    dof_coords = Vh[hp.PARAMETER].tabulate_dof_coordinates()  # (param_dim, 3)
    bottom_dof_idx = np.where(np.isclose(dof_coords[:, 2], 0.0))[0]  # DOFs at z=0

    bottom_coords = dof_coords[bottom_dof_idx]  # (441, 3)
    # np.lexsort: last key is primary → sort by x ascending, break ties by y ascending
    sort_order = np.lexsort((bottom_coords[:, 1], bottom_coords[:, 0]))
    bottom_dof_idx_sorted = bottom_dof_idx[sort_order]

    param_dim = Vh[hp.PARAMETER].dim()
    V_np = np.zeros((param_dim, k))
    for i in range(k):
        V_np[:, i] = V[i].get_local()

    V_bottom = V_np[bottom_dof_idx_sorted, :]  # (441, k), rows ordered as ix*21+iy

    np.save(os.path.join(output_dir, "lam.npy"), lam)
    np.save(os.path.join(output_dir, "V.npy"), V_bottom)
    np.save(os.path.join(output_dir, "bottom_dof_idx.npy"), bottom_dof_idx_sorted)
    print(f"Saved lam.npy (shape={lam.shape}), V.npy (shape={V_bottom.shape}), "
          f"bottom_dof_idx.npy (shape={bottom_dof_idx_sorted.shape}) to {output_dir}/")
