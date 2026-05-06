"""
Noise-level diagnostics for the 98th-percentile Poisson observation.

Tests three indicators that the likelihood noise variance is too low
(posterior too sharp):

  Diagnostic 1 — Normalized misfit at MAP
      Computes ||Bu_MAP - d||^2 / (noise_variance * n_obs).
      For well-calibrated noise this should be O(1).
      << 1  → noise_variance is much larger than the actual data misfit
               (likelihood is looser than needed; posterior artificially broad)
       ~ 1  → noise is well-matched to the data
      >> 1  → noise_variance is smaller than the data misfit
               (likelihood over-constrains; MAP cannot fit the data at the
                assumed noise level — posterior artificially sharp)

  Diagnostic 2 — Misfit Hessian eigenvalue spectrum
      Computes the top-k generalised eigenvalues of the misfit Hessian
      in the prior-preconditioned inner product (the same lam computed for
      the Laplace approximation).  Each lam[i] is the likelihood-to-prior
      variance ratio in the i-th data-informed direction.
      lam[i] >> 1  → likelihood dominates the prior in that direction
      If many eigenvalues are >> 1 the data is over-constraining the
      parameter field — a signature of noise that is too low.

  Diagnostic 3 — Posterior vs prior variance reduction
      variance_reduction[i] = lam[i] / (1 + lam[i])
      This is the fraction by which the posterior variance is smaller than
      the prior variance in each data-informed direction.
      Mean ≈ 1  → posterior has collapsed relative to prior (too sharp)
      Mean ≈ 0  → data barely updates the prior

Usage:
    python poisson_noise_diagnostics_98.py
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import yaml
import h5py
import numpy as np

import dolfin as dl
import hippylib as hp

SEP = "\n" + "=" * 70 + "\n"


# ---------------------------------------------------------------------------
# Helpers (identical to poisson_mcmc_98.py)
# ---------------------------------------------------------------------------

def u_boundary(x, on_boundary):
    return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], 1.0))


def load_targets():
    with h5py.File("data.h5", "r") as f:
        targets = f["/target"][...]
    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("poisson.yaml") as fid:
        inargs = yaml.full_load(fid)

    # -----------------------------------------------------------------------
    # Mesh and function spaces  (identical to poisson_mcmc_98.py)
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
    gamma     = 1.0
    delta     = 9.0
    anis_diff = dl.Identity(2)
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)

    # -----------------------------------------------------------------------
    # Misfit — 98th-percentile observation (data_98.npy)
    # -----------------------------------------------------------------------
    rel_noise = 1e-4

    targets = load_targets()
    data    = np.load("data_98.npy")
    n_obs   = targets.shape[0]

    print(f"Loaded targets from data.h5       (n_obs={n_obs})")
    print(f"Loaded observations from data_98.npy  (shape={data.shape})")
    print(f"noise_variance (set in likelihood) = {rel_noise:.2e}")

    # ---- mismatch note ------------------------------------------------
    # data_98.npy was generated with noise std = rel_noise * MAX(d),
    # so actual noise variance ≈ (rel_noise * MAX)^2, not rel_noise itself.
    # Print both so the gap is visible.
    d_max = float(np.max(np.abs(data)))
    actual_noise_std = rel_noise * d_max
    actual_noise_var = actual_noise_std ** 2
    print(f"||d||_inf = {d_max:.4f}")
    print(f"Noise std added at generation     = rel_noise * ||d||_inf = {actual_noise_std:.2e}")
    print(f"Implied actual noise variance     = {actual_noise_var:.2e}")
    print(f"Ratio (set / actual)              = {rel_noise / actual_noise_var:.2e}  "
          f"(>1 → likelihood too diffuse; <1 → too sharp)")
    # -------------------------------------------------------------------

    misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
    misfit.d.set_local(data)
    misfit.noise_variance = rel_noise

    model = hp.Model(pde, prior, misfit)

    # -----------------------------------------------------------------------
    # MAP estimation
    # -----------------------------------------------------------------------
    print(SEP + "MAP estimation" + SEP)
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
        print(f"Converged in {solver.it} Newton iterations.")
    else:
        print("WARNING: Newton solver did NOT converge.")
    print(f"Final gradient norm : {solver.final_grad_norm:.4e}")
    print(f"Final cost          : {solver.final_cost:.4e}")

    # -----------------------------------------------------------------------
    # Diagnostic 1 — Normalized misfit at MAP
    # -----------------------------------------------------------------------
    print(SEP + "DIAGNOSTIC 1: Normalized misfit at MAP" + SEP)

    # Forward-solve at MAP to get u_MAP
    pde.solveFwd(x[hp.STATE], x)

    # Compute residual Bu_MAP - d
    residual = misfit.d.copy()           # start with d
    Bu_map   = misfit.d.copy()
    misfit.B.mult(x[hp.STATE], Bu_map)   # Bu_MAP
    residual.axpy(-1.0, Bu_map)          # Bu_MAP - d  (sign doesn't matter for norm)
    residual_norm_sq = residual.inner(residual)

    normalized_residual = residual_sq = residual_norm_sq / (misfit.noise_variance * n_obs)

    print(f"||Bu_MAP - d||^2                     = {residual_norm_sq:.4e}")
    print(f"noise_variance * n_obs               = {misfit.noise_variance * n_obs:.4e}")
    print(f"Normalized residual                  = {normalized_residual:.4f}")
    print()
    if normalized_residual < 0.1:
        verdict = "VERY LOW — likelihood is much looser than the actual fit. " \
                  "noise_variance >> true data noise. Posterior may be artificially broad."
    elif normalized_residual < 0.5:
        verdict = "LOW — noise_variance appears larger than the actual data noise level."
    elif normalized_residual <= 2.0:
        verdict = "OK — noise is roughly consistent with the data misfit at the MAP."
    elif normalized_residual <= 10.0:
        verdict = "HIGH — noise_variance may be smaller than the actual data noise. " \
                  "Posterior may be overly sharp."
    else:
        verdict = "VERY HIGH — noise_variance is much too small. " \
                  "The MAP cannot fit the data at this noise level. Posterior is too sharp."
    print(f"  Verdict: {verdict}")

    # -----------------------------------------------------------------------
    # Diagnostic 2 — Misfit Hessian eigenvalue spectrum
    # -----------------------------------------------------------------------
    print(SEP + "DIAGNOSTIC 2: Misfit Hessian eigenvalue spectrum" + SEP)

    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)
    k, p    = 100, 20
    Omega   = hp.MultiVector(x[hp.PARAMETER], k + p)
    hp.parRandom.normal(1.0, Omega)
    lam, V  = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

    print(f"Computed top-{k} generalised eigenvalues of the misfit Hessian.")
    print()
    print(f"  Top-5 eigenvalues : {lam[:5]}")
    print(f"  lam[0]  (largest) : {lam[0]:.4e}")
    print(f"  lam[-1] (smallest): {lam[-1]:.4e}")
    print()

    frac_gt_1    = float(np.mean(lam > 1.0))
    frac_gt_10   = float(np.mean(lam > 10.0))
    frac_gt_100  = float(np.mean(lam > 100.0))
    frac_gt_1e4  = float(np.mean(lam > 1e4))

    print(f"  Fraction of eigenvalues > 1    : {frac_gt_1:.2%}  "
          f"(directions where likelihood > prior)")
    print(f"  Fraction of eigenvalues > 10   : {frac_gt_10:.2%}")
    print(f"  Fraction of eigenvalues > 100  : {frac_gt_100:.2%}")
    print(f"  Fraction of eigenvalues > 1e4  : {frac_gt_1e4:.2%}")
    print()

    if lam[0] > 1e6:
        eig_verdict = "VERY LARGE leading eigenvalue. The likelihood is extremely " \
                      "informative relative to the prior — strongly suggests noise is too low."
    elif lam[0] > 1e3:
        eig_verdict = "Large leading eigenvalue. Likelihood substantially dominates the " \
                      "prior. Consider whether this level of data informativeness is intended."
    elif lam[0] > 10:
        eig_verdict = "Moderate leading eigenvalue. Some directions are data-dominated, " \
                      "which is normal for an informative experiment."
    else:
        eig_verdict = "Small eigenvalues. Data barely updates the prior — noise may be too high."
    print(f"  Verdict: {eig_verdict}")

    # -----------------------------------------------------------------------
    # Diagnostic 3 — Posterior vs prior variance reduction
    # -----------------------------------------------------------------------
    print(SEP + "DIAGNOSTIC 3: Posterior vs prior variance reduction" + SEP)

    var_reduction = lam / (1.0 + lam)   # per-direction fraction of prior variance removed

    mean_vr   = float(np.mean(var_reduction))
    median_vr = float(np.median(var_reduction))
    max_vr    = float(np.max(var_reduction))

    print("  variance_reduction[i] = lam[i] / (1 + lam[i])")
    print("  Interpretation: fraction of prior variance removed in each data-informed direction.")
    print("  Value near 1 → posterior has nearly zero variance there (too sharp).")
    print("  Value near 0 → data barely updates the prior there.")
    print()
    print(f"  Top-5 variance reductions : {var_reduction[:5]}")
    print(f"  Mean   variance reduction : {mean_vr:.4f}")
    print(f"  Median variance reduction : {median_vr:.4f}")
    print(f"  Max    variance reduction : {max_vr:.4f}")
    print()

    frac_vr_gt_99 = float(np.mean(var_reduction > 0.99))
    frac_vr_gt_90 = float(np.mean(var_reduction > 0.90))
    print(f"  Fraction of directions with >99% prior variance removed : {frac_vr_gt_99:.2%}")
    print(f"  Fraction of directions with >90% prior variance removed : {frac_vr_gt_90:.2%}")
    print()

    if mean_vr > 0.95:
        vr_verdict = "VERY HIGH mean variance reduction. The posterior has almost no " \
                     "uncertainty in the data-informed subspace — posterior is too sharp."
    elif mean_vr > 0.75:
        vr_verdict = "HIGH mean variance reduction. The data strongly constrains most " \
                     "data-informed directions."
    elif mean_vr > 0.4:
        vr_verdict = "MODERATE mean variance reduction. Data and prior both contribute " \
                     "meaningfully to the posterior."
    else:
        vr_verdict = "LOW mean variance reduction. The data is only weakly informative " \
                     "relative to the prior — noise may be too high."
    print(f"  Verdict: {vr_verdict}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(SEP + "SUMMARY" + SEP)
    results = {
        "noise_variance_set":          rel_noise,
        "actual_noise_variance_approx": actual_noise_var,
        "noise_ratio_set_over_actual": rel_noise / actual_noise_var,
        "diagnostic_1": {
            "residual_norm_sq":    residual_norm_sq,
            "normalized_residual": normalized_residual,
            "verdict":             verdict,
        },
        "diagnostic_2": {
            "top_5_eigenvalues":      lam[:5].tolist(),
            "lam_max":                float(lam[0]),
            "lam_min_of_top_k":       float(lam[-1]),
            "frac_gt_1":              frac_gt_1,
            "frac_gt_10":             frac_gt_10,
            "frac_gt_100":            frac_gt_100,
            "frac_gt_1e4":            frac_gt_1e4,
            "verdict":                eig_verdict,
        },
        "diagnostic_3": {
            "top_5_variance_reductions":        var_reduction[:5].tolist(),
            "mean_variance_reduction":          mean_vr,
            "median_variance_reduction":        median_vr,
            "max_variance_reduction":           max_vr,
            "frac_directions_gt_99pct_removed": frac_vr_gt_99,
            "frac_directions_gt_90pct_removed": frac_vr_gt_90,
            "verdict":                          vr_verdict,
        },
    }

    out_path = "noise_diagnostics_98.json"
    with open(out_path, "w") as ff:
        json.dump(results, ff, indent=4)
    print(f"Full results saved to {out_path}")

    print()
    print(f"  D1 normalized residual  : {normalized_residual:.4f}  (expect ~1 for calibrated noise)")
    print(f"  D2 leading eigenvalue   : {lam[0]:.4e}  (large → likelihood dominates prior)")
    print(f"  D3 mean var. reduction  : {mean_vr:.4f}  (near 1 → posterior too sharp)")
