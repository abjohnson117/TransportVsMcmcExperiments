#!/usr/bin/env python3
"""
generate_shepp_logan_data.py

Generates the out-of-distribution Shepp-Logan-phantom test point for the
Poisson inverse problem.

True parameter: the standard Shepp-Logan phantom (skimage.data.shepp_logan_phantom),
resized to the 33x33 vertex grid and rescaled from its native [0, 1] intensity
range to [log(3), log(40)]:
  log(40) ≈ 3.69  at the brightest phantom intensity
  log(3)  ≈ 1.10  at the darkest (background) phantom intensity

This is strongly out of distribution relative to the BiLaplacian Gaussian prior
used to generate training data (which produces smooth random fields) — the
phantom has several sharp piecewise-constant ellipses.

PDE setup is identical to poisson/sl.py (and poisson/poisson.py):
  -div(exp(m) grad u) = 1   on [0,1]^2
  u = x[1]  on top/bottom boundaries (Dirichlet)
  observations: 100 pointwise state values on a 10x10 interior grid

Outputs saved to shepp-logan-data/:
  data_sl.npy       concat [y_obs (100,) | u_true_flat (1089,)]
  targets.npy       observation-point coordinates (100, 2)
  map_array.npy     MAP estimate on 33x33 vertex grid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import dolfin as dl
import hippylib as hp

HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "shepp-logan-data")
NX = NY    = 32          # matches poisson.yaml  nelement=32  =>  33x33 vertices


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def u_boundary(x, on_boundary):
    return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], 1.0))


def sl_phantom(n, low=None, high=None):
    """Shepp-Logan phantom of size n x n, rescaled to [low, high].

    Resizes the standard skimage Shepp-Logan phantom (native intensity range
    [0, 1]) to an n x n grid, then linearly rescales intensities so the
    background maps to `low` and the brightest ellipse maps to `high`.
    """
    if low is None:
        low = np.log(3)
    if high is None:
        high = np.log(40)
    img = shepp_logan_phantom()
    img = resize(img, (n, n), anti_aliasing=True)
    img = (img - img.min()) / (img.max() - img.min())
    return low + (high - low) * img


def image_flat_to_dof_flat(V, img_flat, nx, ny):
    """Interpolate a row-major flattened image into FEniCS CG1 DOF order.

    img_flat : 1-D array of length (nx+1)*(ny+1), row-major (y,x).
    """
    coords  = V.tabulate_dof_coordinates().reshape(-1, 2)
    xs, ys  = coords[:, 0], coords[:, 1]
    hx = (xs.max() - xs.min()) / nx
    hy = (ys.max() - ys.min()) / ny
    ix = np.clip(np.round((xs - xs.min()) / hx).astype(int), 0, nx)
    iy = np.clip(np.round((ys - ys.min()) / hy).astype(int), 0, ny)
    return img_flat[iy * (nx + 1) + ix]


def vertex_grid(Vh_param, vec, mesh):
    """Extract FEniCS function values on the vertex grid and return as 2-D array."""
    f = dl.Function(Vh_param, vec)
    C = f.compute_vertex_values(mesh)
    n = int(np.sqrt(C.shape[0]))
    return C.reshape(n, n)


# ─────────────────────────────────────────────────────────────────────────────
# PDE + prior (identical to poisson/sl.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_problem(nx=NX, ny=NY):
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh2  = dl.FunctionSpace(mesh, "Lagrange", 2)
    Vh1  = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh   = [Vh2, Vh1, Vh2]

    u_bdr  = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc     = dl.DirichletBC(Vh[hp.STATE], u_bdr,  u_boundary)
    bc0    = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)
    f_src  = dl.Constant(1.0)

    def pde_varf(u, m, p):
        return (dl.exp(m) * dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx
                - f_src * p * dl.dx)

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Same prior as training data generation
    prior = hp.BiLaplacianPrior(
        Vh[hp.PARAMETER], 1.0, 9.0, dl.Identity(2), robin_bc=True
    )
    return mesh, Vh, pde, prior


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    mesh, Vh, pde, prior = build_problem()
    print(f"DOFs  state={Vh[hp.STATE].dim()}  param={Vh[hp.PARAMETER].dim()}")

    # ── Observation operator: 10x10 interior grid ────────────────────────────
    g1d     = np.linspace(0.05, 0.95, 10)
    Xg, Yg  = np.meshgrid(g1d, g1d)
    targets  = np.vstack([Xg.ravel(), Yg.ravel()]).T          # (100, 2)
    misfit   = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
    np.save(os.path.join(OUTPUT_DIR, "targets.npy"), targets)
    print(f"Observation points: {targets.shape[0]}")

    # ── Shepp-Logan-phantom true parameter ───────────────────────────────────
    m_img_flat = sl_phantom(NX + 1).reshape((NX + 1) * (NY + 1))
    dof_vals   = image_flat_to_dof_flat(Vh[hp.PARAMETER], m_img_flat, NX, NY)

    # Allocate a FEniCS vector via prior, then overwrite with phantom values
    tmp_noise = dl.Vector()
    prior.init_vector(tmp_noise, "noise")
    hp.parRandom.normal(1.0, tmp_noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(tmp_noise, mtrue)
    mtrue.set_local(dof_vals)

    print(f"Shepp-Logan phantom range: [{dof_vals.min():.3f}, {dof_vals.max():.3f}]"
          f"  (log3={np.log(3):.3f}, log40={np.log(40):.3f})")

    # ── Solve forward PDE ─────────────────────────────────────────────────────
    u_state = pde.generate_state()
    x_vec   = [u_state, mtrue, None]
    pde.solveFwd(x_vec[hp.STATE], x_vec)
    print("Forward PDE solved.")

    # ── Apply observation operator, add fixed-std noise (matches poisson_mcmc_98.py) ──
    misfit.B.mult(x_vec[hp.STATE], misfit.d)
    noise_variance = 1e-6
    noise_std_dev  = np.sqrt(noise_variance)   # = 1e-3, absolute (no MAX scaling)
    misfit.noise_variance = noise_variance
    hp.parRandom.normal_perturb(noise_std_dev, misfit.d)

    y_obs       = misfit.d.get_local()                          # (100,)
    u_true_flat = vertex_grid(Vh[hp.PARAMETER], mtrue, mesh).reshape(-1)  # (1089,)
    state_flat  = vertex_grid(Vh[hp.STATE], x_vec[hp.STATE], mesh).reshape(-1)  # (1089,) full PDE solution

    yu = np.concatenate([y_obs, u_true_flat])
    np.save(os.path.join(OUTPUT_DIR, "data_sl.npy"), yu)
    np.save(os.path.join(OUTPUT_DIR, "state_sl.npy"), state_flat)
    print(f"Saved  data_sl.npy   shape={yu.shape}")
    print(f"Saved  state_sl.npy  shape={state_flat.shape}  (full PDE solution on 33x33 grid)")
    print(f"  y_obs  range [{y_obs.min():.5f}, {y_obs.max():.5f}]")
    print(f"  u_true range [{u_true_flat.min():.3f}, {u_true_flat.max():.3f}]")
    print(f"  state  range [{state_flat.min():.4f}, {state_flat.max():.4f}]")

    # ── Prior-mean forward solution (m=0, k=exp(0)=1 everywhere) ─────────────
    # B·u(m=0) is subtracted from observations before FM training/inference to
    # remove the BC-induced y-gradient that causes FM to confuse "high k inside
    # phantom" with "high k at top".
    print("\nComputing prior-mean forward solution (m=0) ...")
    m_zero = dl.Vector()
    prior.init_vector(m_zero, 0)
    m_zero.zero()
    u_prior_state = pde.generate_state()
    x_zero = [u_prior_state, m_zero, None]
    pde.solveFwd(x_zero[hp.STATE], x_zero)
    d_prior = misfit.d.copy()
    misfit.B.mult(x_zero[hp.STATE], d_prior)
    y_prior = d_prior.get_local()
    np.save(os.path.join(OUTPUT_DIR, "y_prior.npy"), y_prior)
    print(f"Saved  y_prior.npy  shape={y_prior.shape}  range=[{y_prior.min():.4f}, {y_prior.max():.4f}]")

    # ── MAP estimate ──────────────────────────────────────────────────────────
    print("\nComputing MAP estimate ...")
    model  = hp.Model(pde, prior, misfit)
    m_init = prior.mean.copy()
    solver = hp.ReducedSpaceNewtonCG(model)
    solver.parameters["rel_tolerance"]      = 1e-6
    solver.parameters["abs_tolerance"]      = 1e-12
    solver.parameters["max_iter"]           = 50
    solver.parameters["GN_iter"]            = 5
    solver.parameters["globalization"]      = "LS"
    solver.parameters["LS"]["c_armijo"]     = 1e-4

    x_map   = solver.solve([None, m_init, None])
    map_arr = vertex_grid(Vh[hp.PARAMETER], x_map[hp.PARAMETER], mesh)
    np.save(os.path.join(OUTPUT_DIR, "map_array.npy"), map_arr)
    print(f"MAP {'converged' if solver.converged else 'DID NOT CONVERGE'}"
          f"  in {solver.it} iter   final cost = {solver.final_cost:.4g}")
    print(f"Saved  map_array.npy  shape={map_arr.shape}")

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, arr, title in zip(
        axes,
        [u_true_flat.reshape(NX + 1, NY + 1), map_arr],
        ["Shepp-Logan phantom (true parameter)", "MAP estimate"],
    ):
        im = ax.imshow(arr, origin="lower", interpolation="bilinear")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sl_true_and_map.png"), dpi=100)
    plt.close()
    print("\nDone.  All outputs in:", OUTPUT_DIR)
