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
The basic part of this code is taken from
https://github.com/hippylib/hippylib/blob/master/applications/poisson/model_subsurf.py.

Input values of this script should be defined in "ppoisson_box.yaml"
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import math
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pickle

import dolfin as dl
import hippylib as hp
import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import hippylib2muq as hm

from nonlinearPPoissonProblem import *


def true_model(prior):
    """
    Define true parameter field.

    In this example, we sample from the prior and take it as the true parameter
    field.
    """
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue

# --- helpers: numpy <-> dl.Vector ---
def vec_to_np(v: dl.Vector) -> np.ndarray:
    return v.get_local().copy()

def func_to_np(f: dl.Function) -> np.ndarray:
    return f.vector().get_local().copy()

def np_to_vec(arr: np.ndarray) -> dl.Vector:
    v = dl.Vector()
    prior.init_vector(v, 0)     # parameter layout
    v.set_local(np.asarray(arr, dtype=float))
    return v

def build_Pk(V_pca: np.ndarray, apply_P) -> np.ndarray:
    # V_pca: (n, k), columns are basis vectors in field space
    PV = np.column_stack([apply_P(V_pca[:, j]) for j in range(V_pca.shape[1])])  # (n, k)
    # print(f"This is the shape of PV: {PV.shape}")
    Pk = V_pca.T @ PV
    return 0.5 * (Pk + Pk.T)  # symmetrize



def export2XDMF(x, Vh, fid):
    fid.parameters["functions_share_mesh"] = True
    fid.parameters["rewrite_function_mesh"] = False

    fun = hp.vector2Function(x, Vh)
    fid.write(fun, 0)


class BottomBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], 0)


class SideBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            dl.near(x[0], 0)
            or dl.near(x[0], Length)
            or dl.near(x[1], 0)
            or dl.near(x[1], Width)
        )


class TopBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], Height)
    

class BottomIndexer:
    """
    Build a regular (ny+1, nx+1) raster of the z=0 vertices by coordinates.
    Works for scalar P1 parameter space on BoxMesh.
    Provides safe DoF<->bottom conversions without artifacts.
    """
    def __init__(self, mesh: dl.Mesh, Vparam: dl.FunctionSpace, nx: int, ny: int,
                 xlen: float = 1.0, ylen: float = 1.0, ztol: float = 1e-12):
        self.mesh, self.V = mesh, Vparam
        self.nx, self.ny = nx, ny
        self.dx = xlen / nx
        self.dy = ylen / ny
        self.ztol = ztol

        coords = mesh.coordinates()                    # (Nvert, 3)
        z = coords[:, 2]
        # vertices on bottom
        bot_verts = np.where(np.abs(z) <= ztol)[0]
        if len(bot_verts) != (nx + 1) * (ny + 1):
            raise ValueError(f"Expected {(nx+1)*(ny+1)} bottom verts, found {len(bot_verts)}")

        # Integer grid indices by rounding to nearest grid line
        x = coords[bot_verts, 0]
        y = coords[bot_verts, 1]
        ix = np.round(x / self.dx).astype(int)
        iy = np.round(y / self.dy).astype(int)

        # Build a (ny+1, nx+1) table of vertex ids in raster order
        table = -np.ones((ny + 1, nx + 1), dtype=int)
        for v, i, j in zip(bot_verts, ix, iy):
            table[j, i] = v
        if (table < 0).any():
            raise RuntimeError("Bottom grid table incomplete (indexing issue).")

        self.vert_table = table                        # (ny+1, nx+1) of vertex ids
        self.vert_ids_raster = table.ravel(order="C")  # row-major: y fast -> x

        # Maps for DoF<->vertex
        self.v2d = dl.vertex_to_dof_map(self.V)        # dof_at_vertex = v2d[vertex_id]
        # Precompute the DoF indices in the same raster order
        self.dof_ids_raster = self.v2d[self.vert_ids_raster]

    # ---------- DoF -> bottom ----------
    def dofs_to_bottom_vec(self, dofs_np: np.ndarray) -> np.ndarray:
        """Full DoF vector -> bottom vector (Nb,) in raster (row-major) order."""
        return dofs_np[self.dof_ids_raster]

    def dofs_to_bottom_img(self, dofs_np: np.ndarray) -> np.ndarray:
        """Full DoF vector -> (ny+1, nx+1) image (row-major)."""
        b = self.dofs_to_bottom_vec(dofs_np)
        return b.reshape(self.ny + 1, self.nx + 1, order="C")

    def func_to_bottom_img(self, f: dl.Function) -> np.ndarray:
        """Alternative read: via compute_vertex_values (avoids DoF/vertex permutation confusion)."""
        vert_vals = f.compute_vertex_values(self.mesh)  # values per vertex id
        b = vert_vals[self.vert_ids_raster]
        return b.reshape(self.ny + 1, self.nx + 1, order="C")

    # ---------- bottom -> DoF ----------
    def bottom_vec_to_dofs(self, bvec: np.ndarray) -> np.ndarray:
        """Bottom vector (Nb,) -> full DoF vector (zeros elsewhere)."""
        out = np.zeros(self.V.dim(), dtype=float)
        out[self.dof_ids_raster] = bvec
        return out

    def bottom_img_to_dofs(self, bimg: np.ndarray) -> np.ndarray:
        """(ny+1, nx+1) -> full DoF vector."""
        return self.bottom_vec_to_dofs(bimg.ravel(order="C"))


def generate_starting():
    """
    Generate an initial parameter sample from the Laplace posterior for the MUQ
    MCMC simulation
    """
    noise = dl.Vector()
    nu.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    pr_s = model.generate_vector(hp.PARAMETER)
    post_s = model.generate_vector(hp.PARAMETER)
    nu.sample(noise, pr_s, post_s, add_mean=True)
    x0 = hm.dlVector2npArray(post_s)
    return x0


def data_file(action, target=None, data=None):
    """
    Read or write the observations.

    :param action: "w" is to write the date to the file named "data.h5" and "r"
                   is to read the data from "data.h5"
    :param target: the location of the observation data
    :param data: the observation data
    """
    f = h5py.File("data.h5", action)
    if action == "w":
        f["/target"] = target
        f["/data"] = data

        f.close()
        return

    elif action == "r":
        target = f["/target"][...]
        data = f["/data"][...]

        f.close()

        return target, data


class ExtractBottomData:
    def __init__(self, mesh, Vh):
        bmesh = dl.BoundaryMesh(mesh, "exterior")
        bmarker = dl.MeshFunction("size_t", mesh, bmesh.topology().dim())
        for c in dl.cells(bmesh):
            if math.isclose(c.midpoint().z(), 0):
                bmarker[c] = 1

        smesh = dl.SubMesh(bmesh, bmarker, 1)

        self.vertex_s2b = smesh.data().array("parent_vertex_indices", 0)
        self.vertex_b2p = bmesh.entity_map(0).array()
        self.vertex2dof = dl.vertex_to_dof_map(Vh)
        self.coordinates = smesh.coordinates()

        self.tria = tri.Triangulation(
            self.coordinates[:, 0], self.coordinates[:, 1], smesh.cells()
        )

    def get_dim(self):
        return self.coordinates.shape[0]

    def get_bottom_data(self, arr):
        return arr[self.vertex2dof[self.vertex_b2p[self.vertex_s2b]]]

    def plot_array(self, arr, vmin=None, vmax=None, cmap=None, fname=None):
        val = arr[self.vertex2dof[self.vertex_b2p[self.vertex_s2b]]]

        if vmax is None:
            vmax = np.max(val)
        if vmin is None:
            vmin = np.min(val)

        plt.tripcolor(self.tria, val, shading="gouraud", vmin=vmin, vmax=vmax)
        if cmap:
            plt.set_cmap(cmap)

        plt.axis("off")
        plt.gca().set_aspect("equal")

        if fname:
            plt.savefig(fname, dpi=100, bbox_inches="tight", pad_inches=0)

        plt.show()


class TracerSideFlux:
    def __init__(self, ds, p, n):
        self.n = dl.FacetNormal(mesh)
        self.ds = ds
        self.p = p

        self.tracer = hp.QoiTracer(n)
        self.ct = 0

    def form(self, u):
        grad_u = dl.nabla_grad(u)
        etah = dl.inner(grad_u, grad_u)

        return etah ** (0.5 * (self.p - 2)) * dl.dot(grad_u, self.n) * self.ds

    def eval(self, u):
        uf = hp.vector2Function(u, Vh[hp.STATE])
        return dl.assemble(self.form(uf))

    def update_tracer(self, state):
        y = self.eval(state)
        self.tracer.append(self.ct, y)
        self.ct += 1


def paramcoord2eigencoord(V, B, x):
    """
    Projection a parameter vector to eigenvector.

    y = V^T * B * x

    :param V multivector: eigenvectors
    :param operator: the right-hand side operator in the generalized eig problem
    :param x np.array: parameter data
    """
    # convert np.array to multivector
    nvec = 1
    Xvecs = hp.MultiVector(pde.generate_parameter(), nvec)
    hm.npArray2dlVector(x, Xvecs[0])

    # multipy B
    BX = hp.MultiVector(Xvecs[0], nvec)
    hp.MatMvMult(B, Xvecs, BX)
    VtBX = BX.dot_mv(V)

    return VtBX.transpose()


if __name__ == "__main__":
    with open("ppoisson_box.yaml") as fid:
        inargs = yaml.full_load(fid)

    sep = "\n" + "#" * 80 + "\n"
    output_dir = "training_dataset"

    #
    #  Set up the mesh and finite element function spaces
    #
    ndim = 3
    Length = 1.0
    Width = Length
    Height = 0.05

    nx = inargs["nelement"][0]
    ny = nx
    nz = inargs["nelement"][1]

    mesh = dl.BoxMesh(dl.Point(0, 0, 0), dl.Point(Length, Width, Height), nx, ny, nz)
    bottom = BottomBoundary()
    side = SideBoundary()
    top = TopBoundary()

    Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh = [Vh2, Vh1, Vh2]

    print(
        "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
            Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()
        )
    )

    extract_bottom = ExtractBottomData(mesh, Vh[hp.PARAMETER])

    # _bottom_idx = extract_bottom.vertex2dof[extract_bottom.vertex_b2p[extract_bottom.vertex_s2b]]
    bottom_idx = BottomIndexer(mesh, Vh[hp.PARAMETER], nx=nx, ny=ny)


    # def dofs_to_bottom(dofs_np: np.ndarray) -> np.ndarray:
    #     """Full parameter DoF vector -> bottom-surface vector (Nb,)."""
    #     return dofs_np[_bottom_idx]

    # def bottom_to_dofs(bottom_np: np.ndarray, ndof: int) -> np.ndarray:
    #     """Scatter bottom-surface vector back into a full DoF vector (zeros elsewhere)."""
    #     out = np.zeros(ndof, dtype=float)
    #     out[_bottom_idx] = bottom_np
    #     return out

    #
    #  Set up the forward problem
    #
    dl.parameters["form_compiler"]["quadrature_degree"] = 3

    bc = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), side)

    #  Bottom and side boundary markers
    boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundary_markers.set_all(0)

    bottom.mark(boundary_markers, 1)
    side.mark(boundary_markers, 2)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    order_ppoisson = 3.0
    functional = NonlinearPPossionForm(order_ppoisson, None, ds(1))
    pde = EnergyFunctionalPDEVariationalProblem(Vh, functional, bc, bc)

    pde.solver = dl.PETScKrylovSolver("cg", "icc")  # amg_method())
    pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", "icc")  # amg_method())
    pde.solver_adj_inc = dl.PETScKrylovSolver("cg", "icc")  # amg_method())
    pde.fwd_solver.solver = dl.PETScKrylovSolver("cg", "icc")  # amg_method())

    pde.fwd_solver.parameters["gdu_tolerance"] = 1e-16
    pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20

    #  pde.solver.parameters["relative_tolerance"] = 1e-15
    #  pde.solver.parameters["absolute_tolerance"] = 1e-20
    #  pde.solver_fwd_inc.parameters = pde.solver.parameters
    #  pde.solver_adj_inc.parameters = pde.solver.parameters
    #  pde.fwd_solver.solver.parameters = pde.solver.parameters

    #
    # Set up the prior
    #
    gamma = 1.0
    delta = 1.0

    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, robin_bc=True)
    print(
        "Prior regularization: (delta_x - gamma*Laplacian)^order: "
        "delta={0}, gamma={1}, order={2}".format(delta, gamma, 2)
    )

    #
    #  Set up the misfit functional and generate synthetic observations
    #
    ntargets = 300
    rel_noise = 0.005

    print("Number of observation points: {0}".format(ntargets))

    if inargs["have_data"]:
        targets, data = data_file("r")
        misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
        misfit.d.set_local(data)

        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        misfit.noise_variance = noise_std_dev * noise_std_dev
        print("Successfully read in targets, data")
    else:
        eps = 0.05
        dummy1 = np.random.uniform(
            Length * (0.0 + eps), Length * (1.0 - eps), [ntargets, 1]
        )
        dummy2 = np.random.uniform(
            Width * (0.0 + eps), Width * (1.0 - eps), [ntargets, 1]
        )
        dummy3 = np.full((ntargets, 1), Height)
        targets = np.concatenate([dummy1, dummy2, dummy3], axis=1)

        misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)

        mtrue = true_model(prior)

        # Export true parameter to mtrue.xdmf file
        with dl.XDMFFile(mesh.mpi_comm(), "mtrue.xdmf") as fid:
            export2XDMF(mtrue, Vh[hp.PARAMETER], fid)

        utrue = pde.generate_state()
        x = [utrue, mtrue, None]
        pde.solveFwd(x[hp.STATE], x)
        misfit.B.mult(x[hp.STATE], misfit.d)
        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        misfit.noise_variance = noise_std_dev * noise_std_dev

        hp.parRandom.normal_perturb(noise_std_dev, misfit.d)

        data_file("w", target=targets, data=misfit.d.get_local())

        #  Export true state solution to uture.xdmf file
        with dl.XDMFFile(mesh.mpi_comm(), "utrue.xdmf") as fid:
            export2XDMF(utrue, Vh[hp.STATE], fid)
        
        # Export targets observed of true state
        # np.save(
        #     os.path.join(output_dir, "true_state_targets.npy"),
        #     misfit.d.get_local(),
        # )

    model = hp.Model(pde, prior, misfit)

    # Get map
    m = prior.mean.copy()

    # Reduced objective & Gauss-Newton solver
    # obj = hp.ReducedFunction(model)                    # uses model J(m), grad J, Hessian actions
    opt_rtol = 1e-6
    opt_atol = 1e-12
    max_iter = 25
    # solver = hp.ReducedNewtonCG(model)
    # solver.parameters["gn_reduction_factor"] = 1.0     # pure GN on this problem is typical
    # solver.parameters["rel_tolerance"] = opt_rtol
    # solver.parameters["abs_tolerance"] = opt_atol
    # solver.parameters["max_iter"] = max_iter
    _solver = None
    for cls in ("ReducedSpaceNewtonCG", "ReducedNewtonCG", "ReducedNewton"):
        if hasattr(hp, cls):
            _solver = getattr(hp, cls)(model)
            break

    if _solver is None:
        raise RuntimeError(
            "No Newton solver class found in this hIPPYlib build. "
            "Available attrs: " + ", ".join(sorted(dir(hp)))
        )

    # Set tolerances if the class exposes a 'parameters' dict
    if hasattr(_solver, "parameters"):
        # _solver.parameters["gn_reduction_factor"] = 1.0
        _solver.parameters["rel_tolerance"] = 1e-6
        _solver.parameters["abs_tolerance"] = 1e-12
        _solver.parameters["max_iter"] = 25
        _solver.parameters["GN_iter"] = 5
        _solver.parameters["globalization"] = "LS"
        _solver.parameters["LS"]["c_armijo"] = 1e-4

    solver = _solver

    # Pack state for solver: x = [state, parameter, adjoint]
    # u = model.generate_vector(hp.STATE)
    # p = model.generate_vector(hp.ADJOINT)
    x = [None, m, None]

    # Solve for MAP
    solver.solve(x)
    m_map = x[hp.PARAMETER]                            # this is the MAP in dl.Vector
    if solver.converged:
        print("\nConverged in ", solver.it, " iterations.")
    else:
        print("\nNot Converged")

    print("Termination reason:  ", solver.termination_reasons[solver.reason])
    print("Final gradient norm: ", solver.final_grad_norm)
    print("Final cost:          ", solver.final_cost)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)

    # Target rank and oversampling; tune k by memory/time (3D -> start moderate)
    k = inargs["MCMC"].get("lr_rank", 150)             # e.g., 100–300 is common
    p_over = inargs["MCMC"].get("lr_oversample", 20)

    # Random probe multivector
    Omega = hp.MultiVector(m_map, k + p_over)
    hp.parRandom.normal(1.0, Omega)

    # Generalized eigendecomp: Hmisfit v = lambda * (prior.R) v  (double pass randomized)
    # prior.R and prior.Rsolver come from BiLaplacianPrior
    lam, V = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

    # Laplace (Gaussian) posterior with low-rank update; set mean to MAP
    nu = hp.GaussianLRPosterior(prior, lam, V) 
    nu.mean = m_map

    # mu = vec_to_np(m_map)  # (n,)
    # Npca = inargs["MCMC"].get("lr_pca_samples", 400)   # 200–800 is fine
    # k_pca = inargs["MCMC"].get("lr_rank", 150) 

    # s_prior = dl.Function(Vh[hp.PARAMETER], name="sample_prior")
    # s_post = dl.Function(Vh[hp.PARAMETER], name="sample_post")
    # noise = dl.Vector(); nu.init_vector(noise, "noise") 

    # npar = Vh[hp.PARAMETER].dim()
    # X_fields = np.empty((Npca, npar), dtype=float)
    # for i in range(Npca):
    #     hp.parRandom.normal(1.0, noise)                   # z ~ N(0, I)
    #     nu.sample(noise, s_prior.vector(), s_post.vector(), add_mean=True)  # post sample
    #     X_fields[i, :] = func_to_np(s_post)

    # flat_length = (nx + 1) * (ny + 1)

    # # mu = mu[:flat_length]

    # # X_fields = X_fields[:, :flat_length]

    # Xc = X_fields - mu[None, :]
    # U_svd, S_svd, Vt_svd = np.linalg.svd(Xc / np.sqrt(max(Npca - 1, 1)),
    #                                     full_matrices=False)
    # V_full = Vt_svd.T                        # (n, r)
    # V_pca  = V_full[:, :k_pca].astype(float) # (n, k)

    # print(f"[Laplace-PCA] PCA computed: k={k_pca}, samples={Npca}")

    # def apply_R_np(v_np: np.ndarray) -> np.ndarray:
    #     x = np_to_vec(v_np)                      # dl.Vector (parameter layout)
    #     y = dl.Vector(); prior.init_vector(y, 0) # allocate output
    #     prior.R.mult(x, y)                       # R y = R x
    #     r = vec_to_np(y)
    #     return r

    # def apply_Hm_np(v_np: np.ndarray) -> np.ndarray:
    #     x = np_to_vec(v_np)
    #     y = dl.Vector(); Hmisfit.init_vector(y, 0)
    #     Hmisfit.mult(x, y)                       # Hmisfit y = Hmisfit x
    #     h = vec_to_np(y)
    #     return h

    # def apply_P_np(v_np: np.ndarray) -> np.ndarray:
    #     # Posterior precision P = R + Hmisfit
    #     return apply_R_np(v_np) + apply_Hm_np(v_np)

    
    # Pk = build_Pk(V_pca, apply_P_np)   # (k, k)

    # os.makedirs(output_dir, exist_ok=True)
    # np.savez_compressed(
    #     os.path.join(output_dir, "laplace_pca_pack.npz"),
    #     mu=mu,           # (n,)
    #     V_pca=V_pca,     # (n, k)
    #     Pk=Pk,           # (k, k)
    #     meta=np.array([
    #         "mu: MAP field (n,)",
    #         "V_pca: PCA basis around MAP (columns) (n,k)",
    #         "Pk: V_pca^T (R + Hmisfit) V_pca (k,k)"
    #     ], dtype=object)
    # )
    # print(f"[Laplace-PCA] Saved to {os.path.join(output_dir, 'laplace_pca_pack.npz')}\n")

    # =========================
# Build & save Laplace-PCA pack IN BOTTOM SPACE
# =========================
print("\n[Laplace-PCA-bottom] Drawing Laplace samples on bottom surface...")

# 1) MAP in field space and its bottom restriction
mu_full = m_map.get_local()
mu_bot = bottom_idx.dofs_to_bottom_vec(mu_full)       # (Nb,)

# 2) Collect bottom-surface Laplace samples
Npca  = inargs["MCMC"].get("lr_pca_samples", 1000)   # 200–800 is fine
k_pca = inargs["MCMC"].get("lr_rank", 150)          # choose <= Nb
npar  = Vh[hp.PARAMETER].dim()

s_post = dl.Function(Vh[hp.PARAMETER], name="sample_post")
s_prior = dl.Function(Vh[hp.PARAMETER], name="sample_prior")
noise  = dl.Vector(); nu.init_vector(noise, "noise")

# Y_bottom will be (Npca, Nb)
Y_bottom = []
for i in range(Npca):
    hp.parRandom.normal(1.0, noise)
    nu.sample(noise, s_prior.vector(), s_post.vector(), add_mean=True)
    y_i = bottom_idx.dofs_to_bottom_vec(func_to_np(s_post))
    Y_bottom.append(y_i)
Y_bottom = np.asarray(Y_bottom, dtype=float)        # (Npca, Nb)

# 3) PCA around mu_bot in bottom space
Xc = Y_bottom - mu_bot[None, :]
U_svd, S_svd, Vt_svd = np.linalg.svd(Xc / np.sqrt(max(Npca - 1, 1)), full_matrices=False)
Vb_full = Vt_svd.T                      # (Nb, r)
Vb      = Vb_full[:, :k_pca].astype(float)  # (Nb, k)

print(f"[Laplace-PCA-bottom] PCA computed: k={k_pca}, samples={Npca}, Nb={Vb.shape[0]}")

# 4) Laplace covariance in bottom PCA coords (estimate from samples)
#    Alpha = bottom PCA coefficients; Ck ≈ cov(Alpha)
Alpha = (Y_bottom - mu_bot[None, :]) @ Vb          # (Npca, k)
Ck    = np.cov(Alpha, rowvar=False, bias=False)    # (k, k)
# small Tikhonov to ensure PD
eps = 1e-6 * (np.trace(Ck) / max(k_pca, 1))
Ck += eps * np.eye(k_pca)
# Lk = np.linalg.cholesky(Ck)                        # store L or C – your choice

# 5) Save bottom-space pack
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(output_dir, "laplace_pca_bottom_pack.npz"),
    mu_bot=mu_bot,     # (Nb,)
    Vb=Vb,             # (Nb, k)
    Ck=Ck,             # (k, k) covariance in PCA coords
    # or save Lk if you prefer
    meta=np.array([
        "mu_bot: MAP restricted to bottom (Nb,)",
        "Vb: bottom-space PCA basis (Nb,k)",
        "Ck: covariance of bottom PCA coeffs under Laplace (k,k)"
    ], dtype=object),
)
print("[Laplace-PCA-bottom] Saved pack:", os.path.join(output_dir, "laplace_pca_bottom_pack.npz"))
