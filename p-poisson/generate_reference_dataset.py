#!/usr/bin/env python3

#  hIPPYlib-MUQ interface for large-scale Bayesian inverse problems
#  Copyright (c) 2019-2020, The University of Texas at Austin,
#  University of California--Merced, Washington University in St. Louis,
#  The United States Army Corps of Engineers, Massachusetts Institute of Technology

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(""))))

import yaml
import numpy as np
import dolfin as dl
import hippylib as hp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math

from nonlinearPPoissonProblem import (
    NonlinearPPossionForm,
    EnergyFunctionalPDEVariationalProblem,
)

ntargets = 300
rel_noise = 0.005
ndim = 3
Length = 1.0
Width = Length
Height = 0.05

def get_data_3d(V, vec, mesh, decimals=12):
    """
    Return a 3-D numpy array of values on the structured BoxMesh grid,
    indexed as [ix, iy, iz] with x slowest and z fastest.

    Parameters
    ----------
    V : dolfin.FunctionSpace
    vec : dolfin.PETScVector (or compatible)
    mesh : dolfin.Mesh
    decimals : int
        round coords to avoid FP jitter when grouping.

    Returns
    -------
    A : np.ndarray  # shape (nx+1, ny+1, nz+1)
    xs, ys, zs : np.ndarray  # unique sorted coordinate axes
    """
    f = dl.Function(V, vec)
    vals = np.array(f.compute_vertex_values(mesh))  # (Nverts,)

    coords = mesh.coordinates()                     # (Nverts, 3)
    x = np.round(coords[:, 0], decimals)
    y = np.round(coords[:, 1], decimals)
    z = np.round(coords[:, 2], decimals)

    xs = np.unique(x); ys = np.unique(y); zs = np.unique(z)

    # indices of each vertex on the Cartesian grid
    ix = np.searchsorted(xs, x)
    iy = np.searchsorted(ys, y)
    iz = np.searchsorted(zs, z)

    A = np.empty((xs.size, ys.size, zs.size), dtype=vals.dtype)
    A[ix, iy, iz] = vals  # place each vertex value at its [ix,iy,iz] slot

    return A, xs, ys, zs


class BottomBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], 0)


class SideBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            dl.near(x[0], 0)
            or dl.near(x[0], 1.0) #TODO: Change for variable length
            or dl.near(x[1], 0)
            or dl.near(x[1], 1.0) #TODO: Change for variable width
        )


class TopBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], 0.05) #TODO: Change for variable height

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

def setup_problem(yamlfile):
    """
    Setup the p-Poisson problem from yaml configuration
    """
    with open(yamlfile, "r") as fid:
        inargs = yaml.safe_load(fid)

    # Set up the mesh and finite element function spaces
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

    # Set up the forward problem
    dl.parameters["form_compiler"]["quadrature_degree"] = 3
    bc = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), side)

    # Bottom and side boundary markers
    boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundary_markers.set_all(0)
    bottom.mark(boundary_markers, 1)
    side.mark(boundary_markers, 2)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    # Setup PDE
    order_ppoisson = 3.0
    functional = NonlinearPPossionForm(order_ppoisson, None, ds(1))
    pde = EnergyFunctionalPDEVariationalProblem(Vh, functional, bc, bc)

    # Configure solvers
    pde.solver = dl.PETScKrylovSolver("cg", "icc")
    pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", "icc")
    pde.solver_adj_inc = dl.PETScKrylovSolver("cg", "icc")
    pde.fwd_solver.solver = dl.PETScKrylovSolver("cg", "icc")

    pde.fwd_solver.parameters["gdu_tolerance"] = 1e-16
    pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20

    # Set up the prior
    gamma = 1.0
    delta = 1.0
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, robin_bc=True)

    return mesh, Vh, pde, prior


def generate_sample(mesh, Vh, pde, prior):
    """
    Generate a single sample with its PDE solution
    """
    try:
        # Sample from prior
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
        hp.parRandom.normal(1.0, noise)
        mtrue = dl.Vector()
        prior.init_vector(mtrue, 0)
        prior.sample(noise, mtrue)

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

        # Solve PDE
        utrue = pde.generate_state()
        x = [utrue, mtrue, None]
        pde.solveFwd(x[hp.STATE], x)
        misfit.B.mult(x[hp.STATE], misfit.d)
        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        misfit.noise_variance = noise_std_dev * noise_std_dev
        hp.parRandom.normal_perturb(noise_std_dev, misfit.d)

        # Convert to numpy arrays using the provided function
        parameter_array, *_ = get_data_3d(Vh[hp.PARAMETER], mtrue, mesh)
        solution_array,  *_ = get_data_3d(Vh[hp.STATE], x[hp.STATE], mesh)

        return parameter_array, solution_array, misfit.d.get_local()

    except Exception as e:
        print(f"Error generating sample: {str(e)}")
        return None, None


def main():
    # Configuration
    here = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(here, "ppoisson_box.yaml")

    num_samples = 350000
    output_dir = "training_dataset"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup the problem
    print("Setting up the p-Poisson problem...")
    mesh, Vh, pde, prior = setup_problem(yaml_file)

    # Initialize arrays to store all samples
    # We'll get the dimensions from the first sample
    print("Generating initial sample to determine dimensions...")
    param_sample, sol_sample, misfit_sample = generate_sample(mesh, Vh, pde, prior)
    if param_sample is None:
        print("Failed to generate initial sample. Exiting.")
        return

    parameters = np.zeros((num_samples, *param_sample.shape))
    solutions = np.zeros((num_samples, *sol_sample.shape))
    misfits = np.zeros((num_samples, *misfit_sample.shape))
    parameters[0] = param_sample
    solutions[0] = sol_sample
    misfits[0] = misfit_sample

    # Generate remaining samples with progress bar
    print(f"Generating {num_samples - 1} more samples...")
    with tqdm(total=num_samples - 1) as pbar:
        successful_samples = 1
        while successful_samples < num_samples:
            param_sample, sol_sample, misfit_sample = generate_sample(mesh, Vh, pde, prior)
            if param_sample is not None:
                parameters[successful_samples] = param_sample
                solutions[successful_samples] = sol_sample
                misfits[successful_samples] = misfit_sample
                successful_samples += 1
                pbar.update(1)

    # Save the datasets
    print("\nSaving datasets...")
    np.save(os.path.join(output_dir, "x0_parameters.npy"), parameters)
    np.save(os.path.join(output_dir, "x0_solutions_full.npy"), solutions)
    np.save(os.path.join(output_dir, "x0_solutions.npy"), misfits)

    print("\nDataset generation complete!")
    print(f"Files are saved in the '{output_dir}' directory as:")
    print("- x0_parameters.npy")
    print("- x0_solutions_full.npy")
    print("- x0_solutions.npy")


if __name__ == "__main__":
    main()
