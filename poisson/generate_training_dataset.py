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
import hippylib2muq as hm

ntargets = 100
# rel_noise = 0.005
# rel_noise = 0.25
rel_noise = 0.001

def get_data(arg, vec, mesh):
    # sqrt_dim = int(np.sqrt(arg.dim()))
    f = dl.Function(arg, vec)
    C = f.compute_vertex_values(mesh)
    reshape_dim = int(np.sqrt(C.shape[0]))
    return C.reshape(reshape_dim, reshape_dim)


def u_boundary(x, on_boundary):
    """
    Define the boundaries that Dirichlet boundary condition is imposed on; in
    this example, they are the top and bottom boundaries.
    """
    return on_boundary and (dl.near(x[1], 0.0) or dl.near(x[1], 1.0))


class bottom_boundary(dl.SubDomain):
    """
    Define the bottom boundary.
    """

    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[1], 0.0)


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


def setup_problem(yamlfile):
    with open(yamlfile, "r") as fid:
        inargs = yaml.full_load(fid)

    nx = ny = inargs["nelement"]
    mesh = dl.UnitSquareMesh(nx, ny)

    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh = [Vh2, Vh1, Vh2]

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

    f = dl.Constant(1.0)

    def pde_varf(u, m, p):
        return (
            dl.exp(m) * dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx
            - f * p * dl.dx
        )

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    gamma = 1.0
    delta = 9.0
    anis_diff = dl.Identity(2)

    prior = hp.BiLaplacianPrior(
        Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True
    )

    return mesh, Vh, pde, prior

def generate_sample(mesh, Vh, pde, prior):
    try:
        mtrue = true_model(prior)
        parameter_array = get_data(Vh[hp.PARAMETER], mtrue, mesh)
        n_side = 10
        grid_1d = np.linspace(0.05, 0.95, n_side)
        X, Y = np.meshgrid(grid_1d, grid_1d)
        targets = np.vstack([X.ravel(), Y.ravel()]).T

        misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)

        utrue = pde.generate_state()
        state_array = get_data(Vh[hp.STATE], utrue, mesh)

        x = [utrue, mtrue, None]
        pde.solveFwd(x[hp.STATE], x)
        misfit.B.mult(x[hp.STATE], misfit.d)
        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        # misfit.noise_variance = noise_std_dev * noise_std_dev
        misfit.noise_variance = rel_noise

        hp.parRandom.normal_perturb(noise_std_dev, misfit.d)

        return parameter_array, state_array, misfit.d.get_local(), targets
    except Exception as e:
        print(f"Error generating sample: {str(e)}")
        return None, None
    
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(here, "poisson.yaml")

    num_samples = 125000
    output_dir = "training_dataset"

    os.makedirs(output_dir, exist_ok=True)

    print("Setting up the Poisson problem...")
    mesh, Vh, pde, prior = setup_problem(yaml_file)

    # Initialize arrays to store all samples
    # We'll get the dimensions from the first sample
    print("Generating initial sample to determine dimensions...")
    param_sample, sol_sample, misfit_sample, targets_sample = generate_sample(mesh, Vh, pde, prior)
    if param_sample is None:
        print("Failed to generate initial sample. Exiting.")
        return

    parameters = np.zeros((num_samples, *param_sample.shape))
    solutions = np.zeros((num_samples, *sol_sample.shape))
    misfits = np.zeros((num_samples, *misfit_sample.shape))
    targets = np.zeros((num_samples, ntargets, 2))
    parameters[0] = param_sample
    solutions[0] = sol_sample
    misfits[0] = misfit_sample
    targets[0] = targets_sample

    # Generate remaining samples with progress bar
    print(f"Generating {num_samples - 1} more samples...")
    with tqdm(total=num_samples - 1) as pbar:
        successful_samples = 1
        while successful_samples < num_samples:
            param_sample, sol_sample, misfit_sample, targets_sample = generate_sample(mesh, Vh, pde, prior)
            if param_sample is not None:
                parameters[successful_samples] = param_sample
                solutions[successful_samples] = sol_sample
                misfits[successful_samples] = misfit_sample
                targets[successful_samples] = targets_sample
                successful_samples += 1
                pbar.update(1)

    # Save the datasets
    print("\nSaving datasets...")
    np.save(os.path.join(output_dir, "parameters.npy"), parameters)
    np.save(os.path.join(output_dir, "solutions_full.npy"), solutions)
    np.save(os.path.join(output_dir, "solutions_grid.npy"), misfits)
    np.save(os.path.join(output_dir, "locations_grid.npy"), targets)

    print("\nDataset generation complete!")
    print(f"Files are saved in the '{output_dir}' directory as:")
    print("- parameters.npy")
    print("- solutions_full.npy")
    print("- solutions_grid.npy")
    print("- locations_grid.npy")

if __name__ == "__main__":
    main()