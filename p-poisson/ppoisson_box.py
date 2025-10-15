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

RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK",
            os.environ.get("PMI_RANK", 0))) + 30
SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
            os.environ.get("PMI_SIZE", 1)))

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


def data_file(action, target=None, data=None, direc=None):
    """
    Read or write the observations.

    :param action: "w" is to write the date to the file named "data.h5" and "r"
                   is to read the data from "data.h5"
    :param target: the location of the observation data
    :param data: the observation data
    """
    if direc is None:
        f = h5py.File("data.h5", action)
    else:
        f = h5py.File(direc, action)
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
    output_root = "training_dataset"
    output_dir  = os.path.join(output_root, f"chain_{RANK:02d}")
    os.makedirs(output_dir, exist_ok=True)

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
    ntargets = 100
    rel_noise = 0.005

    print("Number of observation points: {0}".format(ntargets))

    if inargs["have_data"]:
        targets, data = data_file("r", direc="data_grid.h5")
        misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
        misfit.d.set_local(data)

        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        misfit.noise_variance = noise_std_dev * noise_std_dev
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
        # grid_1d = np.linspace(0.05, 0.95, 10)
        # X, Y = np.meshgrid(grid_1d, grid_1d)
        # targets = np.concatenate([X.ravel()[None, :].T, Y.ravel()[None, :].T, dummy3], axis=1)

        misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)

        mtrue = true_model(prior)

        # Export true parameter to mtrue.xdmf file
        with dl.XDMFFile(mesh.mpi_comm(), "mtrue_grid.xdmf") as fid:
            export2XDMF(mtrue, Vh[hp.PARAMETER], fid)

        utrue = pde.generate_state()
        x = [utrue, mtrue, None]
        pde.solveFwd(x[hp.STATE], x)
        misfit.B.mult(x[hp.STATE], misfit.d)
        MAX = misfit.d.norm("linf")
        noise_std_dev = rel_noise * MAX
        misfit.noise_variance = noise_std_dev * noise_std_dev

        hp.parRandom.normal_perturb(noise_std_dev, misfit.d)

        data_file("w", target=targets, data=misfit.d.get_local(), direc="data_random_grid.h5")

        #  Export true state solution to uture.xdmf file
        with dl.XDMFFile(mesh.mpi_comm(), "utrue_random_grid.xdmf") as fid:
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



    #
    #  Set up ModPieces for implementing MCMC methods
    #
    print(sep, "Set up ModPieces for implementing MCMC methods", sep)

    # a place holder ModPiece for the parameters
    idparam = mm.IdentityOperator(Vh[hp.PARAMETER].dim())

    # log Gaussian Prior ModPiece
    gaussprior = hm.BiLaplaceGaussian(prior)
    log_gaussprior = gaussprior.AsDensity()

    # parameter to log likelihood Modpiece
    param2likelihood = hm.Param2LogLikelihood(model)

    # log target ModPiece
    log_target = mm.DensityProduct(2)

    workgraph = mm.WorkGraph()

    # Identity operator for the parameters
    workgraph.AddNode(idparam, "Identity")

    # Prior model
    workgraph.AddNode(log_gaussprior, "Prior")

    # Likelihood model
    workgraph.AddNode(param2likelihood, "Likelihood")

    # Posterior
    workgraph.AddNode(log_target, "Log_target")

    workgraph.AddEdge("Identity", 0, "Prior", 0)
    workgraph.AddEdge("Prior", 0, "Log_target", 0)

    workgraph.AddEdge("Identity", 0, "Likelihood", 0)
    workgraph.AddEdge("Likelihood", 0, "Log_target", 1)

    # Enable caching
    # if inargs["MCMC"]["name"] not in ("hmala", ""): # This had hpcn before
    log_gaussprior.EnableCache()
    param2likelihood.EnableCache()

    print(f"Starting {inargs['MCMC']['name']} chain...")

    # Construct the problem

    postDens = workgraph.CreateModPiece("Log_target")
    problem = ms.SamplingProblem(postDens)
    options = dict()
    options["NumSamples"] = inargs["MCMC"]["nsamples"]  # Number of MCMC steps to take
    options["BurnIn"] = inargs["MCMC"]["burnin"]  # Number of steps to throw away as burn in
    options["PrintLevel"] = 3

    method_list = dict()

    # h-MALA
    opts = options.copy()
    # opts.update( {'StepSize':0.00000006} ) #TODO: see if there is auto-tuning of parameter during burn-in. Last run I set this to 0.15
    # opts.update({'StepSize': 0.011})
    opts.update({'StepSize': 0.055})
    gauss_hmala = hm.LAPosteriorGaussian(nu, use_zero_mean=True)
    prop = ms.MALAProposal(opts, problem, gauss_hmala)
    kern = ms.MHKernel(opts, problem, prop)
    sampler = ms.SingleChainMCMC(opts, [kern])

    method_list['hMALA'] = {'Options': opts, 'Sampler': sampler}

    # Running MCMC
    # base_seed = 12345
    # np.random.seed(base_seed + RANK)

    # noise = dl.Vector()
    # nu.init_vector(noise, "noise")
    # hp.parRandom.seed(base_seed + RANK)
    # hp.parRandom.normal(1.0, noise)
    # pr_s = model.generate_vector(hp.PARAMETER)
    # post_s = model.generate_vector(hp.PARAMETER)
    # nu.sample(noise, pr_s, post_s, add_mean=True)
    # x0 = hm.dlVector2npArray(
    #     post_s
    # )  # initial guess is a sample from laplace approx posterior.
    def draw_x0_numpy(rank, base_seed=12345):
        np.random.seed(base_seed + rank)  # deterministic per chain

        noise = dl.Vector()
        nu.init_vector(noise, "noise")

        # Fill 'noise' with NumPy-generated values
        nloc = noise.local_size()
        noise_local = np.random.normal(0.0, 1.0, size=nloc)
        noise.set_local(noise_local)
        noise.apply("")

        pr_s   = model.generate_vector(hp.PARAMETER)
        post_s = model.generate_vector(hp.PARAMETER)
        nu.sample(noise, pr_s, post_s, add_mean=True)

        return hm.dlVector2npArray(post_s)
    x0 = draw_x0_numpy(RANK)
    print(f"[chain {RANK}] first 5 entries of x0: {x0[:5]}")

    # Implement MCMC simulations
    for mName, method in method_list.items():
        # Run the MCMC sampler
        print(f"Starting to sample from chain {mName}")
        sampler = method["Sampler"]
        samps = sampler.Run([x0])

        # Save the computed results
        method["Samples"] = samps
        method["ElapsedTime"] = sampler.TotalTime()

        kernel = sampler.Kernels()[0]
        if "AcceptanceRate" in dir(kernel):
            method["AcceptRate"] = kernel.AcceptanceRate()
        elif "AcceptanceRates" in dir(kernel):
            method["AcceptRate"] = kernel.AcceptanceRates()

        print(
            "Drawn ",
            options["NumSamples"] - options["BurnIn"] + 1,
            "MCMC samples using",
            mName,
        )

    samples = method_list["hMALA"]["Samples"].AsMatrix().T
    num_vertices = mesh.num_vertices()
    num_samples = samples.shape[0]

    vertex_samples = np.zeros(
        (num_samples, num_vertices)
    )  # For our example, this should be of size 20,000 x 10201

    # wandb.log({"acceptance rate": method_list["MALA"]["AcceptRate"]}) # TODO: When I increase the number of chains, use wandb

    f = dl.Function(Vh[hp.PARAMETER])
    for i in range(num_samples):
        f.vector().set_local(samples[i])
        vertex_samples[i] = f.compute_vertex_values(mesh)

    np.save(
        os.path.join(output_dir, "hmala_samples_grid.npy"),
        vertex_samples,
    )

    save_list = method_list.copy()
    save_list["hMALA"]["Sampler"] = None
    save_list["hMALA"]["Samples"] = vertex_samples

    with open(os.path.join(output_dir, "method-list-hmala-grid.pkl"), "wb") as f:
        pickle.dump(method_list, f)
