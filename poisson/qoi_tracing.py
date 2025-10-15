import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
from tqdm.auto import tqdm
import statsmodels.api as sm

import hippylib2muq as hm
import dolfin as dl
import hippylib as hp

RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0))) + 30
plt.style.use("ggplot")

def cal_qoiTracer(pde, qoi, muq_samps, pde_solve):
    """
    This function is for tracing the quantity of interest.

    :param hippylib:PDEProblem pde: a hippylib:PDEProblem instance
    :param qoi: the quantity of interest; it should contain the member function
                named as ``eval`` which evaluates the value of qoi
    :param muq_samps: samples generated from ``muq`` sampler
    """
    if isinstance(muq_samps, np.ndarray):
        samps_mat = muq_samps.copy().T
    else:
        samps_mat = muq_samps.AsMatrix().T
    nums = samps_mat.shape[1]
    tracer = hp.QoiTracer(nums)

    ct = 0
    pbar = tqdm(total=nums)
    u = pde.generate_state()
    m = pde.generate_parameter()
    while ct < nums:
        m.set_local(samps_mat[:, ct])
        x = [u, m, None]
        if pde_solve:
            pde.solveFwd(u, x)
        q = qoi.eval(x)
        tracer.append(ct, q)
        ct += 1
        pbar.update(1)
    pbar.close()
    return tracer

def track_qoiTracer(pde, qoi, method_list, max_lag=None):
    """
    This function computes the autocorrelation function and the effective sample
    size of the quantity of interest.

    :param hippylib:PDEProblem pde: a hippylib:PDEProblem instance
    :param qoi: the quantity of interest; it should contain the member function
    :param dictionary method_list: a dictionary containing MCMC methods descriptions
                                   with samples generated from muq sampler
    :param int max_lag: maximum of time lag for computing the autocorrelation
                        function
    """
    qoi_dataset = dict()
    for mName, method in method_list.items():
        qoi_data = dict()
        samps = method["Samples"]
        print(type(samps))

        # Compute QOI
        tracer = cal_qoiTracer(pde, qoi, samps, False)

        # Estimate IAT
        iact, lags, acorrs = hp.integratedAutocorrelationTime(
            tracer.data, max_lag=max_lag
        )

        N = tracer.data.shape[0] if hasattr(tracer.data, "shape") else len(tracer.data)

        # Estimate ESS
        if isinstance(samps, np.ndarray):
            ess = N / iact
        else:
            ess = N / iact

        # Save computed results
        qoi_data["qoi"] = tracer.data
        qoi_data["iact"] = iact
        qoi_data["ess"] = ess

        qoi_dataset[mName] = qoi_data
    return qoi_dataset

# Defining the QOI of interest
class CoordinateQOI(object):
    def __init__(self, index):
        self.index = index

    def eval(self, x):
        """Evaluate the QOI at sample x_t, where t signifies the number of samples generated in the Markov chain.
        The QOI here is simple; it will return the entry of x_t at self.index. 

        Args:
            x_t (hp): The t-th sample from the Marvkon chain.
        """

        x = x[hp.PARAMETER].get_local()
        return x[self.index]
    
# Defining the PDE (necessary for qoi_tracer)
ndim = 2
nx = 32
ny = 32
mesh = dl.UnitSquareMesh(nx, ny)

Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
Vh = [Vh2, Vh1, Vh2]

def all_boundary(x, on_boundary):
    return on_boundary

bc = [dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), all_boundary)]
bc0 = bc.copy()

f = dl.Constant(1.0)

print("Defining PDE...")
def pde_varf(u, m, p):
    return dl.exp(m)*(dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)))*dl.dx - f*p*dl.dx
pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    
# First 50 parameter DOFs
no_dofs = 50
qoi_list = [CoordinateQOI(i) for i in range(no_dofs)]

# Loading samples
output_root = "training_dataset"
output_dir = os.path.join(output_root, f"chain_{RANK:02d}")
hmala_samples = np.load(f"{output_dir}/hmala_samples_grid.npy")

# Loading method lists to calculate
with open(os.path.join(output_dir, "method-list-hmala-grid.pkl"), "rb") as f:
    method_list = pkl.load(f)

# Get QOI results for individual coordinates
print("Starting QOI Tracing...")
max_lag = 600
qoi_datasets = [track_qoiTracer(pde, qoi, method_list, max_lag) for qoi in qoi_list]
# qoi_dataset = track_qoiTracer(pde, qoi_list, method_list)
# print(len(qoi_datasets))

# with open(os.path.join(output_dir, "coordwise_qoi.pkl"), "wb") as f:
#     pkl.dump(qoi_dataset, f)

ess_list = []
for i in range(no_dofs):
    samps = qoi_datasets[i]["hMALA"]["qoi"]
    ess = qoi_datasets[i]["hMALA"]["ess"]
    ess_list.append(ess)

    # Plot auto-correlation
    acf, conf = sm.tsa.stattools.acf(
        samps, nlags=max_lag, alpha=0.1, fft=False,
    )
    xlags = np.arange(acf.size)
    conf0 = conf.T[0]
    conf1 = conf.T[1]
    # plt.fill_between(xlags, conf0, conf1, alpha=0.1)
    plt.plot((0, max_lag), (0,0), "k--")
    plt.plot(xlags, acf, linewidth=2)
    plt.xlim((0, max_lag))
    # ax[1].set_ylim()

plt.title("hMALA - 2D Darcy")
plt.ylabel("ACF")
plt.xlabel("Lag")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "coordwise_plot.png"))
plt.close()

min_ess = np.min(ess_list)
plt.hist(ess_list, bins=20, alpha=0.75);
plt.axvline(min_ess, c="black", linestyle="--", label=f"min ess = {min_ess:,.2f}")
plt.title("ESS Hist for 1st 50 Coordinates")
plt.legend()
plt.savefig(os.path.join(output_dir, "coordwise_ess.png"))