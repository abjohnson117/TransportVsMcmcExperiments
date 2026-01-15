import numpy as np
from scipy.spatial.distance import cdist
from triangular_transport.kernels.kernel_tools import get_gaussianRBF, vectorize_kfunc
from jax import vmap, grad, jit, random
import jax.numpy as jnp
from functools import partial

def median_heuristic_sigma_jax(X, Y=None, max_points=5000, seed=0):
    X = jnp.asarray(X).reshape(X.shape[0], -1)
    if Y is not None:
        Y = jnp.asarray(Y).reshape(Y.shape[0], -1)
        Z = jnp.concatenate([X, Y], axis=0)
    else:
        Z = X

    n = Z.shape[0]
    if n > max_points:
        idx = random.choice(
            random.PRNGKey(seed), n, (max_points,), replace=False
        )
        Z = Z[idx]

    a2 = jnp.sum(Z * Z, axis=1, keepdims=True)
    D2 = a2 + a2.T - 2.0 * (Z @ Z.T)
    D2 = jnp.triu(D2, k=1)  # zero elsewhere
    D = jnp.sqrt(jnp.clip(D2[D2 > 0], a_min=0.0))
    sigma = jnp.median(D)
    return float(sigma)

def imq_kernel(X, Y=None, c=None, beta=0.5, compute_grad=False):
    """ Inverse multiquadric (IMQ) kernel matrix between points in X and Y."""
    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of dimensions.")
    # compute pairwise squared Euclidean distances
    pairwise_dists_sq = cdist(X, Y, 'sqeuclidean')
    # If bandwidth is not provided, use median heuristic
    if c is None:
        c = np.sqrt(np.median(pairwise_dists_sq))

    denom = c**2 + pairwise_dists_sq
    K = denom ** (-beta)

    if compute_grad:
        # Compute gradient of the kernel
        X_expanded = X[:, np.newaxis, :]  # (n, 1, d)
        Y_expanded = Y[np.newaxis, :, :]  # (1, m, d)
        diff = X_expanded - Y_expanded     # (n, m, d)

        K_grad = -beta * diff / denom[..., None] * K[..., None]  # (n, m, d)

        # Compute trace of Hessian (second derivative)
        dim = X.shape[1]
        term1 = 2 * dim / denom
        term2 = 4 * (beta + 1) * pairwise_dists_sq / denom**2
        trK_gradgrad = -beta * K * (term1 - term2)

    # # evaluate kernel
    # K = (c**2 + pairwise_dists_sq) ** (-beta)
    # if compute_grad is True:
    #     # Compute gradient of the kernel
    #     X_expanded = X[:, np.newaxis, :]  # (n, 1, d)
    #     Y_expanded = Y[np.newaxis, :, :]  # (1, m,
    #     diff = X_expanded - Y_expanded  # shape (n, m, d)
    #     K_grad = -beta * diff / (c**2 + pairwise_dists_sq) ** (beta + 1)
    #     K_grad *= K[:, :, None]  # shape (n, m, d)
    #     # compute second trace derivative
    #     trK_gradgrad = (2 * beta * X.shape[1] * K - np.sum(diff ** 2 * K[:, :, None], axis=2) / (c**2 + pairwise_dists_sq) ** (beta + 2)) / (c**2 + pairwise_dists_sq) ** (beta + 1)
        return K, K_grad, trK_gradgrad
    # Return only the kernel matrix
    else:
        return K

def rbf_kernel(X, Y=None, bandwidth=None, compute_grad=False):
    """ RBF (Gaussian) kernel matrix between X and Y points with optional median heuristic for bandwidth."""
    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of dimensions.")
    # compute pairwise squared Euclidean distances
    pairwise_dists = cdist(X, Y, metric='sqeuclidean')
    # If bandwidth is not provided, use median heuristic
    if bandwidth is None:
        bandwidth = np.sqrt(0.5 * np.median(pairwise_dists))
    # evaluate kernel    
    K = np.exp(-pairwise_dists / (2 * bandwidth ** 2))
    # print(f"These are the first 5 elements of pairwise_dists: {pairwise_dists[0, :5]}") #same
    if compute_grad is True:
        # Compute gradient of the kernel
        X_expanded = X[:, np.newaxis, :]  # (n, 1, d)
        Y_expanded = Y[np.newaxis, :, :]  # (1, m, d)
        diff = X_expanded - Y_expanded  # shape (n, m, d)
        K_grad = -diff / (bandwidth ** 2) * K[:, :, None]
        # print(f"this is the first 5 elements of K_grad: {K_grad[0, :5]}") #same
        # compute second trace derivative (Hessian) of the kernel
        trK_gradgrad = K * (pairwise_dists - X.shape[1] * bandwidth**2) / (bandwidth**4)  # (n, m)
        # print(f"this is trace of hessian: {trK_gradgrad}")
        return K, K_grad, trK_gradgrad
    # Return only the kernel matrix
    else:
        return K

def stein_kernel(X, score_func, bandwidth=None, kernel='rbf'):
    """ Stein kernel matrix using RBF kernel and score function. """
    if kernel == 'imq':
        K, K_grad, trK_gradgrad = imq_kernel(X, c=bandwidth, compute_grad=True)
    elif kernel == 'rbf':
        K, K_grad, trK_gradgrad = rbf_kernel(X, bandwidth=bandwidth, compute_grad=True)
    else:
        raise ValueError("Unsupported kernel type. Use 'imq' or 'rbf'.")
    # evaluate score function
    if callable(score_func):
        score_X = score_func(X)  # shape (n, d)
    else:
        raise ValueError("score_func must be a callable function that takes X as input.")
    # Score terms
    term1 = score_X @ score_X.T * K  # s(x)^T s(y) k(x,y)
    # print(f"This is term1: {term1[0, :5]}") #same
    term2 = np.einsum("ik,ijk->ij", score_X, K_grad)  # s(x)^T ∇_x k(x,y)
    term3 = np.einsum("jk,ijk->ij", score_X, -K_grad)  # s(y)^T ∇_y k(x,y)
    # Assemble the Stein kernel matrix
    H = term1 + term2 + term3 + trK_gradgrad
    return H

@partial(jit, static_argnames=("score_func", "kernel"))
def stein_kernel_jax(X, score_func, bandwidth, kernel="rbf"):
    if kernel == "rbf":
        k = get_gaussianRBF(bandwidth)
    kvec = vectorize_kfunc(k)
    K = kvec(X, X)
    # pairwise_dists = (-1 * jnp.log(K)) * (2 * bandwidth ** 2)
    # print(f"These are the first 5 elements of pairwise_dists: {pairwise_dists[0, :5]}") # same

    diff = X[:, None, :] - X[None, :, :]  # shape (n, m, d)
    sqdist = jnp.sum(diff ** 2, axis=-1)
    K_grad = -diff / (bandwidth ** 2) * K[:, :, None]
    # print(f"These are the first 5 elements of K_grad: {K_grad[0, :5]}") #same
    # compute second trace derivative (Hessian) of the kernel
    trK_grad2 = K * (sqdist - X.shape[1] * bandwidth**2) / (bandwidth**4)  # (n, m)
    # print(f"this is trace of hessian: {trK_grad2}") #same
    
    score_X = score_func(X)
    
    term1 = score_X @ score_X.T * K # s(x)^T s(y) k(x,y)
    # print(f"This is term1: {term1[0, :5]}")
    term2 = jnp.einsum("ik, ijk->ij", score_X, K_grad) # s(x)^T ∇_x k(x,y)
    term3 = jnp.einsum("jk, ijk->ij", score_X, -K_grad) # s(y)^T ∇_y k(x,y)

    H = term1 + term2 + term3 + trK_grad2
    return H

def compute_ksd(X, score_func, bandwidth=None, kernel='rbf'):
    """ Computes the Kernel Stein Discrepancy for samples X. """
    H = stein_kernel(X, score_func, bandwidth=bandwidth, kernel=kernel)
    # ksd = np.sqrt(np.sum(H) / (X.shape[0] ** 2))
    ksd = np.sum(H) / (X.shape[0] ** 2)
    return ksd

# def compute_ksd_jax(X, score_func, bandwidth=None, kernel="rbf"):
#     """Computes the Kernel Stein Discrepancy for samples X (target) using jax functionality"""
#     H = stein_kernel_jax(X, score_func, bandwidth, kernel=kernel)
#     ksd2 = np.sum(H) / (X.shape[0] ** 2)
#     return ksd2
def compute_ksd_jax(X, score_func, bandwidth=None, kernel="rbf"):
    """Computes the Kernel Stein Discrepancy for samples X (target) using jax functionality"""
    H = stein_kernel_jax(X, score_func, bandwidth, kernel=kernel)
    H = H - jnp.diag(jnp.diag(H))
    ksd2 = jnp.sum(H) / (X.shape[0] * (X.shape[0] - 1))
    # ksd2 = np.sum(H) / (X.shape[0] ** 2)
    return ksd2

if __name__ == "__main__":

    # define mean and covariance for Gaussian
    dim  = 4
    mean = np.zeros(dim)
    cov  = np.eye(dim)

    # Example: Target is normal
    def score_Gaussian(x, mean, cov):
        return -(np.linalg.inv(cov) @ (x-mean).T).T
    lambda_score_Gaussian = lambda x: score_Gaussian(x, mean, cov)

    for mean_shift in np.arange(0, 5):
        # Generate samples from approximate distribution with shifted mean
        X = np.random.multivariate_normal(mean=mean + mean_shift, cov=cov, size=1000)
        # test the RBF kernel function
        print(f"Mean shift {mean_shift} - KSD (RBF):", compute_ksd(X, score_func=lambda_score_Gaussian))
        # test the IMQ kernel function
        print(f"Mean shift {mean_shift} - KSD (IMQ):", compute_ksd(X, score_func=lambda_score_Gaussian, kernel='imq'))