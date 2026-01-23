import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np
from typing import List, Optional, Sequence

import wandb
from .lobpcg import torch_lobpcg, _maybe_orthonormalize
# from .hvp import make_param_block_hvp
from torch.func import functional_call

import time
import os
from scipy import stats

from scipy.sparse.linalg import LinearOperator, eigsh

from .gauss_newton import ggn_matvec
from .trajectory import (
    SparseJLProjector,
    load_projector,
    make_sparse_jl_projector,
    project_params,
    save_projection_vector,
    save_projector,
)


__all__ = ['param_vector', 'param_length', 'flatt', 'grads_vector', 
           'calculate_all_the_grads', 'compute_eigenvalues', 'compute_grad_H_grad', 
           'calculate_step_batch_lambdamax', 'calculate_averaged_lambdamax', 'create_ntk', 
           'compute_fisher_eigenvalues', 'calculate_all_net_grads',
           'calculate_averaged_grad_H_grad', 'calculate_averaged_grad_H_grad_step', 'calculate_gni',
           'calculate_accuracy', 'calculate_param_distance',
           'EigenvectorCache', 'HessianVectorProduct', 'create_hessian_vector_product', 'compute_multiple_eigenvalues_lobpcg',
           'calculate_gradient_norm_squared_mc', 'calculate_expected_one_step_full_loss_change',
           'calculate_expected_one_step_batch_loss_change', 'compute_gradient_projection_ratios',
           'estimate_hessian_trace', 'gimme_new_rng', 'gimme_random_subset_idx',
           'calculate_second_moment_contraction', 'compute_grad_gauss_newton_grad',
           'compute_gauss_newton_eigenvalues', 'calculate_momentum_batch_sharpness',
           'SparseJLProjector', 'make_sparse_jl_projector', 'project_params',
           'save_projector', 'load_projector', 'save_projection_vector']


class EigenvectorCache:
    """
    A cache for storing eigenvectors to enable warm starts in power iteration methods.
    Designed to be compatible with future LOBPCG implementations.
    """
    def __init__(self, max_eigenvectors=5):
        self.max_eigenvectors = max_eigenvectors
        self.eigenvectors = []   # List of eigenvectors for multi-eigenvalue computations
        self.eigenvalues = []    # Corresponding eigenvalues
        
    def store_eigenvector(self, eigenvector, eigenvalue=None):
        """Store a single eigenvector (and optionally eigenvalue)"""
        if eigenvalue is not None:
            self.eigenvalues = [eigenvalue]
        self.eigenvectors = [eigenvector]
    
    def store_eigenvectors(self, eigenvectors_list, eigenvalues_list=None):
        """Store multiple eigenvectors (for future LOBPCG compatibility)"""
        self.eigenvectors = [v.detach().clone() for v in eigenvectors_list]
        if eigenvalues_list is not None:
            self.eigenvalues = list(eigenvalues_list)
        
        # Trim to maximum size
        if len(self.eigenvectors) > self.max_eigenvectors:
            self.eigenvectors = self.eigenvectors[:self.max_eigenvectors]
            if self.eigenvalues:
                self.eigenvalues = self.eigenvalues[:self.max_eigenvectors]
    
    def get_warm_start_vectors(self, device=None):
        """Get eigenvectors for warm start, optionally moved to specified device"""
        if not self.eigenvectors:
            return None
        
        if device is not None:
            return [v.to(device) for v in self.eigenvectors]
        return self.eigenvectors
    
    def clear(self):
        """Clear all cached eigenvectors"""
        self.eigenvectors = []
        self.eigenvalues = []
    
    def __len__(self):
        return len(self.eigenvectors)
    
    def __contains__(self, key):
        # For backward compatibility with dict-like access
        return hasattr(self, key) and getattr(self, key) is not None



################################################################################
#                                                                              #
#                               HELPER FUNCTIONS                               #
#                                                                              #
################################################################################


def param_vector(net, clone=True):
    '''
    Returns a vector of all the parameters of the network
    If clone=True, returns a detached clone of the parameters
    '''
    # params = list(net.parameters())
    param_vector = T.cat([p.flatten() for p in net.parameters()])
    if clone:
        return param_vector.detach().clone()
    return param_vector

def param_length(net):
    '''
    Returns the number of parameters in the network
    '''
    params = list(net.parameters())
    return sum([p.numel() for p in params])

def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return T.cat([v.flatten() for v in vectors])


def grads_vector(net):  
    # pull out all the gradients from a network as one vector
    grads = []
    for p in net.parameters():
        grads.append(p.grad.flatten().detach().clone())
    return T.cat(grads)


def gimme_new_rng():
    """
    Create a new random number generator with a unique seed.
    """
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    return rng


def gimme_random_subset_idx(dataset_size, subset_size):
    """
    Get random indices for a subset of the dataset.

    Args:
        dataset_size (int): Total size of the dataset.
        subset_size (int): Desired size of the subset.

    Returns:
        Tensor: Random indices for the subset.
    """
    rng = gimme_new_rng()

    shuffle = T.randperm(dataset_size, generator=rng)
    random_idx = shuffle[:subset_size]
    return random_idx


def calculate_param_distance(net, reference_params, p=2):
    """
    Calculate the distance between current network parameters and reference parameters.
    
    Args:
        net (nn.Module): Neural network model
        reference_params (Tensor): Flattened reference parameters (from param_vector())
        p (int, optional): The norm degree. Default: 2 for Euclidean distance
    
    Returns:
        Tensor: The p-norm distance between current and reference parameters
    """
    with torch.no_grad():
        current_params = param_vector(net)
        return T.linalg.vector_norm(current_params - reference_params, ord=p)





def calculate_all_the_grads(net, X, Y, loss_fn, optimizer, storage_device=None):
    # device = net.parameters().__next__().device

    grads = [] # datapoint, parameter
    for x, y in zip(X, Y):
        optimizer.zero_grad()
        y_pred = net(x.unsqueeze(0)).squeeze(dim=-1)
        loss = loss_fn(y_pred, y.unsqueeze(0))
        loss.backward()
        detached_grads = grads_vector(net).detach()
        if storage_device:
            detached_grads = detached_grads.to(storage_device)
        grads.append(detached_grads)
    
    return T.stack(grads)


def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy given the model predictions and target labels.
    
    Args:
        predictions: tensor of shape (num_samples, num_classes) with model predictions
        targets: tensor of shape (num_samples, num_classes) with one-hot encoded labels
                or tensor of shape (num_samples,) with class indices
    
    Returns:
        accuracy: float representing the accuracy (0.0 to 1.0)
    """
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get the predicted class (highest value in each row)
        # this is if we have all the classes
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        # Get the predicted class (sign of the prediction)
        # this is if we have only two classes
        pred_classes = torch.sign(predictions).long()

    
    
    # Check if targets are one-hot encoded or class indices
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # One-hot encoded targets
        true_classes = torch.argmax(targets, dim=1)
    else:
        # Class indices (1D tensor)
        if len(targets.shape) == 1:
            true_classes = torch.round(targets).long()
        else:
            true_classes = targets.long()
    
    # Compare and compute accuracy
    correct = (pred_classes == true_classes).sum().item()
    total = targets.size(0)
    
    return correct / total


def jvp(net, X, Y, loss_fn, vector):
    """
    Computes the Jacobian-vector product (JVP) of the loss with respect to the network parameters.
    
    Args:
        net (nn.Module): The neural network model
        X (Tensor): Input data
        Y (Tensor): Target labels
        loss_fn (callable): Loss function to compute the loss
    
    Returns:
        Tensor: The JVP of the loss with respect to the network parameters
    """
    params = list(net.parameters())
    y_pred = net(X).squeeze(dim=-1)
    loss = loss_fn(y_pred, Y, sampling_vector=vector)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten gradients into a single vector
    grads_vector = flatt(grads).detach()
    return grads_vector


################################################################################
#                                                                              #
#                             HESSIAN-VECTOR PRODUCT                           #
#                                                                              #
################################################################################


class HessianVectorProduct:
    """Callable Hessian-vector product with explicit lifecycle management."""

    def __init__(self,
                 loss,
                 net,
                 params: Optional[Sequence[torch.Tensor]] = None,
                 grads: Optional[Sequence[torch.Tensor]] = None,
                 flat_grads: Optional[torch.Tensor] = None,
                 retain_graph: bool = True):
        if params is None:
            params = list(net.parameters())
        else:
            params = list(params)

        if len(params) == 0:
            raise ValueError("create_hessian_vector_product requires at least one parameter.")

        if grads is None and flat_grads is None:
            grads = torch.autograd.grad(loss, params, create_graph=True)

        if flat_grads is None:
            grads_vector = flatt(grads)
        else:
            grads_vector = flat_grads

        self._params = params
        self._grads = grads
        self._loss_ref = loss  # keep graph alive
        self._retain_graph_default = retain_graph

        grads_vector = grads_vector.view(-1)
        self._grads_vector = grads_vector
        self._device = grads_vector.device
        self._dtype = grads_vector.dtype
        self._numel = grads_vector.numel()
        self._freed = False

    def _ensure_active(self):
        if self._freed:
            raise RuntimeError("HessianVectorProduct.free_memory() has already been called.")

    def _prepare_vec(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() != self._numel:
            raise ValueError("Vector shape mismatch for Hessian-vector product.")
        if vec.device != self._device or vec.dtype != self._dtype:
            vec = vec.to(device=self._device, dtype=self._dtype)
        return vec

    def _apply_single_vector(self, vec: torch.Tensor, retain_flag: bool) -> torch.Tensor:
        self._ensure_active()
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D vector for Hessian application, got {vec.ndim}D tensor.")
        vec = self._prepare_vec(vec)
        grad_v = torch.dot(self._grads_vector, vec)
        Hv = torch.autograd.grad(grad_v, self._params, retain_graph=retain_flag)
        return flatt(Hv)

    def __call__(self, v: torch.Tensor, retain_graph_override: Optional[bool] = None) -> torch.Tensor:
        retain_flag = self._retain_graph_default if retain_graph_override is None else retain_graph_override
        if v.dim() == 1:
            return self._apply_single_vector(v, retain_flag)
        if v.dim() == 2:
            results = []
            num_vecs = v.shape[1]
            for i in range(num_vecs):
                vi = v[:, i]
                needs_retain = True if retain_flag else (i < num_vecs - 1)
                results.append(self._apply_single_vector(vi, needs_retain))
            return torch.stack(results, dim=1)
        raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")

    def free_memory(self):
        if self._freed:
            return
        self._params = None
        self._grads = None
        self._grads_vector = None
        self._loss_ref = None
        self._freed = True


def create_hessian_vector_product(loss,
                                  net,
                                  params: Optional[Sequence[torch.Tensor]] = None,
                                  grads: Optional[Sequence[torch.Tensor]] = None,
                                  flat_grads: Optional[torch.Tensor] = None,
                                  retain_graph: bool = True):
    """
    Create a Hessian-vector product helper for use with LOBPCG and related routines.

    Returns:
        HessianVectorProduct: Callable that also exposes `free_memory()` for manual teardown.
    """
    return HessianVectorProduct(
        loss,
        net,
        params=params,
        grads=grads,
        flat_grads=flat_grads,
        retain_graph=retain_graph,
    )


################################################################################
#                                                                              #
#                             EIGENVALUE FUNCTIONS                             #
#                                                                              #
################################################################################


def compute_eigenvalues(loss, 
                        net, 
                        k=1, 
                        max_iterations=100, 
                        reltol=1e-2,
                        init_vectors=None,
                        batched=None,
                        eigenvector_cache=None,
                        return_eigenvectors: bool = False,
                        use_power_iteration: bool = False,
                        use_lanczos: bool = False,
                        # lanczos_tolerance: float = 1e-6,
                        # lanczos_max_iter: int = 200
                        ):
    """
    Computes the top-k eigenvalues of the Hessian of the loss function at the current point.
    
    Uses LOBPCG by default for better performance, with power iteration as fallback for k=1.

    Args:
        loss (Tensor): The loss value at the current point
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
        reltol (float, optional): relative tolerance threshold for eigenvalue computation. Defaults to 1e-2.
        init_vectors (Tensor, optional): Initial vectors. For k=1, can be 1D vector. For k>1, should be [n_params, k]. 
                                        If None, uses cached or random vectors. Defaults to None.
        batched (Any, optional): Unused parameter. Defaults to None.
        eigenvector_cache (EigenvectorCache, optional): Cache to store/retrieve eigenvectors for warm starts. Defaults to None.
        return_eigenvectors (bool, optional): Whether to return the final eigenvectors. Defaults to False.
        use_power_iteration (bool, optional): If True, force use of power iteration (only works for k=1). Defaults to False.
        use_lanczos (bool, optional): If True, run SciPy Lanczos (`eigsh`) instead of torch LOBPCG. Defaults to False.
        lanczos_tolerance (float, optional): Relative tolerance fed to SciPy `eigsh`. Defaults to 1e-6.
        lanczos_max_iter (int, optional): Max Lanczos iterations. Defaults to 200.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If k=1 and return_eigenvectors=False: Returns single eigenvalue (scalar Tensor)
            - If k=1 and return_eigenvectors=True: Returns (eigenvalue, eigenvector)
            - If k>1 and return_eigenvectors=False: Returns eigenvalues tensor of shape [k]
            - If k>1 and return_eigenvectors=True: Returns (eigenvalues, eigenvectors) where 
              eigenvalues has shape [k] and eigenvectors has shape [n_params, k]

    Note:
        By default, uses LOBPCG for eigenvalue computation for better performance.
        Falls back to power iteration if use_power_iteration=True (only supported for k=1).
        
        If eigenvector_cache is provided, the function will try to reuse previous eigenvectors
        for warm starts and store the final eigenvector(s) for future use.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1. Use LOBPCG (default) for k>1.")
    
    if use_lanczos and use_power_iteration:
        raise ValueError("Cannot use both Lanczos and power iteration simultaneously.")
    
    device = next(net.parameters()).device


    # Choose method: use LOBPCG by default unless explicitly requested to use power iteration
    if use_power_iteration and k == 1:
        # Use the existing power iteration implementation
        return compute_lambdamax_power_iteration(
            loss, net, max_iterations, reltol, init_vectors,
            eigenvector_cache, return_eigenvectors
        )

    if use_lanczos:
        if use_power_iteration:
            raise ValueError("Lanczos path does not support power iteration mode.")
        n_params = param_length(net)
        if n_params == 0:
            raise ValueError("Model must have parameters to compute eigenvalues.")
        if k >= n_params:
            raise ValueError(f"Lanczos requires k < number of params; got k={k}, n_params={n_params}.")

        hvp = create_hessian_vector_product(loss, net)
        param_dtype = next(net.parameters()).dtype
        np_dtype = np.float64 if param_dtype == torch.float64 else np.float32

        def _apply_operator(vec_torch: torch.Tensor) -> torch.Tensor:
            if vec_torch.ndim != 1 or vec_torch.numel() != n_params:
                raise ValueError("Vector shape mismatch for Lanczos matvec.")
            # SciPy works with NumPy arrays, so hop to torch for autograd, then back.
            return hvp(vec_torch).detach()

        def _matvec(vec_np: np.ndarray) -> np.ndarray:
            vec_torch = torch.from_numpy(vec_np).to(device=device, dtype=param_dtype)
            operator_vec = _apply_operator(vec_torch)
            return operator_vec.cpu().numpy().astype(np_dtype, copy=False)

        if n_params == 1:
            basis = torch.ones(n_params, device=device, dtype=param_dtype)
            eigvals_torch = _apply_operator(basis)
            hvp.free_memory()
            if return_eigenvectors:
                if k != 1:
                    raise ValueError("k must be 1 when the parameter space is one-dimensional.")
                return eigvals_torch[0], basis
            return eigvals_torch[0]

        linear_op = LinearOperator(
            shape=(n_params, n_params),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=np_dtype,
        )
        eigvals_np, eigvecs_np = eigsh(
            linear_op,
            k=k,
            which="LM",
            # tol=lanczos_tolerance,
            # maxiter=lanczos_max_iter,
        )
        hvp.free_memory()

        eigvals = torch.from_numpy(eigvals_np.real.astype(np_dtype, copy=False)).to(device=device, dtype=param_dtype)
        eigvecs = torch.from_numpy(eigvecs_np.astype(np_dtype, copy=False)).to(device=device, dtype=param_dtype)

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if k == 1:
            if return_eigenvectors:
                return eigvals[0], eigvecs[:, 0]
            return eigvals[0]
        if return_eigenvectors:
            return eigvals, eigvecs
        return eigvals
    


    else:
        # Use LOBPCG method (default)
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, net, k, max_iterations, reltol, init_vectors, 
            eigenvector_cache, return_eigenvectors=True
        )
        
        if k == 1:
            # For backward compatibility with single eigenvalue case
            eigenvalue = eigenvalues[0]
            if return_eigenvectors:
                return eigenvalue, eigenvectors[:, 0]
            else:
                return eigenvalue
        else:
            # Multiple eigenvalues case
            if return_eigenvectors:
                return eigenvalues, eigenvectors
            else:
                return eigenvalues
    
    





def _run_lobpcg_with_operator(
    operator,
    net,
    k: int,
    max_iterations: int,
    reltol: float,
    init_vectors: Optional[torch.Tensor],
    eigenvector_cache: Optional[EigenvectorCache],
    return_eigenvectors: bool,
):
    param_example = next(net.parameters(), None)
    if param_example is None:
        raise ValueError("Model must have parameters to compute eigenvalues.")
    device = param_example.device
    dtype = param_example.dtype
    n_params = param_length(net)

    if init_vectors is not None:
        X = init_vectors
        if X.shape[1] != k or X.shape[0] != n_params:
            raise ValueError(f"init_vectors must have shape [{n_params}, {k}], got {X.shape}")
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        cached_vectors = eigenvector_cache.get_warm_start_vectors(device)
        if cached_vectors:
            n_cached = min(len(cached_vectors), k)
            chosen = [cached_vectors[i] for i in range(n_cached)]
            if n_cached < k:
                padding = torch.randn(n_params, k - n_cached, device=device, dtype=dtype)
                chosen.extend([padding[:, j] for j in range(padding.shape[1])])
            X = torch.stack(chosen, dim=1)
        else:
            X = torch.randn(n_params, k, device=device, dtype=dtype)
    else:
        X = torch.randn(n_params, k, device=device, dtype=dtype)

    X = X.to(device=device, dtype=dtype).reshape(n_params, k)
    tol = reltol / (20 * max(n_params, 1))

    eigenvalues = None
    eigenvectors = None
    iterations = None
    try:
        eigenvalues, eigenvectors, iterations = torch_lobpcg(
            operator, X, max_iter=max_iterations, tol=tol
        )
    finally:
        pass

    try:
        wandb.log({"lobpcg_iterations": iterations}, commit=False)
    except Exception:
        pass

    if eigenvector_cache is not None:
        eigenvector_list = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        eigenvector_cache.store_eigenvectors(eigenvector_list, eigenvalues.tolist())

    if return_eigenvectors:
        return eigenvalues, eigenvectors
    return eigenvalues


def compute_multiple_eigenvalues_lobpcg(loss, net, k=5, max_iterations=100, reltol=1e-2,
                                       init_vectors=None, eigenvector_cache=None,
                                       return_eigenvectors=False):
    """
    Compute multiple eigenvalues of the Hessian using LOBPCG algorithm.
    
    This function computes the top-k eigenvalues of the Hessian matrix using the
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) algorithm.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 5.
        max_iterations (int, optional): Maximum number of LOBPCG iterations. Defaults to 100.
        reltol (float, optional): Relative tolerance for LOBPCG convergence. Defaults to 2% relative tolerance.

        init_vectors (Tensor, optional): Initial vectors for LOBPCG (shape: [n_params, k]). 
                                       If None, uses random or cached vectors.
        eigenvector_cache (EigenvectorCache, optional): Cache for storing/retrieving eigenvectors.
        return_eigenvectors (bool, optional): Whether to return eigenvectors along with eigenvalues.
        
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If return_eigenvectors is False: Returns eigenvalues tensor of shape [k]
            - If return_eigenvectors is True: Returns tuple of (eigenvalues, eigenvectors)
              where eigenvectors has shape [n_params, k]
    
    Note:
        The eigenvalues are returned in descending order (largest first).
        The function automatically handles the case where k is too large relative to the problem size.
    """
    hessian_matvec = create_hessian_vector_product(loss, net)
    try:
        return _run_lobpcg_with_operator(
            hessian_matvec,
            net,
            k,
            max_iterations,
            reltol,
            init_vectors,
            eigenvector_cache,
            return_eigenvectors,
        )
    finally:
        hessian_matvec.free_memory()



def compute_lambdamax_power_iteration(loss, net, max_iterations, reltol, init_vector,
                                       eigenvector_cache, return_eigenvector):
    """Power iteration implementation of the maximum eigenvalue of the Hessian."""
    device = next(net.parameters()).device

    # compute gradient and keep it for repeated Hessian-vector products
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    hessian_vector_product = create_hessian_vector_product(
        loss,
        net,
        params=params,
        grads=grads,
        retain_graph=True,
    )

    try:
        size = param_length(net)
        
        # Initialize vector with priority: init_vector > cached eigenvector > gradient
        if init_vector is not None:
            v = init_vector
        elif eigenvector_cache is not None:
            # Support both EigenvectorCache objects and dict-style caches
            if isinstance(eigenvector_cache, EigenvectorCache):
                if len(eigenvector_cache) > 0:
                    cached_v = eigenvector_cache.eigenvector
                    if cached_v.device != device:
                        cached_v = cached_v.to(device)
                    v = cached_v.detach()
                else:
                    v = T.randn(size, device=device)
            elif isinstance(eigenvector_cache, dict) and 'eigenvector' in eigenvector_cache:
                # Backward compatibility with dict-style cache
                cached_v = eigenvector_cache['eigenvector']
                if cached_v.device != device:
                    cached_v = cached_v.to(device)
                v = cached_v.detach()
            else:
                v = T.randn(size, device=device)
        else:
            v = T.randn(size, device=device)
        
        with torch.no_grad():
            v = v / T.linalg.norm(v)

        v = v.detach()
        eigenval = 0.0  # Initialize eigenval to avoid undefined variable error
        for i in range(max_iterations):
            Hv = hessian_vector_product(v).detach()

            v = v.detach()
            with T.no_grad():
                rayleigh_quotient = T.dot(Hv, v) / T.dot(v, v)
                eigenval = rayleigh_quotient  # Update eigenval every iteration
                if T.abs(rayleigh_quotient) < 1e-12:
                    break

                residual = Hv - rayleigh_quotient * v
                resid_norm = T.linalg.norm(residual)
                if resid_norm / T.abs(rayleigh_quotient) < reltol:
                    break
                
                v = Hv / T.linalg.norm(Hv)  # Normalize for next iteration

        # Log the number of iterations to wandb
        try:
            wandb.log({"power_iteration_iterations": i + 1}, commit=False)
        except:
            pass

        # Store the final eigenvector in cache for future warm starts
        if eigenvector_cache is not None:
            if isinstance(eigenvector_cache, EigenvectorCache):
                eigenvector_cache.store_eigenvector(v, eigenval)
            else:
                raise ValueError("eigenvector_cache must be an instance of EigenvectorCache")

        results = [eigenval]
        if return_eigenvector:
            results.append(v.detach())
    finally:
        hessian_vector_product.free_memory()

    if len(results) == 1:
        return results[0]
    return tuple(results)


################################################################################
#                                                                              #
#                         GRAD-H-GRAD (BATCH SHARPNESS)                        #
#                                                                              #
################################################################################


def compute_grad_H_grad(loss, net, grad_already_there: bool = False,
                        return_ghg_gg_separately: bool = False):
    """
    Computes g^T H g / ||g||², the Rayleigh quotient of the Hessian H and gradient g.
    
    This function calculates gradient * Hessian * gradient normalized by the squared gradient norm,
    which represents the curvature of the loss in the gradient direction. If taken on a batch, this is 
    step sharpness. Averaging over many batches gives batch sharpness.
    
    Args:
        loss (Tensor): Loss value (must retain computational graph for Hessian computation)
        net (nn.Module): Neural network model
        grad_already_there (bool, optional): Use existing gradients instead of computing new ones. Defaults to False.
        return_ghg_gg_separately (bool, optional): Return (g^T H g, g^T g) separately instead of ratio. Defaults to False.
    
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: Rayleigh quotient g^T H g / ||g||² or separate components if requested
    """
    
    device = next(net.parameters()).device

    params = list(net.parameters())
    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=True)
    else:
        grads = [p.grad for p in params]
        if any(g is None or not g.requires_grad for g in grads):
            grads = torch.autograd.grad(loss, params, create_graph=True)
    
    grads_vector = flatt(grads)
    step_vector = grads_vector.detach()

    hvp = create_hessian_vector_product(
        loss,
        net,
        params=params,
        grads=grads,
        flat_grads=grads_vector,
    )
    try:
        Hv = hvp(step_vector, retain_graph_override=False).detach()
    finally:
        hvp.free_memory()

    if return_ghg_gg_separately:
        return T.dot(step_vector, Hv), T.dot(step_vector, step_vector)
    return T.dot(step_vector, Hv) / T.dot(step_vector, step_vector)


def _flatten_momentum_buffers(optimizer, params, device, dtype):
    """
    Collect momentum buffers for params into a flat vector. Missing buffers default to zero.
    """
    buffers = []
    for p in params:
        state = optimizer.state.get(p, {})
        buf = state.get('momentum_buffer', None)
        if buf is None:
            buf = torch.zeros_like(p, device=device, dtype=dtype)
        buffers.append(buf.reshape(-1))
    return torch.cat(buffers).to(device=device, dtype=dtype)


def compute_momentum_step_sharpness(loss, net, optimizer, momentum_coeff: float,
                                    return_components: bool = False):
    """
    Compute s^T H s / (s^T g) on a single batch, where s = μ v + g for SGD+momentum.

    Returns:
        Tensor: The Rayleigh-like quotient, or (numerator, denominator) if return_components.
    """
    if momentum_coeff <= 0:
        raise ValueError("Momentum step sharpness requires positive momentum.")

    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)
    device = grads_vector.device
    dtype = grads_vector.dtype

    velocity_vector = _flatten_momentum_buffers(optimizer, params, device=device, dtype=dtype)
    step_vector = grads_vector.detach() + momentum_coeff * velocity_vector

    hvp = create_hessian_vector_product(
        loss,
        net,
        params=params,
        grads=grads,
        flat_grads=grads_vector,
    )
    try:
        Hv = hvp(step_vector, retain_graph_override=False).detach()
    finally:
        hvp.free_memory()

    numerator = torch.dot(step_vector, Hv)
    denominator = torch.dot(step_vector, grads_vector.detach())

    if return_components:
        return numerator, denominator

    if denominator.abs().item() < 1e-12:
        return torch.tensor(float('nan'), device=device, dtype=dtype)

    return numerator / denominator


def calculate_averaged_grad_H_grad(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              eps = 0.005, # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96,
                              expectation_inside = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95,
                              use_gauss_newton: bool = False,
                              gauss_newton_loss_type: Optional[str] = None,
                              ): 
    """
    Computes E[g_b H_b g_b / ||g_b||²], which represents batch sharpness, aka the Rayleigh quotient of the 
    batch Hessian and batch gradient.
    The function uses Monte Carlo sampling with adaptive stopping based on relative standard 
    error to efficiently estimate the expectation.
    Args:
        net: Neural network model whose parameters will be used for gradient computation
        X: Input data tensor
        Y: Target labels tensor  
        loss_fn: Loss function to compute gradients from
        batch_size (int): Size of random batches to sample for each estimate
        n_estimates (int, optional): Maximum number of Monte Carlo estimates. Defaults to 500.
        min_estimates (int, optional): Minimum estimates before checking stopping criterion. Defaults to 10.
        eps (float, optional): Relative standard error threshold for early stopping. Defaults to 0.005.
        expectation_inside (bool, optional): If True, computes E[gHg]/E[g²] instead, mostly used for exploratory purposes. Defaults to False.
        with_replacement (bool, optional): Sample batches with replacement. Defaults to False.
        return_confidence_interval (bool, optional): If True, include a confidence interval and related statistics in the return value. Defaults to False.
        confidence_level (float, optional): Confidence level for the interval when `return_confidence_interval` is True. Defaults to 0.95.
        use_gauss_newton (bool, optional): If True, use the Gauss-Newton matrix instead of the exact Hessian.
        gauss_newton_loss_type (str, optional): When using Gauss-Newton, override the inferred loss type ('ce' or 'mse').
    Returns:
        float or dict: The averaged gradient-Hessian-gradient ratio representing batch sharpness. When
            `return_confidence_interval` is True, returns a dictionary with the estimate, confidence interval,
            standard error, confidence level, and number of Monte Carlo samples used.
    Notes:
        - Uses independent random number generator for true randomness (since it is fixed in the main training loop)
        - Implements adaptive stopping based on relative standard error convergence  
        - Logs the number of estimates to wandb if available
        - eps=0.005 approximately gives 1% estimation error
    """
    if use_gauss_newton:
        gn_loss_type = _infer_gauss_newton_loss_type(loss_fn, gauss_newton_loss_type)
    else:
        gn_loss_type = None

    gHg_vals = []
    norm_g_vals = []

    x_vals = gHg_vals
    y_vals = norm_g_vals
    

    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        if with_replacement:
            random_idx = T.randint(0, len(X), (batch_size,), generator=rng)
            
        if batch_size > 128:
            torch.cuda.empty_cache()
         
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        if use_gauss_newton:
            gHg, norm_g = compute_grad_gauss_newton_grad(
                loss,
                net,
                X_batch,
                Y_batch,
                loss_type=gn_loss_type,
                return_gcg_gg_separately=True,
            )
        else:
            gHg, norm_g = compute_grad_H_grad(loss, net, return_ghg_gg_separately=True)
        gHg = gHg.item()
        norm_g = norm_g.item()
        
        
        gHg_vals.append(gHg)
        norm_g_vals.append(norm_g)

        if i < min_estimates:
            continue    

        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x,  var_y  = np.var(x_vals, ddof=1), np.var(y_vals, ddof=1)
        cov_xy = np.cov(x_vals, y_vals, ddof=1)[0, 1]

        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / i

        rse = np.sqrt(var_R) / abs(R)  # relative standard error

        if rse < eps:                    # stopping rule
            break


    num_samples = len(gHg_vals)

    try:
        wandb.log({"number_of_gHg_estimates": num_samples}, commit=False)
    except:
        pass


    if num_samples == 0:
        raise RuntimeError("calculate_averaged_grad_H_grad received no samples; check dataset and parameters.")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1 - confidence_level

    if expectation_inside:
        mean_x = float(np.mean(gHg_vals))
        mean_y = float(np.mean(norm_g_vals))
        if mean_y == 0.0:
            raise ZeroDivisionError("Mean squared gradient is zero; cannot compute batch sharpness.")

        result = mean_x / mean_y

        if not return_confidence_interval:
            return result

        if num_samples < 2:
            stderr = 0.0
            ci = (result, result)
        else:
            var_x = float(np.var(gHg_vals, ddof=1))
            var_y = float(np.var(norm_g_vals, ddof=1))
            cov_xy = float(np.cov(gHg_vals, norm_g_vals, ddof=1)[0, 1])
            var_R = (
                var_x / (mean_y ** 2)
                - 2 * cov_xy * mean_x / (mean_y ** 3)
                + var_y * (mean_x ** 2) / (mean_y ** 4)
            ) / num_samples
            var_R = max(var_R, 0.0)
            stderr = float(np.sqrt(var_R))
            t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1) if num_samples > 1 else 0.0
            if not np.isfinite(t_multiplier):
                t_multiplier = 0.0
            half_width = float(t_multiplier * stderr)
            ci = (result - half_width, result + half_width)

        return {
            "mean": result,
            "ci": ci,
            "stderr": stderr,
            "confidence_level": confidence_level,
            "num_samples": num_samples,
        }

    gHg_normalized = np.array(gHg_vals) / np.array(norm_g_vals)
    result = float(np.mean(gHg_normalized))

    if not return_confidence_interval:
        return result

    if num_samples < 2:
        stderr = 0.0
        ci = (result, result)
    else:
        std = float(np.std(gHg_normalized, ddof=1))
        stderr = float(std / np.sqrt(num_samples))
        t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1)
        if not np.isfinite(t_multiplier):
            t_multiplier = 0.0
        half_width = float(t_multiplier * stderr)
        ci = (result - half_width, result + half_width)

    return {
        "mean": result,
        "ci": ci,
        "stderr": stderr,
        "confidence_level": confidence_level,
        "num_samples": num_samples,
    }


def calculate_averaged_grad_H_grad_step(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 1000,
                              min_estimates = 10,
                              eps = 0.005,
                              log_the_expectation_outside = False,
                              return_ghg_gg_separately = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95,
                              use_gauss_newton: bool = False,
                              gauss_newton_loss_type: Optional[str] = None,
                              ):
    """Backward-compatible wrapper for the batch sharpness estimator E[gHg/g²]."""
    if return_ghg_gg_separately:
        raise NotImplementedError("Returning gHg and g² separately is not supported in this refactor.")

    result = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=batch_size,
        n_estimates=n_estimates,
        min_estimates=min_estimates,
        eps=eps,
        expectation_inside=False,
        with_replacement=with_replacement,
        return_confidence_interval=return_confidence_interval,
        confidence_level=confidence_level,
        use_gauss_newton=use_gauss_newton,
        gauss_newton_loss_type=gauss_newton_loss_type,
    )

    return result


def calculate_momentum_batch_sharpness(
    net,
    optimizer,
    X,
    Y,
    loss_fn,
    batch_size,
    n_estimates: int = 500,
    min_estimates: int = 10,
    eps: float = 0.005,
):
    """
    Estimate E[s^T H s / (s^T g)] over batches where s = μ v + g for SGD+momentum.

    Uses Monte Carlo sampling with an adaptive stop based on relative standard error
    of the sample mean.
    """
    momentum_coeff = optimizer.param_groups[0].get('momentum', 0.0) if optimizer.param_groups else 0.0
    if momentum_coeff is None or momentum_coeff <= 0:
        raise ValueError("Momentum batch sharpness requires SGD with momentum > 0.")

    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    ratios = []

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        if batch_size > 128:
            torch.cuda.empty_cache()

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        numerator, denominator = compute_momentum_step_sharpness(
            loss,
            net,
            optimizer,
            momentum_coeff=momentum_coeff,
            return_components=True,
        )

        if denominator.abs().item() < 1e-12:
            continue

        ratio = (numerator / denominator).detach().item()
        if not np.isfinite(ratio):
            continue

        ratios.append(ratio)

        if len(ratios) < min_estimates:
            continue

        mean_val = float(np.mean(ratios))
        std_val = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
        stderr = std_val / max(np.sqrt(len(ratios)), 1.0)
        rse = stderr / (abs(mean_val) + 1e-12)

        if rse < eps:
            break

    if len(ratios) == 0:
        raise RuntimeError("Momentum batch sharpness could not be estimated (no valid samples).")

    try:
        wandb.log({"number_of_momentum_bs_estimates": len(ratios)}, commit=False)
    except Exception:
        pass

    return float(np.mean(ratios))


################################################################################
#                                                                              #
#                              GAUSS-NEWTON OPS                                #
#                                                                              #
################################################################################


_GAUSS_NEWTON_LOSS_ALIASES = {
    'ce': 'ce',
    'cross_entropy': 'ce',
    'crossentropyloss': 'ce',
    'categorical_crossentropy': 'ce',
    'mse': 'mse',
    'mean_squared_error': 'mse',
    'squared': 'mse',
}


def _normalize_gauss_newton_loss_type(loss_type: str) -> str:
    if not isinstance(loss_type, str):
        raise TypeError("Gauss-Newton loss type override must be a string.")
    key = loss_type.replace('-', '_').lower()
    normalized = _GAUSS_NEWTON_LOSS_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported Gauss-Newton loss type '{loss_type}'. "
            "Supported values: 'ce', 'cross_entropy', 'mse'."
        )
    return normalized


def _infer_gauss_newton_loss_type(loss_fn, override: Optional[str] = None) -> str:
    if override is not None:
        return _normalize_gauss_newton_loss_type(override)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return 'ce'
    if loss_fn.__class__.__name__ == 'SquaredLoss' or isinstance(loss_fn, nn.MSELoss):
        return 'mse'

    raise ValueError(
        "Unable to infer Gauss-Newton loss type from the provided loss_fn. "
        "Pass gauss_newton_loss_type explicitly ('ce' or 'mse')."
    )


def compute_grad_gauss_newton_grad(
    loss,
    net,
    X_batch,
    Y_batch,
    *,
    loss_type: str,
    grad_already_there: bool = False,
    return_gcg_gg_separately: bool = False,
):
    """
    Computes g^T G g / ||g||² where G is the Gauss-Newton matrix.

    Args:
        loss (Tensor): Scalar loss for the current batch.
        net (nn.Module): Model under evaluation.
        X_batch (Tensor): Mini-batch inputs used to form the Gauss-Newton operator.
        Y_batch (Tensor): Mini-batch targets (kept for completeness).
        loss_type (str): Either 'ce' or 'mse' describing the logit Hessian used in GGN.
        grad_already_there (bool): Reuse cached gradients if available.
        return_gcg_gg_separately (bool): If True, return (g^T G g, g^T g) instead of the ratio.

    Returns:
        Tensor or Tuple[Tensor, Tensor]: Rayleigh quotient or its components.
    """
    normalized_loss_type = _normalize_gauss_newton_loss_type(loss_type)

    params = list(net.parameters())
    if not params:
        raise ValueError("compute_grad_gauss_newton_grad requires a model with parameters.")

    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=False)
    else:
        grads = [p.grad for p in params]
        if any(g is None for g in grads):
            grads = torch.autograd.grad(loss, params, create_graph=False)

    grads_vector = flatt(grads).detach()
    if grads_vector.numel() == 0:
        raise ValueError("Gradient vector is empty; ensure the model has trainable parameters.")

    Gg = ggn_matvec(
        model=net,
        x=X_batch,
        y=Y_batch,
        v_flat=grads_vector,
        loss=normalized_loss_type,
        average_over_batch=True,
    ).detach()

    numerator = torch.dot(grads_vector, Gg)
    denominator = torch.dot(grads_vector, grads_vector)
    if return_gcg_gg_separately:
        return numerator, denominator
    return numerator / denominator


class GaussNewtonVectorProduct:
    """
    Callable Gauss-Newton vector product that mimics the HessianVectorProduct interface.
    """

    def __init__(
        self,
        net: nn.Module,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        loss_type: str,
        *,
        average_over_batch: bool = True,
    ):
        param_example = next(net.parameters(), None)
        if param_example is None:
            raise ValueError("Gauss-Newton vector product requires a model with parameters.")

        self._net = net
        self._loss_type = _normalize_gauss_newton_loss_type(loss_type)
        self._average_over_batch = average_over_batch

        device = param_example.device
        dtype = param_example.dtype
        self._X_batch = X_batch.to(device=device, dtype=dtype)
        self._Y_batch = Y_batch.to(device=device)

        self._numel = param_length(net)
        self._device = device
        self._dtype = dtype
        self._freed = False

    def _ensure_active(self):
        if self._freed:
            raise RuntimeError("GaussNewtonVectorProduct.free_memory() has already been called.")

    def _prepare_vec(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() != self._numel:
            raise ValueError(
                f"Vector shape mismatch for Gauss-Newton product: expected {self._numel}, got {vec.numel()}."
            )
        if vec.device != self._device or vec.dtype != self._dtype:
            vec = vec.to(device=self._device, dtype=self._dtype)
        return vec

    def _apply_single_vector(self, vec: torch.Tensor) -> torch.Tensor:
        self._ensure_active()
        vec = self._prepare_vec(vec)
        return ggn_matvec(
            model=self._net,
            x=self._X_batch,
            y=self._Y_batch,
            v_flat=vec,
            loss=self._loss_type,
            average_over_batch=self._average_over_batch,
        ).detach()

    def __call__(self, v: torch.Tensor, retain_graph_override: Optional[bool] = None) -> torch.Tensor:
        if v.dim() == 1:
            return self._apply_single_vector(v)
        if v.dim() == 2:
            cols = []
            for i in range(v.shape[1]):
                cols.append(self._apply_single_vector(v[:, i]))
            return torch.stack(cols, dim=1)
        raise ValueError(f"Gauss-Newton operator expects 1D or 2D tensor, got {v.dim()}D.")

    def free_memory(self):
        if self._freed:
            return
        self._X_batch = None
        self._Y_batch = None
        self._net = None
        self._freed = True


def _compute_gauss_newton_power_iteration(
    operator: GaussNewtonVectorProduct,
    net: nn.Module,
    max_iterations: int,
    reltol: float,
    init_vector: Optional[torch.Tensor],
    eigenvector_cache: Optional[EigenvectorCache],
    return_eigenvector: bool,
):
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    size = param_length(net)

    if init_vector is not None:
        v = init_vector.to(device=device, dtype=dtype)
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        cached = eigenvector_cache.get_warm_start_vectors(device)
        if cached:
            v = cached[0].detach()
        else:
            v = torch.randn(size, device=device, dtype=dtype)
    else:
        v = torch.randn(size, device=device, dtype=dtype)

    with torch.no_grad():
        if torch.linalg.norm(v) == 0:
            raise ValueError("Initialization vector must be non-zero.")
        v = v / torch.linalg.norm(v)

    v = v.detach()
    eigenval = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(max_iterations):
        Gv = operator(v).detach()
        v = v.detach()
        with torch.no_grad():
            denom = torch.dot(v, v)
            if denom.abs() < 1e-20:
                break
            rayleigh = torch.dot(v, Gv) / denom
            eigenval = rayleigh

            residual = Gv - rayleigh * v
            resid_norm = torch.linalg.norm(residual)
            if torch.abs(rayleigh) < 1e-12 or (resid_norm / torch.abs(rayleigh) < reltol):
                break

            Gv_norm = torch.linalg.norm(Gv)
            if Gv_norm.item() == 0:
                break
            v = Gv / Gv_norm

    iterations_run = (i + 1) if "i" in locals() else 0
    try:
        wandb.log({"power_iteration_iterations": iterations_run}, commit=False)
    except Exception:
        pass

    if eigenvector_cache is not None:
        eigenvector_cache.store_eigenvector(v.detach(), eigenval)

    if return_eigenvector:
        return eigenval, v.detach()
    return eigenval


def compute_gauss_newton_eigenvalues(
    net: nn.Module,
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    loss_fn,
    *,
    k: int = 1,
    max_iterations: int = 100,
    reltol: float = 1e-2,
    init_vectors: Optional[torch.Tensor] = None,
    eigenvector_cache: Optional[EigenvectorCache] = None,
    return_eigenvectors: bool = False,
    use_power_iteration: bool = False,
    use_lanczos: bool = False,
    gauss_newton_loss_type: Optional[str] = None,
):
    """
    Compute the top-k eigenvalues of the Gauss-Newton matrix on a specific mini-batch.

    Mirrors ``compute_eigenvalues`` but replaces the Hessian with its Gauss-Newton approximation.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if X_batch.numel() == 0:
        raise ValueError("X_batch must contain at least one sample.")
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1 for Gauss-Newton eigenvalues.")
    if use_lanczos and use_power_iteration:
        raise ValueError("Cannot combine Lanczos with power iteration.")

    loss_type = _infer_gauss_newton_loss_type(loss_fn, gauss_newton_loss_type)
    gn_operator = GaussNewtonVectorProduct(
        net,
        X_batch,
        Y_batch,
        loss_type=loss_type,
        average_over_batch=True,
    )

    try:
        if use_power_iteration and k == 1:
            init_vector_1d = None
            if init_vectors is not None:
                if isinstance(init_vectors, torch.Tensor) and init_vectors.ndim == 1:
                    init_vector_1d = init_vectors
                else:
                    raise ValueError("init_vectors must be a 1D tensor when using power iteration.")
            return _compute_gauss_newton_power_iteration(
                gn_operator,
                net,
                max_iterations,
                reltol,
                init_vector_1d,
                eigenvector_cache,
                return_eigenvectors,
            )

        n_params = param_length(net)
        device = next(net.parameters()).device
        param_dtype = next(net.parameters()).dtype
        np_dtype = np.float64 if param_dtype == torch.float64 else np.float32

        if use_lanczos:
            if n_params == 0:
                raise ValueError("Model must have parameters to compute eigenvalues.")
            if k >= n_params:
                raise ValueError(f"Lanczos requires k < number of params; got k={k}, n_params={n_params}.")
            if n_params == 1:
                basis = torch.ones(n_params, device=device, dtype=param_dtype)
                eigvals_torch = gn_operator(basis).detach()
                if return_eigenvectors:
                    if k != 1:
                        raise ValueError("k must be 1 when parameter space is one-dimensional.")
                    return eigvals_torch[0], basis
                return eigvals_torch[0]

            def _apply_operator(vec_torch: torch.Tensor) -> torch.Tensor:
                if vec_torch.ndim != 1 or vec_torch.numel() != n_params:
                    raise ValueError("Vector shape mismatch for Lanczos matvec.")
                return gn_operator(vec_torch).detach()

            def _matvec(vec_np: np.ndarray) -> np.ndarray:
                vec_torch = torch.from_numpy(vec_np).to(device=device, dtype=param_dtype)
                result = _apply_operator(vec_torch)
                return result.cpu().numpy().astype(np_dtype, copy=False)

            linear_op = LinearOperator(
                shape=(n_params, n_params),
                matvec=_matvec,
                rmatvec=_matvec,
                dtype=np_dtype,
            )
            eigvals_np, eigvecs_np = eigsh(
                linear_op,
                k=k,
                which="LM",
            )

            eigvals = torch.from_numpy(eigvals_np.real.astype(np_dtype, copy=False)).to(device=device, dtype=param_dtype)
            eigvecs = torch.from_numpy(eigvecs_np.astype(np_dtype, copy=False)).to(device=device, dtype=param_dtype)
            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            if k == 1:
                if return_eigenvectors:
                    return eigvals[0], eigvecs[:, 0]
                return eigvals[0]
            if return_eigenvectors:
                return eigvals, eigvecs
            return eigvals

        eigenvalues, eigenvectors = _run_lobpcg_with_operator(
            gn_operator,
            net,
            k,
            max_iterations,
            reltol,
            init_vectors,
            eigenvector_cache,
            True,
        )

        if k == 1:
            eigenvalue = eigenvalues[0]
            eigenvector = eigenvectors[:, 0]
            if return_eigenvectors:
                return eigenvalue, eigenvector
            return eigenvalue

        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues
    finally:
        gn_operator.free_memory()


################################################################################
#                                                                              #
#                       GRADIENT–NOISE INTERACTION (GNI)                       #
#                                                                              #
################################################################################


def calculate_gni(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                            #   max_hessian_iters = 1000,
                            #   hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False,
                              use_subset_of_data: int = None # use only a subset of the dataset to calculate H in GNI - speeds up computations!
                              ): 
    sharpnesses = []

    params = list(net.parameters())


    if use_subset_of_data is not None:
        rng = gimme_new_rng()
        # Take random subset of the dataset
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:use_subset_of_data]
        X = X[random_idx]
        Y = Y[random_idx]

    total_loss = loss_fn(net(X).squeeze(dim=-1), Y)

    total_grads = torch.autograd.grad(total_loss, params, create_graph=True)
    total_grad = flatt(total_grads)
    total_grad_detach = total_grad.detach()

    normalizer = T.dot(total_grad_detach, total_grad_detach).item()

    hvp_total_loss = create_hessian_vector_product(
        total_loss,
        net,
        params=params,
        grads=total_grads,
        flat_grads=total_grad,
    )

    gHg_list = []

    try:
        for i in range(n_estimates):
            rng = gimme_new_rng()

            shuffle = T.randperm(len(X), generator=rng)
            random_idx = shuffle[:batch_size]

            X_batch = X[random_idx]
            Y_batch = Y[random_idx]

            loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

            grads_vector = flatt(torch.autograd.grad(loss, params))
            step_vector = grads_vector.detach()

            Hg = hvp_total_loss(step_vector).detach()

            gHg = T.dot(step_vector, Hg)

            gHg_list.append(gHg.item())
    finally:
        hvp_total_loss.free_memory()

    quantity = np.mean(gHg_list) / normalizer

    return quantity



################################################################################
#                                                                              #
#                               MISCELLANEOUS                                  #
#                                                                              #
################################################################################


def compute_gradient_projection_ratios(grad_vector: torch.Tensor,
                                       eigvecs: torch.Tensor,
                                       max_k: int = 20,
                                       eigenvalues: list = None) -> dict:
    """
    Compute cumulative projection ratios of the full-batch gradient onto the
    subspace spanned by the top-i eigenvectors, i = 1..k, where k = min(K, max_k).

    grad_projection_i = ||Proj_{span(v1..vi)}(g)||_2 / ||g||_2

    Args:
        grad_vector: Flattened full-batch gradient g, shape [n]
        eigvecs:   Matrix of eigenvectors, shape [n, K]
        max_k:     Cap on how many cumulative projections to report (default 20)
        eigenvalues: Optional list of eigenvalues (length K) to ensure proper
                     descending ordering; if provided, will sort eigvecs by it.

    Returns:
        dict mapping names 'grad_projection_01', ..., 'grad_projection_{k:02d}',
        and 'grad_projection_residual' to floats in [0, 1].

    Notes:
        - Uses _maybe_orthonormalize to cheaply verify and, if needed,
          re-orthonormalize the eigenvector block prior to projection.
        - If grad_vector has zero norm, returns all zeros.
    """
    if grad_vector is None or eigvecs is None:
        return {}

    # Ensure 2D [n, K]
    if eigvecs.dim() == 1:
        eigvecs = eigvecs.unsqueeze(1)

    n, K = eigvecs.shape
    if n != grad_vector.numel():
        raise ValueError(f"Dimension mismatch: gradient has {grad_vector.numel()} params, eigenvectors have {n}")

    # Limit to at most max_k eigenvectors
    k = min(K, max_k)

    # If eigenvalues are supplied, sort eigenvectors by descending eigenvalue
    if eigenvalues is not None and len(eigenvalues) >= k:
        # Sort pairs (eigenvalue, column index) descending by value
        import math
        order = sorted(range(len(eigenvalues)), key=lambda idx: (-float(eigenvalues[idx]) if not math.isnan(float(eigenvalues[idx])) else float('inf')))
        order = order[:k]
        V = eigvecs[:, order]
    else:
        V = eigvecs[:, :k]

    # Quick orthonormality check; orthonormalize if necessary
    V = _maybe_orthonormalize(V, assume_ortho=True)

    # Compute projection coefficients c = V^T s
    g = grad_vector.reshape(-1)
    g_norm = torch.linalg.vector_norm(g)
    if g_norm.item() == 0.0:
        # Degenerate step; return zeros
        result = {f"grad_projection_{i:02d}": 0.0 for i in range(1, k + 1)}
        result["grad_projection_residual"] = 0.0
        return result

    c = V.T @ g  # shape [k]
    c2 = c.pow(2)
    # Cumulative squared projection norms
    c2_cum = torch.cumsum(c2, dim=0)
    denom = g_norm.pow(2)
    # Convert to ratios in [0,1]
    ratios = torch.sqrt(torch.clamp(c2_cum / denom, min=0.0, max=1.0))

    result = {}
    for i in range(k):
        result[f"grad_projection_{i+1:02d}"] = float(ratios[i].item())

    # Residual norm ratio for the full k-dimensional subspace
    residual_sq = torch.clamp(1.0 - c2_cum[-1] / denom, min=0.0, max=1.0)
    result["grad_projection_residual"] = float(torch.sqrt(residual_sq).item())

    return result


def estimate_hessian_trace(net,
                           X,
                           Y,
                           loss_fn,
                           max_estimates: int = 512,
                           min_estimates: int = 10,
                           eps: float = 0.01,
                           generator: Optional[torch.Generator] = None,
                           probe_type: str = 'rademacher') -> float:
    """
    Estimate the trace of the full-batch loss Hessian via Hutchinson's method.

    Args:
        net: Neural network model.
        X: Full input tensor used to construct the loss.
        Y: Full target tensor used to construct the loss.
        loss_fn: Callable loss function applied on the full batch.
        max_estimates: Maximum number of probe vectors to use.
        min_estimates: Minimum number of probes before adaptive stopping is checked.
        eps: Relative standard error tolerance for adaptive stopping.
        generator: Optional RNG to make the estimator deterministic (useful in tests).
        probe_type: Distribution for probe vectors. Currently only 'rademacher' is supported.

    Returns:
        float: Estimated trace of the Hessian.
    """

    if max_estimates < 1:
        raise ValueError("max_estimates must be positive")
    if min_estimates < 1:
        raise ValueError("min_estimates must be positive")
    if min_estimates > max_estimates:
        raise ValueError("min_estimates cannot exceed max_estimates")
    if probe_type != 'rademacher':
        raise NotImplementedError(f"Unsupported probe_type: {probe_type}")

    first_param = next(net.parameters())
    device = first_param.device
    dtype = first_param.dtype

    # Evaluate full-batch loss and build Hessian-vector product closure
    preds = net(X).squeeze(dim=-1)
    loss = loss_fn(preds, Y)
    hessian_matvec = create_hessian_vector_product(loss, net)

    n_params = param_length(net)

    if generator is None:
        generator = gimme_new_rng()

    trace_estimates: List[float] = []

    try:
        for i in range(max_estimates):
            # Sample Rademacher probe vector (entries +/-1)
            probe = torch.randint(0, 2, (n_params,), generator=generator, device='cpu', dtype=torch.float32)
            probe = probe.mul_(2.0).sub_(1.0).to(device=device, dtype=dtype)

            Hz = hessian_matvec(probe)
            if Hz.dim() != 1 or Hz.numel() != n_params:
                raise RuntimeError("Hessian-vector product returned unexpected shape")

            trace_component = torch.dot(probe, Hz).detach().item()
            trace_estimates.append(trace_component)

            num_samples = i + 1
            if num_samples < min_estimates:
                continue

            mean_val = float(np.mean(trace_estimates))
            variance = float(np.var(trace_estimates, ddof=1)) if num_samples > 1 else 0.0

            # Avoid division by zero when the estimate is numerically zero
            if abs(mean_val) < 1e-12:
                continue

            sem = np.sqrt(variance / num_samples)
            if sem / abs(mean_val) < eps:
                break
    finally:
        hessian_matvec.free_memory()

    try:
        wandb.log({"hessian_trace_estimates": len(trace_estimates)}, commit=False)
    except Exception:
        pass

    return float(np.mean(trace_estimates))


def calculate_gradient_norm_squared_mc(net,
                                     X,
                                     Y,
                                     loss_fn,
                                     batch_size,
                                     n_estimates=1000,
                                     min_estimates=10,
                                     eps=0.005  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                     ):
    """
    Computes the Monte Carlo estimate of the expected squared norm of mini-batch gradients.
    
    This function estimates E[||∇f_B||²] where f_B is the loss on a mini-batch B,
    using Monte Carlo sampling over random mini-batches.
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor  
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        batch_size (int): Size of mini-batches to sample
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 1000.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        
    Returns:
        float: Monte Carlo estimate of E[||∇f_B||²]
    """
    gradient_norm_squared_vals = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    params = list(net.parameters())
    
    for i in range(n_estimates):
        # Sample random mini-batch
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Compute loss and gradients
        preds = net(X_batch).squeeze(dim=-1)
        loss = loss_fn(preds, Y_batch)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=False)
        grads_vector = flatt(grads)
        
        # Compute squared norm of gradient
        grad_norm_squared = torch.dot(grads_vector, grads_vector).item()
        gradient_norm_squared_vals.append(grad_norm_squared)
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(gradient_norm_squared_vals)
            var_val = np.var(gradient_norm_squared_vals, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(gradient_norm_squared_vals)) / abs(mean_val)
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"gradient_norm_squared_mc_estimates": len(gradient_norm_squared_vals)}, commit=False)
    except:
        pass
    
    return np.mean(gradient_norm_squared_vals)


def calculate_expected_one_step_full_loss_change(net,
                                          X,
                                          Y,
                                          loss_fn,
                                          optimizer,
                                          batch_size,
                                          n_estimates=500,
                                          min_estimates=10,
                                          eps=0.005,  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                          eval_batch_size=None,  # For efficient total loss computation,
                                          use_subset_of_data: int = None # use only a subset of the dataset to calculate total loss - speeds up computations!
                                          ):
    """
    Calculate the expected one-step change in total loss using Monte Carlo estimation.
    
    This function estimates the expected change in total dataset loss when making a 
    gradient step on a randomly sampled mini-batch, then returning to the original parameters.
    
    The process for each estimate:
    1. Compute total loss before step (on entire dataset)
    2. Sample a random mini-batch for gradient computation
    3. Store current parameters
    4. Take one optimization step on the mini-batch
    5. Compute total loss after step (on entire dataset)
    6. Calculate change: (loss_after - loss_before)
    7. Restore original parameters
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        optimizer (torch.optim.Optimizer): Optimizer for taking gradient steps
        batch_size (int): Size of mini-batches to sample for gradient computation
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 500.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        eval_batch_size (int, optional): Batch size for total loss evaluation. If None, uses entire dataset.
        
    Returns:
        float: Monte Carlo estimate of expected total loss change
    """
    loss_changes = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    # Store original parameters
    original_params = param_vector(net).detach().clone()
    
    # Compute total loss before any steps (reused for efficiency)
    # with torch.no_grad():
    if eval_batch_size is None or eval_batch_size >= len(X):
        # Evaluate on entire dataset
        preds_total_before = net(X).squeeze(dim=-1)
        total_loss_before = loss_fn(preds_total_before, Y)
    else:
        raise NotImplementedError("Batched evaluation not implemented")
    # else:
    #     # Evaluate on batches to save memory
    #     total_loss_before = 0.0
    #     n_eval_batches = (len(X) + eval_batch_size - 1) // eval_batch_size
    #     for eval_i in range(n_eval_batches):
    #         start_idx = eval_i * eval_batch_size
    #         end_idx = min((eval_i + 1) * eval_batch_size, len(X))
    #         X_eval = X[start_idx:end_idx]
    #         Y_eval = Y[start_idx:end_idx]
    #         preds_eval = net(X_eval).squeeze(dim=-1)
    #         batch_loss = loss_fn(preds_eval, Y_eval)
    #         total_loss_before += batch_loss.item() * len(X_eval)
    #     total_loss_before = total_loss_before / len(X)

    total_loss_before.backward()
    gradient_norm_squared = sum(p.grad.data.norm(2).item() ** 2 for p in net.parameters())
    eta = optimizer.param_groups[0]['lr']

    for i in range(n_estimates):
        # Sample random mini-batch for gradient step
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Take gradient step on mini-batch
        optimizer.zero_grad()
        preds_batch = net(X_batch).squeeze(dim=-1)
        loss_batch = loss_fn(preds_batch, Y_batch)
        loss_batch.backward()
        optimizer.step()
        
        # Compute total loss after step
        with torch.no_grad():
            if eval_batch_size is None or eval_batch_size >= len(X):
                if use_subset_of_data is not None:
                    random_idx = gimme_random_subset_idx(len(X), use_subset_of_data)

                    X_eval = X[random_idx]
                    Y_eval = Y[random_idx]
                else:
                    X_eval = X
                    Y_eval = Y

                # Evaluate on entire dataset
                preds_total_after = net(X_eval).squeeze(dim=-1)
                total_loss_after = loss_fn(preds_total_after, Y_eval)
            else:
                # Evaluate on batches to save memory
                total_loss_after = 0.0
                n_eval_batches = (len(X) + eval_batch_size - 1) // eval_batch_size
                for eval_i in range(n_eval_batches):
                    start_idx = eval_i * eval_batch_size
                    end_idx = min((eval_i + 1) * eval_batch_size, len(X))
                    X_eval = X[start_idx:end_idx]
                    Y_eval = Y[start_idx:end_idx]
                    preds_eval = net(X_eval).squeeze(dim=-1)
                    batch_loss = loss_fn(preds_eval, Y_eval)
                    total_loss_after += batch_loss.item() * len(X_eval)
                total_loss_after = total_loss_after / len(X)
            
            # Calculate change in total loss
            loss_change = total_loss_after - total_loss_before
            loss_changes.append(loss_change.item() if torch.is_tensor(loss_change) else loss_change)
        
        # Restore original parameters
        with torch.no_grad():
            param_idx = 0
            for param in net.parameters():
                param_size = param.numel()
                param.data.copy_(original_params[param_idx:param_idx + param_size].view_as(param))
                param_idx += param_size
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(loss_changes)
            var_val = np.var(loss_changes, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(loss_changes)) / abs(mean_val) if mean_val != 0 else float('inf')
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"one_step_total_loss_change_estimates": len(loss_changes)}, commit=False)
    except:
        pass
    
    return np.mean(loss_changes) / (eta * gradient_norm_squared)



def calculate_expected_one_step_batch_loss_change(net,
                                          X,
                                          Y,
                                          loss_fn,
                                          optimizer,
                                          batch_size,
                                          n_estimates=500,
                                          min_estimates=10,
                                          eps=0.005  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                          ):
    """
    Calculate the expected one-step change in loss using Monte Carlo estimation.
    
    This function estimates the expected relative change in loss when making a 
    gradient step on a randomly sampled batch, then returning to the original parameters.
    
    The process for each estimate:
    1. Sample a random batch
    2. Store current parameters
    3. Compute loss before step
    4. Take one optimization step
    5. Compute loss after step  
    6. Calculate relative change: (loss_after - loss_before) / loss_before
    7. Restore original parameters
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        optimizer (torch.optim.Optimizer): Optimizer for taking gradient steps
        batch_size (int): Size of mini-batches to sample
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 500.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        
    Returns:
        float: Monte Carlo estimate of expected relative one-step loss change
    """
    loss_changes = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    # Store original parameters
    original_params = param_vector(net).detach().clone()
    
    for i in range(n_estimates):
        # Sample random mini-batch
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Compute loss before step
        optimizer.zero_grad()
        preds_before = net(X_batch).squeeze(dim=-1)
        loss_before = loss_fn(preds_before, Y_batch)
        
        # Take gradient step
        loss_before.backward()
        optimizer.step()
        
        # Compute loss after step (on the same batch)
        with torch.no_grad():
            preds_after = net(X_batch).squeeze(dim=-1)
            loss_after = loss_fn(preds_after, Y_batch)
            
            # Calculate relative change in loss
            relative_change = (loss_after - loss_before) #/ loss_before
            loss_changes.append(relative_change.item())
        
        # Restore original parameters
        current_params = param_vector(net)
        with torch.no_grad():
            param_idx = 0
            for param in net.parameters():
                param_size = param.numel()
                param.data.copy_(original_params[param_idx:param_idx + param_size].view_as(param))
                param_idx += param_size
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(loss_changes)
            var_val = np.var(loss_changes, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(loss_changes)) / abs(mean_val) if mean_val != 0 else float('inf')
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"one_step_loss_change_estimates": len(loss_changes)}, commit=False)
    except:
        pass
    
    return np.mean(loss_changes)


################################################################################
#                                                                              #
#                        GAUSS–NEWTON (=FIM) MATRIX STUFF                      #
#                                                                              #
################################################################################

def calculate_all_net_grads(net, X):

    gradients = []
    params = list(net.parameters())

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)
    del gradients
    return G



def create_ntk(net, X):
    params = list(net.parameters())

    gradients = []

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)

    ntk = G @ G.T
    del G
    # f = lambda v: G.T @ (G @ v) / len(X)

    return ntk


def compute_fisher_eigenvalues(net, X):
    '''
    The trick here is that instead of computing the fisher information matrix, we compute the NTK
    They have the same eigenvalues, but NTK is size n_samples x n_samples, while FIM is size n_params x n_params
    '''

    ntk = create_ntk(net, X)
    # size = param_length(net)

    # device = next(net.parameters()).device
    # eigenval = compute_eigenvalues(operator, size, device, iterations=iterations, epsilon=epsilon)

    eigenval = T.lobpcg(ntk, k=1)
    eigenval = 2/len(X) * eigenval[0]
    
    return eigenval




################################################################################
#                                                                              #
#                                LAMBDA^b_MAX                                  #
#                                                                              #
################################################################################


def calculate_step_batch_lambdamax(
    net: nn.Module,
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    loss_fn,
    max_hessian_iters: int = 100,
    hessian_tolerance: float = 1e-3,
):
    """
    Compute lambda_max on the provided mini-batch only (no expectation).

    Args:
        net: Model under evaluation.
        X_batch: Mini-batch inputs for the current training step.
        Y_batch: Mini-batch targets corresponding to ``X_batch``.
        loss_fn: Loss function used for the training step.
        max_hessian_iters: Maximum iterations for the eigen-solver.
        hessian_tolerance: Relative tolerance for the eigen-solver.

    Returns:
        float: The largest Hessian eigenvalue for the provided mini-batch.
    """
    if X_batch.numel() == 0:
        raise ValueError("X_batch must contain at least one sample.")

    preds = net(X_batch).squeeze(dim=-1)
    loss = loss_fn(preds, Y_batch)

    eigenval = compute_eigenvalues(
        loss,
        net,
        max_iterations=max_hessian_iters,
        reltol=hessian_tolerance,
        # use_lanczos=True,
    )
    return eigenval.item()


def calculate_averaged_lambdamax(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 100,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                              max_hessian_iters = 100,
                              hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False,
                              eigenvector_cache = None
                              ): 
    
    
    sharpnesses = []

    if compute_gHg:
        gHg_values = []
    
    if batch_size is None:
        batch_size = len(X)

    if compute_gHg:
        raise NotImplementedError("compute_gHg=True is not implemented in this refactor.")
    

    for i in range(n_estimates):
        generator = gimme_new_rng()
        shuffle = T.randperm(len(X), generator=generator)
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        sharpness = compute_eigenvalues(loss, 
                        net,
                        max_iterations=max_hessian_iters,
                        reltol=hessian_tolerance,
                        # use_lanczos=True,
                        )
        # if compute_gHg:
        #     sharpness, gHg = sharpness
        #     gHg = gHg.item()
        #     gHg_values.append(gHg)
        
        sharpness = sharpness.item()
        
        sharpnesses.append(sharpness)

        if batch_size >= len(X):
            break

        if len(sharpnesses) > min_estimates:
            mean = np.mean(sharpnesses)
            sem = np.std(sharpnesses) / np.sqrt(len(sharpnesses))

            if sem / mean < tolerance:
                break
    
    # if compute_gHg:
    #     return sharpnesses, gHg_values
    return sharpnesses





################################################################################
#                                                                              #
#        LANCZOS FOR E[(I - ETA * H_B)^2] MAXIMUM EIGENVALUE (BATCHED)         #
#                                                                              #
################################################################################


def calculate_second_moment_contraction_old(
    net: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn,
    batch_size: int,
    eta: float,
    n_batches_in_expectation: int = 4,
    lanczos_tolerance: float = 1e-6,
    lanczos_max_iter: int = 200,
    cache_forward_passes: bool = True,
    batch_index_sets: Optional[Sequence[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Estimate lambda_max(E_B[(I - eta * H_B)^2]) via SciPy Lanczos (eigsh).

    Args:
        net (nn.Module): Model under evaluation.
        X (Tensor): Full dataset inputs.
        Y (Tensor): Full dataset targets.
        loss_fn (Callable): Loss to differentiate.
        batch_size (int): Mini-batch size for Hessian computations.
        eta (float): Step size used in the operator.
        n_batches_in_expectation (int): Number of batches to Monte-Carlo average.
        lanczos_tolerance (float): Tolerance passed to SciPy's eigsh.
        lanczos_max_iter (int): Maximum Lanczos iterations.
        cache_forward_passes (bool): If True, reuse forward graphs across Lanczos steps.
        batch_index_sets (Optional[Sequence[Tensor]]): Pre-selected batch indices.
        generator (Optional[torch.Generator]): RNG for batch sampling.

    Returns:
        float: Largest eigenvalue estimate of the expected operator.
    """
    if LinearOperator is None or eigsh is None:
        raise ImportError("SciPy is required for Lanczos-based measurements.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_batches_in_expectation <= 0:
        raise ValueError("n_batches_in_expectation must be positive.")

    param_example = next(net.parameters(), None)
    if param_example is None:
        raise ValueError("Network has no parameters.")

    device = param_example.device
    param_dtype = param_example.dtype
    n_params = param_length(net)

    if n_params == 0:
        raise ValueError("Param length must be positive.")

    eta = float(eta)
    eta_sq = eta * eta

    def _eval_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_for_loss = preds
        targets_for_loss = targets
        if preds_for_loss.ndim > 1 and preds_for_loss.shape[-1] == 1:
            preds_for_loss = preds_for_loss.squeeze(dim=-1)
        if targets_for_loss.ndim > 1 and targets_for_loss.shape[-1] == 1:
            targets_for_loss = targets_for_loss.squeeze(dim=-1)
        return loss_fn(preds_for_loss, targets_for_loss)

    # Select deterministic batches if provided, otherwise sample with RNG.
    if batch_index_sets is not None:
        if len(batch_index_sets) < n_batches_in_expectation:
            raise ValueError("batch_index_sets shorter than n_batches_in_expectation.")
        index_list = list(batch_index_sets)[:n_batches_in_expectation]
    else:
        if generator is None:
            generator = gimme_new_rng()
        index_list = []
        dataset_size = X.shape[0]
        for _ in range(n_batches_in_expectation):
            idx = torch.randint(
                low=0,
                high=dataset_size,
                size=(batch_size,),
                generator=generator,
            )
            index_list.append(idx)

    batch_entries = []
    for idx in index_list:
        idx = idx.to(device=X.device)

        X_slice = X[idx]
        if torch.is_floating_point(X_slice):
            X_batch = X_slice.to(device=device, dtype=param_dtype)
        else:
            X_batch = X_slice.to(device=device)

        Y_slice = Y[idx]
        if torch.is_floating_point(Y_slice):
            Y_batch = Y_slice.to(device=device, dtype=param_dtype)
        else:
            Y_batch = Y_slice.to(device=device)

        entry = {
            "X": X_batch,
            "Y": Y_batch,
        }

        if cache_forward_passes:
            preds = net(X_batch)
            loss = _eval_loss(preds, Y_batch)
            hvp = create_hessian_vector_product(loss, net)
            entry["hvp"] = hvp
            entry["loss"] = loss  # Keep graph alive for repeated Hessian calls.
        batch_entries.append(entry)

    if not batch_entries:
        raise RuntimeError("No batches were prepared for the expectation operator.")

    def _apply_operator(vec_torch: torch.Tensor) -> torch.Tensor:
        if vec_torch.ndim != 1 or vec_torch.numel() != n_params:
            raise ValueError("Vector shape mismatch for operator application.")

        result = torch.zeros_like(vec_torch)
        for entry in batch_entries:
            hvp_func = entry.get("hvp")
            if hvp_func is None:
                preds = net(entry["X"])
                loss = _eval_loss(preds, entry["Y"])
                hvp_func = create_hessian_vector_product(loss, net)

            u = hvp_func(vec_torch)
            u_detached = u.detach()
            w = hvp_func(u_detached)
            y_batch = vec_torch - 2.0 * eta * u + eta_sq * w
            result = result + y_batch

            if "hvp" not in entry:
                # Release one-off graphs early when caching is disabled.
                hvp_func.free_memory()

        return result / len(batch_entries)

    def _matvec(vec_np: np.ndarray) -> np.ndarray:
        vec_torch = torch.from_numpy(vec_np).to(device=device, dtype=param_dtype)
        operator_vec = _apply_operator(vec_torch).detach()
        return operator_vec.cpu().numpy().astype(np_dtype, copy=False)

    np_dtype = np.float64 if param_dtype == torch.float64 else np.float32

    if n_params == 1:
        basis = torch.ones(n_params, device=device, dtype=param_dtype)
        value = _apply_operator(basis)[0].item()
        lambda_max = value
    else:
        linear_op = LinearOperator(
            shape=(n_params, n_params),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=np_dtype,
        )
        eigvals, _ = eigsh(
            linear_op,
            k=1,
            which="LM",
            tol=lanczos_tolerance,
            maxiter=lanczos_max_iter,
        )
        lambda_max = float(np.max(eigvals.real))

    # Drop strong references so autograd graphs can be freed.
    for entry in batch_entries:
        entry.pop("loss", None)
        cached_hvp = entry.pop("hvp", None)
        if cached_hvp is not None:
            cached_hvp.free_memory()

    return lambda_max




def calculate_second_moment_contraction(
    net: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn,
    batch_size: int,
    eta: float,
    n_batches_in_expectation: int = 4,
    lanczos_tolerance: float = 1e-6,
    lanczos_max_iter: int = 200,
    cache_forward_passes: bool = True,
    batch_index_sets: Optional[Sequence[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    NOTE WE NOW DOING -2*eta*H + eta^2*E_B[H_B^2]
    Estimate lambda_max(E_B[(I - eta * H_B)^2]) via SciPy Lanczos (eigsh).

    Args:
        net (nn.Module): Model under evaluation.
        X (Tensor): Full dataset inputs.
        Y (Tensor): Full dataset targets.
        loss_fn (Callable): Loss to differentiate.
        batch_size (int): Mini-batch size for Hessian computations.
        eta (float): Step size used in the operator.
        n_batches_in_expectation (int): Number of batches to Monte-Carlo average.
        lanczos_tolerance (float): Tolerance passed to SciPy's eigsh.
        lanczos_max_iter (int): Maximum Lanczos iterations.
        cache_forward_passes (bool): If True, reuse forward graphs across Lanczos steps.
        batch_index_sets (Optional[Sequence[Tensor]]): Pre-selected batch indices.
        generator (Optional[torch.Generator]): RNG for batch sampling.

    Returns:
        float: Largest eigenvalue estimate of the expected operator.
    """
    if LinearOperator is None or eigsh is None:
        raise ImportError("SciPy is required for Lanczos-based measurements.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_batches_in_expectation <= 0:
        raise ValueError("n_batches_in_expectation must be positive.")

    param_example = next(net.parameters(), None)
    if param_example is None:
        raise ValueError("Network has no parameters.")

    device = param_example.device
    param_dtype = param_example.dtype
    n_params = param_length(net)

    if n_params == 0:
        raise ValueError("Param length must be positive.")

    eta = float(eta)
    # eta_sq = eta * eta

    def _eval_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_for_loss = preds
        targets_for_loss = targets
        if preds_for_loss.ndim > 1 and preds_for_loss.shape[-1] == 1:
            preds_for_loss = preds_for_loss.squeeze(dim=-1)
        if targets_for_loss.ndim > 1 and targets_for_loss.shape[-1] == 1:
            targets_for_loss = targets_for_loss.squeeze(dim=-1)
        return loss_fn(preds_for_loss, targets_for_loss)

    # Select deterministic batches if provided, otherwise sample with RNG.
    if batch_index_sets is not None:
        if len(batch_index_sets) < n_batches_in_expectation:
            raise ValueError("batch_index_sets shorter than n_batches_in_expectation.")
        index_list = list(batch_index_sets)[:n_batches_in_expectation]
    else:
        if generator is None:
            generator = gimme_new_rng()
        index_list = []
        dataset_size = X.shape[0]
        for _ in range(n_batches_in_expectation):
            idx = torch.randint(
                low=0,
                high=dataset_size,
                size=(batch_size,),
                generator=generator,
            )
            index_list.append(idx)

    batch_entries = []
    for idx in index_list:
        idx = idx.to(device=X.device)

        X_slice = X[idx]
        if torch.is_floating_point(X_slice):
            X_batch = X_slice.to(device=device, dtype=param_dtype)
        else:
            X_batch = X_slice.to(device=device)

        Y_slice = Y[idx]
        if torch.is_floating_point(Y_slice):
            Y_batch = Y_slice.to(device=device, dtype=param_dtype)
        else:
            Y_batch = Y_slice.to(device=device)

        entry = {
            "X": X_batch,
            "Y": Y_batch,
        }

        if cache_forward_passes:
            preds = net(X_batch)
            loss = _eval_loss(preds, Y_batch)
            hvp = create_hessian_vector_product(loss, net)
            entry["hvp"] = hvp
            entry["loss"] = loss  # Keep graph alive for repeated Hessian calls.
        batch_entries.append(entry)

    if not batch_entries:
        raise RuntimeError("No batches were prepared for the expectation operator.")

    # do full-batch hessian too
    if cache_forward_passes:
        preds = net(X)
        loss = _eval_loss(preds, Y)
        full_hvp = create_hessian_vector_product(loss, net)

        batch_entries[0]["hvp_full"] = full_hvp
        batch_entries[0]["loss_full"] = loss  # Keep graph alive for repeated Hessian calls.

    def _apply_operator(vec_torch: torch.Tensor) -> torch.Tensor:
        if vec_torch.ndim != 1 or vec_torch.numel() != n_params:
            raise ValueError("Vector shape mismatch for operator application.")

        H_B_squared_v_array = torch.zeros_like(vec_torch)

        full_hvp_func = batch_entries[0].get("hvp_full")
        if full_hvp_func is not None:
            Hv = full_hvp_func(vec_torch)
            Hv = Hv.detach()
        else:
            raise RuntimeError("eeeh, I just didn't implement it")
        
        for entry in batch_entries:
            hvp_func = entry.get("hvp")
            if hvp_func is None:
                preds = net(entry["X"])
                loss = _eval_loss(preds, entry["Y"])
                hvp_func = create_hessian_vector_product(loss, net)

            H_B_v = hvp_func(vec_torch)
            H_B_v = H_B_v.detach()
            H_B_squared_v = hvp_func(H_B_v)
            H_B_squared_v = H_B_squared_v.detach()

            H_B_squared_v_array += H_B_squared_v

            if "hvp" not in entry:
                # Release one-off graphs early when caching is disabled.
                hvp_func.free_memory()
            
        E_H_B_squared_v = H_B_squared_v_array / len(batch_entries)
        
        result = -2.0 * eta * Hv + eta**2 * E_H_B_squared_v

        return result

        # return result / len(batch_entries)

    def _matvec(vec_np: np.ndarray) -> np.ndarray:
        vec_torch = torch.from_numpy(vec_np).to(device=device, dtype=param_dtype)
        operator_vec = _apply_operator(vec_torch).detach()
        return operator_vec.cpu().numpy().astype(np_dtype, copy=False)

    np_dtype = np.float64 if param_dtype == torch.float64 else np.float32

    if n_params == 1:
        basis = torch.ones(n_params, device=device, dtype=param_dtype)
        value = _apply_operator(basis)[0].item()
        lambda_max = value
    else:
        linear_op = LinearOperator(
            shape=(n_params, n_params),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=np_dtype,
        )
        eigvals, _ = eigsh(
            linear_op,
            k=1,
            which="LA",
            tol=lanczos_tolerance,
            maxiter=lanczos_max_iter,
        )
        lambda_max = float(np.max(eigvals.real))

    # Drop strong references so autograd graphs can be freed.
    for entry in batch_entries:
        entry.pop("loss", None)
        cached_hvp = entry.pop("hvp", None)
        if cached_hvp is not None:
            cached_hvp.free_memory()
    full_entry = batch_entries[0]
    full_hvp_cached = full_entry.pop("hvp_full", None)
    if full_hvp_cached is not None:
        full_hvp_cached.free_memory()
    full_entry.pop("loss_full", None)

    return lambda_max
