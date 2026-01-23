"""
Gauss-Newton Generalized (GGN) Matrix-Vector Product Implementation.

This module provides efficient computation of GGN matrix-vector products using PyTorch's
automatic differentiation tools (torch.func). The GGN matrix is a curvature approximation
that replaces the loss Hessian in the full Hessian with a per-example Hessian in
logit space

For a neural network f(θ; x) → z (logits) with loss ℓ(z, y), the Hessian can be decomposed as:
    H = J^T H_ℓ J + second-order terms
where J is the Jacobian of the network output w.r.t. parameters, and H_ℓ is the Hessian of
the loss w.r.t. logits. The GGN approximation drops the second-order terms:
    G = J^T H_ℓ J

For classification with softmax cross-entropy:
    H_ℓ = Diag(p) - p p^T  (where p = softmax(z))
For MSE:
    H_ℓ = 2I

The GGN matrix-vector product G v is computed efficiently via:
1. Forward-mode AD (JVP): u = J v  (Jacobian-vector product)
2. Apply logit Hessian: s = H_ℓ u  (cheap in logit space)
3. Reverse-mode AD (VJP): G v = J^T s  (vector-Jacobian product)

This is batched and averaged over training examples, making it suitable for computing
sharpness measures and optimization dynamics analysis.

Requires PyTorch ≥ 2.0 for torch.func API (functional_call, jvp, vjp).
I think we even need at least 2.9 as Torch.func is super beta.
"""
# PyTorch 2.9 — GGN matvec (flat -> flat) with torch.func.functional_call
import torch
from torch.func import functional_call, jvp, vjp
from dataclasses import dataclass

@dataclass
class _PInfo:
    name: str
    shape: torch.Size
    numel: int
    dtype: torch.dtype
    device: torch.device

def _collect_state(model: torch.nn.Module):
    """Gather param info, primals (as a tuple), and buffers (dict)."""
    infos, params = [], []
    for n, p in model.named_parameters():
        infos.append(_PInfo(n, p.shape, p.numel(), p.dtype, p.device))
        params.append(p.detach())
    params_tuple = tuple(params)
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    return infos, params_tuple, buffers

def _tuple_to_paramdict(infos, params_tuple):
    return {info.name: p for info, p in zip(infos, params_tuple)}

def _unflatten_like(v_flat: torch.Tensor, infos):
    """Split a flat vector into a tuple of tensors shaped like params."""
    v_flat = v_flat.view(-1)
    out, off = [], 0
    for info in infos:
        chunk = v_flat.narrow(0, off, info.numel)
        out.append(chunk.to(dtype=info.dtype, device=info.device).view(info.shape))
        off += info.numel
    return tuple(out)

def _flatten_tuple(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])

def _apply_H_logit(u, z, loss: str):
    # z, u : [B, C]
    if loss == 'ce':
        p = z.softmax(dim=-1)
        pu = (p * u).sum(dim=-1, keepdim=True)
        return p * (u - pu)
    elif loss == 'mse':
        # we need the 2xId, as we are not doing the 1/2 in the MSE loss
        return 2*u
    else:
        raise NotImplementedError(f"loss={loss!r}")

def ggn_matvec(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor | None,           # not needed for CE; kept for API symmetry
    v_flat: torch.Tensor,             # 1-D vector over ALL params
    loss: str = 'ce',
    average_over_batch: bool = True,  # True -> divide by B
) -> torch.Tensor:
    """
    Compute G v (GGN matvec) for a minibatch, returning a flat vector.
    Uses JVP (u=Jv) -> H_ℓ(z) -> VJP (J^T s). No per-example loops.
    """
    was_training = model.training
    model.eval()  # stabilize curvature (BN/Dropout off)

    # param/buffer primals + packing spec
    infos, params_tuple, buffers = _collect_state(model)
    total = sum(i.numel for i in infos)
    if v_flat.numel() != total:
        raise ValueError(f"v_flat has {v_flat.numel()} elems, but model has {total} params.")

    # tangent tuple shaped like params
    v_tuple = _unflatten_like(v_flat, infos)

    # define logits function using functional_call
    def f(*params_tensors):
        params_dict = _tuple_to_paramdict(infos, params_tensors)
        # You can also merge params+buffers into one dict; 2-dict form is fine in 2.9
        logits = functional_call(model, (params_dict, buffers), (x,), strict=True)
        return logits  # [B, C]

    # 1) JVP: u = J v
    z, u = jvp(f, params_tuple, v_tuple)

    # 2) s = H_ℓ(z) @ u   (cheap; C is small for CIFAR-10)
    s = _apply_H_logit(u, z, loss=loss)

    # 3) VJP: J^T s
    _, vjp_fn = vjp(f, *params_tuple)
    scale = (1.0 / x.shape[0]) if average_over_batch else 1.0
    Gv_tuple = vjp_fn(s * scale)

    # pack back to flat
    Gv_flat = _flatten_tuple(Gv_tuple)

    if was_training:
        model.train()
    return Gv_flat