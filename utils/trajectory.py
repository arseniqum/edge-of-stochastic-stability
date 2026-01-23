from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector

DEFAULT_TARGET_DIM = 5000
DEFAULT_K = 4
DEFAULT_SEED = 1337
_COPRIME_CACHE = {}


def _ensure_generator(
    generator: Optional[torch.Generator],
    seed: Optional[int],
    device: Optional[torch.device],
) -> torch.Generator:
    """
    Return a torch.Generator without mutating the caller's generator.
    """
    if generator is None:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        return generator

    cloned = torch.Generator(device=device or generator.device)
    cloned.set_state(generator.get_state())
    if seed is not None:
        cloned.manual_seed(seed)
    return cloned


def _coprime_choices(modulus: int) -> torch.Tensor:
    """
    Precompute integers in [1, modulus) that are coprime to modulus.
    """
    if modulus in _COPRIME_CACHE:
        return _COPRIME_CACHE[modulus]
    coprimes = [i for i in range(1, modulus) if math.gcd(i, modulus) == 1]
    if not coprimes:
        raise ValueError(f"No coprime strides available for modulus={modulus}.")
    tensor = torch.tensor(coprimes, dtype=torch.long)
    _COPRIME_CACHE[modulus] = tensor
    return tensor


def _trajectory_base_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the directory used for local trajectory artifacts.
    Defaults to $RESULTS/trajectory or ./trajectory.
    """
    if base_dir is not None:
        root = Path(base_dir)
    else:
        results_root = os.environ.get("RESULTS")
        root = Path(results_root) if results_root else Path.cwd()
    return (root / "trajectory").resolve()


class SparseJLProjector:
    """
    Sparse Johnson–Lindenstrauss projector with k nonzeros per column.
    """

    def __init__(
        self,
        param_dim: int,
        target_dim: int = DEFAULT_TARGET_DIM,
        k: int = DEFAULT_K,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = DEFAULT_SEED,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be positive.")
        if target_dim < k:
            raise ValueError("target_dim must be at least k.")
        self.param_dim = int(param_dim)
        self.target_dim = int(target_dim)
        self.k = int(k)
        self.seed = seed
        self._generator = _ensure_generator(generator, seed, device)
        self._generator_seed = int(self._generator.initial_seed())
        self.matrix = self._build_projection_matrix(device=device, dtype=dtype)

    def _build_projection_matrix(
        self,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        """
        Build a k-sparse projection matrix with unique rows per column.
        """
        coprimes = _coprime_choices(self.target_dim)
        stride_idx = torch.randint(
            0,
            len(coprimes),
            (self.param_dim,),
            generator=self._generator,
            device=device,
        )
        strides = coprimes.to(device=device)[stride_idx]

        base = torch.randint(
            0,
            self.target_dim,
            (self.param_dim,),
            generator=self._generator,
            device=device,
        )
        offsets = torch.arange(self.k, device=device)
        rows = (base.unsqueeze(1) + strides.unsqueeze(1) * offsets) % self.target_dim

        cols = torch.arange(self.param_dim, device=device).repeat_interleave(self.k)
        signs = torch.randint(
            0,
            2,
            (self.param_dim * self.k,),
            generator=self._generator,
            device=device,
            dtype=torch.int8,
        )
        signs = signs.mul_(2).sub_(1).to(dtype=dtype or torch.float32)
        values = signs * (1.0 / math.sqrt(self.k))

        indices = torch.stack([rows.reshape(-1), cols], dim=0)
        matrix = torch.sparse_coo_tensor(
            indices,
            values,
            (self.target_dim, self.param_dim),
            dtype=dtype or torch.float32,
            device=device,
        )
        return matrix.coalesce()

    def project_vector(self, vector: Tensor) -> Tensor:
        """
        Apply the projection to a 1D parameter vector.
        """
        if vector.dim() != 1:
            raise ValueError("Expected a 1D parameter vector.")
        if vector.numel() != self.param_dim:
            raise ValueError(f"Vector length {vector.numel()} does not match param_dim={self.param_dim}.")
        if vector.device != self.matrix.device or vector.dtype != self.matrix.dtype:
            vector = vector.to(device=self.matrix.device, dtype=self.matrix.dtype)
        return torch.sparse.mm(self.matrix, vector.unsqueeze(1)).squeeze(1)

    def project_parameters(self, net: torch.nn.Module) -> Tensor:
        """
        Flatten network parameters and project them.
        """
        flat_params = parameters_to_vector(net.parameters())
        return self.project_vector(flat_params)

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SparseJLProjector":
        """
        Return a copy of the projector moved to the requested device/dtype.
        """
        matrix = self.matrix
        if device is not None or dtype is not None:
            matrix = matrix.to(device=device or matrix.device, dtype=dtype or matrix.dtype)
        return self.__class__.from_components(
            param_dim=self.param_dim,
            target_dim=self.target_dim,
            k=self.k,
            matrix=matrix,
            seed=self.seed,
            generator_seed=self._generator_seed,
        )

    def state_dict(self) -> dict:
        """
        Minimal state for serialization.
        """
        return {
            "param_dim": self.param_dim,
            "target_dim": self.target_dim,
            "k": self.k,
            "seed": self.seed,
            "generator_seed": self._generator_seed,
            "matrix": self.matrix.cpu(),
        }

    @classmethod
    def from_components(
        cls,
        param_dim: int,
        target_dim: int,
        k: int,
        matrix: torch.Tensor,
        seed: Optional[int] = None,
        generator_seed: Optional[int] = None,
    ) -> "SparseJLProjector":
        obj = cls.__new__(cls)
        obj.param_dim = int(param_dim)
        obj.target_dim = int(target_dim)
        obj.k = int(k)
        obj.seed = seed
        obj._generator_seed = generator_seed
        obj._generator = None
        obj.matrix = matrix.coalesce()
        return obj

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
        map_location: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SparseJLProjector":
        matrix = state["matrix"]
        if map_location is not None or dtype is not None:
            matrix = matrix.to(device=map_location, dtype=dtype or matrix.dtype)
        return cls.from_components(
            param_dim=state["param_dim"],
            target_dim=state["target_dim"],
            k=state["k"],
            matrix=matrix,
            seed=state.get("seed"),
            generator_seed=state.get("generator_seed"),
        )


def make_sparse_jl_projector(
    param_dim: int,
    target_dim: int = DEFAULT_TARGET_DIM,
    k: int = DEFAULT_K,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = DEFAULT_SEED,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SparseJLProjector:
    """
    Convenience constructor mirroring measurement helpers.
    """
    return SparseJLProjector(
        param_dim=param_dim,
        target_dim=target_dim,
        k=k,
        generator=generator,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def save_projector(
    projector: SparseJLProjector,
    path: Optional[Union[str, Path]] = None,
    name: str = "projection_matrix.pt",
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Persist the projector locally so it can be re-used without wandb.
    """
    if path is None:
        path = _trajectory_base_dir(base_dir) / name
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), path)
    return path


def load_projector(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> SparseJLProjector:
    """
    Load a saved projector from disk.
    """
    state = torch.load(path, map_location=map_location)
    return SparseJLProjector.from_state_dict(state, map_location=map_location, dtype=dtype)


def project_params(
    net: torch.nn.Module,
    projector: SparseJLProjector,
) -> Tensor:
    """
    Flatten and project network parameters using a provided projector.
    """
    return projector.project_parameters(net)


def save_projection_vector(
    projection: Tensor,
    run_name: str,
    step: Union[int, str],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save a projected parameter vector for later comparison or artifact upload.
    """
    root = _trajectory_base_dir(base_dir) / run_name
    root.mkdir(parents=True, exist_ok=True)
    filename = f"step_{step}.pt"
    path = root / filename
    payload = {
        "projection": projection.detach().cpu(),
        "step": step,
        "run": run_name,
    }
    torch.save(payload, path)
    return path
