import os
import json
from pathlib import Path
import secrets
from typing import Any, Dict, Mapping

import numpy as np
import torch

try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when wandb missing
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


from utils.naming import sanitize_run_name_part as _sanitize_run_name_part
from utils.naming import compose_run_name as _compose_run_name


def is_wandb_available() -> bool:
    """Return whether the wandb library is importable."""

    return WANDB_AVAILABLE


def generate_run_id() -> str:
    """Generate a run identifier compatible with wandb directories."""

    if WANDB_AVAILABLE:
        # When a run already exists, reuse its id to keep checkpoint paths stable.
        if wandb.run:  # type: ignore[truthy-bool]
            return wandb.run.id  # type: ignore[return-value]
        return wandb.util.generate_id()  # type: ignore[attr-defined]

    return secrets.token_hex(4)


def init_wandb(args, step_to_start):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing experiment configuration

    step_to_start: int
        calculated start step
    
    Returns:
    --------
    wandb.run
        Initialized wandb run object
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is not available; call init_wandb only when enabled.")

    # Create a name for the run based on dataset, model, batch size, and learning rate
    run_name = _compose_run_name(args)

    tags = []
    if hasattr(args, 'wandb_tag') and args.wandb_tag is not None:
        tags.append(args.wandb_tag)

    # Handle continuation from existing wandb run
    if hasattr(args, 'cont_run_id') and args.cont_run_id is not None:
        # Resume existing run
        wandb_config = vars(args)
        wandb_config['fork_from'] = args.cont_run_id
        wandb_config['fork_step'] = step_to_start

        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "eoss"),
            mode=os.getenv("WANDB_MODE", "offline"),
            # id=args.cont_run_id,
            name=run_name,
            config=wandb_config,
            fork_from=f"{args.cont_run_id}?_step={step_to_start}",
            save_code=True,
            tags=tags if tags else None,
            notes=getattr(args, 'wandb_notes', None)
        )
        print(f"Resumed wandb run: {args.cont_run_id}")
    else:
        # Create new run
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "eoss"),
            mode=os.getenv("WANDB_MODE", "offline"),   # honours your env var
            name=run_name,
            config=vars(args),      # captures all CLI flags
            save_code=True,         # snapshot calling file & git commit
            tags=tags if tags else None,
            notes=getattr(args, 'wandb_notes', None)
        )
        print(f"Started new wandb run: {run.id}")
        

    # --- metric definitions so W&B plots look correct -------------
    # use global step (column 1) as the x-axis for everything else
    wandb.define_metric("step")
    for m in [
        "batch_loss","full_loss","batch_lambda_max","lambda_max","lambda_max_gn",
        "step_batch_lambda_max", "max_batch_lambda_max",
        "step_sharpness","batch_sharpn","batch_sharpn_gn","momentum_bs","grad_H_grad","batch_fisher_eigenval",
        "total_fisher_eigenval","sharpness_static","GNI","accuracy",
        "hessian_trace","second_moment_contraction",
        "param_distance","gradient_norm_squared","quadratic_loss_gn","proj_grad_ratio",
        "trajectory_proj_norm",
        "final_lambda_max","final_full_loss","final_accuracy"
    ]:
        wandb.define_metric(m, step_metric="step")

    # Define gradient projection metrics up to k=20 and residual so they plot correctly
    for i in range(1, 21):
        wandb.define_metric(f"grad_projection_{i:02d}", step_metric="step")
    wandb.define_metric("grad_projection_residual", step_metric="step")
    
    return run


def _maybe_scalar(value: Any) -> Any:
    """Convert 0-D torch/np tensors to scalars for logging."""

    if isinstance(value, torch.Tensor) and value.dim() == 0:
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def log_metrics(metrics: Mapping[str, Any]) -> None:
    """Log a dictionary of metrics to wandb after basic sanitisation."""

    if not WANDB_AVAILABLE:
        return

    if metrics is None:
        return

    metrics_dict = dict(metrics)

    all_eigenvalues = metrics_dict.pop('all_eigenvalues', None)
    grad_projections = metrics_dict.pop('grad_projections', None)

    filtered_metrics: Dict[str, Any] = {}

    for key, value in metrics_dict.items():
        value = _maybe_scalar(value)

        if value is None:
            continue

        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue

        filtered_metrics[key] = value

    if all_eigenvalues is not None:
        if hasattr(all_eigenvalues, 'tolist'):
            eigenvalues = all_eigenvalues.tolist()
        elif isinstance(all_eigenvalues, (list, tuple)):
            eigenvalues = list(all_eigenvalues)
        else:
            eigenvalues = [all_eigenvalues]

        for idx, eigenval in enumerate(eigenvalues, start=1):
            eigenval = _maybe_scalar(eigenval)
            if eigenval is None:
                continue
            if isinstance(eigenval, (float, np.floating)) and np.isnan(eigenval):
                continue
            filtered_metrics[f"eigenvalue_{idx}"] = eigenval

    if isinstance(grad_projections, Mapping):
        for key, value in grad_projections.items():
            value = _maybe_scalar(value)
            if value is None:
                continue
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            filtered_metrics[key] = value

    if filtered_metrics:
        wandb.log(filtered_metrics)


def save_checkpoint_wandb(model, optimizer, step, epoch, loss, run_id=None, save_every_n_steps=None):
    """
    Save model checkpoint to separate wandb_checkpoints directory organized by run ID.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to checkpoint
    optimizer : torch.optim.Optimizer  
        The optimizer to checkpoint
    step : int
        Current training step
    epoch : int
        Current epoch
    loss : float
        Current loss value
    run_id : str, optional
        Wandb run ID. If None, uses current run
    save_every_n_steps : int, optional
        Frequency of checkpointing. If provided, only saves if step % save_every_n_steps == 0
    
    Returns:
    --------
    Path or None
        Path to saved checkpoint if saved, None if skipped
    """
    if save_every_n_steps is not None and step % save_every_n_steps != 0:
        return None
        
    if run_id is None:
        if not WANDB_AVAILABLE or wandb.run is None:
            raise RuntimeError("run_id must be provided when wandb is disabled.")
        run_id = wandb.run.id
    
    # Create checkpoint directory separate from wandb runs
    wandb_dir = Path(os.environ.get("WANDB_DIR", "."))
    checkpoint_base_dir = wandb_dir / "wandb_checkpoints"
    checkpoint_dir = checkpoint_base_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint with step-based naming
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'run_id': run_id
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    
    # Update checkpoint metadata
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'checkpoints': []}
    
    # Add/update checkpoint info
    checkpoint_info = {
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'filename': f"checkpoint_step_{step}.pt",
        'path': str(checkpoint_path)
    }
    
    # Remove any existing checkpoint with same step
    metadata['checkpoints'] = [c for c in metadata['checkpoints'] if c['step'] != step]
    metadata['checkpoints'].append(checkpoint_info)
    
    # Sort by step for easier lookup
    metadata['checkpoints'].sort(key=lambda x: x['step'])
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return checkpoint_path


def _trajectory_dir(run_id: str) -> Path:
    """
    Return the directory where trajectory projections are stored for a run.
    Mirrors checkpoint directory structure for consistency.
    """
    wandb_dir = Path(os.environ.get("WANDB_DIR", "."))
    trajectory_base_dir = wandb_dir / "wandb_trajectories"
    return trajectory_base_dir / run_id


def save_trajectory_projection(projection: torch.Tensor, step: int, run_id: str = None, save_every_n_steps: int = None):
    """
    Save a parameter projection for trajectory comparison, analogous to checkpoint saving.

    Parameters
    ----------
    projection : torch.Tensor
        Projected parameter vector (will be moved to CPU before saving).
    step : int
        Training step.
    run_id : str, optional
        Wandb run ID. If None, uses current wandb run when available.
    save_every_n_steps : int, optional
        If provided, only save when step % save_every_n_steps == 0.

    Returns
    -------
    Path or None
        Path to saved projection, or None if skipped due to cadence.
    """
    if save_every_n_steps is not None and step % save_every_n_steps != 0:
        return None

    if run_id is None:
        if not WANDB_AVAILABLE or wandb.run is None:  # type: ignore[truthy-bool]
            raise RuntimeError("run_id must be provided when wandb is disabled.")
        run_id = wandb.run.id  # type: ignore[assignment]

    trajectory_dir = _trajectory_dir(run_id)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    projection_path = trajectory_dir / f"trajectory_step_{step}.pt"
    payload = {
        "projection": projection.detach().cpu(),
        "step": step,
        "run_id": run_id,
    }
    torch.save(payload, projection_path)

    metadata_path = trajectory_dir / "trajectory_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'trajectories': []}

    entry = {
        "step": step,
        "filename": projection_path.name,
        "path": str(projection_path),
    }
    metadata['trajectories'] = [c for c in metadata['trajectories'] if c['step'] != step]
    metadata['trajectories'].append(entry)
    metadata['trajectories'].sort(key=lambda x: x['step'])

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return projection_path


def find_closest_trajectory_projection(target_step: int, run_id: str = None, trajectory_dir: Path = None):
    """
    Find the trajectory projection with step closest to (but not exceeding) target_step.
    """
    if trajectory_dir is None:
        if run_id is None:
            if not WANDB_AVAILABLE or wandb.run is None:  # type: ignore[truthy-bool]
                raise RuntimeError("run_id must be provided when wandb is disabled.")
            run_id = wandb.run.id  # type: ignore[assignment]
        trajectory_dir = _trajectory_dir(run_id)
        if not trajectory_dir.exists():
            return None
    else:
        trajectory_dir = Path(trajectory_dir)

    metadata_path = trajectory_dir / "trajectory_metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    trajectories = metadata.get('trajectories', [])
    if not trajectories:
        return None

    candidates = [c for c in trajectories if c['step'] <= target_step]
    if not candidates:
        return None

    return max(candidates, key=lambda x: x['step'])


def load_trajectory_projection(projection_info, map_location=None):
    """
    Load a saved trajectory projection payload.
    """
    projection_path = Path(projection_info['path'])
    if not projection_path.exists():
        raise FileNotFoundError(f"Trajectory projection not found: {projection_path}")
    payload = torch.load(projection_path, map_location=map_location)
    return payload


def get_trajectory_dir_for_run(run_id: str):
    """
    Get trajectory projection directory for a specific run ID.
    """
    trajectory_dir = _trajectory_dir(run_id)
    if trajectory_dir.exists():
        return trajectory_dir
    return None


def find_closest_checkpoint_wandb(target_step, run_id=None, checkpoint_dir=None):
    """
    Find the checkpoint with step closest to (but not exceeding) target_step.
    
    Parameters:
    -----------
    target_step : int
        Target step to restore from
    run_id : str, optional
        Wandb run ID. If None, uses current run
    checkpoint_dir : Path, optional
        Checkpoint directory. If None, uses wandb_checkpoints/{run_id}
        
    Returns:
    --------
    dict or None
        Checkpoint info dict with keys: step, epoch, loss, filename, path
        None if no suitable checkpoint found
    """
    if checkpoint_dir is None:
        if run_id is None:
            if not WANDB_AVAILABLE or wandb.run is None:
                raise RuntimeError("run_id must be provided when wandb is disabled.")
            run_id = wandb.run.id
        
        checkpoint_base_dir = Path("wandb_checkpoints")
        checkpoint_dir = checkpoint_base_dir / run_id
        
        if not checkpoint_dir.exists():
            return None
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    checkpoints = metadata.get('checkpoints', [])
    if not checkpoints:
        return None
    
    # Find closest checkpoint not exceeding target_step
    suitable_checkpoints = [c for c in checkpoints if c['step'] <= target_step]
    
    if not suitable_checkpoints:
        return None
    
    # Return checkpoint with highest step that doesn't exceed target
    closest_checkpoint = max(suitable_checkpoints, key=lambda x: x['step'])
    
    return closest_checkpoint


def load_checkpoint_wandb(checkpoint_info, model, optimizer=None):
    """
    Load model and optimizer from wandb checkpoint.
    
    Parameters:
    -----------
    checkpoint_info : dict
        Checkpoint info from find_closest_checkpoint_wandb
    model : torch.nn.Module
        Model to load state into
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into
        
    Returns:
    --------
    dict
        Loaded checkpoint data with keys: step, epoch, loss, run_id
    """
    checkpoint_path = Path(checkpoint_info['path'])
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    device = next(model.parameters()).device
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint_data['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    return {
        'step': checkpoint_data['step'],
        'epoch': checkpoint_data['epoch'], 
        'loss': checkpoint_data['loss'],
        'run_id': checkpoint_data.get('run_id')
    }


def get_checkpoint_dir_for_run(run_id):
    """
    Get checkpoint directory for a specific run ID.
    
    Parameters:
    -----------
    run_id : str
        Wandb run ID
        
    Returns:
    --------
    Path or None
        Path to checkpoint directory if found, None otherwise
    """
    wandb_dir = Path(os.environ.get("WANDB_DIR", "."))
    checkpoint_base_dir = wandb_dir / "wandb_checkpoints"
    checkpoint_dir = checkpoint_base_dir / run_id
    
    if checkpoint_dir.exists():
        return checkpoint_dir
    
    return None
