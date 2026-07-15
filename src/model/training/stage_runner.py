import csv
import os
import re
import glob
import datetime
import logging
import hashlib
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar

# comes from your project
from model.architecture import (
    h5py_dset,
    Dataloader,
    full_vector_loss,
    angle_loss,
    mag_p_error,
    inclination_error,
    declination_error,
)
#Loading helper functions for loss and error calculation.
#Also includes the data handling classes h5py_dset and Dataloader.

# Streaming HDF5 loaders are not part of this release.
H5StreamingDataset = None
H5ShardIndex = None


LOGGER = logging.getLogger(__name__)


def _tensor_summary(tensor: torch.Tensor) -> dict[str, float | str]:
    tensor_value = tensor.detach()
    if tensor_value.numel() == 0:
        return {"dtype": str(tensor_value.dtype), "min": float("nan"), "max": float("nan")}
    return {
        "dtype": str(tensor_value.dtype),
        "min": float(tensor_value.min().item()),
        "max": float(tensor_value.max().item()),
    }


def _write_step_stats(
    writer: csv.DictWriter,
    *,
    epoch_index: int,
    batch_index: int,
    global_step: int,
    loss_value: float,
    output_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> None:
    output_summary = _tensor_summary(output_tensor)
    target_summary = _tensor_summary(target_tensor)
    writer.writerow(
        {
            "epoch": epoch_index,
            "batch": batch_index,
            "step": global_step,
            "loss": loss_value,
            "output_dtype": output_summary["dtype"],
            "output_min": output_summary["min"],
            "output_max": output_summary["max"],
            "target_dtype": target_summary["dtype"],
            "target_min": target_summary["min"],
            "target_max": target_summary["max"],
        }
    )


def _build_best_checkpoint_path(checkpoint_prefix: str) -> Path:
    """Return the checkpoint path that stores the lowest validation loss."""

    checkpoint_prefix_path = Path(checkpoint_prefix).expanduser()
    checkpoint_directory = checkpoint_prefix_path.parent
    if checkpoint_directory and not checkpoint_directory.exists():
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_file_base_name = checkpoint_prefix_path.name
    return checkpoint_directory / f"{checkpoint_file_base_name}_min_loss.pth"


def _save_state_dict_with_fallback(state_dict: dict, target_path: Path) -> Path:
    """Persist ``state_dict`` to ``target_path`` with a hashed fallback name if needed."""

    resolved_primary_path = target_path
    torch_primary_path = resolve_torch_io_path(resolved_primary_path)
    try:
        torch.save(state_dict, torch_primary_path)
        return resolved_primary_path
    except (OSError, RuntimeError) as primary_exc:
        LOGGER.warning(
            "Primary checkpoint save path failed: %s :: %s",
            resolved_primary_path,
            primary_exc,
        )
        fallback_stem = hashlib.sha256(str(resolved_primary_path).encode("utf-8")).hexdigest()
        fallback_path = resolved_primary_path.with_name(f"{fallback_stem}.pth")
        fallback_directory = fallback_path.parent
        if fallback_directory and not fallback_directory.exists():
            fallback_directory.mkdir(parents=True, exist_ok=True)
        torch_fallback_path = resolve_torch_io_path(fallback_path)
        try:
            torch.save(state_dict, torch_fallback_path)
        except (OSError, RuntimeError) as fallback_exc:
            raise OSError(
                "Unable to save checkpoint to either the primary or fallback location."
                f" Primary: '{resolved_primary_path}', Fallback: '{fallback_path}'."
            ) from fallback_exc
        warnings.warn(
            "Saved checkpoint to hashed fallback filename due to primary path failure.",
            RuntimeWarning,
            stacklevel=2,
        )
        LOGGER.warning("Checkpoint stored at fallback path: %s", fallback_path)
        return fallback_path


def resolve_torch_io_path(path: Path | str) -> str:
    """Return a path string that Torch can reliably open on all platforms."""

    candidate_path = Path(path).expanduser()
    if os.name == "nt":
        # resolve(strict=False) keeps non-existent paths while producing an absolute path
        absolute_path = candidate_path.resolve(strict=False)
        path_str = str(absolute_path)
        if path_str.startswith("\\\\?\\"):
            return path_str
        if path_str.startswith("\\\\"):
            # UNC path: convert "\\server" -> "\\?\\UNC\\server"
            unc_prefix = "\\\\?\\UNC\\"
            path_without_leading_backslashes = path_str.lstrip("\\")
            return f"{unc_prefix}{path_without_leading_backslashes}"
        return f"\\\\?\\{path_str}"
    return str(candidate_path)


def _max_readable_index(dset) -> int:
    low = 0
    high = len(dset) - 1
    best = -1
    while low <= high:
        mid = (low + high) // 2
        try:
            _ = dset[mid]
            best = mid
            low = mid + 1
        except OSError:
            high = mid - 1
    return best


# ---------- data ----------

def load_xy_from_h5(
    folder_path: str,
    w_n_data: int,
    device: torch.device | None,
    *,
    seed: int | None = None,
    shuffle: bool = True,
    move_to_device: bool = False,
):
    """
    Load 'I' and 'm' tensors from a dataset H5 file, optionally moving to device.
    Args:
        folder_path: Directory containing dataset.h5 or path to dataset.h5.
        w_n_data: Number of samples to load.
        device: Torch device to move tensors to.
        seed: Optional random seed for deterministic sampling.
        shuffle: Whether to randomize the selected samples.
        move_to_device: If True, move tensors to the provided device immediately.
    Returns:
        x: Input tensor.
        y: Target tensor.
        n: Number of samples loaded.
    """
    candidate_path = os.path.normpath(folder_path)
    if os.path.isdir(candidate_path):
        h5 = os.path.join(candidate_path, "dataset.h5")
    else:
        h5 = candidate_path
    h5 = os.path.normpath(h5)
    LOGGER.info("Resolving dataset path to '%s'", h5)
    if not os.path.isfile(h5):
        raise FileNotFoundError(f"H5 dataset not found at '{h5}'")

    x = h5py_dset(h5, 'I') #Extract input data 'I' (Bfield images, shape: (N, H, W, C)), N defn number of samples, H height, W width, C channels (3 for vector ml)
    y = h5py_dset(h5, 'm') #Extract target data 'm' (Dipole moment vectors, shape: (N, 3)), N defn number of samples, 3 for (mx, my, mz)

    total_samples = min(len(x), len(y))
    if total_samples > 0:
        try:
            _ = x[total_samples - 1]
        except OSError:
            max_I = _max_readable_index(x)
            if max_I < 0:
                raise
            total_samples = min(total_samples, max_I + 1)
            LOGGER.warning(
                "Detected unreadable tail in H5 'I'; trimming total_samples to %d",
                total_samples,
            )
    n = min(w_n_data, total_samples) #Determine number of samples to load, limited by w_n_data and available data
    if n <= 0:
        return torch.empty((0, 3, 0, 0)), torch.empty((0, 3)), 0

    if shuffle:
        rng = np.random.default_rng(seed)
        if n < total_samples:
            indices = rng.choice(total_samples, size=n, replace=False)
        else:
            indices = np.arange(total_samples)
            rng.shuffle(indices)
    else:
        indices = slice(0, n)

    if isinstance(indices, np.ndarray):
        # h5py requires fancy indices to be in increasing order; restore shuffle afterward.
        sorter = np.argsort(indices)
        sorted_indices = indices[sorter]
        x = x[sorted_indices].clone().detach().float().permute(0, 3, 1, 2)
        y = y[sorted_indices].clone().detach().float()
        if sorter.size and not np.all(sorter == np.arange(sorter.size)):
            unsorter = np.argsort(sorter)
            unsorter_t = torch.as_tensor(unsorter, dtype=torch.long)
            x = x[unsorter_t]
            y = y[unsorter_t]
    else:
        x = x[indices].clone().detach().float().permute(0, 3, 1, 2) #Load samples, convert to float, permute to (N, C, H, W)
        y = y[indices].clone().detach().float() #Load samples, convert to float

    if device is not None:
        if move_to_device and device.type != "cpu":
            print(f"Moving data to device: {device}") #Print device info
            x = x.to(device)
            y = y.to(device)
        elif device.type == "cpu":
            print("Using CPU, this may be slow.") #Warn if using CPU
        else:
            print(f"Keeping data on CPU; batches will move to device: {device}")
    return x, y, n #Return input tensor, target tensor, and number of samples

def make_loaders(
    x,
    y,
    batch_size: int = 32,
    train_val_split: float = 0.8,
    *,
    seed: int | None = None,
):
    """
    Return train and val dataloaders using your Dataloader helper.
    Args:
        x: Input tensor.
        y: Target tensor.
        batch_size: Batch size for loaders.
        train_val_split: Fraction of data for training.
        seed: Optional seed for deterministic splitting/shuffling.
    Returns:
        train_loader, val_loader: DataLoader objects.
    """
    loaders = Dataloader(x, y) #Create Dataloader instance with input and target tensors
    dataset = loaders._dataset
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = None
    shuffle_generator = None
    if seed is not None:
        split_generator = torch.Generator()
        split_generator.manual_seed(seed)
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(seed + 1)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    return (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=shuffle_generator,
        ),
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    )


# ---------- model ----------

# The legacy CNN builders (SimpleCNN and variants) are not part of this
# release; the published model is AttentionDipoleModel (--dipolenet).


# ---------- training ----------

def train_one_stage(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    criterion = full_vector_loss,
    checkpoint_prefix: str = "checkpoint_minval",
    log_every: int = 50,
    show_progress: bool = True,
    step_stats_csv_path: str | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    clip_grad_norm: float | None = None,
):
    """
    Train for `epochs`. Save weights when validation loss hits a new min.
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of epochs.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (optional).
        criterion: Loss function.
        checkpoint_prefix: Prefix for checkpoint file.
        log_every: Logging frequency.
        show_progress: Show progress bar.
        clip_grad_norm: Max L2 norm for gradient clipping (set <=0 or None to disable).
    Returns:
        metrics: Dict with training metrics.
    """
    LOGGER.info("[%s] entering train_one_stage", getattr(model, "name", "trial"))

    loss_tr, mag_tr, inc_tr, dec_tr = [], [], [], [] #Lists to store training metrics over epochs
    val_loss_history, val_mag_history, val_inc_history, val_dec_history = [], [], [], []
    dec_valid_fraction_tr: list[float] = []
    dec_valid_fraction_val: list[float] = []
    best_val = float("inf") #Initialize best validation loss to infinity
    best_ckpt = None #Variable to store best checkpoint path
    best_checkpoint_path_template = _build_best_checkpoint_path(checkpoint_prefix)
    early_stop_patience = early_stopping_patience if early_stopping_patience and early_stopping_patience > 0 else None
    epochs_since_improvement = 0
    early_stop_epoch = None

    bar_ctx = alive_bar(epochs, title=f"Training on {checkpoint_prefix}") if show_progress else None #Progress bar context if enabled
    progress_bar_update_func = bar_ctx.__enter__() if bar_ctx else None #Callable returned from the progress bar context

    if hasattr(model, 'device'):
        if model.device == torch.device('cpu'):
            print("Using CPU for training, this may be slow.")
        target_device = model.device
    else:
        raise AttributeError("Model must have a 'device' attribute indicating its torch.device.")

    gradient_history: list[dict[str, float]] = []
    step_stats_handle = None
    step_stats_writer = None
    global_step = 0

    if step_stats_csv_path:
        step_stats_path = Path(step_stats_csv_path)
        step_stats_path.parent.mkdir(parents=True, exist_ok=True)
        step_stats_handle = step_stats_path.open("w", encoding="utf-8", newline="")
        step_stats_writer = csv.DictWriter(
            step_stats_handle,
            fieldnames=[
                "epoch",
                "batch",
                "step",
                "loss",
                "output_dtype",
                "output_min",
                "output_max",
                "target_dtype",
                "target_min",
                "target_max",
            ],
        )
        step_stats_writer.writeheader()

    clip_parameters = None
    if clip_grad_norm is not None and clip_grad_norm > 0:
        clip_parameters = [p for group in optimizer.param_groups for p in group["params"]]

    def _run_training_step(
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        if isinstance(optimizer, torch.optim.LBFGS):
            step_state: dict[str, torch.Tensor | float] = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_inputs)
                loss_value = criterion(outputs, batch_targets)
                output_gradients = torch.autograd.grad(
                    loss_value,
                    outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                loss_value.backward()
                if clip_parameters is not None:
                    torch.nn.utils.clip_grad_norm_(clip_parameters, max_norm=clip_grad_norm)
                step_state["loss"] = float(loss_value.detach().item())
                step_state["outputs"] = outputs.detach()
                step_state["output_gradients"] = output_gradients.detach()
                return loss_value

            optimizer.step(closure)
            if "loss" not in step_state:
                raise RuntimeError("LBFGS step did not populate training-step state.")
            return (
                float(step_state["loss"]),
                step_state["outputs"],
                step_state["output_gradients"],
            )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_inputs)
        loss_value = criterion(outputs, batch_targets)
        output_gradients = torch.autograd.grad(
            loss_value,
            outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        loss_value.backward()
        if clip_parameters is not None:
            torch.nn.utils.clip_grad_norm_(clip_parameters, max_norm=clip_grad_norm)
        optimizer.step()
        return float(loss_value.detach().item()), outputs.detach(), output_gradients.detach()

    try:
        for epoch in range(epochs):
            model.train() #Set model to train
            train_loss_sum = 0.0
            train_magnitude_error_sum = 0.0
            train_inclination_error_sum = 0.0
            train_declination_error_sum = 0.0
            train_total_samples = 0
            train_valid_declination_samples = 0.0
            component_gradient_sum = torch.zeros(3, device=target_device)
            component_gradient_batches = 0

            if epoch % log_every == 0: #Log learning rate at specified intervals
                lr = optimizer.param_groups[0]['lr'] #Get current learning rate from optimizer
                print(f"Epoch {epoch} | LR {lr:.6f}") 

            # Training loop
            for batch_index, (bx, by) in enumerate(train_loader): #For batchx, batchy in training loader
                # Move batch data to the model's declared device so inputs and targets align
                bx = bx.to(target_device)
                by = by.to(target_device)
                loss_value, out, output_gradients = _run_training_step(bx, by)
                component_gradient_sum += output_gradients.abs().mean(dim=0)
                component_gradient_batches += 1
                if step_stats_writer:
                    _write_step_stats(
                        step_stats_writer,
                        epoch_index=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        loss_value=loss_value,
                        output_tensor=out,
                        target_tensor=by,
                    )
                global_step += 1

                batch_size = by.size(0)
                train_loss_sum += loss_value * batch_size
                train_magnitude_error_sum += mag_p_error(out, by).item() * batch_size
                train_inclination_error_sum += inclination_error(out, by).item() * batch_size
                declination_batch_error, declination_valid_fraction = declination_error(
                    out,
                    by,
                    return_fraction=True,
                )
                valid_declination_samples = declination_valid_fraction.item() * batch_size
                train_declination_error_sum += declination_batch_error.item() * valid_declination_samples
                train_total_samples += batch_size
                train_valid_declination_samples += valid_declination_samples

            if scheduler is not None:
                scheduler.step()  # Step the learning rate scheduler if provided

            # Store average metrics for this epoch
            train_loss_average = train_loss_sum / train_total_samples if train_total_samples else 0.0
            train_magnitude_error_average = (
                train_magnitude_error_sum / train_total_samples if train_total_samples else 0.0
            )
            train_inclination_error_average = (
                train_inclination_error_sum / train_total_samples if train_total_samples else 0.0
            )
            train_declination_error_average = (
                train_declination_error_sum / train_valid_declination_samples
                if train_valid_declination_samples
                else 0.0
            )
            loss_tr.append(train_loss_average)  # Average training loss
            mag_tr.append(train_magnitude_error_average)    # Average magnitude error
            inc_tr.append(train_inclination_error_average)  # Average inclination error
            dec_tr.append(train_declination_error_average)  # Average declination error
            train_declination_valid_fraction = (
                train_valid_declination_samples / train_total_samples if train_total_samples else 0.0
            )
            dec_valid_fraction_tr.append(train_declination_valid_fraction)
            if component_gradient_batches:
                grad_components = (
                    component_gradient_sum / component_gradient_batches
                ).detach().cpu().tolist()
            else:
                grad_components = [0.0, 0.0, 0.0]

            gradient_metrics = {
                "Mx": grad_components[0],
                "My": grad_components[1],
                "Mz": grad_components[2],
            }
            gradient_history.append(gradient_metrics)

            # Validation loop
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for validation
                val_loss_sum = 0.0
                val_magnitude_error_sum = 0.0
                val_inclination_error_sum = 0.0
                val_declination_error_sum = 0.0
                val_total_samples = 0
                val_valid_declination_samples = 0.0
                for vx, vy in val_loader:
                    vx = vx.to(target_device)
                    vy = vy.to(target_device)  # Move validation batch to the model device
                    vout = model(vx)  # Forward pass for validation batch
                    batch_size_val = vy.size(0)
                    val_loss_sum += criterion(vout, vy).item() * batch_size_val
                    val_magnitude_error_sum += mag_p_error(vout, vy).item() * batch_size_val
                    val_inclination_error_sum += inclination_error(vout, vy).item() * batch_size_val
                    dec_error_val, dec_valid_fraction_val_batch = declination_error(
                        vout,
                        vy,
                        return_fraction=True,
                    )
                    valid_declination_samples_val = dec_valid_fraction_val_batch.item() * batch_size_val
                    val_declination_error_sum += dec_error_val.item() * valid_declination_samples_val
                    val_total_samples += batch_size_val
                    val_valid_declination_samples += valid_declination_samples_val

                v_loss = val_loss_sum / val_total_samples if val_total_samples else 0.0
                v_mag = val_magnitude_error_sum / val_total_samples if val_total_samples else 0.0
                v_inc = val_inclination_error_sum / val_total_samples if val_total_samples else 0.0
                v_dec = (
                    val_declination_error_sum / val_valid_declination_samples
                    if val_valid_declination_samples
                    else 0.0
                )
                validation_declination_valid_fraction = (
                    val_valid_declination_samples / val_total_samples if val_total_samples else 0.0
                )
                val_loss_history.append(v_loss)
                val_mag_history.append(v_mag)
                val_inc_history.append(v_inc)
                val_dec_history.append(v_dec)
                dec_valid_fraction_val.append(validation_declination_valid_fraction)

            if not np.isfinite(v_loss):
                raise ValueError(f"Validation loss is not finite at epoch {epoch}: {v_loss}")

            # Save checkpoint if validation loss improves
            if v_loss < best_val - early_stopping_min_delta:
                best_val = v_loss  # Update best validation loss
                best_ckpt_path = best_checkpoint_path_template
                checkpoint_directory = best_ckpt_path.parent
                if checkpoint_directory and not checkpoint_directory.exists():
                    checkpoint_directory.mkdir(parents=True, exist_ok=True)
                try:
                    saved_checkpoint_path = _save_state_dict_with_fallback(
                        model.state_dict(),
                        best_ckpt_path,
                    )
                except OSError as exc:
                    raise OSError(
                        "Unable to save checkpoint to"
                        f" '{best_ckpt_path}'. Ensure the directory exists and is writable."
                    ) from exc
                best_ckpt = str(saved_checkpoint_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Print validation metrics at specified logging interval
            if epoch % log_every == 0:
                # Angle printed from last batch for continuity with your script
                ang = angle_loss(vout, vy).item()
                print(
                    f"Epoch {epoch} | Val Loss {v_loss:.4e} | "
                    f"Mag %Err {v_mag*100:.2f}% | "
                    f"Incl {np.degrees(v_inc):.2f} deg | "
                    f"Decl {np.degrees(v_dec):.2f} deg "
                    f"(valid {validation_declination_valid_fraction*100:.1f}% of samples)"
                )
                print(
                    "    Avg |∂L/∂M| per component:",
                    ", ".join(
                        f"{comp}={gradient_metrics[comp]:.3e}" for comp in ("Mx", "My", "Mz")
                    ),
                )

            if progress_bar_update_func:
                progress_bar_update_func()

            if early_stop_patience is not None and epochs_since_improvement >= early_stop_patience:
                early_stop_epoch = epoch
                print(
                    "Early stopping triggered after",
                    epochs_since_improvement,
                    "epochs without improvement.",
                )
                break

    finally:
        if bar_ctx:
            bar_ctx.__exit__(None, None, None)
        if step_stats_handle:
            step_stats_handle.close()

    metrics = {
        "train_loss": loss_tr,
        "train_mag_p": mag_tr,
        "train_incl": inc_tr,
        "train_decl": dec_tr,
        "train_decl_valid_fraction": dec_valid_fraction_tr,
        "val_decl_valid_fraction": dec_valid_fraction_val,
        "val_loss": val_loss_history,
        "val_mag_p": val_mag_history,
        "val_incl": val_inc_history,
        "val_decl": val_dec_history,
        "train_gradient_components": gradient_history,
        "best_val_loss": best_val,
        "best_ckpt": best_ckpt,
        "early_stop_epoch": early_stop_epoch,
    }
    LOGGER.info(
        "[%s] exiting train_one_stage after %d epochs (steps per epoch ~%d)",
        getattr(model, "name", "trial"),
        epochs,
        len(train_loader),
    )
    return metrics

# ---------- eval and reporting ----------

def plot_training(metrics: dict, out_png: str):
    """
    Plot raw curves for loss, mag, incl, decl to out_png.
    Args:
        metrics: Dict with training metrics.
        out_png: Output PNG file path.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    loss = np.array(metrics["train_loss"])
    mag  = np.array(metrics["train_mag_p"])
    inc  = np.degrees(np.array(metrics["train_incl"]))
    dec  = np.degrees(np.array(metrics["train_decl"]))

    val_loss = np.array(metrics.get("val_loss", []))
    val_mag  = np.array(metrics.get("val_mag_p", []))
    val_inc  = np.degrees(np.array(metrics.get("val_incl", [])))
    val_dec  = np.degrees(np.array(metrics.get("val_decl", [])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Loss ---
    axes[0, 0].plot(loss, label="Train")
    if val_loss.size:
        axes[0, 0].plot(val_loss, label="Validation")
        axes[0, 0].legend()
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    # --- Magnitude % Error ---
    axes[0, 1].plot(mag, label="Train")
    if val_mag.size:
        axes[0, 1].plot(val_mag, label="Validation")
        axes[0, 1].legend()
    axes[0, 1].set_title("Mag % Error")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("% Error")

    # --- Inclination Error ---
    axes[1, 0].plot(inc, label="Train")
    if val_inc.size:
        axes[1, 0].plot(val_inc, label="Validation")
        axes[1, 0].legend()
    axes[1, 0].set_title("Inclination Error")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Deg. Error")

    # --- Declination Error ---
    axes[1, 1].plot(dec, label="Train")
    if val_dec.size:
        axes[1, 1].plot(val_dec, label="Validation")
        axes[1, 1].legend()
    axes[1, 1].set_title("Declination Error")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Deg. Error")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model(model: torch.nn.Module, out_path: str):
    """
    Save model weights to file.
    Args:
        model: Model to save.
        out_path: Output file path.
    """
    destination_path = Path(out_path).expanduser()
    destination_directory = destination_path.parent
    if destination_directory and not destination_directory.exists():
        destination_directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), resolve_torch_io_path(destination_path))
