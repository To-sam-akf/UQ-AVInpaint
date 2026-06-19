"""Train deterministic convolutional Mel autoencoder.

Phase 1 (warmup): L1 reconstruction only.
Phase 2: L1 + time-gradient + random-boundary loss.

Usage:
    python train_mel_autoencoder.py --name MelAE --data_root /path/to/data \
        --batch_size 16 --nepochs 50 --ae_latent_dim 8
    # or via main.py:
    python main.py train-mel-ae -- --name MelAE --data_root /path/to/data
"""

import os
import sys
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import Options_inpainting
from Data_loaders.uq_av_loader import get_uq_av_data_loaders
from Models.Mel_Autoencoder import MelAEModel
from utils.viai_a_metrics import (
    compute_viai_a_metrics,
    write_mel_images,
    compose_inpainted_mel,
    _as_bchw,
)

# ---------------------------------------------------------------------------
hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def _configure_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "MelAE"
    if not _arg_was_passed("--train_split_name"):
        hparams.train_split_name = "train_av_split.txt"
    if not _arg_was_passed("--val_split_name"):
        hparams.val_split_name = "val_av_split.txt"
    if not _arg_was_passed("--checkpoint_dir"):
        hparams.checkpoint_dir = os.path.join(
            getattr(hparams, "checkpoints_dir", "./checkpoints"), "mel_ae",
        )
    if not _arg_was_passed("--log_event_path"):
        hparams.log_event_path = os.path.join(hparams.checkpoint_dir, "events")


def _run_phase(model, data_loader, phase, global_step, writer, global_epoch):
    train = phase == "train"
    totals = {"loss_total": 0.0, "loss_l1": 0.0, "loss_grad": 0.0,
               "loss_boundary": 0.0, "psnr_full": 0.0, "psnr_missing": 0.0}
    ssim_full_sum = 0.0
    sample_count = 0
    ssim_sample_count = 0
    batch_count = 0
    stop_training = False

    progress = tqdm(
        data_loader, desc=f"[MelAE {phase}] epoch={global_epoch + 1}",
        unit="batch", dynamic_ncols=True,
    )
    for batch in progress:
        iter_start = time.time()

        # Use clean mel_target (no mask injection for AE training)
        mel = batch["mel_target"].float().to(device)
        model.set_input(mel)

        if train:
            model.optimize_parameters(global_step)
            global_step += 1
        else:
            model.test(global_step=global_step)

        model.get_loss_items()

        # Compute PSNR/SSIM metrics on full Mel
        metrics = compute_viai_a_metrics(
            model.mel_recon, model.mel_target,
            torch.zeros_like(model.mel_target[:, :1, :, :]),  # no missing region
            compute_ssim=not train or (global_step % max(1, int(getattr(hparams, "metric_freq", 100))) == 0),
        )

        totals["loss_total"] += model.loss_total_item
        totals["loss_l1"] += model.loss_l1_item
        totals["loss_grad"] += model.loss_grad_item
        totals["loss_boundary"] += model.loss_boundary_item
        totals["psnr_full"] += metrics["psnr_full_sum"]
        totals["psnr_missing"] += metrics["psnr_missing_sum"]
        sample_count += metrics["num_samples"]
        if metrics["ssim_full_sum"] is not None:
            ssim_full_sum += metrics["ssim_full_sum"]
            ssim_sample_count += metrics["num_samples"]

        progress.set_postfix(
            step=global_step,
            loss=f"{model.loss_total_item:.4f}",
            l1=f"{model.loss_l1_item:.4f}",
            psnr=f"{metrics['psnr_full']:.2f}",
            ssim=f"{metrics.get('ssim_full', 'N/A')}",
        )

        if train:
            model.TF_writer(writer, global_step, prefix=phase)

            if (global_step > 0
                    and global_step % getattr(hparams, "tb_image_freq", 500) == 0):
                write_mel_images(
                    writer, phase, global_step,
                    model.mel_target, model.mel_recon, model.mel_target,
                    max_items=getattr(hparams, "tb_image_count", 4),
                )

            if (global_step > 0
                    and global_step % getattr(hparams, "print_freq", 100) == 0):
                elapsed = (time.time() - iter_start) / max(1, hparams.batch_size)
                tqdm.write(
                    f"[MelAE train] step={global_step} "
                    f"loss={model.loss_total_item:.6f} "
                    f"l1={model.loss_l1_item:.6f} "
                    f"grad={model.loss_grad_item:.6f} "
                    f"bdy={model.loss_boundary_item:.6f} "
                    f"psnr={metrics['psnr_full']:.3f} "
                    f"warmup={'on' if global_step < model.warmup_steps else 'off'} "
                    f"time={elapsed:.4f}s/sample"
                )

            if (global_step > 0
                    and global_step % getattr(hparams, "checkpoint_interval", 1000) == 0):
                model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)

        batch_count += 1

        if (train and getattr(hparams, "max_train_steps", None) is not None
                and global_step >= hparams.max_train_steps):
            tqdm.write(f"Reached max_train_steps={hparams.max_train_steps}")
            stop_training = True
            break

    if batch_count == 0:
        return global_step, stop_training, None

    averages = {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_l1": totals["loss_l1"] / batch_count,
        "loss_grad": totals["loss_grad"] / batch_count,
        "loss_boundary": totals["loss_boundary"] / batch_count,
        "psnr_full": totals["psnr_full"] / max(1, sample_count),
        "ssim_full": (ssim_full_sum / max(1, ssim_sample_count)
                       if ssim_sample_count > 0 else None),
    }
    tqdm.write(
        f"[MelAE {phase}] "
        f"loss={averages['loss_total']:.6f} "
        f"l1={averages['loss_l1']:.6f} "
        f"grad={averages['loss_grad']:.6f} "
        f"bdy={averages['loss_boundary']:.6f} "
        f"psnr={averages['psnr_full']:.3f} "
        f"ssim={averages.get('ssim_full', 'N/A')}"
    )
    return global_step, stop_training, averages


def train_loop(model, data_loaders, writer, start_step=0, start_epoch=0):
    global_step = start_step
    global_epoch = start_epoch
    while global_epoch < hparams.nepochs:
        if "train" in data_loaders:
            global_step, stop, _ = _run_phase(
                model, data_loaders["train"], "train",
                global_step, writer, global_epoch,
            )
            if stop:
                break

        if "val" in data_loaders:
            with torch.no_grad():
                _run_phase(
                    model, data_loaders["val"], "val",
                    global_step, writer, global_epoch,
                )

        global_epoch += 1
        print(f"[MelAE] epoch={global_epoch} lr={model.current_lr}")

    # Compute latent statistics on training set
    print("[MelAE] Computing latent normalisation statistics on training set ...")
    stats = model.compute_latent_stats(data_loaders.get("train"), max_batches=100)
    print(f"[MelAE] latent mean: {stats['mean'].tolist()}")
    print(f"[MelAE] latent std:  {stats['std'].tolist()}")

    # Final save with latent stats
    model.save_checkpoint(
        global_step, global_epoch, hparams.checkpoint_dir,
        latent_stats=stats,
    )


def main():
    _configure_defaults()

    split_names = {
        "train": hparams.train_split_name,
        "val": hparams.val_split_name,
    }
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)

    print(f"[MelAE] data_root={hparams.data_root}")
    print(f"[MelAE] checkpoint_dir={hparams.checkpoint_dir}")
    print(f"[MelAE] latent_dim={getattr(hparams, 'ae_latent_dim', 8)}")

    data_loaders = get_uq_av_data_loaders(
        hparams.data_root, split_names,
        phases=("train", "val"),
        batch_size=hparams.batch_size,
        num_workers=getattr(hparams, "num_workers", 4),
        pin_memory=getattr(hparams, "pin_memory", True),
    )

    model = MelAEModel(hparams, device=device)

    global_step = 0
    global_epoch = 0
    if hparams.resume and hparams.resume_path:
        global_step, global_epoch = model.load_checkpoint(
            hparams.resume_path,
            reset_optimizer=getattr(hparams, "reset_optimizer", False),
        )
        print(f"[MelAE] resumed step={global_step} epoch={global_epoch}")

    writer = SummaryWriter(log_dir=hparams.log_event_path)
    try:
        train_loop(model, data_loaders, writer,
                   start_step=global_step, start_epoch=global_epoch)
    except KeyboardInterrupt:
        print("[MelAE] Interrupted! Saving checkpoint ...")
        model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)
    finally:
        writer.close()
    print("[MelAE] Finished.")


if __name__ == "__main__":
    main()
    sys.exit(0)
