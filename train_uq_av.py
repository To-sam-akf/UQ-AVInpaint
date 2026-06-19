"""Train UQ-AV K=1 Latent Diffusion (P3).

Requires a pre-trained frozen Mel autoencoder (P2 checkpoint).

Usage (via main.py):
    python main.py train-uq-av -- \
      --name UQ-AV-K1 \\
      --data_root /path/to/data \\
      --train_split_name train_av_split.txt \\
      --val_split_name val_av_split.txt \\
      --ae_checkpoint checkpoints/mel_ae/MelAE_checkpoint_step000005000.pth.tar \\
      --batch_size 8 --nepochs 100 \\
      --uq_lambda_boundary 0.1

Key defaults:
  - 1000 diffusion timesteps, linear schedule.
  - DDIM inference with 50 steps.
  - K=1 only (multi-candidate in P4).
  - AE is frozen and used only for encode/decode.
"""

import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import Options_inpainting
from Data_loaders.uq_av_loader import get_uq_av_data_loaders
from Models.UQ_AV_Diffusion import UQAVDiffusionModel
from utils.viai_a_metrics import (
    compute_viai_a_metrics,
    write_mel_images,
    compose_inpainted_mel,
)

# ---------------------------------------------------------------------------
hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(
        arg == name or arg.startswith(name + "=") for arg in sys.argv[1:]
    )


def _configure_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "UQ-AV-K1"
    if not _arg_was_passed("--train_split_name"):
        hparams.train_split_name = "train_av_split.txt"
    if not _arg_was_passed("--val_split_name"):
        hparams.val_split_name = "val_av_split.txt"
    if not _arg_was_passed("--checkpoint_dir"):
        hparams.checkpoint_dir = os.path.join(
            getattr(hparams, "checkpoints_dir", "./checkpoints"),
            "uq_av_k1",
        )
    if not _arg_was_passed("--log_event_path"):
        hparams.log_event_path = os.path.join(
            hparams.checkpoint_dir, "events",
        )


def _run_phase(model, data_loader, phase, global_step, writer,
               global_epoch):
    train = phase == "train"
    totals = {
        "loss_total": 0.0, "loss_diff": 0.0,
        "loss_boundary": 0.0, "loss_sync": 0.0,
    }
    psnr_full_sum = 0.0
    psnr_missing_sum = 0.0
    ssim_full_sum = 0.0
    sample_count = 0
    ssim_sample_count = 0
    batch_count = 0
    stop_training = False

    progress = tqdm(
        data_loader,
        desc=f"[UQ-AV {phase}] epoch={global_epoch + 1}",
        unit="batch", dynamic_ncols=True,
    )
    for batch in progress:
        iter_start = time.time()

        model.set_input(batch)

        if train:
            model.optimize_parameters(global_step)
            global_step += 1
        else:
            model.test(global_step=global_step)

        model.get_loss_items()

        # Metrics
        metrics = {}
        if not train or global_step % max(
            1, int(getattr(hparams, "metric_freq", 100))
        ) == 0:
            metrics = compute_viai_a_metrics(
                model.mel_pred if model.mel_pred is not None
                else model.mel_corrupted,
                model.mel_target,
                model.missing_mask,
                compute_ssim=True,
            )
            psnr_full_sum += metrics["psnr_full_sum"]
            psnr_missing_sum += metrics["psnr_missing_sum"]
            sample_count += metrics["num_samples"]
            if metrics.get("ssim_full_sum") is not None:
                ssim_full_sum += metrics["ssim_full_sum"]
                ssim_sample_count += metrics["num_samples"]

        totals["loss_total"] += model.loss_total_item
        totals["loss_diff"] += model.loss_diff_item
        totals["loss_boundary"] += model.loss_boundary_item
        totals["loss_sync"] += model.loss_sync_item

        postfix = {
            "step": global_step,
            "diff": f"{model.loss_diff_item:.4f}",
            "bdy": f"{model.loss_boundary_item:.4f}",
        }
        if metrics:
            postfix["psnr"] = f"{metrics.get('psnr_full', 'N/A')}"
            postfix["psnr_miss"] = f"{metrics.get('psnr_missing', 'N/A')}"
        progress.set_postfix(**postfix)

        if train:
            model.TF_writer(writer, global_step, prefix=phase)

            if (global_step > 0
                    and global_step % getattr(hparams, "tb_image_freq", 500) == 0
                    and model.mel_pred is not None):
                mel_completed = compose_inpainted_mel(
                    model.mel_corrupted, model.mel_pred,
                    model.missing_mask,
                )
                write_mel_images(
                    writer, phase, global_step,
                    model.mel_corrupted, mel_completed, model.mel_target,
                    max_items=getattr(hparams, "tb_image_count", 4),
                )

            if (global_step > 0
                    and global_step % getattr(hparams, "print_freq", 100) == 0):
                elapsed = (time.time() - iter_start) / max(
                    1, hparams.batch_size
                )
                tqdm.write(
                    f"[UQ-AV train] step={global_step} "
                    f"loss={model.loss_total_item:.6f} "
                    f"diff={model.loss_diff_item:.6f} "
                    f"bdy={model.loss_boundary_item:.6f} "
                    f"sync={model.loss_sync_item:.6f} "
                    f"psnr={metrics.get('psnr_full', 'N/A')} "
                    f"psnr_miss={metrics.get('psnr_missing', 'N/A')} "
                    f"lr={model.current_lr:.2e} "
                    f"time={elapsed:.4f}s/sample"
                )

            if (global_step > 0
                    and global_step % getattr(
                        hparams, "checkpoint_interval", 1000
                    ) == 0):
                model.save_checkpoint(
                    global_step, global_epoch, hparams.checkpoint_dir,
                )

        batch_count += 1

        if (train
                and getattr(hparams, "max_train_steps", None) is not None
                and global_step >= hparams.max_train_steps):
            tqdm.write(
                f"Reached max_train_steps={hparams.max_train_steps}"
            )
            stop_training = True
            break

    if batch_count == 0:
        return global_step, stop_training, None

    averages = {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_diff": totals["loss_diff"] / batch_count,
        "loss_boundary": totals["loss_boundary"] / batch_count,
        "loss_sync": totals["loss_sync"] / batch_count,
        "psnr_full": (
            psnr_full_sum / max(1, sample_count)
            if sample_count > 0 else None
        ),
        "psnr_missing": (
            psnr_missing_sum / max(1, sample_count)
            if sample_count > 0 else None
        ),
        "ssim_full": (
            ssim_full_sum / max(1, ssim_sample_count)
            if ssim_sample_count > 0 else None
        ),
    }
    tqdm.write(
        f"[UQ-AV {phase}] "
        f"loss={averages['loss_total']:.6f} "
        f"diff={averages['loss_diff']:.6f} "
        f"bdy={averages['loss_boundary']:.6f} "
        f"psnr={averages.get('psnr_full', 'N/A')} "
        f"psnr_miss={averages.get('psnr_missing', 'N/A')}"
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

        if "val" in data_loaders and global_epoch % max(
            1, int(getattr(hparams, "test_eval_epoch_interval", 1))
        ) == 0:
            with torch.no_grad():
                _run_phase(
                    model, data_loaders["val"], "val",
                    global_step, writer, global_epoch,
                )

        global_epoch += 1
        print(
            f"[UQ-AV] epoch={global_epoch} lr={model.current_lr}"
        )

    # Final save
    model.save_checkpoint(
        global_step, global_epoch, hparams.checkpoint_dir,
    )


def main():
    _configure_defaults()

    split_names = {
        "train": hparams.train_split_name,
        "val": hparams.val_split_name,
    }
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)

    ae_ckpt = getattr(hparams, "ae_checkpoint", None)
    if ae_ckpt is None:
        raise RuntimeError(
            "--ae_checkpoint is required for P3 training. "
            "Point to a trained Mel AE checkpoint from P2."
        )

    print(f"[UQ-AV] data_root={hparams.data_root}")
    print(f"[UQ-AV] checkpoint_dir={hparams.checkpoint_dir}")
    print(f"[UQ-AV] ae_checkpoint={ae_ckpt}")

    data_loaders = get_uq_av_data_loaders(
        hparams.data_root, split_names,
        phases=("train", "val"),
        batch_size=hparams.batch_size,
        num_workers=getattr(hparams, "num_workers", 4),
        pin_memory=getattr(hparams, "pin_memory", True),
    )

    model = UQAVDiffusionModel(hparams, device=device)
    model.load_ae_checkpoint(ae_ckpt)

    global_step = 0
    global_epoch = 0
    if hparams.resume and hparams.resume_path:
        global_step, global_epoch = model.load_checkpoint(
            hparams.resume_path,
            reset_optimizer=getattr(hparams, "reset_optimizer", False),
        )
        print(
            f"[UQ-AV] resumed step={global_step} epoch={global_epoch}"
        )

    writer = SummaryWriter(log_dir=hparams.log_event_path)
    try:
        train_loop(
            model, data_loaders, writer,
            start_step=global_step, start_epoch=global_epoch,
        )
    except KeyboardInterrupt:
        print("[UQ-AV] Interrupted! Saving checkpoint ...")
        model.save_checkpoint(
            global_step, global_epoch, hparams.checkpoint_dir,
        )
    finally:
        writer.close()
    print("[UQ-AV] Finished.")


if __name__ == "__main__":
    main()
    sys.exit(0)
