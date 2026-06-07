import os
import sys
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import Options_inpainting
from Data_loaders import viai_a_loader
from Models.VIAI_A_inpainting import VIAIAModel
from utils.viai_a_metrics import compute_viai_a_metrics, write_mel_images


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def configure_viai_a_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "VIAI-A-PatchGAN" if getattr(hparams, "use_gan", False) else "VIAI-A"
    if not _arg_was_passed("--train_split_name"):
        hparams.train_split_name = "train_viai_a_split.txt"
    if not _arg_was_passed("--val_split_name"):
        hparams.val_split_name = "val_viai_a_split.txt"
    if not _arg_was_passed("--test_split_name"):
        hparams.test_split_name = "test_viai_a_split.txt"
    if not _arg_was_passed("--log_event_path"):
        event_name = "events_viai_a_patchgan" if getattr(hparams, "use_gan", False) else "events_viai_a"
        hparams.log_event_path = os.path.join(hparams.checkpoint_dir, event_name)


def _positive_interval(name):
    return max(0, int(getattr(hparams, name, 0)))


def _should_compute_ssim(train, step):
    if not train:
        return True
    metric_freq = _positive_interval("metric_freq")
    return step <= 1 or (metric_freq > 0 and step % metric_freq == 0)


def _should_write_images(train, step, batch_index):
    image_freq = _positive_interval("tb_image_freq")
    if train:
        return step <= 1 or (image_freq > 0 and step % image_freq == 0)
    return batch_index == 0


def _write_train_monitoring(writer, prefix, step, model, metrics):
    if writer is None:
        return
    writer.add_scalar(f"{prefix}/psnr_full", metrics["psnr_full"], step)
    writer.add_scalar(f"{prefix}/psnr_missing", metrics["psnr_missing"], step)
    if metrics["ssim_full"] is not None:
        writer.add_scalar(f"{prefix}/ssim_full", metrics["ssim_full"], step)
    writer.add_scalar(f"{prefix}/blank_frames", model.blank_length, step)
    writer.add_scalar(f"{prefix}/lr", model.current_lr, step)


def _write_phase_averages(writer, prefix, step, averages, current_lr):
    if writer is None or averages is None:
        return
    writer.add_scalar(f"{prefix}/loss_total", averages["loss_total"], step)
    writer.add_scalar(f"{prefix}/loss_full_l1", averages["loss_full_l1"], step)
    writer.add_scalar(f"{prefix}/loss_missing_l1", averages["loss_missing_l1"], step)
    writer.add_scalar(f"{prefix}/psnr_full", averages["psnr_full"], step)
    writer.add_scalar(f"{prefix}/psnr_missing", averages["psnr_missing"], step)
    if averages["ssim_full"] is not None:
        writer.add_scalar(f"{prefix}/ssim_full", averages["ssim_full"], step)
    writer.add_scalar(f"{prefix}/blank_frames", averages["blank_frames"], step)
    writer.add_scalar(f"{prefix}/lr", current_lr, step)
    if "loss_recon" in averages:
        writer.add_scalar(f"{prefix}/loss_recon", averages["loss_recon"], step)
        writer.add_scalar(f"{prefix}/loss_g_gan", averages["loss_g_gan"], step)
        writer.add_scalar(f"{prefix}/weighted_loss_recon", averages["weighted_loss_recon"], step)
        writer.add_scalar(f"{prefix}/weighted_loss_gan", averages["weighted_loss_gan"], step)
        writer.add_scalar(f"{prefix}/beta_recon", averages["beta_recon"], step)
        writer.add_scalar(f"{prefix}/loss_d", averages["loss_d"], step)
        writer.add_scalar(f"{prefix}/loss_d_real", averages["loss_d_real"], step)
        writer.add_scalar(f"{prefix}/loss_d_fake", averages["loss_d_fake"], step)
        writer.add_scalar(f"{prefix}/d_real_mean", averages["d_real_mean"], step)
        writer.add_scalar(f"{prefix}/d_fake_mean", averages["d_fake_mean"], step)


def run_phase(model, data_loader, phase, global_step, writer, global_epoch):
    train = phase == "train"
    totals = {
        "loss_total": 0.0,
        "loss_full_l1": 0.0,
        "loss_missing_l1": 0.0,
        "psnr_full": 0.0,
        "psnr_missing": 0.0,
        "ssim_full": 0.0,
    }
    if model.use_gan:
        totals.update(
            {
                "loss_recon": 0.0,
                "loss_g_gan": 0.0,
                "weighted_loss_recon": 0.0,
                "weighted_loss_gan": 0.0,
                "beta_recon": 0.0,
                "loss_d": 0.0,
                "loss_d_real": 0.0,
                "loss_d_fake": 0.0,
                "d_real_mean": 0.0,
                "d_fake_mean": 0.0,
            }
        )
    sample_count = 0
    ssim_sample_count = 0
    batch_count = 0
    stop_training = False

    progress = tqdm(
        data_loader,
        desc=f"[VIAI-A {phase}] epoch={global_epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
    )
    for data in progress:
        iter_start_time = time.time()
        if train:
            model.get_blank_space_length(global_step)
        model.set_inputs(data, deterministic_missing=not train)
        if train:
            model.optimize_parameters(global_step)
            global_step += 1
        else:
            model.test(global_step=global_step)
        model.get_loss_items()
        metrics = compute_viai_a_metrics(
            model.mel_pred,
            model.mel_target_4d,
            model.missing_mask,
            compute_ssim=_should_compute_ssim(train, global_step),
        )

        totals["loss_total"] += model.loss_total_item
        totals["loss_full_l1"] += model.loss_full_l1_item
        totals["loss_missing_l1"] += model.loss_missing_l1_item
        if model.use_gan:
            totals["loss_recon"] += model.loss_recon_item
            totals["loss_g_gan"] += model.loss_G_GAN_item
            totals["weighted_loss_recon"] += model.weighted_loss_recon_item
            totals["weighted_loss_gan"] += model.weighted_loss_gan_item
            totals["beta_recon"] += model.beta_recon_item
            totals["loss_d"] += model.loss_D_item
            totals["loss_d_real"] += model.loss_D_real_item
            totals["loss_d_fake"] += model.loss_D_fake_item
            totals["d_real_mean"] += model.d_real_mean_item
            totals["d_fake_mean"] += model.d_fake_mean_item
        totals["psnr_full"] += metrics["psnr_full_sum"]
        totals["psnr_missing"] += metrics["psnr_missing_sum"]
        sample_count += metrics["num_samples"]
        if metrics["ssim_full_sum"] is not None:
            totals["ssim_full"] += metrics["ssim_full_sum"]
            ssim_sample_count += metrics["num_samples"]

        postfix = dict(
            step=global_step,
            loss=f"{model.loss_total_item:.4f}",
            full_l1=f"{model.loss_full_l1_item:.4f}",
            missing_l1=f"{model.loss_missing_l1_item:.4f}",
            psnr=f"{metrics['psnr_full']:.2f}",
            psnr_miss=f"{metrics['psnr_missing']:.2f}",
            blank=model.blank_length,
        )
        if metrics["ssim_full"] is not None:
            postfix["ssim"] = f"{metrics['ssim_full']:.4f}"
        if model.use_gan:
            postfix["recon"] = f"{model.loss_recon_item:.4f}"
            postfix["g_gan"] = f"{model.loss_G_GAN_item:.4f}"
            postfix["d"] = f"{model.loss_D_item:.4f}"
        progress.set_postfix(**postfix)

        if train and global_step % hparams.print_freq == 0:
            elapsed = (time.time() - iter_start_time) / max(1, hparams.batch_size)
            ssim_text = (
                f" ssim={metrics['ssim_full']:.6f}"
                if metrics["ssim_full"] is not None
                else ""
            )
            gan_text = (
                f" recon={model.loss_recon_item:.6f} "
                f"g_gan={model.loss_G_GAN_item:.6f} "
                f"d={model.loss_D_item:.6f}"
                if model.use_gan
                else ""
            )
            tqdm.write(
                f"[VIAI-A train] step={global_step} "
                f"loss={model.loss_total_item:.6f} "
                f"full_l1={model.loss_full_l1_item:.6f} "
                f"missing_l1={model.loss_missing_l1_item:.6f} "
                f"eta1={model.eta1_item:.6f}"
                f"{gan_text} "
                f"psnr={metrics['psnr_full']:.3f} "
                f"psnr_missing={metrics['psnr_missing']:.3f}"
                f"{ssim_text} "
                f"time_per_sample={elapsed:.4f}s"
            )
        if train and global_step > 0 and global_step % hparams.checkpoint_interval == 0:
            model.save_checkpoint(global_step, 0, hparams.checkpoint_dir)
        if train:
            model.TF_writer(writer, global_step, prefix=phase)
            _write_train_monitoring(writer, phase, global_step, model, metrics)
        if _should_write_images(train, global_step, batch_count):
            write_mel_images(
                writer,
                phase,
                global_step,
                model.mel_input_4d,
                model.mel_pred,
                model.mel_target_4d,
                max_items=getattr(hparams, "tb_image_count", 4),
            )

        batch_count += 1

        if train and hparams.max_train_steps is not None and global_step >= hparams.max_train_steps:
            tqdm.write(f"Reached VIAI-A smoke-test max_train_steps={hparams.max_train_steps}")
            stop_training = True
            break

    if batch_count == 0:
        return global_step, stop_training, None
    averages = {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_full_l1": totals["loss_full_l1"] / batch_count,
        "loss_missing_l1": totals["loss_missing_l1"] / batch_count,
        "psnr_full": totals["psnr_full"] / max(1, sample_count),
        "psnr_missing": totals["psnr_missing"] / max(1, sample_count),
        "ssim_full": None
        if ssim_sample_count == 0
        else totals["ssim_full"] / ssim_sample_count,
        "blank_frames": float(model.blank_length),
    }
    if model.use_gan:
        averages.update(
            {
                "loss_recon": totals["loss_recon"] / batch_count,
                "loss_g_gan": totals["loss_g_gan"] / batch_count,
                "weighted_loss_recon": totals["weighted_loss_recon"] / batch_count,
                "weighted_loss_gan": totals["weighted_loss_gan"] / batch_count,
                "beta_recon": totals["beta_recon"] / batch_count,
                "loss_d": totals["loss_d"] / batch_count,
                "loss_d_real": totals["loss_d_real"] / batch_count,
                "loss_d_fake": totals["loss_d_fake"] / batch_count,
                "d_real_mean": totals["d_real_mean"] / batch_count,
                "d_fake_mean": totals["d_fake_mean"] / batch_count,
            }
        )
    ssim_summary = (
        f" ssim={averages['ssim_full']:.6f}"
        if averages["ssim_full"] is not None
        else ""
    )
    gan_summary = (
        f" recon={averages['loss_recon']:.6f} "
        f"g_gan={averages['loss_g_gan']:.6f} "
        f"d={averages['loss_d']:.6f} "
        f"d_real_mean={averages['d_real_mean']:.4f} "
        f"d_fake_mean={averages['d_fake_mean']:.4f}"
        if model.use_gan
        else ""
    )
    tqdm.write(
        f"[VIAI-A {phase}] "
        f"loss={averages['loss_total']:.6f} "
        f"full_l1={averages['loss_full_l1']:.6f} "
        f"missing_l1={averages['loss_missing_l1']:.6f} "
        f"{gan_summary} "
        f"psnr={averages['psnr_full']:.3f} "
        f"psnr_missing={averages['psnr_missing']:.3f}"
        f"{ssim_summary}"
    )
    if not train:
        _write_phase_averages(writer, phase, global_step, averages, model.current_lr)
    return global_step, stop_training, averages


def train_loop(model, data_loaders, writer, start_step=0, start_epoch=0):
    global_step = start_step
    global_epoch = start_epoch
    while global_epoch < hparams.nepochs:
        if "train" in data_loaders:
            global_step, stop_training, _ = run_phase(
                model,
                data_loaders["train"],
                "train",
                global_step,
                writer,
                global_epoch,
            )
            if stop_training:
                break
        if "val" in data_loaders:
            with torch.no_grad():
                run_phase(model, data_loaders["val"], "val", global_step, writer, global_epoch)
        global_epoch += 1
        print(f"[VIAI-A] epoch={global_epoch} current_lr={model.current_lr}")

    model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)


def main():
    configure_viai_a_defaults()
    if getattr(hparams, "use_gan", False):
        print("[VIAI-A] PatchGAN enabled; checkpoints will include netD and optimizer_D.")
    else:
        print(
            "[VIAI-A] PatchGAN disabled (audio-only); "
            "checkpoints will not contain PatchGAN discriminator weights."
        )
    if hparams.resume and hparams.init_from_viai_a is not None:
        raise RuntimeError(
            "Use either --resume/--resume_path to continue a run, or "
            "--init_from_viai_a to start a new run from pretrained weights; not both."
        )

    os.makedirs(hparams.checkpoint_dir, exist_ok=True)
    data_loaders = viai_a_loader.get_data_loaders(hparams.data_root, phases=("train", "val"))
    model = VIAIAModel(hparams, device=device)

    global_step = 0
    global_epoch = 0
    if hparams.resume and hparams.resume_path is not None:
        global_step, global_epoch = model.load_checkpoint(
            hparams.resume_path,
            reset_optimizer=hparams.reset_optimizer,
        )
        print(f"[VIAI-A] resumed checkpoint step={global_step} epoch={global_epoch}")
    elif hparams.init_from_viai_a is not None:
        source_step, source_epoch = model.load_init_checkpoint(hparams.init_from_viai_a)
        global_step = 0
        global_epoch = 0
        print(
            "[VIAI-A] initialized from checkpoint "
            f"source_step={source_step} source_epoch={source_epoch}; "
            "new run starts at step=0 epoch=0"
        )

    log_event_path = hparams.log_event_path
    if log_event_path is None:
        log_event_path = "log/run-test-" + hparams.name + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path:", log_event_path)
    writer = SummaryWriter(log_dir=log_event_path)

    try:
        train_loop(model, data_loaders, writer, start_step=global_step, start_epoch=global_epoch)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        writer.close()
    print("Finished VIAI-A training")


if __name__ == "__main__":
    main()
    sys.exit(0)
