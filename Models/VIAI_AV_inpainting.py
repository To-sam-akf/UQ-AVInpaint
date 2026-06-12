import os
import random

import torch
import torch.nn as nn

from Data_loaders import mel_loader
from loss_functions import GANLoss, L2ContrastiveLoss
from networks import Discriminator_Networks
from networks import Image_Embedding
from networks import Inpainting_Networks
from networks import New_Inpainting_Networks
from utils import util


def _copy_matching_state(module, source_state, label):
    target_state = module.state_dict()
    copied = 0
    skipped = 0
    for name, value in source_state.items():
        if isinstance(value, nn.Parameter):
            value = value.data
        if name not in target_state:
            continue
        if target_state[name].shape != value.shape:
            skipped += 1
            continue
        target_state[name].copy_(value)
        copied += 1
    print(f"[VIAI-AV] loaded {copied} tensors into {label}; skipped_shape={skipped}")
    return copied


class VIAIAVModel(object):
    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")
        self.use_gan = bool(getattr(hparams, "use_gan", False))
        self.enable_sync_loss = not bool(getattr(hparams, "disable_sync_loss", False))
        self.enable_probe_loss = not bool(getattr(hparams, "disable_probe_loss", False))

        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams).to(self.device)
        self.VideoEncoder = Image_Embedding.ImageEmbedding(hparams=hparams).to(self.device)
        self.Mel_Decoder = New_Inpainting_Networks.MelDecoderImage(hparams=hparams).to(self.device)
        self.netD = None
        if self.use_gan:
            self.netD = Discriminator_Networks.MelDiscriminator().to(self.device)

        self.criterion_l1 = nn.L1Loss()
        self.criterion_gan = GANLoss(use_lsgan=False, device=self.device) if self.use_gan else None
        self.criterion_sync = L2ContrastiveLoss(
            margin=getattr(hparams, "sync_margin", 1.0),
            max_violation=False,
        )

        generator_params = (
            list(self.Mel_Encoder.parameters())
            + list(self.VideoEncoder.parameters())
            + list(self.Mel_Decoder.parameters())
        )
        self.optimizer_G = torch.optim.Adam(
            generator_params,
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
        )
        self.optimizer_D = None
        if self.use_gan:
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=hparams.lr,
                betas=(hparams.beta1, hparams.beta2),
            )
        self.current_lr = hparams.lr
        self.blank_length = getattr(hparams, "min_blank_frames", 20)

        self.loss_total_item = 0.0
        self.loss_av_gen_item = 0.0
        self.loss_recon_item = 0.0
        self.loss_full_l1_item = 0.0
        self.loss_missing_l1_item = 0.0
        self.loss_G_GAN_item = 0.0
        self.weighted_loss_recon_item = 0.0
        self.weighted_loss_gan_item = 0.0
        self.loss_sync_item = 0.0
        self.loss_probe_gen_item = 0.0
        self.loss_probe_recon_item = 0.0
        self.loss_probe_full_l1_item = 0.0
        self.loss_probe_missing_l1_item = 0.0
        self.loss_probe_G_GAN_item = 0.0
        self.weighted_loss_probe_gen_item = 0.0
        self.loss_D_item = 0.0
        self.loss_D_real_item = 0.0
        self.loss_D_fake_item = 0.0
        self.eta1_item = 0.0
        self.eta2_item = 0.0
        self._print_loss_configuration()

    def _print_loss_configuration(self):
        lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
        lambda_recon = getattr(self.hparams, "lambda_recon", 1.0)
        lambda_sync = getattr(self.hparams, "lambda_sync", 1.0)
        lambda_probe = getattr(self.hparams, "lambda_probe", 1.0)
        print(
            "[VIAI-AV] loss weights: "
            f"use_gan={self.use_gan} "
            f"lambda_gan={lambda_gan} "
            f"lambda_recon={lambda_recon} "
            f"lambda_sync={lambda_sync} "
            f"lambda_probe={lambda_probe} "
            f"enable_sync_loss={self.enable_sync_loss} "
            f"enable_probe_loss={self.enable_probe_loss}"
        )
        if self.use_gan:
            print(
                "[VIAI-AV] generator formula: "
                "loss_av_gen = lambda_gan * loss_g_gan + lambda_recon * loss_recon"
            )
        else:
            print("[VIAI-AV] generator formula: loss_av_gen = lambda_recon * loss_recon")
        if self.enable_sync_loss or self.enable_probe_loss:
            print(
                "[VIAI-AV] total formula: "
                "loss_total = loss_av_gen + lambda_sync * loss_sync "
                "+ lambda_probe * eta2 * loss_probe_gen"
            )

    def _eta(self, step, base, interval, floor):
        if interval <= 0:
            return floor
        return max(floor, base ** (float(step) / float(interval)))

    def _to_device(self, tensor):
        if tensor is None:
            return None
        return tensor.float().to(self.device)

    def get_blank_space_length(self, global_step):
        min_blank = max(1, int(getattr(self.hparams, "min_blank_frames", 20)))
        max_blank = max(min_blank, int(getattr(self.hparams, "max_blank_frames", 50)))
        self.blank_length = random.randint(min_blank, max_blank)
        return self.blank_length

    def set_inputs(self, data, mask_specs=None):
        if len(data) < 8:
            raise ValueError("Expected 8-tuple batch from audio_loader.collate_fn")
        (
            video_batch,
            flow_batch,
            c_batch,
            x_batch,
            y_batch,
            g_batch,
            input_lengths,
            path_batch,
        ) = data
        self.video_batch = self._to_device(video_batch)
        self.flow_batch = self._to_device(flow_batch)
        self.mel_target = self._to_device(c_batch)
        self.audio_input = self._to_device(x_batch)
        self.audio_target = self._to_device(y_batch)
        self.g_batch = g_batch.to(self.device) if g_batch is not None else None
        self.input_lengths = input_lengths.to(self.device)
        self.path_batch = path_batch

        if self.mel_target is None:
            raise ValueError("Local conditioning Mel target is required for VIAI-AV training")
        if mask_specs is None:
            self.mel_input, self.missing_mask, self.missing_span = (
                mel_loader.corrupt_mel_spectrogram(
                    self.mel_target,
                    self.blank_length,
                )
            )
        else:
            self.mel_input, self.missing_mask, self.missing_span = (
                mel_loader.corrupt_mel_spectrogram_batch(
                    self.mel_target,
                    mask_specs,
                )
            )
            self.blank_length = int(mask_specs[0]["gap_frames"])
        self.mel_target_4d = self.mel_target.unsqueeze(1)
        self.mel_input_4d = self.mel_input.unsqueeze(1)
        self.missing_mask = self.missing_mask.to(self.device)

    def _forward_inpainter(self):
        self.mel_features = self.Mel_Encoder(self.mel_input)
        self.video_feature = self.VideoEncoder(self.video_batch, self.flow_batch)
        self.mel_target_features = self.Mel_Encoder(self.mel_target)
        self.mel_target_feature_flat = self.mel_target_features[-1].flatten(1)
        self.video_feature_flat = self.video_feature.flatten(1)
        self.mel_net_norm = util.l2_norm(self.mel_target_feature_flat.detach())
        self.video_net_norm = util.l2_norm(self.video_feature_flat)
        self.mel_pred = self.Mel_Decoder(
            self.mel_features,
            self.mel_input_4d.size(),
            self.video_feature,
        )
        if self.enable_probe_loss:
            self.mel_probe_pred = self.Mel_Decoder(
                self.mel_features,
                self.mel_input_4d.size(),
                self.mel_target_features[-1],
            )
        else:
            self.mel_probe_pred = None
        return self.mel_pred

    def _zero_loss_like(self, reference):
        return torch.zeros((), device=self.device, dtype=reference.dtype)

    def _reconstruction_losses(self, prediction):
        loss_full_l1 = self.criterion_l1(prediction, self.mel_target_4d)
        masked_abs = torch.abs(prediction - self.mel_target_4d) * self.missing_mask
        loss_missing_l1 = masked_abs.sum() / torch.clamp(self.missing_mask.sum(), min=1.0)
        loss_recon = self.eta1 * loss_full_l1 + loss_missing_l1
        return loss_recon, loss_full_l1, loss_missing_l1

    def _compute_losses(self, global_step):
        self.eta1 = self._eta(
            global_step,
            getattr(self.hparams, "recon_decay_base", 0.9),
            getattr(self.hparams, "recon_decay_interval", 1000.0),
            getattr(self.hparams, "recon_decay_floor", 0.1),
        )
        self.eta2 = self._eta(
            global_step,
            getattr(self.hparams, "probe_decay_base", getattr(self.hparams, "sync_decay_base", 0.9)),
            getattr(
                self.hparams,
                "probe_decay_interval",
                getattr(self.hparams, "sync_decay_interval", 1000.0),
            ),
            getattr(self.hparams, "probe_decay_floor", getattr(self.hparams, "sync_decay_floor", 0.1)),
        )
        self.loss_recon, self.loss_full_l1, self.loss_missing_l1 = self._reconstruction_losses(
            self.mel_pred
        )
        lambda_recon = getattr(self.hparams, "lambda_recon", 1.0)
        self.weighted_loss_recon = lambda_recon * self.loss_recon

        if self.use_gan:
            pred_fake = self.netD(self.mel_pred)
            self.loss_G_GAN = self.criterion_gan(pred_fake, True)
            lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
            self.weighted_loss_gan = lambda_gan * self.loss_G_GAN
        else:
            self.loss_G_GAN = self._zero_loss_like(self.loss_recon)
            self.weighted_loss_gan = self._zero_loss_like(self.loss_recon)
        self.loss_av_gen = self.weighted_loss_gan + self.weighted_loss_recon

        if self.enable_sync_loss:
            self.loss_sync = self.criterion_sync(self.mel_net_norm, self.video_net_norm)
        else:
            self.loss_sync = self._zero_loss_like(self.loss_av_gen)

        if self.enable_probe_loss and self.mel_probe_pred is not None:
            (
                self.loss_probe_recon,
                self.loss_probe_full_l1,
                self.loss_probe_missing_l1,
            ) = self._reconstruction_losses(self.mel_probe_pred)
            if self.use_gan:
                pred_probe_fake = self.netD(self.mel_probe_pred)
                self.loss_probe_G_GAN = self.criterion_gan(pred_probe_fake, True)
                lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
                self.loss_probe_gen = (
                    lambda_gan * self.loss_probe_G_GAN
                    + lambda_recon * self.loss_probe_recon
                )
            else:
                self.loss_probe_G_GAN = self._zero_loss_like(self.loss_probe_recon)
                self.loss_probe_gen = lambda_recon * self.loss_probe_recon
        else:
            self.loss_probe_recon = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_full_l1 = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_missing_l1 = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_G_GAN = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_gen = self._zero_loss_like(self.loss_av_gen)

        lambda_sync = getattr(self.hparams, "lambda_sync", 1.0)
        lambda_probe = getattr(self.hparams, "lambda_probe", 1.0)
        self.weighted_loss_probe_gen = self.eta2 * self.loss_probe_gen
        self.loss_total = (
            self.loss_av_gen
            + lambda_sync * self.loss_sync
            + lambda_probe * self.weighted_loss_probe_gen
        )
        if not self.use_gan:
            self.loss_D_real = self._zero_loss_like(self.loss_av_gen)
            self.loss_D_fake = self._zero_loss_like(self.loss_av_gen)
            self.loss_D = self._zero_loss_like(self.loss_av_gen)

    def _compute_discriminator_loss(self):
        if not self.use_gan:
            self.loss_D_real = self._zero_loss_like(self.loss_av_gen)
            self.loss_D_fake = self._zero_loss_like(self.loss_av_gen)
            self.loss_D = self._zero_loss_like(self.loss_av_gen)
            return
        pred_real = self.netD(self.mel_target_4d)
        fake_losses = [self.criterion_gan(self.netD(self.mel_pred.detach()), False, softlabel=True)]
        if self.enable_probe_loss and self.mel_probe_pred is not None:
            fake_losses.append(
                self.criterion_gan(self.netD(self.mel_probe_pred.detach()), False, softlabel=True)
            )
        self.loss_D_real = self.criterion_gan(pred_real, True, softlabel=True)
        self.loss_D_fake = sum(fake_losses) / len(fake_losses)
        self.loss_D = 0.5 * (self.loss_D_real + self.loss_D_fake)

    def optimize_parameters(self, global_step):
        self.Mel_Encoder.train()
        self.VideoEncoder.train()
        self.Mel_Decoder.train()
        if self.use_gan:
            self.netD.eval()
            for parameter in self.netD.parameters():
                parameter.requires_grad = False

        self._forward_inpainter()
        self._compute_losses(global_step)
        self.optimizer_G.zero_grad()
        self.loss_total.backward()
        self.optimizer_G.step()

        if self.use_gan:
            self.netD.train()
            for parameter in self.netD.parameters():
                parameter.requires_grad = True
            self.optimizer_D.zero_grad()
            self._compute_discriminator_loss()
            self.loss_D.backward()
            self.optimizer_D.step()
        self.current_lr = self.optimizer_G.param_groups[0]["lr"]

    def test(self, global_step=0):
        self.Mel_Encoder.eval()
        self.VideoEncoder.eval()
        self.Mel_Decoder.eval()
        if self.use_gan:
            self.netD.eval()
        with torch.no_grad():
            self._forward_inpainter()
            self._compute_losses(global_step=global_step)
            self._compute_discriminator_loss()

    def get_loss_items(self):
        self.loss_total_item = float(self.loss_total.detach().cpu().item())
        self.loss_av_gen_item = float(self.loss_av_gen.detach().cpu().item())
        self.loss_recon_item = float(self.loss_recon.detach().cpu().item())
        self.loss_full_l1_item = float(self.loss_full_l1.detach().cpu().item())
        self.loss_missing_l1_item = float(self.loss_missing_l1.detach().cpu().item())
        self.loss_G_GAN_item = float(self.loss_G_GAN.detach().cpu().item())
        self.weighted_loss_recon_item = float(self.weighted_loss_recon.detach().cpu().item())
        self.weighted_loss_gan_item = float(self.weighted_loss_gan.detach().cpu().item())
        self.loss_sync_item = float(self.loss_sync.detach().cpu().item())
        self.loss_probe_gen_item = float(self.loss_probe_gen.detach().cpu().item())
        self.loss_probe_recon_item = float(self.loss_probe_recon.detach().cpu().item())
        self.loss_probe_full_l1_item = float(self.loss_probe_full_l1.detach().cpu().item())
        self.loss_probe_missing_l1_item = float(self.loss_probe_missing_l1.detach().cpu().item())
        self.loss_probe_G_GAN_item = float(self.loss_probe_G_GAN.detach().cpu().item())
        self.weighted_loss_probe_gen_item = float(
            self.weighted_loss_probe_gen.detach().cpu().item()
        )
        self.loss_D_item = float(self.loss_D.detach().cpu().item())
        self.loss_D_real_item = float(self.loss_D_real.detach().cpu().item())
        self.loss_D_fake_item = float(self.loss_D_fake.detach().cpu().item())
        self.eta1_item = float(self.eta1)
        self.eta2_item = float(self.eta2)

    def TF_writer(self, writer, step, prefix="train"):
        if writer is None:
            return
        writer.add_scalar(f"{prefix}/loss_total", self.loss_total_item, step)
        writer.add_scalar(f"{prefix}/loss_av_gen", self.loss_av_gen_item, step)
        writer.add_scalar(f"{prefix}/loss_recon", self.loss_recon_item, step)
        writer.add_scalar(f"{prefix}/loss_full_l1", self.loss_full_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_missing_l1", self.loss_missing_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_g_gan", self.loss_G_GAN_item, step)
        writer.add_scalar(f"{prefix}/weighted_loss_recon", self.weighted_loss_recon_item, step)
        writer.add_scalar(f"{prefix}/weighted_loss_gan", self.weighted_loss_gan_item, step)
        writer.add_scalar(f"{prefix}/loss_sync", self.loss_sync_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_gen", self.loss_probe_gen_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_recon", self.loss_probe_recon_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_full_l1", self.loss_probe_full_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_missing_l1", self.loss_probe_missing_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_g_gan", self.loss_probe_G_GAN_item, step)
        if self.enable_probe_loss:
            writer.add_scalar(
                f"{prefix}/weighted_loss_probe_gen",
                self.weighted_loss_probe_gen_item,
                step,
            )
        writer.add_scalar(f"{prefix}/loss_d", self.loss_D_item, step)
        writer.add_scalar(f"{prefix}/loss_d_real", self.loss_D_real_item, step)
        writer.add_scalar(f"{prefix}/loss_d_fake", self.loss_D_fake_item, step)
        writer.add_scalar(f"{prefix}/eta1", self.eta1_item, step)
        writer.add_scalar(f"{prefix}/eta2", self.eta2_item, step)

    def save_checkpoint(self, global_step, global_epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.hparams.name}_checkpoint_step{global_step:09d}.pth.tar",
        )
        checkpoint = {
            "Mel_Encoder": self.Mel_Encoder.state_dict(),
            "VideoEncoder": self.VideoEncoder.state_dict(),
            "Mel_Decoder": self.Mel_Decoder.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict()
            if self.hparams.save_optimizer_state
            else None,
            "global_step": global_step,
            "global_epoch": global_epoch,
            "use_gan": self.use_gan,
            "stage": "VIAI-AV-stage4-sync-probe",
            "enable_sync_loss": self.enable_sync_loss,
            "enable_probe_loss": self.enable_probe_loss,
        }
        if self.use_gan:
            checkpoint["netD"] = self.netD.state_dict()
            checkpoint["optimizer_D"] = (
                self.optimizer_D.state_dict()
                if self.hparams.save_optimizer_state
                else None
            )
        torch.save(checkpoint, checkpoint_path)
        print("Saved VIAI-AV checkpoint:", checkpoint_path)
        return checkpoint_path

    def _checkpoint_use_gan(self, checkpoint):
        if "use_gan" in checkpoint:
            return bool(checkpoint["use_gan"])
        return "netD" in checkpoint

    def _assert_checkpoint_gan_mode(self, checkpoint, checkpoint_path):
        checkpoint_use_gan = self._checkpoint_use_gan(checkpoint)
        if checkpoint_use_gan == self.use_gan:
            return
        if checkpoint_use_gan:
            hint = "This checkpoint includes PatchGAN weights; rerun with --use_gan."
        else:
            hint = "This checkpoint was trained without PatchGAN; rerun without --use_gan."
        raise RuntimeError(
            "VIAI-AV checkpoint GAN mode mismatch: "
            f"checkpoint use_gan={checkpoint_use_gan}, current use_gan={self.use_gan}. "
            f"{hint} checkpoint={checkpoint_path}"
        )
    # 导入共有的结构权重，主要是Mel_Encoder和Mel_Decoder的权重，VideoEncoder不导入，因为VIAI-A没有视频编码器；如果有netD且当前模型使用GAN，则导入netD权重，否则不导入netD权重
    def load_checkpoint(self, checkpoint_path, reset_optimizer=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.checkpoint_stage = checkpoint.get("stage", "")
        self.checkpoint_enable_probe_loss = bool(
            checkpoint.get("enable_probe_loss", False)
        )
        self._assert_checkpoint_gan_mode(checkpoint, checkpoint_path)
        self.Mel_Encoder.load_state_dict(checkpoint["Mel_Encoder"])
        self.Mel_Decoder.load_state_dict(checkpoint["Mel_Decoder"])
        if self.use_gan:
            self.netD.load_state_dict(checkpoint["netD"])
        if "VideoEncoder" not in checkpoint:
            raise RuntimeError(
                "This VIAI-AV checkpoint does not contain VideoEncoder. "
                "Use a stage3 checkpoint saved by train-viai-av."
            )
        self.VideoEncoder.load_state_dict(checkpoint["VideoEncoder"])
        if not reset_optimizer:
            if checkpoint.get("optimizer_G") is not None:
                self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            if self.use_gan and checkpoint.get("optimizer_D") is not None:
                self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        return int(checkpoint.get("global_step", 0)), int(checkpoint.get("global_epoch", 0))

    def load_viai_a_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"VIAI-A initialization checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "Mel_Encoder" not in checkpoint or "Mel_Decoder" not in checkpoint:
            raise RuntimeError(
                "VIAI-A initialization checkpoint must contain Mel_Encoder and Mel_Decoder."
            )
        print(f"[VIAI-AV] initializing audio branch from: {checkpoint_path}")
        _copy_matching_state(self.Mel_Encoder, checkpoint["Mel_Encoder"], "Mel_Encoder")
        _copy_matching_state(self.Mel_Decoder, checkpoint["Mel_Decoder"], "MelDecoderImage")
        if hasattr(self.Mel_Decoder, "init_deconv_1_1_1"):
            self.Mel_Decoder.init_deconv_1_1_1()
            print("[VIAI-AV] initialized MelDecoderImage fusion stem from audio decoder stem")
        if self.use_gan and "netD" in checkpoint:
            _copy_matching_state(self.netD, checkpoint["netD"], "MelDiscriminator")
        elif self.use_gan:
            print(
                "[VIAI-AV] source checkpoint has no netD; "
                "MelDiscriminator remains randomly initialized and will be trained by VIAI-AV"
            )
        return int(checkpoint.get("global_step", 0)), int(checkpoint.get("global_epoch", 0))
