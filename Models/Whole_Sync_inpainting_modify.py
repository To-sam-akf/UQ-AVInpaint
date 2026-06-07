import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from Data_loaders import mel_loader
from loss_functions import GANLoss, L2ContrastiveLoss
from networks import Discriminator_Networks
from networks import Image_Embedding
from networks import Inpainting_Networks
from networks import New_Inpainting_Networks
from utils import util


class AudioModel(object):
    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")
        self.use_visual = bool(getattr(hparams, "image", True) or getattr(hparams, "flow", True))

        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams).to(self.device)
        if self.use_visual:
            self.VideoEncoder = Image_Embedding.ImageEmbedding(hparams=hparams).to(self.device)
            self.Mel_Decoder = New_Inpainting_Networks.MelDecoderImage(hparams=hparams).to(self.device)
        else:
            self.VideoEncoder = None
            self.Mel_Decoder = New_Inpainting_Networks.MelDecoder(hparams=hparams).to(self.device)
        self.netD = Discriminator_Networks.MelDiscriminator().to(self.device)

        self.criterion_gan = GANLoss(use_lsgan=False, device=self.device)
        self.criterion_sync = L2ContrastiveLoss(
            margin=getattr(hparams, "sync_margin", 1.0),
            max_violation=False,
        )
        self.criterion_l1 = nn.L1Loss()

        generator_params = list(self.Mel_Encoder.parameters()) + list(self.Mel_Decoder.parameters())
        if self.VideoEncoder is not None:
            generator_params += list(self.VideoEncoder.parameters())

        self.optimizer_G = torch.optim.Adam(
            generator_params,
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
        )
        self.current_lr = hparams.lr

        self.train = 1
        self.update_wavenet = False
        self.blank_length = getattr(hparams, "min_blank_frames", 20)

        self.reconstruct_loss_item = 0.0
        self.loss_mel_L1_item = 0.0
        self.EmbeddingL2_item = 0.0

        self.mel_net_norm = torch.zeros(1, hparams.length_feature, device=self.device)
        self.video_net_norm = torch.zeros(1, hparams.length_feature, device=self.device)

    def _to_device(self, tensor):
        if tensor is None:
            return None
        return tensor.float().to(self.device)

    def _eta(self, step, base, interval, floor):
        if interval <= 0:
            return floor
        return max(floor, base ** (float(step) / float(interval)))

    def get_blank_space_length(self, global_step):
        min_blank = max(1, int(getattr(self.hparams, "min_blank_frames", 20)))
        max_blank = max(min_blank, int(getattr(self.hparams, "max_blank_frames", 50)))
        self.blank_length = random.randint(min_blank, max_blank)
        return self.blank_length

    def set_inputs(self, data):
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
            raise ValueError("Local conditioning mel target is required for VIAI training")

        self.mel_input, self.missing_mask, self.missing_span = mel_loader.corrupt_mel_spectrogram(
            self.mel_target, self.blank_length
        )
        self.mel_target_4d = self.mel_target.unsqueeze(1)
        self.mel_input_4d = self.mel_input.unsqueeze(1)
        self.missing_mask = self.missing_mask.to(self.device)

    def _forward_inpainter(self):
        mel_features = self.Mel_Encoder(self.mel_input)
        if self.use_visual and self.VideoEncoder is not None:
            video_feature = self.VideoEncoder(self.video_batch, self.flow_batch)
            mel_pred = self.Mel_Decoder(mel_features, self.mel_input_4d.size(), video_feature)
            video_feat_flat = video_feature.flatten(1)
        else:
            video_feat_flat = None
            mel_pred = self.Mel_Decoder(mel_features, self.mel_input_4d.size())

        with torch.no_grad():
            target_features = self.Mel_Encoder(self.mel_target)

        input_feat_flat = mel_features[-1].flatten(1)
        target_feat_flat = target_features[-1].flatten(1)

        self.mel_input_feature = input_feat_flat
        self.mel_target_feature = target_feat_flat
        self.video_feature = video_feat_flat

        self.mel_net_norm = util.l2_norm(target_feat_flat)
        if video_feat_flat is None:
            self.video_net_norm = self.mel_net_norm.detach()
        else:
            self.video_net_norm = util.l2_norm(video_feat_flat)

        self.mel_pred = mel_pred
        return mel_pred

    def _compute_losses(self, global_step):
        recon_full = self.criterion_l1(self.mel_pred, self.mel_target_4d)
        masked_abs = torch.abs(self.mel_pred - self.mel_target_4d) * self.missing_mask
        denom = torch.clamp(self.missing_mask.sum(), min=1.0)
        recon_missing = masked_abs.sum() / denom
        eta1 = self._eta(
            global_step,
            getattr(self.hparams, "recon_decay_base", 0.9),
            getattr(self.hparams, "recon_decay_interval", 1000.0),
            getattr(self.hparams, "recon_decay_floor", 0.1),
        )
        self.loss_mel_L1 = eta1 * recon_full + recon_missing

        pred_fake = self.netD(self.mel_pred)
        self.loss_G_GAN = self.criterion_gan(pred_fake, True)

        if self.use_visual and self.video_feature is not None:
            eta2 = self._eta(
                global_step,
                getattr(self.hparams, "sync_decay_base", 0.9),
                getattr(self.hparams, "sync_decay_interval", 1000.0),
                getattr(self.hparams, "sync_decay_floor", 0.1),
            )
            self.EmbeddingL2 = eta2 * self.criterion_sync(self.mel_net_norm.detach(), self.video_net_norm)
        else:
            self.EmbeddingL2 = torch.zeros(1, device=self.device, dtype=self.loss_mel_L1.dtype).squeeze(0)

        lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
        lambda_sync = getattr(self.hparams, "lambda_sync", 1.0)
        lambda_recon = getattr(self.hparams, "lambda_recon", 1.0)
        self.loss_G = (
            lambda_recon * self.loss_mel_L1
            + lambda_gan * self.loss_G_GAN
            + lambda_sync * self.EmbeddingL2
        )

        pred_real = self.netD(self.mel_target_4d)
        pred_fake_detach = self.netD(self.mel_pred.detach())
        self.loss_D_real = self.criterion_gan(pred_real, True, softlabel=True)
        self.loss_D_fake = self.criterion_gan(pred_fake_detach, False, softlabel=True)
        self.loss_D = 0.5 * (self.loss_D_real + self.loss_D_fake)

    def optimize_parameters(self, global_step):
        self.Mel_Encoder.train()
        self.Mel_Decoder.train()
        self.netD.train()
        if self.VideoEncoder is not None:
            self.VideoEncoder.train()

        self._forward_inpainter()
        self._compute_losses(global_step)

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.current_lr = self.optimizer_G.param_groups[0]["lr"]

    def test(self):
        self.Mel_Encoder.eval()
        self.Mel_Decoder.eval()
        self.netD.eval()
        if self.VideoEncoder is not None:
            self.VideoEncoder.eval()

        self._forward_inpainter()
        self._compute_losses(global_step=0)

    def eval_model(self, step, eval_dir):
        self.eval_model_test(step, eval_dir)

    def eval_model_test(self, step, eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
        with torch.no_grad():
            self.test()

    def get_loss_items(self):
        self.reconstruct_loss_item = float(self.loss_G.detach().cpu().item())
        self.loss_mel_L1_item = float(self.loss_mel_L1.detach().cpu().item())
        if torch.is_tensor(self.EmbeddingL2):
            self.EmbeddingL2_item = float(self.EmbeddingL2.detach().cpu().item())
        else:
            self.EmbeddingL2_item = float(self.EmbeddingL2)

    def get_current_visuals(self):
        return {
            "input_mel": util.tensor2image(self.mel_input_4d.detach().cpu()),
            "pred_mel": util.tensor2image(self.mel_pred.detach().cpu()),
            "target_mel": util.tensor2image(self.mel_target_4d.detach().cpu()),
        }

    def get_current_errors(self):
        return {
            "loss_mel_l1": self.loss_mel_L1_item,
            "loss_sync": self.EmbeddingL2_item,
            "loss_recon_total": self.reconstruct_loss_item,
        }

    def TF_writer(self, writer, step):
        if writer is None:
            return
        writer.add_scalar("loss/recon_total", self.reconstruct_loss_item, step)
        writer.add_scalar("loss/mel_l1", self.loss_mel_L1_item, step)
        writer.add_scalar("loss/sync", self.EmbeddingL2_item, step)

    def save_inpainting_checkpoint(self, global_step, global_test_step, checkpoint_dir, global_epoch, hparams=None):
        util.save_inpainting_checkpoint(
            self, global_step, global_test_step, checkpoint_dir, global_epoch, hparams=self.hparams
        )

    def load_inpainting_checkpoint(self, checkpoint_path, reset_optimizer=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "Mel_Encoder" in checkpoint:
            self.Mel_Encoder = util.copy_state_dict(checkpoint["Mel_Encoder"], self.Mel_Encoder)
        if "Mel_Decoder" in checkpoint:
            self.Mel_Decoder = util.copy_state_dict(checkpoint["Mel_Decoder"], self.Mel_Decoder)
        if "netD" in checkpoint:
            self.netD = util.copy_state_dict(checkpoint["netD"], self.netD)
        if self.VideoEncoder is not None and "VideoEncoder" in checkpoint:
            self.VideoEncoder = util.copy_state_dict(checkpoint["VideoEncoder"], self.VideoEncoder)

        if not reset_optimizer:
            if "optimizer_G" in checkpoint and checkpoint["optimizer_G"] is not None:
                self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            if "optimizer_D" in checkpoint and checkpoint["optimizer_D"] is not None:
                self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])

        global_step = int(checkpoint.get("global_step", 0))
        global_epoch = int(checkpoint.get("global_epoch", 0))
        global_test_step = int(checkpoint.get("global_test_step", 0))
        return global_step, global_epoch, global_test_step

    def load_part_checkpoint(self):
        pretrain_path = getattr(self.hparams, "wavenet_pretrain", None)
        if pretrain_path is None or not os.path.exists(pretrain_path):
            return
        util.load_part_checkpoint(pretrain_path, self)

    def del_no_need(self):
        # Kept for compatibility with original training loop.
        return
