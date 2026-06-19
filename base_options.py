import argparse
import os
import torch.nn as nn


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        # Basic
        self.parser.add_argument("--name", type=str, default="VIAI-AV")
        self.parser.add_argument("--isTrain", type=bool, default=True)
        self.parser.add_argument("--data_root", type=str, default="/root/shared-nvme/data")
        self.parser.add_argument("--image_path", type=str, default="/root/shared-nvme/data")
        self.parser.add_argument("--speaker_id", type=int, default=None)
        self.parser.add_argument("--test_size", type=float, default=0.05)
        self.parser.add_argument("--metadata_name", type=str, default="metadata.csv")
        self.parser.add_argument("--new_split_name", type=str, default="_new_split.txt")
        self.parser.add_argument("--train_split_name", type=str, default="train_av_split.txt")
        self.parser.add_argument("--val_split_name", type=str, default="val_av_split.txt")
        self.parser.add_argument("--test_split_name", type=str, default="test_av_split.txt")

        # Resume / checkpoint
        self.parser.add_argument("--resume", action="store_true")
        self.parser.add_argument("--resume_path", type=str, default=None)
        self.parser.add_argument(
            "--init_from_viai_a",
            type=str,
            default=None,
            help=(
                "Initialize audio-side weights from a VIAI-A or VIAI-A-PatchGAN "
                "checkpoint without inheriting step, epoch, or optimizer state. "
                "For train-viai-av, when omitted it uses the latest VIAI-A "
                "checkpoint for baseline runs; with --use_gan, it prefers "
                "VIAI-A-PatchGAN and falls back to VIAI-A."
            ),
        )
        self.parser.add_argument("--load_pretrain", action="store_true")
        self.parser.add_argument("--wavenet_pretrain", type=str, default=None)
        self.parser.add_argument("--reset_optimizer", action="store_true")
        self.parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--log_event_path", type=str, default=None)
        self.parser.add_argument(
            "--bad_sample_log",
            "--bad-sample-log",
            dest="bad_sample_log",
            type=str,
            default=None,
        )
        self.parser.add_argument(
            "--strict_av_samples",
            "--strict-av-samples",
            dest="strict_av_samples",
            action="store_true",
        )
        self.parser.add_argument(
            "--results_dir",
            type=str,
            default="./checkpoints/viai_a_test_results",
            help="Directory for VIAI-A test JSON files and CSV summaries.",
        )
        self.parser.add_argument("--save_optimizer_state", type=bool, default=True)

        # Runtime
        self.parser.add_argument("--num_workers", type=int, default=4)
        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--pin_memory", type=bool, default=True)
        self.parser.add_argument("--mul_gpu", type=bool, default=True)
        self.parser.add_argument("--cuda_on", type=bool, default=True)

        # Visualizer
        self.parser.add_argument("--display_id", type=int, default=0)
        self.parser.add_argument("--display_freq", type=int, default=200)
        self.parser.add_argument("--print_freq", type=int, default=100)
        self.parser.add_argument("--display_winsize", type=int, default=256)
        self.parser.add_argument("--display_port", type=int, default=8097)
        self.parser.add_argument("--display_single_pane_ncols", type=int, default=0)
        self.parser.add_argument(
            "--metric_freq",
            type=int,
            default=100,
            help="For VIAI-A training, compute CPU-heavy SSIM every N steps.",
        )
        self.parser.add_argument(
            "--tb_image_freq",
            type=int,
            default=500,
            help="For VIAI-A training, write Mel comparison images every N steps.",
        )
        self.parser.add_argument(
            "--tb_image_count",
            type=int,
            default=4,
            help="Maximum number of VIAI-A Mel samples to write to TensorBoard.",
        )

        # Optimization
        self.parser.add_argument("--nepochs", type=int, default=100)
        self.parser.add_argument("--lr", type=float, default=1e-4)
        self.parser.add_argument("--beta1", type=float, default=0.5)
        self.parser.add_argument("--beta2", type=float, default=0.999)
        self.parser.add_argument("--checkpoint_interval", type=int, default=1000)
        self.parser.add_argument("--train_eval_interval", type=int, default=1000)
        self.parser.add_argument("--test_eval_epoch_interval", type=int, default=1)
        self.parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Stop after this many optimizer steps. Intended for local smoke tests only.",
        )

        # Loss weights (paper-style objective)
        self.parser.add_argument(
            "--use_gan",
            action="store_true",
            help="Enable PatchGAN discriminator and GAN loss for VIAI-A or VIAI-AV.",
        )
        self.parser.add_argument("--lambda_gan", type=float, default=1.0)
        self.parser.add_argument(
            "--lambda_recon",
            type=float,
            default=1.0,
            help="Backward-compatible name for the VIAI reconstruction weight beta.",
        )
        self.parser.add_argument(
            "--beta_recon",
            type=float,
            default=None,
            help="Paper-style beta weight for reconstruction loss in L_GAN + beta * L_re.",
        )
        self.parser.add_argument("--lambda_sync", type=float, default=1.0)
        self.parser.add_argument("--lambda_probe", type=float, default=1.0)
        self.parser.add_argument("--sync_margin", type=float, default=1.0)
        self.parser.add_argument(
            "--disable_sync_loss",
            action="store_true",
            help="For VIAI-AV stage 4 ablation, disable contrastive audio-video sync loss.",
        )
        self.parser.add_argument(
            "--disable_probe_loss",
            action="store_true",
            help="For VIAI-AV stage 4 ablation, disable the VIAI-AA' probe branch loss.",
        )
        self.parser.add_argument("--recon_decay_base", type=float, default=0.9)
        self.parser.add_argument("--recon_decay_interval", type=float, default=1000.0)
        self.parser.add_argument("--recon_decay_floor", type=float, default=0.1)
        self.parser.add_argument("--sync_decay_base", type=float, default=0.9)
        self.parser.add_argument("--sync_decay_interval", type=float, default=1000.0)
        self.parser.add_argument("--sync_decay_floor", type=float, default=0.1)
        self.parser.add_argument("--probe_decay_base", type=float, default=None)
        self.parser.add_argument("--probe_decay_interval", type=float, default=None)
        self.parser.add_argument("--probe_decay_floor", type=float, default=None)

        # Missing-region setup (4s inputs, 0.4s~1.0s gaps)
        self.parser.add_argument("--max_time_sec", type=float, default=None)
        self.parser.add_argument("--max_time_steps", type=int, default=64000)
        self.parser.add_argument("--min_blank_frames", type=int, default=20)
        self.parser.add_argument("--max_blank_frames", type=int, default=50)
        self.parser.add_argument(
            "--baseline-mask-manifest",
            "--baseline_mask_manifest",
            dest="baseline_mask_manifest",
            type=str,
            default=None,
            help="Optional fixed per-sample mask manifest for reproducible baseline evaluation.",
        )
        self.parser.add_argument(
            "--baseline-protocol-json",
            "--baseline_protocol_json",
            dest="baseline_protocol_json",
            type=str,
            default=None,
        )
        self.parser.add_argument(
            "--eval-seed",
            "--eval_seed",
            dest="eval_seed",
            type=int,
            default=1234,
        )
        self.parser.add_argument(
            "--deterministic-eval",
            "--deterministic_eval",
            dest="deterministic_eval",
            action="store_true",
        )
        self.parser.add_argument(
            "--eval-branch",
            "--eval_branch",
            dest="eval_branch",
            choices=["av", "probe"],
            default="av",
        )

        # Mel / audio features
        self.parser.add_argument("--sample_rate", type=int, default=16000)
        self.parser.add_argument("--input_type", type=str, default="raw")
        self.parser.add_argument("--quantize_channels", type=int, default=65536)
        self.parser.add_argument("--rescaling", type=bool, default=True)
        self.parser.add_argument("--rescaling_max", type=float, default=0.999)
        self.parser.add_argument("--silence_threshold", type=float, default=2)
        self.parser.add_argument("--cin_channels", type=int, default=80)
        self.parser.add_argument("--num_mels", type=int, default=80)
        self.parser.add_argument("--max_mel_lengths", type=int, default=200)
        self.parser.add_argument("--fft_size", type=int, default=1280)
        self.parser.add_argument("--hop_size", type=int, default=320)
        self.parser.add_argument("--frame_shift_ms", type=float, default=None)
        self.parser.add_argument("--fmin", type=float, default=125.0)
        self.parser.add_argument("--fmax", type=float, default=7600.0)
        self.parser.add_argument("--min_level_db", type=float, default=-100.0)
        self.parser.add_argument("--ref_level_db", type=float, default=20.0)
        self.parser.add_argument("--allow_clipping_in_normalization", type=bool, default=True)

        # Optional Mel-to-waveform output for test/demo runs.
        self.parser.add_argument(
            "--use_vocoder",
            action="store_true",
            help="During test, synthesize wav files from reconstructed Mel spectrograms.",
        )
        self.parser.add_argument(
            "--vocoder_backend",
            type=str,
            default="griffin_lim",
            choices=["griffin_lim"],
            help="Mel-to-waveform backend. Current route B implementation uses Griffin-Lim.",
        )
        self.parser.add_argument(
            "--vocoder_n_iter",
            type=int,
            default=32,
            help="Number of Griffin-Lim iterations when --use_vocoder is enabled.",
        )
        self.parser.add_argument(
            "--vocoder_max_samples",
            type=int,
            default=None,
            help="Optional cap on the number of wav samples generated during one test run.",
        )
        self.parser.add_argument(
            "--vocoder_output_dir",
            type=str,
            default=None,
            help="Optional wav output directory. Defaults to <results_dir>/wav/stepXXXXXXXXX.",
        )

        # Visual stream and fusion encoder
        self.parser.add_argument("--feature_length", type=int, default=256)
        self.parser.add_argument("--length_feature", type=int, default=256)
        self.parser.add_argument("--image_size", type=int, default=256)
        self.parser.add_argument("--image_rescal_size", type=int, default=256)
        self.parser.add_argument("--image_channel_size", type=int, default=3)
        self.parser.add_argument("--image_hope_size", type=int, default=1)
        self.parser.add_argument("--visual_frame_count", type=int, default=50)
        self.parser.add_argument("--visual_frame_interval_sec", type=float, default=0.08)
        self.parser.add_argument("--image", type=bool, default=True)
        self.parser.add_argument("--flow", type=bool, default=True)
        self.parser.add_argument("--load_num", type=int, default=1)
        self.parser.add_argument("--resnet_pretrain", type=bool, default=False)
        self.parser.add_argument("--resnet_pretrain_path", type=str, default=None)

        # Decoder / GAN / WaveNet
        self.parser.add_argument("--norm_type", type=str, default="batch")
        self.parser.add_argument("--out_channels", type=int, default=30)
        self.parser.add_argument("--decode_layers", type=int, default=24)
        self.parser.add_argument("--decode_stacks", type=int, default=3)
        self.parser.add_argument("--residual_channels", type=int, default=256)
        self.parser.add_argument("--gate_channels", type=int, default=256)
        self.parser.add_argument("--skip_out_channels", type=int, default=256)
        self.parser.add_argument("--kernel_size", type=int, default=3)
        self.parser.add_argument("--dropout", type=float, default=0.05)
        self.parser.add_argument("--weight_normalization", type=bool, default=True)
        self.parser.add_argument("--upsample_conditional_features", type=bool, default=True)
        self.parser.add_argument("--upsample_scales", type=int, nargs="+", default=[5, 8, 8])
        self.parser.add_argument("--freq_axis_kernel_size", type=int, default=3)
        self.parser.add_argument("--n_speakers", type=int, default=None)
        self.parser.add_argument("--gin_channels", type=int, default=-1)
        self.parser.add_argument("--file_channel", type=int, default=-1)
        self.parser.add_argument("--log_scale_min", type=float, default=-7.0)

        # -------------------------------------------------------------------
        # UQ-AVInpaint (P3-P6) — Mel autoencoder, latent diffusion, multi-hypothesis
        # -------------------------------------------------------------------
        self.parser.add_argument(
            "--ae_checkpoint",
            type=str,
            default=None,
            help="Path to a pre-trained Mel AE checkpoint (required for P3+).",
        )
        self.parser.add_argument(
            "--ae_latent_dim",
            type=int,
            default=8,
            help="Mel AE latent channel dimension (used when building the AE/UNet).",
        )
        self.parser.add_argument(
            "--ae_base_channels",
            type=int,
            default=32,
            help="Base channel count for Mel AE encoder/decoder.",
        )
        self.parser.add_argument(
            "--ae_warmup_steps",
            type=int,
            default=2000,
            help="Mel AE warmup steps with L1-only loss before enabling gradient/boundary terms.",
        )
        self.parser.add_argument(
            "--uq_no_video",
            action="store_true",
            help="Run in audio-only mode (use VideoConditionDummy).",
        )
        self.parser.add_argument(
            "--uq_video_dim",
            type=int,
            default=256,
            help="Video token dimension for cross-attention in the diffusion U-Net.",
        )
        self.parser.add_argument(
            "--uq_unet_base_channels",
            type=int,
            default=64,
            help="Base channel count for the latent diffusion U-Net.",
        )
        self.parser.add_argument(
            "--uq_time_emb_dim",
            type=int,
            default=256,
            help="Dimension of the sinusoidal time embedding in the diffusion U-Net.",
        )
        self.parser.add_argument(
            "--uq_attn_heads",
            type=int,
            default=4,
            help="Number of attention heads for cross-attention over video tokens.",
        )
        self.parser.add_argument(
            "--uq_lambda_boundary",
            type=float,
            default=0.1,
            help="Weight of the boundary-consistency auxiliary loss.",
        )
        self.parser.add_argument(
            "--uq_lambda_sync",
            type=float,
            default=0.0,
            help="Weight of the audio-visual sync auxiliary loss (P5+).",
        )
        self.parser.add_argument(
            "--uq_lr",
            type=float,
            default=None,
            help="Learning rate for UQ diffusion training (falls back to --lr).",
        )
        self.parser.add_argument(
            "--uq_grad_clip",
            type=float,
            default=1.0,
            help="Max gradient norm for diffusion U-Net training.",
        )
        self.parser.add_argument(
            "--uq_inference_steps",
            type=int,
            default=50,
            help="Number of DDIM steps during inference.",
        )
        self.parser.add_argument(
            "--uq_diffusion_timesteps",
            type=int,
            default=1000,
            help="Total diffusion timesteps (training schedule length).",
        )
        self.parser.add_argument(
            "--uq_beta_start",
            type=float,
            default=1e-4,
            help="Initial beta for the UQ diffusion noise schedule.",
        )
        self.parser.add_argument(
            "--uq_beta_end",
            type=float,
            default=0.02,
            help="Final beta for the UQ diffusion noise schedule.",
        )
        self.parser.add_argument(
            "--uq_num_candidates",
            type=int,
            default=1,
            help="Number of inpainting candidates K (P4: 4/8/16).",
        )
        self.parser.add_argument(
            "--uq_ddim_eta",
            type=float,
            default=0.0,
            help="DDIM stochasticity parameter (0 = deterministic).",
        )
        self.parser.add_argument(
            "--uq_beta_schedule",
            type=str,
            choices=["linear", "cosine"],
            default="linear",
            help="Diffusion noise schedule type.",
        )
        self.parser.add_argument(
            "--uq_schedule_type",
            type=str,
            choices=["linear", "cosine"],
            default="linear",
            help="Alias for --uq_beta_schedule.",
        )
        self.parser.add_argument(
            "--uq_lambda_diversity",
            type=float,
            default=0.0,
            help="Diversity-target loss weight (P5+).",
        )
        self.parser.add_argument(
            "--uq_video_degradation",
            type=str,
            default="original",
            choices=["original", "blur", "occlusion", "frame_drop",
                     "temporal_shift", "wrong_video", "no_video"],
            help="Visual degradation condition for evaluation.",
        )
        self.parser.add_argument(
            "--uq_metadata_dir",
            type=str,
            default=None,
            help="Optional UQ metadata directory containing mask manifests and onsets.",
        )
        self.parser.add_argument(
            "--aq_audio_branch_channels",
            type=int,
            default=256,
            help="Base channels for the internal audio-condition encoder.",
        )
        self.parser.add_argument(
            "--test_image_batch_interval",
            type=int,
            default=10,
            help="Write Mel comparison images every N validation/test batches.",
        )

        self.initialized = True
        return self.parser

    def gather_options(self, args=None):
        if not self.initialized:
            self.initialize()
        # Ignore unknown args to make config import-safe in notebooks/tools.
        opt, unknown = self.parser.parse_known_args(args=args)
        deprecated_args = [arg for arg in unknown if arg == "--beta_gan" or arg.startswith("--beta_gan=")]
        if deprecated_args:
            self.parser.error(
                "--beta_gan has been removed. Use --lambda_recon for the "
                "reconstruction loss weight."
            )
        return opt

    def parse(self, args=None):
        opt = self.gather_options(args=args)

        if opt.beta_recon is None:
            opt.beta_recon = opt.lambda_recon
        else:
            opt.lambda_recon = opt.beta_recon

        if opt.length_feature != opt.feature_length:
            opt.length_feature = opt.feature_length
        opt.feature_length = opt.length_feature

        # Keep compatibility with two historical field names.
        if opt.file_channel != -1 and opt.gin_channels == -1:
            opt.gin_channels = opt.file_channel
        if opt.gin_channels != -1 and opt.file_channel == -1:
            opt.file_channel = opt.gin_channels

        if opt.max_time_steps is not None and opt.hop_size > 0:
            opt.max_mel_lengths = opt.max_time_steps // opt.hop_size
        opt.num_mels = opt.cin_channels

        if opt.probe_decay_base is None:
            opt.probe_decay_base = opt.sync_decay_base
        if opt.probe_decay_interval is None:
            opt.probe_decay_interval = opt.sync_decay_interval
        if opt.probe_decay_floor is None:
            opt.probe_decay_floor = opt.sync_decay_floor

        if opt.norm_type.lower() == "instance":
            opt.normlayer = nn.InstanceNorm2d
        else:
            opt.normlayer = nn.BatchNorm2d

        if opt.uq_schedule_type != "linear" and opt.uq_beta_schedule == "linear":
            opt.uq_beta_schedule = opt.uq_schedule_type
        opt.uq_schedule_type = opt.uq_beta_schedule

        if not opt.checkpoint_dir:
            opt.checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not opt.log_event_path:
            opt.log_event_path = os.path.join(opt.checkpoint_dir, "events")
        if not opt.bad_sample_log:
            opt.bad_sample_log = os.path.join(opt.data_root, "viai_av_bad_samples.csv")

        return opt
