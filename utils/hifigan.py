import math
import os
from types import SimpleNamespace

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


LRELU_SLOPE = 0.1


def get_hifigan_config(hparams):
    return SimpleNamespace(
        num_mels=int(getattr(hparams, "num_mels", getattr(hparams, "cin_channels", 80))),
        sample_rate=int(getattr(hparams, "sample_rate", 16000)),
        fft_size=int(getattr(hparams, "fft_size", 1280)),
        hop_size=int(getattr(hparams, "hop_size", 320)),
        fmin=float(getattr(hparams, "fmin", 125.0)),
        fmax=float(getattr(hparams, "fmax", 7600.0)),
        min_level_db=float(getattr(hparams, "min_level_db", -100.0)),
        ref_level_db=float(getattr(hparams, "ref_level_db", 20.0)),
        upsample_rates=[8, 8, 5],
        upsample_kernel_sizes=[16, 16, 10],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
    )


def normalized_mel_to_log_condition(mel, hparams, eps=1e-5):
    if not torch.is_tensor(mel):
        mel = torch.as_tensor(mel, dtype=torch.float32)
    mel = mel.float()
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    if mel.dim() == 4:
        if mel.size(1) != 1:
            raise ValueError("Expected 4D Mel tensor with singleton channel axis.")
        mel = mel[:, 0]
    num_mels = int(getattr(hparams, "num_mels", getattr(hparams, "cin_channels", 80)))
    if mel.size(1) != num_mels and mel.size(2) == num_mels:
        mel = mel.transpose(1, 2)
    if mel.size(1) != num_mels:
        raise ValueError(f"Expected {num_mels} Mel bins, got shape {tuple(mel.shape)}.")

    min_level_db = float(getattr(hparams, "min_level_db", -100.0))
    ref_level_db = float(getattr(hparams, "ref_level_db", 20.0))
    mel_db = torch.clamp(mel, 0.0, 1.0) * (-min_level_db) + min_level_db
    mel_amp = torch.pow(torch.tensor(10.0, device=mel.device, dtype=mel.dtype), (mel_db + ref_level_db) * 0.05)
    return torch.log(torch.clamp(mel_amp, min=eps))


def _get_mel_basis(hparams, device, dtype):
    key = (
        int(getattr(hparams, "sample_rate", 16000)),
        int(getattr(hparams, "fft_size", 1280)),
        int(getattr(hparams, "num_mels", getattr(hparams, "cin_channels", 80))),
        float(getattr(hparams, "fmin", 125.0)),
        float(getattr(hparams, "fmax", 7600.0)),
    )
    if not hasattr(_get_mel_basis, "cache"):
        _get_mel_basis.cache = {}
    if key not in _get_mel_basis.cache:
        basis = librosa.filters.mel(
            sr=key[0],
            n_fft=key[1],
            n_mels=key[2],
            fmin=key[3],
            fmax=key[4],
        ).astype(np.float32)
        _get_mel_basis.cache[key] = torch.from_numpy(basis)
    return _get_mel_basis.cache[key].to(device=device, dtype=dtype)


def waveform_to_log_mel_condition(wav, hparams, eps=1e-5):
    if wav.dim() == 3:
        wav = wav[:, 0]
    n_fft = int(getattr(hparams, "fft_size", 1280))
    hop = int(getattr(hparams, "hop_size", 320))
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = torch.abs(spec)
    mel_basis = _get_mel_basis(hparams, wav.device, wav.dtype)
    mel = torch.matmul(mel_basis, magnitude)
    return torch.log(torch.clamp(mel, min=eps))


def init_weights(module, mean=0.0, std=0.01):
    classname = module.__class__.__name__
    if "Conv" in classname:
        module.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=(kernel_size * dilation - dilation) // 2,
                    )
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in dilations
            ]
        )
        self.apply(init_weights)

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = conv1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = conv2(x)
            x = x + residual
        return x


class HifiGanGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(config.num_mels, config.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = config.upsample_initial_channel
        for index, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            out_channels = channels // 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        out_channels,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            for kernel_size, dilations in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(out_channels, kernel_size, dilations))
            channels = out_channels
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3))
        self.apply(init_weights)

    def forward(self, mel_condition):
        x = self.conv_pre(mel_condition)
        for index, upsample in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = upsample(x)
            xs = None
            for block_index in range(self.num_kernels):
                block = self.resblocks[index * self.num_kernels + block_index]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        batch, channels, time = x.shape
        if time % self.period != 0:
            pad = self.period - (time % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            time += pad
        x = x.view(batch, channels, time // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(period) for period in [2, 3, 5, 7, 11]])

    def forward(self, real, fake):
        real_outputs, fake_outputs, real_maps, fake_maps = [], [], [], []
        for discriminator in self.discriminators:
            real_output, real_map = discriminator(real)
            fake_output, fake_map = discriminator(fake)
            real_outputs.append(real_output)
            fake_outputs.append(fake_output)
            real_maps.append(real_map)
            fake_maps.append(fake_map)
        return real_outputs, fake_outputs, real_maps, fake_maps


class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(self, real, fake):
        real_outputs, fake_outputs, real_maps, fake_maps = [], [], [], []
        for index, discriminator in enumerate(self.discriminators):
            if index != 0:
                real = self.meanpools[index - 1](real)
                fake = self.meanpools[index - 1](fake)
            real_output, real_map = discriminator(real)
            fake_output, fake_map = discriminator(fake)
            real_outputs.append(real_output)
            fake_outputs.append(fake_output)
            real_maps.append(real_map)
            fake_maps.append(fake_map)
        return real_outputs, fake_outputs, real_maps, fake_maps


def feature_loss(real_maps, fake_maps):
    loss = 0.0
    for real_group, fake_group in zip(real_maps, fake_maps):
        for real, fake in zip(real_group, fake_group):
            loss = loss + F.l1_loss(fake, real.detach())
    return loss * 2.0


def discriminator_loss(real_outputs, fake_outputs):
    loss = 0.0
    for real, fake in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((1.0 - real) ** 2) + torch.mean(fake ** 2)
    return loss


def generator_loss(fake_outputs):
    loss = 0.0
    for fake in fake_outputs:
        loss = loss + torch.mean((1.0 - fake) ** 2)
    return loss


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("generator", "model_g", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_matching_generator_checkpoint(path, generator, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    source = _checkpoint_state_dict(checkpoint)
    if not isinstance(source, dict):
        raise ValueError(f"Unsupported HiFi-GAN checkpoint format: {path}")

    target = generator.state_dict()
    copied, skipped_shape, skipped_missing = 0, 0, 0
    for name, value in source.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("generator."):
            name = name[len("generator.") :]
        if name not in target:
            skipped_missing += 1
            continue
        if target[name].shape != value.shape:
            skipped_shape += 1
            continue
        target[name].copy_(value)
        copied += 1
    generator.load_state_dict(target)
    report = {
        "loaded": copied,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "path": os.path.abspath(path),
    }
    print(
        "[HiFi-GAN] partial checkpoint load: "
        f"loaded={copied} skipped_shape={skipped_shape} "
        f"skipped_missing={skipped_missing} path={path}"
    )
    return report


def save_hifigan_checkpoint(path, generator, mpd=None, msd=None, optim_g=None, optim_d=None, step=0, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "generator": generator.state_dict(),
        "step": int(step),
        "config": vars(config) if config is not None and hasattr(config, "__dict__") else config,
    }
    if mpd is not None:
        payload["mpd"] = mpd.state_dict()
    if msd is not None:
        payload["msd"] = msd.state_dict()
    if optim_g is not None:
        payload["optim_g"] = optim_g.state_dict()
    if optim_d is not None:
        payload["optim_d"] = optim_d.state_dict()
    torch.save(payload, path)
    return path


def infer_hifigan_waveform(generator, mel, hparams, device):
    generator.eval()
    with torch.no_grad():
        condition = normalized_mel_to_log_condition(mel, hparams).to(device)
        wav = generator(condition)
    return wav.squeeze(1).detach().cpu().numpy()


def missing_span_from_mask(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim == 4:
        array = array[0, 0]
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D/3D/4D mask, got shape {array.shape}.")
    active = np.where(array.mean(axis=0) > 0.5)[0]
    if active.size == 0:
        return None
    return int(active[0]), int(active[-1] + 1)


def splice_missing_audio(original, generated, mask, hparams, crossfade_ms=20.0):
    original = np.asarray(original, dtype=np.float32).reshape(-1)
    generated = np.asarray(generated, dtype=np.float32).reshape(-1)
    target_length = original.shape[0]
    if generated.shape[0] < target_length:
        generated = np.pad(generated, (0, target_length - generated.shape[0]), mode="constant")
    else:
        generated = generated[:target_length]

    span = missing_span_from_mask(mask)
    if span is None:
        return original.copy()
    hop = int(getattr(hparams, "hop_size", 320))
    start = max(0, span[0] * hop)
    end = min(target_length, span[1] * hop)
    if end <= start:
        return original.copy()

    output = original.copy()
    output[start:end] = generated[start:end]
    fade = int(float(crossfade_ms) * 0.001 * int(getattr(hparams, "sample_rate", 16000)))
    fade = max(0, min(fade, (end - start) // 2))
    if fade > 0:
        fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        left = slice(start, start + fade)
        output[left] = original[left] * (1.0 - fade_in) + generated[left] * fade_in
        fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        right = slice(end - fade, end)
        output[right] = original[right] * (1.0 - fade_out) + generated[right] * fade_out
    return output
