import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.io import wavfile
from tensorboardX import SummaryWriter
from torch.utils import data as data_utils
from tqdm import tqdm

import Options_inpainting
from utils.hifigan import (
    HifiGanGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
    get_hifigan_config,
    load_matching_generator_checkpoint,
    normalized_mel_to_log_condition,
    save_hifigan_checkpoint,
    splice_missing_audio,
    waveform_to_log_mel_condition,
)
from utils.vocoder import peak_normalize


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available() and bool(getattr(hparams, "cuda_on", True))
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def read_split_rows(data_root, split_name):
    split_path = os.path.join(data_root, split_name)
    rows = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 split columns in {split_path}: {line}")
            sample_dir, mel_path, audio_path, mel_frames = parts
            rows.append(
                {
                    "sample_dir": sample_dir,
                    "mel_path": os.path.join(data_root, mel_path),
                    "audio_path": os.path.join(data_root, audio_path),
                    "mel_frames": int(mel_frames),
                }
            )
    return rows


def pad_2d_time_first(array, target_len):
    if array.shape[0] >= target_len:
        return array[:target_len]
    pad_len = target_len - array.shape[0]
    return np.pad(array, [(0, pad_len), (0, 0)], mode="constant", constant_values=0.0)


def pad_1d(array, target_len):
    if array.shape[0] >= target_len:
        return array[:target_len]
    pad_len = target_len - array.shape[0]
    return np.pad(array, (0, pad_len), mode="constant", constant_values=0.0)


class VocoderSplitDataset(data_utils.Dataset):
    def __init__(self, data_root, split_name, train=True, hparams=hparams):
        self.rows = read_split_rows(data_root, split_name)
        self.train = train
        self.hparams = hparams
        self.segment_mel_frames = int(getattr(hparams, "hifigan_segment_mel_frames", 32))
        if self.segment_mel_frames <= 0:
            raise ValueError("--hifigan_segment_mel_frames must be > 0.")
        self.hop_size = int(hparams.hop_size)

    def __len__(self):
        return len(self.rows)

    def _choose_start(self, mel_frames):
        if mel_frames <= self.segment_mel_frames:
            return 0
        max_start = mel_frames - self.segment_mel_frames
        if self.train:
            return int(np.random.randint(0, max_start + 1))
        return max_start // 2

    def __getitem__(self, index):
        row = self.rows[index]
        mel = np.load(row["mel_path"]).astype(np.float32)
        audio = np.load(row["audio_path"]).astype(np.float32)
        start = self._choose_start(mel.shape[0])
        end = start + self.segment_mel_frames
        mel_window = pad_2d_time_first(mel[start:end], self.segment_mel_frames)
        audio_start = start * self.hop_size
        audio_len = self.segment_mel_frames * self.hop_size
        audio_window = pad_1d(audio[audio_start : audio_start + audio_len], audio_len)
        return {
            "mel": torch.from_numpy(mel_window.T).float(),
            "audio": torch.from_numpy(audio_window[None]).float(),
            "path": row["sample_dir"],
        }


def collate_fn(batch):
    return {
        "mel": torch.stack([item["mel"] for item in batch], dim=0),
        "audio": torch.stack([item["audio"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }


def make_loader(split_name, train):
    split_path = os.path.join(hparams.data_root, split_name)
    if not os.path.exists(split_path):
        if train:
            raise RuntimeError(f"Training split does not exist: {split_path}")
        print(f"[HiFi-GAN] validation split missing, skipping: {split_path}")
        return None
    dataset = VocoderSplitDataset(hparams.data_root, split_name, train=train, hparams=hparams)
    if len(dataset) == 0:
        if train:
            raise RuntimeError(f"Training split is empty: {split_path}")
        return None
    loader = data_utils.DataLoader(
        dataset,
        batch_size=int(hparams.batch_size),
        shuffle=train,
        num_workers=int(hparams.num_workers),
        pin_memory=bool(getattr(hparams, "pin_memory", True)),
        collate_fn=collate_fn,
        drop_last=train and len(dataset) >= int(hparams.batch_size),
    )
    print(f"[HiFi-GAN] {split_name}: samples={len(dataset)} train={train}")
    return loader


def _align_mel_frames(generated_mel, target_mel):
    frames = min(generated_mel.size(-1), target_mel.size(-1))
    return generated_mel[..., :frames], target_mel[..., :frames]


def train_step(batch, generator, mpd, msd, optim_g, optim_d):
    mel = batch["mel"].to(device)
    real = batch["audio"].to(device)
    condition = normalized_mel_to_log_condition(mel, hparams).to(device)
    fake = generator(condition)
    if fake.size(-1) != real.size(-1):
        samples = min(fake.size(-1), real.size(-1))
        fake = fake[..., :samples]
        real = real[..., :samples]

    optim_d.zero_grad(set_to_none=True)
    real_outputs, fake_outputs, _, _ = mpd(real, fake.detach())
    loss_disc_mpd = discriminator_loss(real_outputs, fake_outputs)
    real_outputs, fake_outputs, _, _ = msd(real, fake.detach())
    loss_disc_msd = discriminator_loss(real_outputs, fake_outputs)
    loss_disc = loss_disc_mpd + loss_disc_msd
    loss_disc.backward()
    optim_d.step()

    optim_g.zero_grad(set_to_none=True)
    fake_mel = waveform_to_log_mel_condition(fake, hparams)
    fake_mel, target_mel = _align_mel_frames(fake_mel, condition)
    loss_mel = torch.nn.functional.l1_loss(fake_mel, target_mel)

    real_outputs, fake_outputs, real_maps, fake_maps = mpd(real, fake)
    loss_fm_mpd = feature_loss(real_maps, fake_maps)
    loss_gen_mpd = generator_loss(fake_outputs)
    real_outputs, fake_outputs, real_maps, fake_maps = msd(real, fake)
    loss_fm_msd = feature_loss(real_maps, fake_maps)
    loss_gen_msd = generator_loss(fake_outputs)
    loss_fm = loss_fm_mpd + loss_fm_msd
    loss_adv = loss_gen_mpd + loss_gen_msd
    loss_gen = (
        loss_adv
        + float(hparams.hifigan_feature_loss_weight) * loss_fm
        + float(hparams.hifigan_mel_loss_weight) * loss_mel
    )
    loss_gen.backward()
    optim_g.step()

    return {
        "loss_gen": float(loss_gen.detach().cpu()),
        "loss_disc": float(loss_disc.detach().cpu()),
        "loss_mel": float(loss_mel.detach().cpu()),
        "loss_adv": float(loss_adv.detach().cpu()),
        "loss_fm": float(loss_fm.detach().cpu()),
    }


def save_eval_audio(val_loader, generator, step):
    if val_loader is None or int(getattr(hparams, "hifigan_eval_samples", 0)) <= 0:
        return
    output_dir = os.path.join(hparams.checkpoint_dir, "audio", f"step{step:09d}")
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    written = 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            real = batch["audio"].to(device)
            condition = normalized_mel_to_log_condition(mel, hparams).to(device)
            fake = generator(condition)
            for index in range(fake.size(0)):
                if written >= int(hparams.hifigan_eval_samples):
                    generator.train()
                    return
                target = real[index, 0].detach().cpu().numpy()
                generated = fake[index, 0].detach().cpu().numpy()
                mask = torch.ones(1, 1, mel.size(1), mel.size(2))
                spliced = splice_missing_audio(
                    target,
                    generated,
                    mask,
                    hparams,
                    crossfade_ms=float(getattr(hparams, "vocoder_crossfade_ms", 20.0)),
                )
                wavfile.write(
                    os.path.join(output_dir, f"{written:03d}_target.wav"),
                    int(hparams.sample_rate),
                    (peak_normalize(target) * 32767.0).astype(np.int16),
                )
                wavfile.write(
                    os.path.join(output_dir, f"{written:03d}_generated.wav"),
                    int(hparams.sample_rate),
                    (peak_normalize(generated) * 32767.0).astype(np.int16),
                )
                wavfile.write(
                    os.path.join(output_dir, f"{written:03d}_spliced.wav"),
                    int(hparams.sample_rate),
                    (peak_normalize(spliced) * 32767.0).astype(np.int16),
                )
                written += 1
    generator.train()


def main():
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)
    log_event_path = hparams.log_event_path or os.path.join(
        hparams.checkpoint_dir,
        "events_hifigan_" + str(datetime.now()).replace(" ", "_"),
    )
    writer = SummaryWriter(log_dir=log_event_path)

    config = get_hifigan_config(hparams)
    generator = HifiGanGenerator(config).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    if hparams.hifigan_pretrained_generator:
        load_matching_generator_checkpoint(hparams.hifigan_pretrained_generator, generator, device=device)

    optim_g = torch.optim.AdamW(generator.parameters(), lr=float(hparams.lr), betas=(hparams.beta1, hparams.beta2))
    disc_params = list(mpd.parameters()) + list(msd.parameters())
    optim_d = torch.optim.AdamW(disc_params, lr=float(hparams.lr), betas=(hparams.beta1, hparams.beta2))

    train_loader = make_loader(hparams.train_split_name, train=True)
    val_loader = make_loader(hparams.val_split_name, train=False)

    max_steps = hparams.max_steps if hparams.max_steps is not None else hparams.max_train_steps
    if max_steps is None:
        max_steps = 50000
    max_steps = int(max_steps)
    global_step = 0
    generator.train()
    mpd.train()
    msd.train()

    try:
        while global_step < max_steps:
            progress = tqdm(train_loader, desc=f"[HiFi-GAN] step={global_step}/{max_steps}", dynamic_ncols=True)
            for batch in progress:
                metrics = train_step(batch, generator, mpd, msd, optim_g, optim_d)
                global_step += 1
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                progress.set_postfix(step=global_step, **{k: f"{v:.4f}" for k, v in metrics.items()})
                if global_step % int(hparams.print_freq) == 0:
                    tqdm.write(
                        "[HiFi-GAN] "
                        f"step={global_step} "
                        f"loss_gen={metrics['loss_gen']:.6f} "
                        f"loss_disc={metrics['loss_disc']:.6f} "
                        f"loss_mel={metrics['loss_mel']:.6f}"
                    )
                if global_step % int(hparams.checkpoint_interval) == 0 or global_step == max_steps:
                    save_hifigan_checkpoint(
                        os.path.join(hparams.checkpoint_dir, f"hifigan_generator_step{global_step:09d}.pth.tar"),
                        generator,
                        mpd=mpd,
                        msd=msd,
                        optim_g=optim_g,
                        optim_d=optim_d,
                        step=global_step,
                        config=config,
                    )
                    save_eval_audio(val_loader, generator, global_step)
                if global_step >= max_steps:
                    break
    except KeyboardInterrupt:
        print("[HiFi-GAN] interrupted; saving latest checkpoint.")
    finally:
        save_hifigan_checkpoint(
            os.path.join(hparams.checkpoint_dir, f"hifigan_generator_step{global_step:09d}.pth.tar"),
            generator,
            mpd=mpd,
            msd=msd,
            optim_g=optim_g,
            optim_d=optim_d,
            step=global_step,
            config=config,
        )
        writer.close()
    print(f"[HiFi-GAN] finished at step={global_step}")


if __name__ == "__main__":
    main()
    sys.exit(0)
