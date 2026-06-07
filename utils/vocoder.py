import os
import re

import librosa
import numpy as np
from scipy.io import wavfile


def _as_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _as_bct(mel):
    array = _as_numpy(mel).astype(np.float32)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError("Expected 4D Mel tensor with a singleton channel axis.")
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("Expected Mel with shape (B, C, T), (B, 1, C, T), or (C, T).")
    return array


def _as_bt(audio):
    if audio is None:
        return None
    array = _as_numpy(audio).astype(np.float32)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[:, :, 0]
        elif array.shape[1] == 1:
            array = array[:, 0, :]
        else:
            raise ValueError("Expected audio with shape (B, T, 1), (B, 1, T), or (B, T).")
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("Expected audio with shape (B, T), (B, T, 1), or (B, 1, T).")
    return array


def safe_audio_stem(sample_path, fallback="sample"):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_path)).strip("_")
    return stem if stem else fallback


def normalized_mel_to_amplitude(mel, hparams):
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        raise ValueError("Expected a 2D Mel spectrogram.")

    num_mels = int(getattr(hparams, "num_mels", getattr(hparams, "cin_channels", 80)))
    if mel.shape[0] != num_mels and mel.shape[1] == num_mels:
        mel = mel.T
    if mel.shape[0] != num_mels:
        raise ValueError(f"Expected {num_mels} Mel bins, got shape {mel.shape}.")

    min_level_db = float(getattr(hparams, "min_level_db", -100.0))
    ref_level_db = float(getattr(hparams, "ref_level_db", 20.0))
    mel_db = np.clip(mel, 0.0, 1.0) * (-min_level_db) + min_level_db
    return np.power(10.0, (mel_db + ref_level_db) * 0.05).astype(np.float32)


def _fix_waveform_length(wav, target_length):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] > target_length:
        return wav[:target_length]
    if wav.shape[0] < target_length:
        return np.pad(wav, (0, target_length - wav.shape[0]), mode="constant")
    return wav


def peak_normalize(wav, peak=0.99):
    wav = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    max_abs = float(np.max(np.abs(wav))) if wav.size else 0.0
    if max_abs > 0.0:
        wav = wav * (float(peak) / max_abs)
    return np.clip(wav, -float(peak), float(peak)).astype(np.float32)


def mel_to_waveform(mel, hparams, backend="griffin_lim", n_iter=32):
    backend = str(backend).lower()
    if backend not in {"griffin_lim", "griffin-lim", "griffinlim"}:
        raise ValueError(f"Unsupported vocoder backend: {backend}")

    mel_amp = normalized_mel_to_amplitude(mel, hparams)
    sample_rate = int(getattr(hparams, "sample_rate", 16000))
    fft_size = int(getattr(hparams, "fft_size", 1280))
    hop_size = int(getattr(hparams, "hop_size", 320))
    fmin = float(getattr(hparams, "fmin", 125.0))
    fmax = float(getattr(hparams, "fmax", sample_rate / 2.0))

    wav = librosa.feature.inverse.mel_to_audio(
        mel_amp,
        sr=sample_rate,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        fmin=fmin,
        fmax=fmax,
        power=1.0,
        n_iter=int(n_iter),
        center=False,
        dtype=np.float32,
    )
    return _fix_waveform_length(wav, mel_amp.shape[1] * hop_size)


def save_waveform(path, wav, sample_rate, normalize=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if normalize:
        wav = peak_normalize(wav)
    else:
        wav = np.clip(np.nan_to_num(np.asarray(wav, dtype=np.float32)), -1.0, 1.0)
    wavfile.write(path, int(sample_rate), (wav * 32767.0).astype(np.int16))
    return path


def save_vocoder_batch(
    output_dir,
    start_index,
    paths,
    mel_input,
    mel_pred,
    missing_mask,
    target_audio,
    hparams,
    backend="griffin_lim",
    n_iter=32,
    max_items=None,
):
    os.makedirs(output_dir, exist_ok=True)
    mel_input = np.clip(_as_bct(mel_input), 0.0, 1.0)
    mel_pred = np.clip(_as_bct(mel_pred), 0.0, 1.0)
    missing_mask = np.clip(_as_bct(missing_mask), 0.0, 1.0)
    target_audio = _as_bt(target_audio)

    batch_size = min(mel_input.shape[0], mel_pred.shape[0], missing_mask.shape[0])
    if max_items is not None:
        batch_size = min(batch_size, max(0, int(max_items)))

    sample_rate = int(getattr(hparams, "sample_rate", 16000))
    written = []
    for index in range(batch_size):
        sample_path = paths[index] if index < len(paths) else f"sample_{start_index + index}"
        stem = f"{start_index + index:06d}_{safe_audio_stem(sample_path)}"
        reconstructed_mel = mel_input[index] * (1.0 - missing_mask[index]) + mel_pred[index] * missing_mask[index]
        reconstructed_wav = mel_to_waveform(
            reconstructed_mel,
            hparams,
            backend=backend,
            n_iter=n_iter,
        )

        reconstructed_path = os.path.join(output_dir, f"{stem}_reconstructed.wav")
        target_path = os.path.join(output_dir, f"{stem}_target.wav")
        save_waveform(reconstructed_path, reconstructed_wav, sample_rate, normalize=True)
        if target_audio is not None and index < target_audio.shape[0]:
            save_waveform(target_path, target_audio[index], sample_rate, normalize=True)
        else:
            target_path = ""

        written.append(
            {
                "reconstructed": reconstructed_path,
                "target": target_path,
            }
        )
    return written
