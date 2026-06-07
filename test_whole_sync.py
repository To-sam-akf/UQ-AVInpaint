import os
import re
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Data_loaders.audio_loader as data_loader_utils
import Models.Whole_Sync_inpainting_modify as Audio_model
import Options_inpainting
from utils import util


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def checkpoint_step(path):
    match = re.search(r"checkpoint_step(\d+)", os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def resolve_checkpoint_path(resume_path, checkpoint_dir, name):
    search_dirs = []
    if resume_path is not None:
        candidate = os.path.abspath(resume_path)
        if os.path.exists(candidate):
            return candidate
        search_dirs.append(os.path.dirname(candidate))
    search_dirs.append(os.path.abspath(checkpoint_dir))

    candidates = []
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.startswith(f"{name}_checkpoint_step") and filename.endswith(".pth.tar"):
                candidates.append(os.path.join(directory, filename))

    if not candidates:
        raise RuntimeError(
            "No checkpoint found. Pass --resume_path or place a "
            f"{name}_checkpoint_step*.pth.tar file under {checkpoint_dir}."
        )
    return sorted(candidates, key=lambda path: (checkpoint_step(path), os.path.getmtime(path)))[-1]


def evaluate(model, data_loader):
    loss_total = 0.0
    mel_l1_total = 0.0
    sync_total = 0.0
    batch_count = 0
    audio_ebds = []
    image_ebds = []

    for data in tqdm(data_loader, desc="[test] evaluating", unit="batch"):
        model.set_inputs(data)
        with torch.no_grad():
            model.test()
        model.get_loss_items()
        loss_total += model.reconstruct_loss_item
        mel_l1_total += model.loss_mel_L1_item
        sync_total += model.EmbeddingL2_item
        audio_ebds.append(util.to_np(model.mel_net_norm))
        image_ebds.append(util.to_np(model.video_net_norm))
        batch_count += 1
        model.del_no_need()

    if batch_count == 0:
        raise RuntimeError("Test dataloader is empty.")

    audio_ebds = np.concatenate(audio_ebds, axis=0)
    image_ebds = np.concatenate(image_ebds, axis=0)
    video_metrics = util.L2retrieval(audio_ebds, image_ebds)
    audio_metrics = util.L2retrieval(image_ebds, audio_ebds)
    return {
        "reconstruction": loss_total / batch_count,
        "mel_l1": mel_l1_total / batch_count,
        "sync": sync_total / batch_count,
        "num_samples": int(audio_ebds.shape[0]),
        "video_metrics": video_metrics,
        "audio_metrics": audio_metrics,
    }


def main():
    data_loaders = data_loader_utils.get_data_loaders(
        hparams.data_root,
        hparams.speaker_id,
        test_shuffle=False,
        phases=("test",),
    )
    if "test" not in data_loaders:
        raise RuntimeError(
            f"Test split is missing or empty: {os.path.join(hparams.data_root, hparams.test_split_name)}"
        )

    model = Audio_model.AudioModel(hparams, device=device)
    checkpoint_path = resolve_checkpoint_path(hparams.resume_path, hparams.checkpoint_dir, hparams.name)
    global_step, global_epoch, global_test_step = model.load_inpainting_checkpoint(
        checkpoint_path,
        reset_optimizer=True,
    )
    print(
        f"[test] loaded checkpoint: {checkpoint_path} "
        f"(step={global_step}, epoch={global_epoch}, test_step={global_test_step})"
    )

    results = evaluate(model, data_loaders["test"])
    print(
        "[test] losses: "
        f"reconstruction={results['reconstruction']:.6f}, "
        f"mel_l1={results['mel_l1']:.6f}, "
        f"sync={results['sync']:.6f}"
    )
    info = (
        "[test] Video Retrieval ({num} samples): "
        "R@1: {m[0]:.2f}, R@5: {m[1]:.2f}, R@10: {m[2]:.2f}, "
        "R@50: {m[3]:.2f}, MedR: {m[4]:.1f}, MeanR: {m[5]:.1f}"
    )
    info_inv = (
        "[test] Audio Retrieval ({num} samples): "
        "R@1: {m[0]:.2f}, R@5: {m[1]:.2f}, R@10: {m[2]:.2f}, "
        "R@50: {m[3]:.2f}, MedR: {m[4]:.1f}, MeanR: {m[5]:.1f}"
    )
    print(info.format(num=results["num_samples"], m=results["video_metrics"]))
    print(info_inv.format(num=results["num_samples"], m=results["audio_metrics"]))


if __name__ == "__main__":
    main()
    sys.exit(0)
