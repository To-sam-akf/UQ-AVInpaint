import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Models.VIAI_AV_inpainting import VIAIAVModel


def _build_mask(batch_size, mel_bins, mel_steps, span):
    start, end = span
    mask = torch.zeros(batch_size, 1, mel_bins, mel_steps)
    mask[:, :, :, start:end] = 1.0
    return mask


def _make_loss_model(
    candidates,
    target,
    missing_mask,
    missing_span,
    lambda_min_k=1.0,
    lambda_mean_k=0.1,
    lambda_boundary=0.05,
):
    model = object.__new__(VIAIAVModel)
    model.device = target.device
    model.hparams = SimpleNamespace(
        lambda_min_k=lambda_min_k,
        lambda_mean_k=lambda_mean_k,
        lambda_boundary=lambda_boundary,
    )
    model.criterion_l1 = torch.nn.L1Loss()
    model.eta1 = 1.0
    model.mel_candidates = candidates
    model.mel_completed_candidates = candidates.clone()
    model.mel_target_4d = target
    model.missing_mask = missing_mask
    model.missing_span = missing_span
    model.evidence_score = torch.ones(target.size(0), 1)
    return model


def test_best_of_k_and_mean_k_missing_l1():
    batch_size, num_candidates, mel_bins, mel_steps = 2, 4, 3, 10
    target = torch.linspace(0.0, 1.0, steps=mel_bins * mel_steps).view(
        1,
        1,
        mel_bins,
        mel_steps,
    )
    target = target.repeat(batch_size, 1, 1, 1)
    mask = _build_mask(batch_size, mel_bins, mel_steps, (3, 7))

    candidates = target.unsqueeze(1).repeat(1, num_candidates, 1, 1, 1)
    offsets = torch.tensor([0.0, 0.25, 0.5, 1.0]).view(1, num_candidates, 1, 1, 1)
    candidates = candidates + offsets * mask.unsqueeze(1)

    model = _make_loss_model(candidates, target, mask, (3, 7))
    model._multi_candidate_losses()

    assert torch.isfinite(model.loss_min_k)
    assert torch.isfinite(model.loss_mean_k)
    assert model.loss_min_k < model.loss_mean_k
    assert model.best_of_k_missing_l1 <= model.mean_k_missing_l1

    single_model = _make_loss_model(candidates[:, :1], target, mask, (3, 7))
    single_model._multi_candidate_losses()
    assert torch.allclose(single_model.loss_min_k, single_model.loss_mean_k)
    assert torch.allclose(
        single_model.best_of_k_missing_l1,
        single_model.mean_k_missing_l1,
    )

    zero_weight_model = _make_loss_model(
        candidates,
        target,
        mask,
        (3, 7),
        lambda_min_k=0.0,
        lambda_mean_k=0.0,
        lambda_boundary=0.0,
    )
    zero_weight_model._multi_candidate_losses()
    assert torch.allclose(
        zero_weight_model.loss_multi_candidate,
        torch.zeros_like(zero_weight_model.loss_multi_candidate),
    )


def test_boundary_loss_skips_out_of_range_edges():
    batch_size, num_candidates, mel_bins, mel_steps = 2, 4, 3, 10
    target = torch.linspace(0.0, 1.0, steps=mel_bins * mel_steps).view(
        1,
        1,
        mel_bins,
        mel_steps,
    )
    target = target.repeat(batch_size, 1, 1, 1)

    for span in [(0, 3), (3, 7), (7, 10)]:
        start, end = span
        mask = _build_mask(batch_size, mel_bins, mel_steps, span)
        candidates = target.unsqueeze(1).repeat(1, num_candidates, 1, 1, 1)
        candidates[:, 1:, :, :, start:end] += 0.5

        model = _make_loss_model(candidates, target, mask, span)
        model._multi_candidate_losses()

        assert torch.isfinite(model.loss_boundary)
        assert torch.isfinite(model.loss_multi_candidate)


if __name__ == "__main__":
    test_best_of_k_and_mean_k_missing_l1()
    test_boundary_loss_skips_out_of_range_edges()
