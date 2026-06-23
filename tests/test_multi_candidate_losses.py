import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Models.VIAI_AV_inpainting import VIAIAVModel
from networks.EC_VIAI_Modules import CandidateScorer, UncertaintyHead


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
    assert model.oracle_gain >= 0.0
    assert model.best_of_k_missing_l1 <= model.top1_missing_l1
    assert model.best_of_k_missing_l1 <= model.candidate0_missing_l1
    assert model.best_of_k_missing_l1 <= model.random_expected_missing_l1

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


def test_candidate_scorer_outputs_distribution_and_argmax_anchor():
    torch.manual_seed(17)
    scorer = CandidateScorer(feature_channels=4, candidate_stat_dim=3, hidden_channels=8)
    candidate_stats = torch.randn(2, 4, 3)
    audio_bottleneck = torch.randn(2, 4, 1, 5)
    video_feature = torch.randn(2, 4, 1, 5)
    evidence = torch.tensor([[0.8], [0.2]])

    logits, pi = scorer(candidate_stats, audio_bottleneck, video_feature, evidence)

    assert tuple(logits.shape) == (2, 4)
    assert tuple(pi.shape) == (2, 4)
    assert torch.allclose(pi.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.equal(torch.argmax(pi, dim=1), torch.zeros(2, dtype=torch.long))


def test_uncertainty_head_outputs_bounded_sample_score():
    torch.manual_seed(23)
    head = UncertaintyHead(feature_channels=4, stats_dim=7, hidden_channels=8)
    audio_bottleneck = torch.randn(2, 4, 1, 5)
    video_feature = torch.randn(2, 4, 1, 5)
    uncertainty_stats = torch.randn(2, 7)

    uncertainty = head(audio_bottleneck, video_feature, uncertainty_stats)

    assert tuple(uncertainty.shape) == (2, 1)
    assert bool(torch.all(uncertainty >= 0.0) and torch.all(uncertainty <= 1.0))


def test_calibration_loss_targets_true_best_candidate():
    model = object.__new__(VIAIAVModel)
    model.hparams = SimpleNamespace(lambda_calib=0.5, calib_error_tau=0.1)
    model.use_candidate_scorer = True
    model.criterion_uncertainty_calib = torch.nn.SmoothL1Loss()
    model.candidate_missing_l1 = torch.tensor(
        [[0.5, 0.1, 0.2], [0.3, 0.4, 0.1]],
        dtype=torch.float32,
    )
    model.candidate_logits = torch.tensor(
        [[0.1, 0.8, 0.2], [0.2, 0.0, 0.9]],
        dtype=torch.float32,
    )
    model.uncertainty_score = torch.full((2, 1), 0.5)

    model._calibration_losses()

    expected_targets = torch.tensor([1, 2])
    expected_ce = F.cross_entropy(model.candidate_logits, expected_targets)
    assert torch.allclose(model.loss_candidate_scorer, expected_ce)
    assert model.loss_calib.item() > 0.0
    assert torch.allclose(model.weighted_loss_calib, 0.5 * model.loss_calib)

    model.hparams.lambda_calib = 0.0
    model._calibration_losses()
    assert torch.allclose(
        model.weighted_loss_calib,
        torch.zeros_like(model.weighted_loss_calib),
    )


if __name__ == "__main__":
    test_best_of_k_and_mean_k_missing_l1()
    test_boundary_loss_skips_out_of_range_edges()
    test_candidate_scorer_outputs_distribution_and_argmax_anchor()
    test_uncertainty_head_outputs_bounded_sample_score()
    test_calibration_loss_targets_true_best_candidate()
