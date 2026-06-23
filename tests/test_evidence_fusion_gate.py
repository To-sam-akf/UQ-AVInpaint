import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import Options_inpainting
from Models.VIAI_AV_inpainting import VIAIAVModel
from networks.EC_VIAI_Modules import (
    BottleneckAdapter,
    EvidenceFusionGate,
    apply_visual_evidence_augmentation,
)


def _build_mask(batch_size, mel_bins, mel_steps, span):
    start, end = span
    mask = torch.zeros(batch_size, 1, mel_bins, mel_steps)
    mask[:, :, :, start:end] = 1.0
    return mask


def _make_loss_model(candidates, target, missing_mask, evidence_score):
    model = object.__new__(VIAIAVModel)
    model.device = target.device
    model.hparams = SimpleNamespace(
        lambda_min_k=0.0,
        lambda_mean_k=0.0,
        lambda_boundary=0.0,
        lambda_diversity=1.0,
        lambda_gate_evidence=0.0,
        evidence_diversity_d_min=0.02,
        evidence_diversity_alpha=0.08,
        evidence_gate_low=0.24,
        evidence_gate_high=0.34,
    )
    model.criterion_l1 = torch.nn.L1Loss()
    model.eta1 = 1.0
    model.mel_candidates = candidates
    model.mel_completed_candidates = candidates.clone()
    model.mel_target_4d = target
    model.missing_mask = missing_mask
    model.missing_span = (2, 6)
    model.evidence_score = evidence_score
    return model


def test_evidence_fusion_gate_shapes_range_and_gradients():
    torch.manual_seed(3)
    gate = EvidenceFusionGate(feature_channels=4, hidden_channels=8)
    audio_bottleneck = torch.randn(2, 4, 1, 5, requires_grad=True)
    video_feature = torch.randn(2, 4, 1, 5, requires_grad=True)
    evidence = torch.tensor([[0.9], [0.1]])

    calibrated, gate_value, audio_prior = gate(audio_bottleneck, video_feature, evidence)

    assert calibrated.shape == video_feature.shape
    assert audio_prior.shape == video_feature.shape
    assert tuple(gate_value.shape) == (2, 1, 1, 1)
    assert bool(torch.all(gate_value >= 0.0) and torch.all(gate_value <= 1.0))
    assert torch.allclose(audio_prior, audio_bottleneck.detach(), atol=1e-6)

    loss = calibrated.mean() + gate_value.mean()
    loss.backward()
    assert audio_bottleneck.grad is not None
    assert video_feature.grad is not None
    assert gate.gate_mlp[0].weight.grad is not None
    assert gate.audio_prior.weight.grad is not None


def test_evidence_diversity_target_is_larger_for_low_evidence():
    batch_size, num_candidates, mel_bins, mel_steps = 2, 3, 4, 8
    target = torch.zeros(batch_size, 1, mel_bins, mel_steps)
    mask = _build_mask(batch_size, mel_bins, mel_steps, (2, 6))
    candidates = target.unsqueeze(1).repeat(1, num_candidates, 1, 1, 1)
    candidates[:, 1, :, :, 2:6] = 0.05
    candidates[:, 2, :, :, 2:6] = 0.10
    evidence = torch.tensor([[1.0], [0.0]])

    model = _make_loss_model(candidates, target, mask, evidence)
    model._multi_candidate_losses()

    assert model.evidence_diversity_target[1].item() > model.evidence_diversity_target[0].item()
    assert torch.isfinite(model.loss_evidence_div)
    assert torch.isfinite(model.weighted_loss_evidence_div)


def test_pairwise_distance_uses_missing_region_only():
    batch_size, num_candidates, mel_bins, mel_steps = 1, 2, 3, 8
    target = torch.zeros(batch_size, 1, mel_bins, mel_steps)
    mask = _build_mask(batch_size, mel_bins, mel_steps, (2, 6))
    candidates = target.unsqueeze(1).repeat(1, num_candidates, 1, 1, 1)
    candidates[:, 1, :, :, :2] = 1.0
    candidates[:, 1, :, :, 6:] = 1.0

    model = _make_loss_model(candidates, target, mask, torch.ones(batch_size, 1))
    known_only_distance = model._candidate_pairwise_distance_per_sample(candidates)
    assert torch.allclose(known_only_distance, torch.zeros_like(known_only_distance))

    candidates[:, 1, :, :, 2:6] = 1.0
    missing_distance = model._candidate_pairwise_distance_per_sample(candidates)
    assert missing_distance.item() > 0.0


def test_zero_lambda_diversity_does_not_change_total_multi_loss():
    batch_size, num_candidates, mel_bins, mel_steps = 1, 2, 3, 8
    target = torch.zeros(batch_size, 1, mel_bins, mel_steps)
    mask = _build_mask(batch_size, mel_bins, mel_steps, (2, 6))
    candidates = target.unsqueeze(1).repeat(1, num_candidates, 1, 1, 1)
    candidates[:, 1, :, :, 2:6] = 0.5

    model = _make_loss_model(candidates, target, mask, torch.zeros(batch_size, 1))
    model.hparams.lambda_diversity = 0.0
    model._multi_candidate_losses()

    expected = (
        model.weighted_loss_min_k
        + model.weighted_loss_mean_k
        + model.weighted_loss_boundary
    )
    assert torch.allclose(model.loss_multi_candidate, expected)
    assert torch.allclose(
        model.weighted_loss_evidence_div,
        torch.zeros_like(model.weighted_loss_evidence_div),
    )


def test_visual_evidence_augmentation_modes_preserve_shapes():
    video = torch.randn(2, 5, 3, 4, 4)
    flow = torch.randn(2, 5, 2, 4, 4)
    for mode in [
        "flow_75",
        "flow_50",
        "flow_25",
        "flow_zero",
        "static_video_zero_flow",
    ]:
        aug_video, aug_flow = apply_visual_evidence_augmentation(video, flow, mode)
        assert aug_video.shape == video.shape
        assert aug_flow.shape == flow.shape

    _, zero_flow = apply_visual_evidence_augmentation(video, flow, "flow_zero")
    assert torch.allclose(zero_flow, torch.zeros_like(flow))

    static_video, static_flow = apply_visual_evidence_augmentation(
        video,
        flow,
        "static_video_zero_flow",
    )
    assert torch.allclose(static_flow, torch.zeros_like(flow))
    assert torch.allclose(static_video[:, 0:1].expand_as(static_video), static_video)


def test_gate_target_range_and_order():
    model = object.__new__(VIAIAVModel)
    model.hparams = SimpleNamespace(evidence_gate_low=0.24, evidence_gate_high=0.34)
    evidence = torch.tensor([[0.10], [0.24], [0.29], [0.34], [0.50]])

    target = model._gate_target_from_evidence(evidence)

    assert bool(torch.all(target >= 0.0) and torch.all(target <= 1.0))
    assert target[0].item() == 0.0
    assert target[1].item() == 0.0
    assert target[2].item() > target[1].item()
    assert target[3].item() == 1.0
    assert target[4].item() == 1.0


def test_zero_lambda_gate_evidence_does_not_add_loss():
    model = object.__new__(VIAIAVModel)
    model.hparams = SimpleNamespace(lambda_gate_evidence=0.0)
    model.use_evidence_gate = True
    model.criterion_gate_evidence = torch.nn.SmoothL1Loss()
    model.loss_av_gen = torch.tensor(1.0)
    model.gate_value = torch.ones(2, 1, 1, 1)
    model.gate_target = torch.zeros(2, 1, 1, 1)

    weighted = model._gate_evidence_loss()

    assert model.loss_gate_evidence.item() > 0.0
    assert torch.allclose(weighted, torch.zeros_like(weighted))


def test_freeze_gate_evidence_backbone_excludes_encoder_params_from_optimizer():
    hparams = Options_inpainting.Inpainting_Config(
        force_reload=True,
        args=[
            "--enable_ec_viai_av",
            "--stochastic_adapter",
            "--enable_evidence_gate",
            "--freeze_gate_evidence_backbone",
            "--num_candidates",
            "2",
            "--test_num_candidates",
            "2",
        ],
    )
    model = VIAIAVModel(hparams, device=torch.device("cpu"))

    optimizer_param_ids = {
        id(parameter)
        for group in model.optimizer_G.param_groups
        for parameter in group["params"]
    }
    mel_encoder_param_ids = {id(parameter) for parameter in model.Mel_Encoder.parameters()}
    video_encoder_param_ids = {
        id(parameter) for parameter in model.VideoEncoder.parameters()
    }
    decoder_param_ids = {id(parameter) for parameter in model.Mel_Decoder.parameters()}

    assert model.freeze_gate_evidence_backbone
    assert not (optimizer_param_ids & mel_encoder_param_ids)
    assert not (optimizer_param_ids & video_encoder_param_ids)
    assert optimizer_param_ids & decoder_param_ids
    assert all(not parameter.requires_grad for parameter in model.Mel_Encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.VideoEncoder.parameters())


def test_frozen_backbone_train_modes_keep_encoders_eval():
    model = object.__new__(VIAIAVModel)
    model.freeze_gate_evidence_backbone = True
    model.Mel_Encoder = torch.nn.BatchNorm1d(1)
    model.VideoEncoder = torch.nn.BatchNorm1d(1)
    model.Mel_Decoder = torch.nn.BatchNorm1d(1)
    model.BottleneckAdapter = torch.nn.BatchNorm1d(1)
    model.EvidenceFusionGate = torch.nn.BatchNorm1d(1)
    model.use_bottleneck_adapter = True
    model.use_evidence_gate = True

    model._set_generator_train_modes()

    assert not model.Mel_Encoder.training
    assert not model.VideoEncoder.training
    assert model.Mel_Decoder.training
    assert model.BottleneckAdapter.training
    assert model.EvidenceFusionGate.training

    model.freeze_gate_evidence_backbone = False
    model._set_generator_train_modes()

    assert model.Mel_Encoder.training
    assert model.VideoEncoder.training


def test_bottleneck_adapter_sigma_scale_is_optional_and_broadcasts():
    torch.manual_seed(11)
    adapter = BottleneckAdapter(feature_channels=3, hidden_channels=4)
    mu = torch.zeros(2, 3, 1, 2)
    logvar = torch.zeros_like(mu)

    z_default, sigma_default = adapter.sample_latent(
        mu,
        logvar,
        num_candidates=4,
        sigma_min=0.0,
        sigma_max=2.0,
    )
    z_scaled, sigma_scaled = adapter.sample_latent(
        mu,
        logvar,
        num_candidates=4,
        sigma_min=0.0,
        sigma_max=2.0,
        sigma_scale=torch.tensor([[[[0.5]]], [[[2.0]]]]),
    )

    assert tuple(z_default.shape) == (2, 4, 3, 1, 2)
    assert sigma_default.shape == mu.shape
    assert sigma_scaled.shape == mu.shape
    assert torch.allclose(sigma_default, torch.ones_like(sigma_default))
    assert torch.allclose(sigma_scaled[0], torch.full_like(sigma_scaled[0], 0.5))
    assert torch.allclose(sigma_scaled[1], torch.full_like(sigma_scaled[1], 2.0))
    assert torch.allclose(z_scaled[:, 0], mu)


def test_sigma_scale_from_gate_target_is_larger_for_low_evidence():
    model = object.__new__(VIAIAVModel)
    model.hparams = SimpleNamespace(
        evidence_sigma_scale_min=0.5,
        evidence_sigma_scale_max=2.0,
    )
    gate_target = torch.tensor([[[[1.0]]], [[[0.5]]], [[[0.0]]]])

    sigma_scale = model._sigma_scale_from_gate_target(gate_target)

    assert sigma_scale[0].item() == 0.5
    assert sigma_scale[1].item() == 1.25
    assert sigma_scale[2].item() == 2.0
    assert sigma_scale[0].item() < sigma_scale[1].item() < sigma_scale[2].item()
