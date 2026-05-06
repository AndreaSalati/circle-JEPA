"""Tests for Phase 3: model module (encoder, predictor, EMA, CircadianJEPA)."""

import math

import pytest
import torch

from circadian_jepa.model.ema import EMATeacher
from circadian_jepa.model.encoder import Encoder
from circadian_jepa.model.jepa import CircadianJEPA
from circadian_jepa.model.predictor import RotationPredictor


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def test_encoder_output_shape():
    enc = Encoder(n_genes=15, hidden_dims=[64, 32], embedding_dim=2)
    x = torch.randn(32, 15)
    z = enc(x)
    assert z.shape == (32, 2)


def test_encoder_phase_known_input():
    enc = Encoder(n_genes=1)
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    phases = enc.phase(z)
    assert torch.isclose(phases[0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(phases[1], torch.tensor(math.pi / 2), atol=1e-5)
    assert torch.isclose(phases[2].abs(), torch.tensor(math.pi), atol=1e-5)


def test_encoder_amplitude_known_input():
    enc = Encoder(n_genes=1)
    z = torch.tensor([[1.0, 0.0], [3.0, 4.0]])
    amps = enc.amplitude(z)
    assert torch.isclose(amps[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(amps[1], torch.tensor(5.0), atol=1e-5)


def test_encoder_normalize_output():
    enc = Encoder(n_genes=15, normalize_output=True)
    x = torch.randn(16, 15)
    z = enc(x)
    norms = torch.norm(z, dim=-1)
    assert torch.allclose(norms, torch.ones(16), atol=1e-5)


# ---------------------------------------------------------------------------
# RotationPredictor
# ---------------------------------------------------------------------------

def test_rotation_predictor_identity():
    pred = RotationPredictor(learn_delta=False)
    z = torch.randn(8, 2)
    out = pred(z)
    assert torch.equal(out, z)


def test_rotation_predictor_rotate_known():
    z = torch.tensor([[1.0, 0.0]])
    delta = torch.tensor([math.pi / 2])
    rotated = RotationPredictor.rotate(z, delta)
    expected = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_rotation_predictor_rotate_360():
    z = torch.randn(4, 2)
    delta = torch.full((4,), 2 * math.pi)
    rotated = RotationPredictor.rotate(z, delta)
    assert torch.allclose(rotated, z, atol=1e-5)


def test_rotation_predictor_apply_delta():
    pred = RotationPredictor(learn_delta=True)
    z = torch.tensor([[1.0, 0.0]])
    delta = torch.tensor([math.pi])
    out = pred(z, delta)
    expected = torch.tensor([[-1.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# EMATeacher
# ---------------------------------------------------------------------------

def test_ema_teacher_no_grad():
    enc = Encoder(n_genes=15)
    ema = EMATeacher(enc, momentum=0.99)
    for p in ema.teacher.parameters():
        assert not p.requires_grad


def test_ema_teacher_update_moves_toward_student():
    enc = Encoder(n_genes=15)
    ema = EMATeacher(enc, momentum=0.0)  # full copy each step

    # Perturb student params so teacher and student differ.
    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(2.0)
        for p in ema.teacher.parameters():
            p.fill_(0.0)

    ema.update(enc)

    for t_p, s_p in zip(ema.teacher.parameters(), enc.parameters()):
        assert torch.allclose(t_p.data, s_p.data, atol=1e-6)


def test_ema_teacher_partial_update():
    enc = Encoder(n_genes=15)
    momentum = 0.5
    ema = EMATeacher(enc, momentum=momentum)

    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(1.0)
        for p in ema.teacher.parameters():
            p.fill_(0.0)

    ema.update(enc)

    for t_p in ema.teacher.parameters():
        # Expected: 0.5*0 + 0.5*1 = 0.5
        assert torch.allclose(t_p.data, torch.tensor(0.5), atol=1e-6)


def test_ema_teacher_forward_no_grad():
    enc = Encoder(n_genes=15)
    ema = EMATeacher(enc, momentum=0.99)
    x = torch.randn(4, 15)
    out = ema.forward(x)
    assert out.grad_fn is None


# ---------------------------------------------------------------------------
# CircadianJEPA
# ---------------------------------------------------------------------------

def test_jepa_forward_shapes():
    model = CircadianJEPA(n_genes=15)
    view_a = torch.randn(64, 15)
    view_b = torch.randn(64, 15)
    out = model(view_a, view_b)
    assert out["z_a"].shape == (64, 2)
    assert out["z_b_target"].shape == (64, 2)
    assert out["z_a_pred"].shape == (64, 2)


def test_jepa_gradient_flow():
    model = CircadianJEPA(n_genes=15)
    view_a = torch.randn(8, 15)
    view_b = torch.randn(8, 15)
    out = model(view_a, view_b)

    # z_a and z_a_pred must have grad_fn (connected to student)
    assert out["z_a"].grad_fn is not None
    assert out["z_a_pred"].grad_fn is not None

    # z_b_target must NOT have grad_fn (teacher runs under no_grad)
    assert out["z_b_target"].grad_fn is None


def test_jepa_step_ema():
    model = CircadianJEPA(n_genes=15)
    # Grab teacher param snapshot before step
    t_before = [p.data.clone() for p in model.ema.teacher.parameters()]

    # Perturb student
    with torch.no_grad():
        for p in model.student_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    model.step_ema()

    t_after = [p.data for p in model.ema.teacher.parameters()]
    s_params = list(model.student_encoder.parameters())

    for tb, ta, sp in zip(t_before, t_after, s_params):
        expected = model.ema.momentum * tb + (1 - model.ema.momentum) * sp.data
        assert torch.allclose(ta, expected, atol=1e-6)


def test_jepa_embed():
    model = CircadianJEPA(n_genes=15)
    x = torch.randn(10, 15)
    z = model.embed(x)
    assert z.shape == (10, 2)
    assert z.grad_fn is None
