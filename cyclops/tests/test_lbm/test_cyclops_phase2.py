from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cyclops.models.config import CyclopsConfig
from cyclops.models.model import CyclopsModel


def bare_model() -> CyclopsModel:
    model = CyclopsModel.__new__(CyclopsModel)
    nn.Module.__init__(model)
    return model


def test_phase_configuration_guards():
    with pytest.raises(ValueError, match="Phase 1"):
        CyclopsConfig(training_phase="phase1", temporal_conditioning_mode="scheduled")
    with pytest.raises(ValueError, match="Phase 2"):
        CyclopsConfig(training_phase="phase2", temporal_conditioning_mode="teacher")
    cfg = CyclopsConfig(training_phase="phase2", temporal_conditioning_mode="scheduled")
    assert cfg.teacher_forcing_start == 1.0
    assert cfg.teacher_forcing_end == 0.2


def test_first_frame_null_token_receives_gradient():
    model = bare_model()
    model.target_key = "image"
    model.training_phase = "phase1"
    model.temporal_conditioning_mode = "teacher"
    model.use_scheduled_sampling = True
    model.null_previous_latent = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def encode(self, batch, key, reference_key=None):
        return batch[key]

    model._encode = MethodType(encode, model)
    reference = torch.zeros(2, 4, 3, 3)
    batch = {
        "image_prev": torch.ones_like(reference),
        "is_first_frame": torch.tensor([True, True]),
    }
    condition, previous_gt, first = model._previous_condition(batch, reference)
    condition.sum().backward()
    assert first.all()
    assert not previous_gt.requires_grad
    assert model.null_previous_latent.grad is not None
    assert model.null_previous_latent.grad.abs().sum() > 0


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.states = []

    def forward(self, sample, timestep, conditioning, source_latent, previous_latent):
        sample.retain_grad()
        self.states.append(sample)
        return self.scale * sample + 0.01 * source_latent + 0.01 * previous_latent


def test_four_step_rollout_backpropagates_through_every_state():
    model = bare_model()
    model.ode_num_steps = 4
    model.denoiser = TinyVelocity()

    def conditioning(self, batch, device):
        return {}

    model._conditioning = MethodType(conditioning, model)
    start = torch.randn(1, 4, 2, 2, requires_grad=True)
    source = torch.randn_like(start)
    previous = torch.randn_like(start)
    terminal = model._rollout(start, source, previous, {}, differentiable=True)
    terminal.square().mean().backward()

    assert len(model.denoiser.states) == 4
    assert all(state.grad is not None for state in model.denoiser.states)
    assert model.denoiser.scale.grad is not None
    assert model.denoiser.scale.grad.abs() > 0


def test_teacher_forcing_schedule_reaches_configured_endpoints():
    model = bare_model()
    model.training_phase = "phase2"
    model.use_scheduled_sampling = True
    model.config = SimpleNamespace(
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.2,
        teacher_forcing_anneal_steps=11,
    )
    model.teacher_forcing_total_steps = 11
    model.current_training_step = 0
    assert model.teacher_forcing_probability() == pytest.approx(1.0)
    model.current_training_step = 10
    assert model.teacher_forcing_probability() == pytest.approx(0.2)
