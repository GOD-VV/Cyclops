import torch

from cyclops.models.bridge_conditioning import BridgeConditioningLevel
from cyclops.models.config import CyclopsConfig


def test_component_switches_are_enabled_by_default():
    cfg = CyclopsConfig()
    assert not cfg.use_lbm_core
    assert cfg.use_source_attention
    assert cfg.use_temporal_attention
    assert cfg.use_scheduled_sampling
    assert cfg.use_lpips_loss
    assert cfg.use_gradient_loss
    assert cfg.use_color_statistics_loss
    assert cfg.use_terminal_reward


def test_attention_switches_bypass_disabled_branches():
    level = BridgeConditioningLevel(
        channels=32, latent_channels=4, attention_dim=8, heads=2,
        use_source_attention=False, use_temporal_attention=False,
    )
    feature = torch.randn(1, 32, 4, 4)
    source = torch.randn(1, 4, 2, 2)
    previous = torch.randn(1, 4, 2, 2)
    assert torch.equal(level(feature, source, previous), feature)
