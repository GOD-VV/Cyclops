from typing import List, Literal, Optional, Tuple
from pydantic.dataclasses import dataclass
from ..base import ModelConfig


@dataclass
class LBMConfig(ModelConfig):
    source_key: str = "source_image"
    target_key: str = "target_image"
    mask_key: Optional[str] = None

    latent_loss_weight: float = 1.0
    latent_loss_type: Literal["l2", "l1"] = "l2"
    pixel_loss_type: Literal["l2", "l1", "lpips"] = "l2"
    pixel_loss_max_size: int = 512
    pixel_loss_weight: float = 0.0

    timestep_sampling: Literal["uniform", "log_normal", "custom_timesteps"] = "uniform"
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    selected_timesteps: Optional[List[float]] = None
    prob: Optional[List[float]] = None
    bridge_noise_sigma: float = 0.001

    use_temporal_stability: bool = True
    temporal_stability_weight: float = 0.1
    temporal_noise_smoothing: float = 0.5

    use_color_loss: bool = True
    color_loss_weight: float = 0.1
    use_grad_loss: bool = True
    grad_loss_weight: float = 0.1
    grad_loss_type: Literal["l1", "l2"] = "l1"

    use_rl_finetune: bool = True
    rl_learning_rate: float = 1e-5
    rl_sample_steps: int = 4
    rl_reward_weight: float = 0.5

    use_densification: bool = False
    densification_pretrained_path: Optional[str] = None
    freeze_densification: bool = True
    densification_input_key: str = "sparse_intensity"
    densification_model_type: str = "intensity"
    densification_fusion_mode: Literal["concat", "cross_attn", "feature_injection"] = (
        "concat"
    )
    densification_feature_level: Literal["encoder", "decoder_mid", "decoder_late"] = (
        "encoder"
    )
    densification_feature_weight: float = 1.0

    densification_z0_vae_weight: float = 0.85
    freeze_projection_layer: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.timestep_sampling == "log_normal":
            assert isinstance(self.logit_mean, float) and isinstance(
                self.logit_std, float
            ), (
                "logit_mean and logit_std should be float for log_normal timestep sampling"
            )

        if self.timestep_sampling == "custom_timesteps":
            assert isinstance(self.selected_timesteps, list) and isinstance(
                self.prob, list
            ), (
                "timesteps and prob should be list for custom_timesteps timestep sampling"
            )
            assert len(self.selected_timesteps) == len(self.prob), (
                "timesteps and prob should be of same length for custom_timesteps timestep sampling"
            )
            assert abs(sum(self.prob) - 1.0) < 1e-6, (
                "prob should sum to 1 for custom_timesteps timestep sampling"
            )
