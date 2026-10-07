from typing import Literal, Tuple

from pydantic.dataclasses import dataclass

from lbm.models.lbm.lbm_config_cylops import LBMConfig


@dataclass
class CyclopsConfig(LBMConfig):
    ode_num_steps: int = 4
    bridge_time_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    bridge_noise_sigma: float = 0.005
    latent_attention_dim: int = 64
    latent_attention_heads: int = 4
    training_phase: Literal["phase1", "phase2"] = "phase1"
    temporal_conditioning_mode: Literal["teacher", "scheduled"] = "teacher"
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.2
    teacher_forcing_anneal_steps: int = 20_000
    lpips_weight: float = 1.0
    gradient_weight: float = 0.1
    color_weight: float = 0.05
    reward_weight: float = 0.8
    reward_fidelity_weight: float = 1.0
    reward_temporal_weight: float = 0.5
    vae_num_channels: int = 4


    use_lbm_core: bool = False
    use_source_attention: bool = True
    use_temporal_attention: bool = True
    use_scheduled_sampling: bool = True
    use_lpips_loss: bool = True
    use_gradient_loss: bool = True
    use_color_statistics_loss: bool = True
    use_terminal_reward: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ode_num_steps != 4:
            raise ValueError("Cyclops is trained with M=4 Euler steps")
        if tuple(self.bridge_time_values) != (0.0, 0.25, 0.5, 0.75):
            raise ValueError("Cyclops bridge times must be (0, 0.25, 0.5, 0.75)")
        if self.latent_attention_dim % self.latent_attention_heads:
            raise ValueError("latent_attention_dim must be divisible by latent_attention_heads")
        if not 0.0 <= self.teacher_forcing_end <= self.teacher_forcing_start <= 1.0:
            raise ValueError("teacher-forcing probabilities must satisfy 0 <= end <= start <= 1")
        if self.teacher_forcing_anneal_steps <= 0:
            raise ValueError("teacher_forcing_anneal_steps must be positive")
        if self.training_phase == "phase1" and self.temporal_conditioning_mode != "teacher":
            raise ValueError("Phase 1 must use teacher temporal conditioning")
        if (
            self.training_phase == "phase2"
            and self.use_scheduled_sampling
            and self.temporal_conditioning_mode != "scheduled"
        ):
            raise ValueError(
                "Phase 2 scheduled sampling requires temporal_conditioning_mode='scheduled'"
            )
