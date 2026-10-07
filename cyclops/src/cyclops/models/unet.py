
from typing import Dict, List, Optional, Union

import torch

from lbm.models.unets import DiffusersUNet2DCondWrapper

from .bridge_conditioning import MultiScaleBridgeConditioner, condition_block_output


class CyclopsUNet2DCondWrapper(DiffusersUNet2DCondWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge_conditioner = None
        self._bridge_source_latent = None
        self._bridge_previous_latent = None
        self._bridge_hook_handles = []

    def configure_bridge_conditioning(
        self, latent_channels: int = 4, attention_dim: int = 64, heads: int = 4,
        use_source_attention: bool = True, use_temporal_attention: bool = True,
    ) -> None:
        if self.bridge_conditioner is not None:
            return
        channels = tuple(int(value) for value in self.config.block_out_channels)
        self.bridge_conditioner = MultiScaleBridgeConditioner(
            channels, latent_channels, attention_dim, heads,
            use_source_attention=use_source_attention,
            use_temporal_attention=use_temporal_attention,
        ).to(device=self.conv_in.weight.device, dtype=self.conv_in.weight.dtype)

        def hook(_module, _inputs, output):
            return condition_block_output(
                output,
                self.bridge_conditioner,
                self._bridge_source_latent,
                self._bridge_previous_latent,
            )

        blocks = list(self.down_blocks) + [self.mid_block] + list(self.up_blocks)
        self._bridge_hook_handles = [block.register_forward_hook(hook) for block in blocks]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor],
        ip_adapter_cond_embedding: Optional[List[torch.Tensor]] = None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        down_intrablock_additional_residuals=None,
        source_latent: Optional[torch.Tensor] = None,
        previous_latent: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        self._bridge_source_latent = source_latent
        self._bridge_previous_latent = previous_latent
        try:
            return super().forward(
                sample=sample,
                timestep=timestep,
                conditioning=conditioning,
                ip_adapter_cond_embedding=ip_adapter_cond_embedding,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                *args,
                **kwargs,
            )
        finally:
            self._bridge_source_latent = None
            self._bridge_previous_latent = None
