
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCrossAttention(nn.Module):
    def __init__(self, feature_channels: int, latent_channels: int, attention_dim: int, heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, feature_channels)
        self.query = nn.Conv2d(feature_channels, attention_dim, 1)
        self.key = nn.Conv2d(latent_channels, attention_dim, 1)
        self.value = nn.Conv2d(latent_channels, attention_dim, 1)
        self.attention = nn.MultiheadAttention(attention_dim, heads, batch_first=True)
        self.output = nn.Conv2d(attention_dim, feature_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(context, feature.shape[-2:], mode="bilinear", align_corners=False)
        query = self.query(self.norm(feature)).flatten(2).transpose(1, 2)
        key = self.key(context).flatten(2).transpose(1, 2)
        value = self.value(context).flatten(2).transpose(1, 2)
        attended, _ = self.attention(query, key, value, need_weights=False)
        attended = attended.transpose(1, 2).reshape(feature.shape[0], -1, *feature.shape[-2:])
        return self.output(attended)


class BridgeConditioningLevel(nn.Module):

    def __init__(
        self, channels: int, latent_channels: int, attention_dim: int, heads: int,
        use_source_attention: bool = True, use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.use_source_attention = bool(use_source_attention)
        self.use_temporal_attention = bool(use_temporal_attention)
        self.source_attention = LatentCrossAttention(channels, latent_channels, attention_dim, heads)
        self.temporal_attention = LatentCrossAttention(channels, latent_channels, attention_dim, heads)

    def forward(self, feature, source_latent, previous_latent):
        conditioned = feature
        if self.use_source_attention and source_latent is not None:
            conditioned = conditioned + self.source_attention(conditioned, source_latent)
        if self.use_temporal_attention and previous_latent is not None:
            conditioned = conditioned + self.temporal_attention(conditioned, previous_latent)
        return conditioned


class MultiScaleBridgeConditioner(nn.Module):
    def __init__(
        self, block_channels, latent_channels: int = 4, attention_dim: int = 64, heads: int = 4,
        use_source_attention: bool = True, use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.levels = nn.ModuleDict(
            {
                str(channel): BridgeConditioningLevel(
                    channel, latent_channels, attention_dim, heads,
                    use_source_attention=use_source_attention,
                    use_temporal_attention=use_temporal_attention,
                )
                for channel in sorted(set(block_channels))
            }
        )

    def forward(self, feature, source_latent, previous_latent):
        return self.levels[str(feature.shape[1])](feature, source_latent, previous_latent)


def condition_block_output(output, conditioner, source, previous):

    if isinstance(output, torch.Tensor):
        return conditioner(output, source, previous)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        hidden = conditioner(output[0], source, previous)
        remainder = list(output[1:])
        if remainder and isinstance(remainder[0], tuple) and remainder[0]:
            residuals = list(remainder[0])
            if residuals[-1].shape == hidden.shape:
                residuals[-1] = hidden
            remainder[0] = tuple(residuals)
        return (hidden, *remainder)
    return output
