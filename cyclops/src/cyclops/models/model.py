
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from lbm.models.lbm.lbm_model_cylops import LBMModel


class CyclopsModel(LBMModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        self.ode_num_steps = int(config.ode_num_steps)
        self.training_phase = config.training_phase
        self.temporal_conditioning_mode = config.temporal_conditioning_mode
        self.use_lbm_core = bool(config.use_lbm_core)
        self.use_source_attention = bool(config.use_source_attention)
        self.use_temporal_attention = bool(config.use_temporal_attention)
        self.use_scheduled_sampling = bool(config.use_scheduled_sampling)
        self.use_lpips_loss = bool(config.use_lpips_loss)
        self.use_gradient_loss = bool(config.use_gradient_loss)
        self.use_color_statistics_loss = bool(config.use_color_statistics_loss)
        self.use_terminal_reward = bool(config.use_terminal_reward)
        self.null_previous_latent = torch.nn.Parameter(
            torch.zeros(1, int(config.vae_num_channels), 1, 1)
        )
        if not hasattr(self.denoiser, "configure_bridge_conditioning"):
            raise TypeError("Cyclops requires CyclopsUNet2DCondWrapper")
        if not self.use_lbm_core and (self.use_source_attention or self.use_temporal_attention):
            self.denoiser.configure_bridge_conditioning(
                latent_channels=int(config.vae_num_channels),
                attention_dim=int(config.latent_attention_dim),
                heads=int(config.latent_attention_heads),
                use_source_attention=self.use_source_attention,
                use_temporal_attention=self.use_temporal_attention,
            )

    def _encode(self, batch: Dict[str, Any], key: str, reference_key: Optional[str] = None):
        image = batch[key].to(dtype=self.dtype)
        reference_key = reference_key or self.target_key
        if reference_key in batch and image.shape[-2:] != batch[reference_key].shape[-2:]:
            image = F.interpolate(
                image,
                batch[reference_key].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.vae.encode(image) if self.vae is not None else image

    def _null_previous(self, reference: torch.Tensor):
        return self.null_previous_latent.to(reference).expand_as(reference)

    def _first_frame_mask(self, batch, reference):
        value = batch.get("is_first_frame")
        if value is None:
            return torch.zeros(reference.shape[0], device=reference.device, dtype=torch.bool)
        mask = torch.as_tensor(value, device=reference.device, dtype=torch.bool).flatten()
        return mask.expand(reference.shape[0]) if mask.numel() == 1 else mask

    def teacher_forcing_probability(self) -> float:
        if self.training_phase == "phase1" or not self.use_scheduled_sampling:
            return 1.0
        start = float(self.config.teacher_forcing_start)
        end = float(self.config.teacher_forcing_end)
        total_steps = int(
            getattr(
                self,
                "teacher_forcing_total_steps",
                self.config.teacher_forcing_anneal_steps,
            )
        )
        duration = max(1, total_steps - 1)
        current_step = float(getattr(self, "current_training_step", 0))
        progress = min(1.0, current_step / duration)
        return start + (end - start) * progress

    def _predict_previous(self, batch):

        previous_source = self._encode(
            batch, f"{self.source_key}_prev", f"{self.target_key}_prev"
        )
        previous2_target_key = f"{self.target_key}_prev_prev"
        previous_is_first = torch.as_tensor(
            batch.get("is_previous_first_frame", False),
            device=previous_source.device,
            dtype=torch.bool,
        ).flatten()
        if previous_is_first.numel() == 1:
            previous_is_first = previous_is_first.expand(previous_source.shape[0])
        if previous2_target_key in batch:
            previous2_target = self._encode(batch, previous2_target_key)
        else:
            previous2_target = self._null_previous(previous_source)
        history = torch.where(
            previous_is_first[:, None, None, None],
            self._null_previous(previous_source),
            previous2_target.detach(),
        )
        return self._rollout(
            previous_source,
            previous_source,
            history.detach(),
            batch,
            differentiable=False,
        )

    def _previous_condition(self, batch, z_reference) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self._first_frame_mask(batch, z_reference)
        z_previous_gt = self._encode(batch, f"{self.target_key}_prev")
        condition = z_previous_gt
        if (
            self.training_phase == "phase2"
            and self.temporal_conditioning_mode == "scheduled"
            and self.use_scheduled_sampling
        ):
            predicted = self._predict_previous(batch).detach()
            if self.training:
                use_teacher = (
                    torch.rand(z_reference.shape[0], device=z_reference.device)
                    < self.teacher_forcing_probability()
                )
                condition = torch.where(
                    use_teacher[:, None, None, None], z_previous_gt, predicted
                )
            else:
                condition = predicted


        condition = condition.detach()
        condition = torch.where(
            first[:, None, None, None], self._null_previous(z_reference), condition
        )
        return condition, z_previous_gt.detach(), first

    def _conditioning(self, batch, device):
        return self._get_conditioning(batch, set_ucg_rate_zero=True, device=device)

    def _velocity(self, latent, tau, z_source, z_previous, conditioning):
        if isinstance(tau, torch.Tensor):
            timestep = tau.to(device=latent.device, dtype=torch.float32).flatten()
            if timestep.numel() == 1:
                timestep = timestep.expand(latent.shape[0])
        else:
            timestep = torch.full(
                (latent.shape[0],), float(tau), device=latent.device, dtype=torch.float32
            )
        return self.denoiser(
            sample=latent,
            timestep=timestep,
            conditioning=conditioning,
            source_latent=z_source,
            previous_latent=z_previous,
        )

    def _bridge_matching(self, batch, z_source, z_target, z_previous):
        times = torch.as_tensor(
            self.config.bridge_time_values, device=z_target.device, dtype=z_target.dtype
        )
        indices = torch.randint(0, len(times), (z_target.shape[0],), device=z_target.device)
        tau = times[indices].view(-1, 1, 1, 1)
        noise = torch.randn_like(z_target)
        z_tau = (
            (1.0 - tau) * z_source
            + tau * z_target
            + float(self.config.bridge_noise_sigma)
            * torch.sqrt(tau * (1.0 - tau))
            * noise
        )
        conditioning = self._conditioning(batch, z_target.device)
        prediction = self._velocity(z_tau, tau.flatten(), z_source, z_previous, conditioning)
        target_velocity = (z_target - z_tau) / (1.0 - tau).clamp_min(1e-6)
        lbm_loss = F.mse_loss(prediction, target_velocity)
        terminal_estimate = z_tau + (1.0 - tau) * prediction
        return lbm_loss, terminal_estimate

    def _spatial_losses(self, terminal_estimate, target_rgb):
        prediction_rgb = self.vae.decode(terminal_estimate).clamp(-1, 1)
        if prediction_rgb.shape[-2:] != target_rgb.shape[-2:]:
            prediction_rgb = F.interpolate(
                prediction_rgb, target_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        valid_mask = torch.ones_like(target_rgb, dtype=torch.bool)
        zero = prediction_rgb.new_zeros(())
        lpips_loss = (
            self.lpips_loss(prediction_rgb, target_rgb).mean()
            if self.use_lpips_loss else zero
        )
        gradient_loss = (
            self.grad_loss(prediction_rgb, target_rgb, valid_mask, "l1").mean()
            if self.use_gradient_loss else zero
        )
        if self.use_color_statistics_loss:
            pred_mean = prediction_rgb.mean(dim=(-2, -1))
            target_mean = target_rgb.mean(dim=(-2, -1))
            pred_std = prediction_rgb.std(dim=(-2, -1), unbiased=False)
            target_std = target_rgb.std(dim=(-2, -1), unbiased=False)
            color_loss = (
                pred_mean.sub(target_mean).abs() + pred_std.sub(target_std).abs()
            ).sum(dim=1).mean()
        else:
            color_loss = zero
        return prediction_rgb, lpips_loss, gradient_loss, color_loss

    def _rollout(
        self, z_start, z_source, z_previous, batch, differentiable=True, num_steps=None
    ):
        steps = int(num_steps or self.ode_num_steps)
        if steps not in (1, 2, 4):
            raise ValueError("Supported NFE values are {1, 2, 4}")

        def integrate():
            latent = z_start
            conditioning = self._conditioning(batch, latent.device)
            delta_tau = 1.0 / steps
            for step in range(steps):
                velocity = self._velocity(
                    latent, step / steps, z_source, z_previous, conditioning
                )
                latent = latent + velocity * delta_tau
            return latent

        if differentiable:
            return integrate()
        with torch.no_grad():
            return integrate()

    def _terminal_penalty(self, z_hat, z_target, z_previous, z_previous_gt, first):
        fidelity_per_sample = (z_hat - z_target).square().flatten(1).mean(1)
        temporal_per_sample = (
            (z_hat - z_previous) - (z_target - z_previous_gt)
        ).abs().flatten(1).mean(1)
        valid = ~first
        temporal = (
            temporal_per_sample[valid].mean() if valid.any() else z_hat.new_zeros(())
        )
        fidelity = fidelity_per_sample.mean()
        total = (
            float(self.config.reward_fidelity_weight) * fidelity
            + float(self.config.reward_temporal_weight) * temporal
        )
        return total, fidelity, temporal

    def forward(self, batch: Dict[str, Any], step=0, *args, **kwargs):
        if self.use_lbm_core:
            return super().forward(batch, step=step, *args, **kwargs)
        if self.training:
            self.num_iterations.add_(1)
            self.current_training_step = int(step)
        z_target = self._encode(batch, self.target_key)
        z_source = self._encode(batch, self.source_key, self.target_key)
        z_previous, z_previous_gt, first = self._previous_condition(batch, z_target)

        lbm_loss, terminal_estimate = self._bridge_matching(
            batch, z_source, z_target, z_previous
        )
        prediction_rgb, lpips_loss, gradient_loss, color_loss = self._spatial_losses(
            terminal_estimate, batch[self.target_key].to(dtype=self.dtype)
        )
        phase1_loss = (
            lbm_loss
            + float(self.config.lpips_weight) * lpips_loss
            + float(self.config.gradient_weight) * gradient_loss
            + float(self.config.color_weight) * color_loss
        )

        reward_penalty = z_target.new_zeros(())
        reward_fidelity = z_target.new_zeros(())
        reward_temporal = z_target.new_zeros(())
        total_loss = phase1_loss
        if self.training_phase == "phase2" and self.use_terminal_reward:
            z_hat = self._rollout(
                z_source, z_source, z_previous, batch, differentiable=self.training
            )
            reward_penalty, reward_fidelity, reward_temporal = self._terminal_penalty(
                z_hat, z_target, z_previous, z_previous_gt, first
            )
            total_loss = total_loss + float(self.config.reward_weight) * reward_penalty

        return {
            "loss": total_loss,
            "lbm_loss": lbm_loss,
            "lpips_loss": lpips_loss,
            "gradient_loss": gradient_loss,
            "color_loss": color_loss,
            "terminal_reward_penalty": reward_penalty,
            "reward_fidelity": reward_fidelity,
            "reward_temporal": reward_temporal,
            "teacher_forcing_probability": z_target.new_tensor(
                self.teacher_forcing_probability()
            ),
            "predicted_hr": prediction_rgb,
        }

    @torch.no_grad()
    def sample(
        self,
        z,
        num_steps=4,
        conditioner_inputs=None,
        max_samples=None,
        verbose=False,
        previous_sample=None,
    ):
        if self.use_lbm_core:
            return super().sample(
                z, num_steps=num_steps, conditioner_inputs=conditioner_inputs,
                max_samples=max_samples, verbose=verbose, previous_sample=previous_sample,
            )
        batch = conditioner_inputs or {}
        z_source = z[:max_samples] if max_samples else z
        if previous_sample is None:
            z_previous = self._null_previous(z_source)
        else:
            z_previous = previous_sample[:max_samples] if max_samples else previous_sample
        latent = self._rollout(
            z_source,
            z_source,
            z_previous.detach(),
            batch,
            differentiable=False,
            num_steps=num_steps,
        )
        decoded = self.vae.decode(latent) if self.vae is not None else latent
        reference = batch.get(self.target_key, batch.get(self.source_key))
        if reference is not None and decoded.shape[-2:] != reference.shape[-2:]:
            decoded = F.interpolate(
                decoded, reference.shape[-2:], mode="bilinear", align_corners=False
            )
        return decoded
