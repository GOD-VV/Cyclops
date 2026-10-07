from typing import Any, Dict, List, Optional, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper
from ..vae import AutoencoderKLDiffusers
from .lbm_config_cylops import LBMConfig


class LBMModel(BaseModel):
    @classmethod
    def load_from_config(cls, config: LBMConfig):
        return cls(config=config)

    def __init__(
        self,
        config: LBMConfig,
        denoiser: Union[
            DiffusersUNet2DWrapper,
            DiffusersUNet2DCondWrapper,
        ] = None,
        training_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        sampling_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        vae: AutoencoderKLDiffusers = None,
        conditioner: ConditionerWrapper = None,
    ):
        BaseModel.__init__(self, config)

        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.sampling_noise_scheduler = sampling_noise_scheduler
        self.training_noise_scheduler = training_noise_scheduler
        self.timestep_sampling = config.timestep_sampling
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.pixel_loss_type = config.pixel_loss_type
        self.pixel_loss_max_size = config.pixel_loss_max_size
        self.pixel_loss_weight = config.pixel_loss_weight
        self.logit_mean = config.logit_mean
        self.logit_std = config.logit_std
        self.prob = config.prob
        self.selected_timesteps = config.selected_timesteps
        self.source_key = config.source_key
        self.target_key = config.target_key
        self.mask_key = config.mask_key
        self.bridge_noise_sigma = config.bridge_noise_sigma

        self.use_temporal_stability: bool = getattr(
            config, "use_temporal_stability", True
        )
        temporal_stability_weight = getattr(config, "temporal_stability_weight", 0.1)

        self.temporal_stability_weight = (
            temporal_stability_weight if self.use_temporal_stability else 0.0
        )
        self.temporal_noise_smoothing = getattr(config, "temporal_noise_smoothing", 0.5)

        self.use_color_loss: bool = getattr(config, "use_color_loss", True)
        self.color_loss_weight: float = getattr(config, "color_loss_weight", 0.1)
        self.use_grad_loss: bool = getattr(config, "use_grad_loss", True)
        self.grad_loss_weight: float = getattr(config, "grad_loss_weight", 0.1)
        self.grad_loss_type: str = getattr(config, "grad_loss_type", "l1")

        self.register_buffer("num_iterations", torch.tensor([0], dtype=torch.long))

        if self.pixel_loss_type == "lpips" and self.pixel_loss_weight > 0:
            self.lpips_loss = lpips.LPIPS(net="vgg")
        else:
            self.lpips_loss = None

        self.register_buffer(
            "_sobel_kernel_x", self._make_sobel_kernel_x(), persistent=False
        )
        self.register_buffer(
            "_sobel_kernel_y", self._make_sobel_kernel_y(), persistent=False
        )
        self._eps = 1e-6

        self.use_pose_warp = getattr(config, "use_pose_warp", True)
        self.require_depth_for_warp = getattr(config, "require_depth_for_warp", False)
        self.plane_depth_estimate = getattr(config, "plane_depth_estimate", "auto")
        self.fixed_plane_depth = getattr(config, "fixed_plane_depth", 1.0)

        self.use_densification = getattr(config, "use_densification", False)
        self.densification_model = None
        self.densification_encoder = None
        self.densification_input_key = getattr(
            config, "densification_input_key", "sparse_intensity"
        )
        self.freeze_densification = getattr(config, "freeze_densification", True)
        self.densification_fusion_mode = getattr(
            config, "densification_fusion_mode", "concat"
        )
        self.densification_feature_level = getattr(
            config, "densification_feature_level", "encoder"
        )
        self.densification_feature_weight = getattr(
            config, "densification_feature_weight", 1.0
        )
        self.densification_z0_vae_weight = float(
            max(0.0, min(1.0, getattr(config, "densification_z0_vae_weight", 0.85)))
        )

        if self.use_densification:
            self._init_densification_model(config)

    def _init_densification_model(self, config):

        try:
            import sys
            from pathlib import Path

            scripts_lbm_path = (
                Path(__file__).parent.parent.parent.parent.parent.parent / "scripts_lbm"
            )
            if scripts_lbm_path.exists() and str(scripts_lbm_path) not in sys.path:
                sys.path.insert(0, str(scripts_lbm_path))

            model_type = getattr(config, "densification_model_type", "intensity")
            pretrained_path = getattr(config, "densification_pretrained_path", None)

            try:
                try:
                    from s2d_woafm_depth_256x455 import DepthCompletionNet  # type: ignore
                except ImportError:
                    import importlib.util

                    model_file = scripts_lbm_path / "s2d_woafm_depth_256x455.py"
                    if model_file.exists():
                        spec = importlib.util.spec_from_file_location(
                            "s2d_woafm_depth_256x455", model_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            DepthCompletionNet = module.DepthCompletionNet
                        else:
                            raise ImportError("Unable to load s2d_woafm_depth_256x455")
                    else:
                        raise ImportError(f"Model file does not exist: {model_file}")

                full_model = DepthCompletionNet()

                self.densification_encoder = nn.Sequential(
                    full_model.conv1_d,
                    full_model.conv2,
                    full_model.conv3,
                    full_model.conv4,
                    full_model.conv5,
                    full_model.conv6,
                )

                if pretrained_path and Path(pretrained_path).exists():
                    self._load_densification_weights(pretrained_path, full_model)

                del full_model

                latent_channels = getattr(config, "vae_num_channels", 4)
                self.densification_projection = nn.Conv2d(
                    512, latent_channels, kernel_size=1
                )

                print(
                    f"Densification encoder initialized (fusion mode: {self.densification_fusion_mode}, feature level: {self.densification_feature_level})"
                )

            except ImportError as e:
                print(f"Warning: failed to import densification model: {e}")
                print("Using placeholder encoder")
                self.densification_encoder = nn.Identity()
                self.densification_projection = None

            if not hasattr(self, "densification_projection"):
                self.densification_projection = None

            if self.freeze_densification:
                self._freeze_densification()

        except Exception as e:
            print(f"Warning: failed to initialize densification model: {e}")
            print("Using placeholder encoder")
            self.densification_encoder = nn.Identity()

    def _load_densification_weights(self, path: str, full_model: nn.Module):

        try:
            checkpoint = torch.load(path, map_location="cpu")

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            try:
                full_model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained densification weights: {path}")
            except Exception as e:
                print(f"Warning: mismatched densification weights: {e}")

                model_dict = full_model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(pretrained_dict)
                full_model.load_state_dict(model_dict)

            encoder_state_dict = {}
            for name, module in full_model.named_modules():
                if name in ["conv1_d", "conv2", "conv3", "conv4", "conv5", "conv6"]:
                    for param_name, param in module.named_parameters():
                        full_name = f"{name}.{param_name}" if name else param_name
                        encoder_state_dict[full_name] = param.data.clone()

            if encoder_state_dict:
                sequential_state_dict = {}
                for i, (name, module) in enumerate(
                    self.densification_encoder.named_children()
                ):
                    for key, value in encoder_state_dict.items():
                        if key.startswith(name):
                            new_key = key.replace(f"{name}.", f"{i}.")
                            sequential_state_dict[new_key] = value

                if sequential_state_dict:
                    self.densification_encoder.load_state_dict(
                        sequential_state_dict, strict=False
                    )
                    print("Extracted densification encoder weights")

        except Exception as e:
            print(f"Error: failed to load pretrained weights {path}: {e}")

    def _freeze_densification(self):

        if self.densification_encoder is not None:
            self.densification_encoder.eval()
            for param in self.densification_encoder.parameters():
                param.requires_grad = False

    def _unfreeze_densification(self):

        if self.densification_encoder is not None:
            for param in self.densification_encoder.parameters():
                param.requires_grad = True

    def _extract_densification_features(
        self, sparse_input: torch.Tensor
    ) -> Optional[torch.Tensor]:

        if not self.use_densification or self.densification_encoder is None:
            return None

        if sparse_input.shape[1] == 3:
            sparse_input = sparse_input.mean(dim=1, keepdim=True)
        elif sparse_input.shape[1] != 1:
            sparse_input = sparse_input[:, 0:1, :, :]

        if self.freeze_densification:
            with torch.no_grad():
                features = self.densification_encoder(sparse_input)
        else:
            features = self.densification_encoder(sparse_input)
        return features * self.densification_feature_weight

    def _project_densification_to_latent(
        self, features: torch.Tensor, target_shape: Tuple[int, ...]
    ) -> torch.Tensor:

        if getattr(self, "densification_projection", None) is None:
            raise AttributeError(
                "densification_projection is not initialized; enable use_densification and load the encoder"
            )
        # [B, 512, H', W'] -> [B, 4, H', W']
        out = self.densification_projection(features)

        if out.shape[2:] != target_shape[2:]:
            out = F.interpolate(
                out, size=target_shape[2:], mode="bilinear", align_corners=False
            )
        return out.to(features.dtype)

    def _get_flow_start_latent(
        self,
        batch: Dict[str, Any],
        source_key: str,
        target_key: str,
        z_ref: torch.Tensor,
    ) -> torch.Tensor:

        dt = self.dtype
        source_image = torch.nn.functional.interpolate(
            batch[source_key].to(dtype=dt),
            size=batch[target_key].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z_vae = self.vae.encode(source_image) if self.vae is not None else source_image

        w = self.densification_z0_vae_weight
        if w >= 1.0:
            return z_vae
        if (
            not self.use_densification
            or self.densification_input_key not in batch
            or getattr(self, "densification_projection", None) is None
        ):
            return z_vae

        sparse_input = batch[self.densification_input_key].to(dtype=dt)
        densification_features = self._extract_densification_features(sparse_input)
        if densification_features is None:
            return z_vae

        z_dens = self._project_densification_to_latent(
            densification_features, target_shape=z_ref.shape
        ).to(dtype=z_ref.dtype)

        if w <= 0.0:
            return z_dens
        return w * z_vae + (1.0 - w) * z_dens

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        super().on_fit_start(device=device, *args, **kwargs)
        if self.vae is not None:
            self.vae.on_fit_start(device=device, *args, **kwargs)
        if self.conditioner is not None:
            self.conditioner.on_fit_start(device=device, *args, **kwargs)
        if self.densification_encoder is not None:
            self.densification_encoder.to(device)
        if getattr(self, "densification_projection", None) is not None:
            self.densification_projection.to(device)

    def _compute_frame_loss(
        self,
        batch: Dict[str, Any],
        source_key: str,
        target_key: str,
        mask_key: Optional[str] = None,
        compute_auxiliary: bool = True,
        **kwargs,
    ):

        if self.vae is not None:
            z = self.vae.encode(batch[target_key].to(dtype=self.dtype))
            downsampling_factor = self.vae.downsampling_factor
        else:
            z = batch[target_key]
            downsampling_factor = 1

        if mask_key and mask_key in batch:
            valid_mask = batch[mask_key].bool()[:, 0, :, :].unsqueeze(1)
            invalid_mask = ~valid_mask
            valid_mask_for_latent = ~torch.max_pool2d(
                invalid_mask.float(),
                downsampling_factor,
                downsampling_factor,
            ).bool()
            valid_mask_for_latent = valid_mask_for_latent.repeat((1, z.shape[1], 1, 1))
        else:
            valid_mask = torch.ones_like(batch[target_key]).bool()
            valid_mask_for_latent = torch.ones_like(z).bool()

        z_0 = self._get_flow_start_latent(batch, source_key, target_key, z)

        conditioning = self._get_conditioning(batch, latent_shape=z.shape, **kwargs)

        timestep = self._timestep_sampling(n_samples=z.shape[0], device=z.device)
        sigmas = self._get_sigmas(
            self.training_noise_scheduler, timestep, n_dim=4, device=z.device
        )

        noisy_sample = (
            sigmas * z_0
            + (1.0 - sigmas) * z
            + self.bridge_noise_sigma
            * (sigmas * (1.0 - sigmas)) ** 0.5
            * torch.randn_like(z)
        )

        for i, t in enumerate(timestep):
            if t.item() == self.training_noise_scheduler.timesteps[0]:
                noisy_sample[i] = z_0[i]

        prediction = self.denoiser(
            sample=noisy_sample, timestep=timestep, conditioning=conditioning, **kwargs
        )

        target_v = z_0 - z
        denoised_sample = self._predicted_x_0(prediction, noisy_sample, sigmas)

        latent_loss_val = torch.tensor(0.0, device=z.device)
        if self.latent_loss_weight > 0:
            latent_loss_val = self.latent_loss(
                prediction, target_v.detach(), valid_mask_for_latent
            ).mean()

        results = {
            "loss_sum": latent_loss_val,
            "latent_recon_loss": latent_loss_val,
            "denoised_sample": denoised_sample,
            "noisy_sample": noisy_sample,
        }

        if compute_auxiliary:
            pixel_loss_val = torch.tensor(0.0, device=z.device)
            if self.pixel_loss_weight > 0:
                pixel_loss_val = self.pixel_loss(
                    denoised_sample, batch[target_key].detach(), valid_mask
                )
                results["loss_sum"] += self.pixel_loss_weight * pixel_loss_val
                results["pixel_recon_loss"] = pixel_loss_val.mean()

            if self.use_color_loss or self.use_grad_loss:
                decoded_prediction_full = self.vae.decode(denoised_sample).clamp(-1, 1)

                if self.use_color_loss:
                    c_loss = self.color_loss(
                        decoded_prediction_full, batch[target_key], valid_mask
                    ).mean()
                    results["loss_sum"] += self.color_loss_weight * c_loss
                    results["color_loss"] = c_loss

                if self.use_grad_loss:
                    g_loss = self.grad_loss(
                        decoded_prediction_full,
                        batch[target_key],
                        valid_mask,
                        self.grad_loss_type,
                    ).mean()
                    results["loss_sum"] += self.grad_loss_weight * g_loss
                    results["grad_loss"] = g_loss

        return results

    def compute_pose_warp(
        self,
        prev_latent: torch.Tensor,
        pose_prev: torch.Tensor,
        pose_curr: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_prev: Optional[torch.Tensor] = None,
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = prev_latent.shape

        if latent_height is not None:
            H = latent_height
        if latent_width is not None:
            W = latent_width

        pose_rel = torch.bmm(
            torch.inverse(pose_prev),  # [B, 4, 4]
            pose_curr,  # [B, 4, 4]
        )  # [B, 4, 4]

        R = pose_rel[:, :3, :3]  # [B, 3, 3]
        t = pose_rel[:, :3, 3:4]  # [B, 3, 1]

        device = prev_latent.device
        dtype = prev_latent.dtype

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # [H, W] -> [B, H, W]
        x_coords = x_coords.unsqueeze(0).repeat(B, 1, 1)
        y_coords = y_coords.unsqueeze(0).repeat(B, 1, 1)

        if self.vae is not None:
            downsampling_factor = self.vae.downsampling_factor
        else:
            downsampling_factor = 1

        x_img = x_coords * downsampling_factor
        y_img = y_coords * downsampling_factor

        fx = intrinsics[:, 0, 0]  # [B]
        fy = intrinsics[:, 1, 1]  # [B]
        cx = intrinsics[:, 0, 2]  # [B]
        cy = intrinsics[:, 1, 2]  # [B]

        if depth_prev is not None:
            if depth_prev.shape[2] != H or depth_prev.shape[3] != W:
                depth_latent = F.interpolate(
                    depth_prev, size=(H, W), mode="bilinear", align_corners=False
                )
            else:
                depth_latent = depth_prev
            z_cam = depth_latent.squeeze(1)  # [B, H, W]
        else:
            if self.plane_depth_estimate == "fixed":
                z_cam = (
                    torch.ones(B, H, W, device=device, dtype=dtype)
                    * self.fixed_plane_depth
                )

            elif self.plane_depth_estimate == "focal_length":
                focal_length = (fx + fy) / 2.0  # [B]
                z_cam = (
                    focal_length.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)
                )  # [B, H, W]

            else:  # "auto"
                t_norm = torch.norm(t.squeeze(-1), dim=1)

                base_depth = 1.0

                adaptive_depth = base_depth / (
                    1.0 + t_norm.unsqueeze(-1).unsqueeze(-1) * 10.0
                )  # [B, 1, 1]
                z_cam = adaptive_depth.expand(B, H, W)  # [B, H, W]

                z_cam = torch.clamp(z_cam, min=0.1, max=10.0)

        x_cam = (
            (x_img - cx.unsqueeze(-1).unsqueeze(-1))
            * z_cam
            / fx.unsqueeze(-1).unsqueeze(-1)
        )
        y_cam = (
            (y_img - cy.unsqueeze(-1).unsqueeze(-1))
            * z_cam
            / fy.unsqueeze(-1).unsqueeze(-1)
        )

        points_3d_prev = torch.stack([x_cam, y_cam, z_cam], dim=1)  # [B, 3, H, W]

        points_3d_prev_flat = points_3d_prev.view(B, 3, -1)  # [B, 3, H*W]
        points_3d_curr = torch.bmm(R, points_3d_prev_flat) + t  # [B, 3, H*W]

        x_proj = points_3d_curr[:, 0] / (
            points_3d_curr[:, 2] + self._eps
        ) * fx.unsqueeze(-1) + cx.unsqueeze(-1)
        y_proj = points_3d_curr[:, 1] / (
            points_3d_curr[:, 2] + self._eps
        ) * fy.unsqueeze(-1) + cy.unsqueeze(-1)

        x_proj_latent = x_proj.view(B, H, W) / downsampling_factor
        y_proj_latent = y_proj.view(B, H, W) / downsampling_factor

        grid_x = 2.0 * x_proj_latent / W - 1.0
        grid_y = 2.0 * y_proj_latent / H - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

        prev_latent_warped = F.grid_sample(
            prev_latent,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        depth_curr = points_3d_curr[:, 2].view(B, H, W)  # [B, H, W]
        valid_mask = (
            (
                (grid_x >= -1)
                & (grid_x <= 1)
                & (grid_y >= -1)
                & (grid_y <= 1)
                & (depth_curr > 0)
            )
            .float()
            .unsqueeze(1)
        )  # [B, 1, H, W]

        return prev_latent_warped, valid_mask

    def compute_rl_reward(
        self,
        current_latent: torch.Tensor,
        prev_latent: torch.Tensor,
        gt_latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        fidelity_loss = F.mse_loss(current_latent, gt_latent, reduction="none")
        if mask is not None:
            fidelity_loss = (fidelity_loss * mask).sum() / (mask.sum() + self._eps)
        else:
            fidelity_loss = fidelity_loss.mean()
        reward_f = -fidelity_loss

        temporal_diff = torch.abs(current_latent - prev_latent)
        reward_t = -temporal_diff.mean()

        total_reward = reward_f + self.temporal_stability_weight * reward_t
        return total_reward

    def rl_training_step(self, batch):

        dt_enc = self.dtype
        target_z = self.vae.encode(batch[self.target_key].to(dtype=dt_enc))

        prev_target_key = f"{self.target_key}_prev"
        if prev_target_key not in batch:
            print("Unable to build the LBM training target.")
            return torch.tensor(0.0, device=target_z.device)

        prev_target_z = self.vae.encode(batch[prev_target_key].to(dtype=dt_enc))

        source_image = batch[self.source_key].to(dtype=dt_enc)
        current_z = self.vae.encode(source_image).detach().clone()

        conditioning = self._get_conditioning(batch, latent_shape=current_z.shape)

        dt = 1.0 / self.config.rl_sample_steps
        for step in range(self.config.rl_sample_steps):
            t = torch.ones(current_z.shape[0], device=current_z.device) * (
                1.0 - step * dt
            )
            v_pred = self.denoiser(
                sample=current_z, timestep=t, conditioning=conditioning
            )
            current_z = current_z - v_pred * dt

        reward = self.compute_rl_reward(
            current_latent=current_z, prev_latent=prev_target_z, gt_latent=target_z
        )

        return -reward.mean() * self.config.rl_reward_weight

    def forward(self, batch: Dict[str, Any], step=0, batch_idx=0, *args, **kwargs):
        self.num_iterations.add_(1)

        res_t = self._compute_frame_loss(
            batch,
            self.source_key,
            self.target_key,
            self.mask_key,
            compute_auxiliary=True,
            **kwargs,
        )
        total_loss = res_t["loss_sum"]

        loss_temporal = torch.tensor(
            0.0, device=total_loss.device, dtype=total_loss.dtype
        )

        prev_source_key = f"{self.source_key}_prev"
        if prev_source_key in batch:
            try:
                res_prev = self._compute_frame_loss(
                    batch,
                    source_key=prev_source_key,
                    target_key=f"{self.target_key}_prev",
                    mask_key=f"{self.mask_key}_prev"
                    if f"{self.mask_key}_prev" in batch
                    else None,
                    compute_auxiliary=False,
                    **kwargs,
                )

                if self.use_pose_warp:
                    pose_prev_key = (
                        "pose_prev"
                        if "pose_prev" in batch
                        else f"{self.source_key}_pose_prev"
                    )
                    pose_curr_key = (
                        "pose_curr"
                        if "pose_curr" in batch
                        else f"{self.source_key}_pose"
                    )
                    intrinsics_key = (
                        "intrinsics" if "intrinsics" in batch else "camera_intrinsics"
                    )
                    depth_prev_key = (
                        "depth_prev"
                        if "depth_prev" in batch
                        else f"{self.source_key}_depth_prev"
                    )

                    if (
                        pose_prev_key in batch
                        and pose_curr_key in batch
                        and intrinsics_key in batch
                    ):
                        pose_prev = batch[pose_prev_key]  # [B, 4, 4]
                        pose_curr = batch[pose_curr_key]  # [B, 4, 4]
                        intrinsics = batch[intrinsics_key]  # [B, 3, 3]

                        depth_prev = batch.get(depth_prev_key, None)

                        if pose_prev.dim() == 2:
                            pose_prev = pose_prev.unsqueeze(0).repeat(
                                res_prev["denoised_sample"].shape[0], 1, 1
                            )
                        if pose_curr.dim() == 2:
                            pose_curr = pose_curr.unsqueeze(0).repeat(
                                res_t["denoised_sample"].shape[0], 1, 1
                            )
                        if intrinsics.dim() == 2:
                            intrinsics = intrinsics.unsqueeze(0).repeat(
                                res_t["denoised_sample"].shape[0], 1, 1
                            )

                        if self.require_depth_for_warp and depth_prev is None:
                            loss_temporal = F.mse_loss(
                                res_t["denoised_sample"], res_prev["denoised_sample"]
                            )
                        else:
                            prev_latent_warped, valid_mask = self.compute_pose_warp(
                                res_prev["denoised_sample"],
                                pose_prev,
                                pose_curr,
                                intrinsics,
                                depth_prev=depth_prev,
                            )

                            if valid_mask.sum() > 0:
                                diff = (
                                    res_t["denoised_sample"] - prev_latent_warped
                                ) ** 2
                                loss_temporal = (diff * valid_mask).sum() / (
                                    valid_mask.sum() + self._eps
                                )
                            else:
                                loss_temporal = F.mse_loss(
                                    res_t["denoised_sample"],
                                    res_prev["denoised_sample"],
                                )
                    else:
                        loss_temporal = F.mse_loss(
                            res_t["denoised_sample"], res_prev["denoised_sample"]
                        )
                else:
                    loss_temporal = F.mse_loss(
                        res_t["denoised_sample"], res_prev["denoised_sample"]
                    )
            except Exception as e:
                loss_temporal = torch.tensor(
                    0.0, device=total_loss.device, dtype=total_loss.dtype
                )
                if self.training and torch.distributed.is_initialized():
                    import warnings

                    warnings.warn(
                        f"Failed to compute temporal loss: {e}. Using zero loss to maintain synchronization."
                    )

            total_loss += self.temporal_stability_weight * loss_temporal

        rl_loss = torch.tensor(0.0, device=total_loss.device)
        if getattr(self.config, "use_rl_finetune", False) and self.training:
            rl_loss = self.rl_training_step(batch)
            total_loss += rl_loss

        return {
            "loss": total_loss,
            "rl_loss": rl_loss,
            "latent_recon_loss": res_t["latent_recon_loss"],
            "pixel_recon_loss": res_t.get("pixel_recon_loss", torch.tensor(0.0)),
            "color_loss": res_t.get("color_loss", torch.tensor(0.0)),
            "grad_loss": res_t.get("grad_loss", torch.tensor(0.0)),
            "predicted_hr": res_t["denoised_sample"],
        }

    def latent_loss(self, prediction, model_input, valid_latent_mask):
        if self.latent_loss_type == "l2":
            return torch.mean(
                (
                    (prediction * valid_latent_mask - model_input * valid_latent_mask)
                    ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )
        elif self.latent_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    prediction * valid_latent_mask - model_input * valid_latent_mask
                ).reshape(model_input.shape[0], -1),
                1,
            )
        else:
            raise NotImplementedError(
                f"Loss type {self.latent_loss_type} not implemented"
            )

    def pixel_loss(self, prediction, model_input, valid_mask):
        latent_crop = self.pixel_loss_max_size // self.vae.downsampling_factor
        input_crop = self.pixel_loss_max_size
        crop_h = max((prediction.shape[2] - latent_crop), 0)
        crop_w = max((prediction.shape[3] - latent_crop), 0)
        input_crop_h = max((model_input.shape[2] - self.pixel_loss_max_size), 0)
        input_crop_w = max((model_input.shape[3] - self.pixel_loss_max_size), 0)

        offset_h = 0 if crop_h == 0 else torch.randint(0, crop_h, (1,)).item()
        offset_w = 0 if crop_w == 0 else torch.randint(0, crop_w, (1,)).item()
        input_offset_h, input_offset_w = (
            offset_h * self.vae.downsampling_factor,
            offset_w * self.vae.downsampling_factor,
        )

        prediction = prediction[
            :, :, offset_h : offset_h + latent_crop, offset_w : offset_w + latent_crop
        ]
        model_input = model_input[
            :,
            :,
            input_offset_h : input_offset_h + input_crop,
            input_offset_w : input_offset_w + input_crop,
        ]
        valid_mask = valid_mask[
            :,
            :,
            input_offset_h : input_offset_h + input_crop,
            input_offset_w : input_offset_w + input_crop,
        ]

        decoded_prediction = self.vae.decode(prediction).clamp(-1, 1)
        if self.pixel_loss_type == "l2":
            return torch.mean(
                (
                    (decoded_prediction * valid_mask - model_input * valid_mask) ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )
        elif self.pixel_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    decoded_prediction * valid_mask - model_input * valid_mask
                ).reshape(model_input.shape[0], -1),
                1,
            )
        elif self.pixel_loss_type == "lpips":
            return self.lpips_loss(
                decoded_prediction * valid_mask, model_input * valid_mask
            ).mean()

    def color_loss(self, pred_rgb, gt_rgb, mask):
        mu_pred, std_pred = self._masked_mean_std(pred_rgb, mask)
        mu_gt, std_gt = self._masked_mean_std(gt_rgb, mask)
        mean_term = torch.sqrt(((mu_pred - mu_gt) ** 2).sum(dim=[1, 2, 3]) + self._eps)
        std_term = torch.sqrt(((std_pred - std_gt) ** 2).sum(dim=[1, 2, 3]) + self._eps)
        return mean_term + std_term

    def grad_loss(self, pred_rgb, gt_rgb, mask, loss_type="l1"):
        gx_p, gy_p = self._image_grad(pred_rgb)
        gx_g, gy_g = self._image_grad(gt_rgb)
        m_exp = (
            mask.float().repeat(1, pred_rgb.shape[1], 1, 1)
            if mask.shape[1] == 1
            else mask.float()
        )
        diff = (
            (gx_p - gx_g) ** 2 + (gy_p - gy_g) ** 2
            if loss_type == "l2"
            else torch.abs(gx_p - gx_g) + torch.abs(gy_p - gy_g)
        )
        masked = diff * m_exp
        return masked.reshape(masked.shape[0], -1).mean(dim=1)

    def _masked_mean_std(self, x, mask):
        m = (
            mask.float().repeat(1, x.shape[1], 1, 1)
            if mask.shape[1] == 1
            else mask.float()
        )
        denom = torch.clamp(m.sum(dim=[2, 3], keepdim=True), min=self._eps)
        mean = (x * m).sum(dim=[2, 3], keepdim=True) / denom
        var = ((x - mean) ** 2 * m).sum(dim=[2, 3], keepdim=True) / denom
        return mean, torch.sqrt(var + self._eps)

    def _image_grad(self, x):
        C = x.shape[1]
        kx = self._sobel_kernel_x.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
        ky = self._sobel_kernel_y.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
        return F.conv2d(x, kx, padding=1, groups=C), F.conv2d(
            x, ky, padding=1, groups=C
        )

    def _make_sobel_kernel_x(self):
        return torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def _make_sobel_kernel_y(self):
        return torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    def _get_conditioning(
        self,
        batch,
        ucg_keys=None,
        set_ucg_rate_zero=False,
        latent_shape=None,
        *args,
        **kwargs,
    ):

        conditioning = None
        if self.conditioner is not None:
            conditioning = self.conditioner(
                batch,
                ucg_keys=ucg_keys,
                set_ucg_rate_zero=set_ucg_rate_zero,
                vae=self.vae,
                *args,
                **kwargs,
            )

        return conditioning

    def _timestep_sampling(self, n_samples=1, device="cpu"):
        if self.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                self.training_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device="cpu",
            )
            return self.training_noise_scheduler.timesteps[idx].to(device=device)
        elif self.timestep_sampling == "log_normal":
            u = torch.nn.functional.sigmoid(
                torch.normal(
                    mean=self.logit_mean,
                    std=self.logit_std,
                    size=(n_samples,),
                    device="cpu",
                )
            )
            indices = (
                u * self.training_noise_scheduler.config.num_train_timesteps
            ).long()
            return self.training_noise_scheduler.timesteps[indices].to(device=device)
        elif self.timestep_sampling == "custom_timesteps":
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)
            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

    def _predicted_x_0(self, model_output, sample, sigmas=None):
        return sample - model_output * sigmas

    def _get_sigmas(
        self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"
    ):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        step_indices = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps.to(device)
        ]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # @torch.no_grad()
    # def sample(self, z, num_steps=20, conditioner_inputs=None, max_samples=None, verbose=False, previous_sample=None):
    #     self.sampling_noise_scheduler.set_timesteps(sigmas=np.linspace(1, 1/num_steps, num_steps))
    #     sample = z[:max_samples] if max_samples else z
    #     conditioning = self._get_conditioning(conditioner_inputs, set_ucg_rate_zero=True, device=z.device)
    #     if conditioning and max_samples:
    #         conditioning["cond"] = {k: v[:max_samples] for k, v in conditioning["cond"].items()}

    #     for i, t in tqdm(enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose):
    #         denoiser_input = self.sampling_noise_scheduler.scale_model_input(sample, t) if hasattr(self.sampling_noise_scheduler, "scale_model_input") else sample
    #         pred = self.denoiser(sample=denoiser_input, timestep=t.to(z.device).repeat(denoiser_input.shape[0]), conditioning=conditioning)
    #         sample = self.sampling_noise_scheduler.step(pred, t, sample, return_dict=False)[0]

    #         if i < len(self.sampling_noise_scheduler.timesteps) - 1:
    #             sigmas = self._get_sigmas(self.sampling_noise_scheduler, self.sampling_noise_scheduler.timesteps[i+1].repeat(sample.shape[0]), n_dim=4, device=z.device)
    #             noise_scale = self.bridge_noise_sigma * (sigmas * (1.0 - sigmas)) ** 0.5

    #             random_noise = torch.randn_like(sample)
    #             if previous_sample is not None and self.temporal_stability_weight > 0.0 and previous_sample.shape == sample.shape:
    #                 diff = sample - previous_sample
    #                 temporal_direction = diff / (torch.norm(diff.view(diff.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8)
    #                 temporal_noise = temporal_direction * (torch.norm(random_noise.view(random_noise.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8)
    #                 blended_noise = (1.0 - self.temporal_noise_smoothing) * random_noise + self.temporal_noise_smoothing * temporal_noise
    #                 noise = (1.0 - self.temporal_stability_weight) * random_noise + self.temporal_stability_weight * blended_noise
    #             else:
    #                 noise = random_noise

    #             sample = (sample + noise_scale * noise).to(z.dtype)

    #     return self.vae.decode(sample) if self.vae is not None else sample
    @torch.no_grad()
    def sample(
        self,
        z,
        num_steps=20,
        conditioner_inputs=None,
        max_samples=None,
        verbose=False,
        previous_sample=None,
    ):
        self.sampling_noise_scheduler.set_timesteps(
            sigmas=np.linspace(1, 1 / num_steps, num_steps)
        )
        sample = z[:max_samples] if max_samples else z
        conditioning = self._get_conditioning(
            conditioner_inputs, set_ucg_rate_zero=True, device=z.device
        )
        if conditioning and max_samples:
            conditioning["cond"] = {
                k: v[:max_samples] for k, v in conditioning["cond"].items()
            }

        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):
            timestep_curr = (
                self.sampling_noise_scheduler.timesteps[i]
                .to(z.device)
                .repeat(sample.shape[0])
            )
            sigmas_curr = self._get_sigmas(
                self.sampling_noise_scheduler, timestep_curr, n_dim=4, device=z.device
            )

            denoiser_input = (
                self.sampling_noise_scheduler.scale_model_input(sample, t)
                if hasattr(self.sampling_noise_scheduler, "scale_model_input")
                else sample
            )
            pred = self.denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                conditioning=conditioning,
            )

            # new_sample = self.sampling_noise_scheduler.step(pred, t, sample, return_dict=False)[0]

            dt = 1.0 / num_steps
            new_sample = sample - pred * dt

            if i < len(self.sampling_noise_scheduler.timesteps) - 1:
                timestep_next = (
                    self.sampling_noise_scheduler.timesteps[i + 1]
                    .to(z.device)
                    .repeat(sample.shape[0])
                )
                sigmas_next = self._get_sigmas(
                    self.sampling_noise_scheduler,
                    timestep_next,
                    n_dim=4,
                    device=z.device,
                )

                scale = self.bridge_noise_sigma**2 * sigmas_next * (1.0 - sigmas_next)
                scale_curr = (
                    self.bridge_noise_sigma**2 * sigmas_curr * (1.0 - sigmas_curr)
                )
                noise_scale = (
                    1
                    - scale
                    * (1.0 - sigmas_curr)
                    / (1.0 - sigmas_next) ** 2
                    / self.bridge_noise_sigma**2
                    / sigmas_curr
                ) * scale

                noise = torch.randn_like(sample)
                sample = new_sample + noise_scale**0.5 * noise
            else:
                sample = new_sample

            sample = sample.to(z.dtype)

        return self.vae.decode(sample) if self.vae is not None else sample

    def log_samples(self, batch, input_shape=None, max_samples=None, num_steps=20):
        if isinstance(num_steps, int):
            num_steps = [num_steps]
        N = max_samples if max_samples is not None else len(batch[self.source_key])
        batch = {k: v[:N] for k, v in batch.items()}
        logs = {}
        for num_step in num_steps:
            z_ref = (
                self.vae.encode(batch[self.target_key].to(dtype=self.dtype))
                if self.vae is not None
                else batch[self.target_key]
            )
            z_start = self._get_flow_start_latent(
                batch, self.source_key, self.target_key, z_ref
            )
            with torch.autocast(dtype=self.dtype, device_type="cuda"):
                logs[f"samples_{num_step}_steps"] = self.sample(
                    z_start, num_steps=num_step, conditioner_inputs=batch, max_samples=N
                )
        return logs
