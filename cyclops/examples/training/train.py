import datetime
import logging
import os
import random
import re
import shutil
import sys
from typing import List, Optional, Dict, Any


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import fire
import torch
import yaml
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode

from cyclops.data import CyclopsDataConfig, CyclopsDataModule
from cyclops.models import CyclopsConfig, CyclopsModel, CyclopsUNet2DCondWrapper
from cyclops.trainer import CyclopsTrainingPipeline
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig
from lbm.trainer.loggers import WandbSampleLogger
from lbm.trainer.utils import StateDictAdapter


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def get_model(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    source_key: str = "normal",
    target_key: str = "image",
    mask_key: str = "mask",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,

    temporal_stability_weight: float = 0.1,
    temporal_noise_smoothing: float = 0.5,
    use_temporal_stability: bool = True,
    use_color_loss: bool = True,
    color_loss_weight: float = 0.1,
    use_grad_loss: bool = True,
    grad_loss_weight: float = 0.1,
    grad_loss_type: str = "l1",

    use_pose_warp: bool = True,
    require_depth_for_warp: bool = False,
    plane_depth_estimate: str = "auto",
    fixed_plane_depth: float = 1.0,

    use_densification: bool = False,
    densification_pretrained_path: Optional[str] = None,
    freeze_densification: bool = True,
    densification_input_key: str = "sparse_intensity",
    densification_model_type: str = "intensity",
    densification_fusion_mode: str = "concat",
    densification_feature_level: str = "encoder",
    densification_feature_weight: float = 1.0,
    **kwargs
):
    conditioners = []


    pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )


    denoiser = CyclopsUNet2DCondWrapper(
        in_channels=unet_input_channels,
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)


    state_dict = pipe.unet.state_dict()
    for k in ["add_embedding.linear_1.weight", "add_embedding.linear_1.bias",
              "add_embedding.linear_2.weight", "add_embedding.linear_2.bias"]:
        if k in state_dict: del state_dict[k]

    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )
    denoiser.load_state_dict(state_dict, strict=True)
    del pipe


    if conditioning_images_keys or conditioning_masks_keys:
        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    conditioner = ConditionerWrapper(conditioners=conditioners)


    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)


    config = CyclopsConfig(
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
        temporal_stability_weight=temporal_stability_weight,
        temporal_noise_smoothing=temporal_noise_smoothing,
        use_temporal_stability=use_temporal_stability,
        use_color_loss=use_color_loss,
        color_loss_weight=color_loss_weight,
        use_grad_loss=use_grad_loss,
        grad_loss_weight=grad_loss_weight,
        grad_loss_type=grad_loss_type,

        use_pose_warp=use_pose_warp,
        require_depth_for_warp=require_depth_for_warp,
        plane_depth_estimate=plane_depth_estimate,
        fixed_plane_depth=fixed_plane_depth,

        use_densification=use_densification,
        densification_pretrained_path=densification_pretrained_path,
        freeze_densification=freeze_densification,
        densification_input_key=densification_input_key,
        densification_model_type=densification_model_type,
        densification_fusion_mode=densification_fusion_mode,
        densification_feature_level=densification_feature_level,
        densification_feature_weight=densification_feature_weight,
        ode_num_steps=int(kwargs.get("ode_num_steps", 4)),
        bridge_time_values=tuple(kwargs.get("bridge_time_values", (0.0, 0.25, 0.5, 0.75))),
        temporal_conditioning_mode=kwargs.get("temporal_conditioning_mode", "teacher"),
        latent_attention_dim=int(kwargs.get("latent_attention_dim", 64)),
        latent_attention_heads=int(kwargs.get("latent_attention_heads", 4)),
        training_phase=kwargs.get("training_phase", "phase1"),
        teacher_forcing_start=float(kwargs.get("teacher_forcing_start", 1.0)),
        teacher_forcing_end=float(kwargs.get("teacher_forcing_end", 0.2)),
        teacher_forcing_anneal_steps=int(kwargs.get("teacher_forcing_anneal_steps", 20000)),
        lpips_weight=float(kwargs.get("lpips_weight", 1.0)),
        gradient_weight=float(kwargs.get("gradient_weight", 0.1)),
        color_weight=float(kwargs.get("color_weight", 0.05)),
        reward_weight=float(kwargs.get("reward_weight", 0.8)),
        reward_fidelity_weight=float(kwargs.get("reward_fidelity_weight", 1.0)),
        reward_temporal_weight=float(kwargs.get("reward_temporal_weight", 0.5)),
        vae_num_channels=vae_num_channels,
        use_lbm_core=bool(kwargs.get("use_lbm_core", False)),
        use_source_attention=bool(kwargs.get("use_source_attention", True)),
        use_temporal_attention=bool(kwargs.get("use_temporal_attention", True)),
        use_scheduled_sampling=bool(kwargs.get("use_scheduled_sampling", True)),
        use_lpips_loss=bool(kwargs.get("use_lpips_loss", True)),
        use_gradient_loss=bool(kwargs.get("use_gradient_loss", True)),
        use_color_statistics_loss=bool(kwargs.get("use_color_statistics_loss", True)),
        use_terminal_reward=bool(kwargs.get("use_terminal_reward", True)),
    )
    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature, subfolder="scheduler"
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature, subfolder="scheduler"
    )

    model = CyclopsModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model


def get_filter_mappers():

    key_map = {
        "jpg": "image",
        "jpg_prev": "image_prev",
        "normal_aligned.png": "normal",
        "normal_prev_aligned.png": "normal_prev",
        "jpg_prev_prev": "image_prev_prev",
        "normal_prev_prev_aligned.png": "normal_prev_prev",
        "mask.png": "mask",
        "mask_prev.png": "mask_prev",
        "pose_curr": "pose_curr",
        "pose_prev": "pose_prev",
        "intrinsics": "intrinsics",
        "sparse_intensity": "sparse_intensity",

    }

    image_like_keys = [
        "image", "image_prev", "image_prev_prev",
        "normal", "normal_prev", "normal_prev_prev",
    ]
    mask_like_keys = ["mask", "mask_prev"]
    target_size = (256, 455)

    filters_mappers = [


        MapperWrapper([
            KeyRenameMapper(KeyRenameMapperConfig(key_map=key_map)),


            *[
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key=k,
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {"size": target_size, "interpolation": InterpolationMode.BICUBIC},
                        ],
                    )
                ) for k in image_like_keys
            ],


            *[
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key=k,
                        transforms=["ToTensor", "Resize", "Normalize"],
                        transforms_kwargs=[
                            {},
                            {"size": target_size, "interpolation": InterpolationMode.NEAREST},
                            {"mean": 0.0, "std": 1.0},
                        ],
                    )
                ) for k in mask_like_keys
            ],


            *[RescaleMapper(RescaleMapperConfig(key=k)) for k in image_like_keys]
        ]),
    ]
    return filters_mappers


def get_data_module(
    train_data_path: str,
    validation_data_path: str,
    batch_size: int,
    num_workers: int = 4,
    camera_intrinsics: Optional[List] = None,
    densification_input_size: Optional[tuple] = (256, 455),
):

    train_data_config = CyclopsDataConfig(
        data_path=train_data_path,
        per_worker_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )


    validation_data_config = CyclopsDataConfig(
        data_path=validation_data_path,
        per_worker_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return CyclopsDataModule(
        train_config=train_data_config,
        train_transforms=get_filter_mappers(),
        validation_config=validation_data_config,
        validation_transforms=get_filter_mappers(),
    )


def main(
    train_data_path: str = None,
    validation_data_path: str = None,
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "normal",
    target_key: str = "image",
    mask_key: str = "mask",
    wandb_project: str = "lbm-temporal",
    batch_size: int = 4,
    num_steps: List[int] = [1, 4],
    learning_rate: float = 4e-5,
    save_ckpt_path: str = "./checkpoints",
    log_interval: int = 100,
    resume_from_checkpoint: bool = True,
    pretrained_weights_path: str = None,
    max_epochs: int = 50,
    save_interval: int = 1000,
    path_config: str = None,
    num_gpus: int = 1,
    **kwargs
):
    seed_everything(int(kwargs.get("seed", 42)), workers=True)


    os.makedirs(save_ckpt_path, exist_ok=True)
    log_dir = os.path.join(save_ckpt_path, "logs")
    os.makedirs(log_dir, exist_ok=True)


    log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_train.log"
    log_filepath = os.path.join(log_dir, log_filename)


    log_file = open(log_filepath, 'a', encoding='utf-8', buffering=1)


    original_stdout = sys.stdout
    original_stderr = sys.stderr


    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)


    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)


    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)


    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    print("=" * 80)
    print(f"训练日志将保存到: {log_filepath}")
    print(f"训练配置 - 数据路径: {train_data_path}, 保存路径: {save_ckpt_path}, GPU数量: {num_gpus}")
    print("=" * 80)

    try:

        model = get_model(
            backbone_signature=backbone_signature,
            vae_num_channels=vae_num_channels,
            unet_input_channels=unet_input_channels,
            source_key=source_key,
            target_key=target_key,
            mask_key=mask_key,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
        if bool(kwargs.get("gradient_checkpointing", False)):
            enable_gc = getattr(model.denoiser, "enable_gradient_checkpointing", None)
            if enable_gc is None:
                raise RuntimeError("The configured denoiser does not support gradient checkpointing")
            enable_gc()
            print("已启用 U-Net gradient checkpointing")


        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"加载预训练权重: {pretrained_weights_path}")
            try:
                state_dict = load_file(pretrained_weights_path)
                print(f"权重文件包含 {len(state_dict)} 个参数")


                if any(k.startswith("model.") for k in state_dict.keys()):
                    print("检测到 PyTorch Lightning 格式权重（包含 'model.' 前缀），正在转换...")
                    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}


                model_state_dict = model.state_dict()
                model_keys = set(model_state_dict.keys())
                weight_keys = set(state_dict.keys())


                filtered_state_dict = {}
                shape_mismatch_keys = []

                for key in weight_keys:
                    if key in model_keys:

                        weight_shape = state_dict[key].shape
                        model_shape = model_state_dict[key].shape
                        if weight_shape == model_shape:
                            filtered_state_dict[key] = state_dict[key]
                        else:
                            shape_mismatch_keys.append(f"{key} (权重: {weight_shape}, 模型: {model_shape})")
                    else:

                        pass


                matched_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - weight_keys
                unexpected_keys = weight_keys - model_keys

                print(f"权重匹配情况: {len(matched_keys)}/{len(model_keys)} 个参数匹配")
                if shape_mismatch_keys:
                    print(f"跳过 {len(shape_mismatch_keys)} 个形状不匹配的参数:")
                    for key_info in shape_mismatch_keys[:5]:
                        print(f"  - {key_info}")
                    if len(shape_mismatch_keys) > 5:
                        print(f"  ... 还有 {len(shape_mismatch_keys) - 5} 个")
                if missing_keys:
                    print(f"警告: {len(missing_keys)} 个模型参数未在权重中找到（将使用随机初始化）")
                if unexpected_keys:
                    print(f"信息: {len(unexpected_keys)} 个权重参数未使用（可能是不同配置的权重）")


                missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                if missing:
                    print(f"未匹配的参数: {len(missing)} 个")
                if unexpected:
                    print(f"未使用的权重: {len(unexpected)} 个")

                print("预训练权重加载完成")
            except Exception as e:
                print(f"错误: 加载预训练权重失败: {e}")
                import traceback
                traceback.print_exc()
                print("将从头开始训练")


        camera_intrinsics = kwargs.get("camera_intrinsics", None)
        densification_input_size = kwargs.get("densification_input_size", (256, 455))
        data_module = get_data_module(
            train_data_path=train_data_path,
            validation_data_path=validation_data_path or train_data_path,
            batch_size=batch_size,
            num_workers=int(kwargs.get("num_workers", 4)),
            camera_intrinsics=camera_intrinsics,
            densification_input_size=densification_input_size,
        )


        trainable_params = [r"^denoiser\."]
        if not getattr(model, "use_lbm_core", False):
            trainable_params.append(r"^null_previous_latent$")
        training_config = TrainingConfig(
            learning_rate=learning_rate,
            log_keys=["image", "normal", "mask"],
            trainable_params=trainable_params,
            optimizer_name=kwargs.get("optimizer", "AdamW"),
            log_samples_model_kwargs={"num_steps": num_steps},
        )

        pipeline = CyclopsTrainingPipeline(model=model, pipeline_config=training_config)


        init_from_checkpoint_path = kwargs.get("init_from_checkpoint_path", None)
        if init_from_checkpoint_path:
            init_from_checkpoint_path = os.path.abspath(os.path.expanduser(init_from_checkpoint_path))
            if not os.path.exists(init_from_checkpoint_path):
                raise FileNotFoundError(
                    f"init_from_checkpoint_path 不存在: {init_from_checkpoint_path}"
                )
            try:
                ckpt = torch.load(
                    init_from_checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                ckpt = torch.load(init_from_checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
            critical_missing = [
                key for key in missing
                if key.startswith("model.denoiser.") or key == "model.null_previous_latent"
            ]
            if critical_missing:
                raise RuntimeError(
                    "Phase-I checkpoint is incompatible; missing trainable keys: "
                    f"{critical_missing[:20]}"
                )
            print(f"从 checkpoint 初始化模型权重: {init_from_checkpoint_path}")
            if missing:
                print(f"初始化时缺失 {len(missing)} 个非训练参数（前 10 个）: {missing[:10]}")
            if unexpected:
                print(f"初始化时多余 {len(unexpected)} 个参数（前 10 个）: {unexpected[:10]}")


        start_ckpt = None
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):

            print("使用预训练权重，从头开始训练（不包含优化器状态）")
        elif resume_from_checkpoint and os.path.exists(f"{save_ckpt_path}/last.ckpt"):

            start_ckpt = f"{save_ckpt_path}/last.ckpt"
            print(f"断点续传: {start_ckpt}")
        elif init_from_checkpoint_path:
            print("使用 init_from_checkpoint_path 初始化权重，并从当前 yaml 的训练设置开始微调")
        else:
            print("从头开始训练（未找到预训练权重或断点续传文件）")


        training_signature = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + "-LBM-Temporal"


        if num_gpus > 1:
            from pytorch_lightning.strategies import DDPStrategy
            ddp_strategy = DDPStrategy(find_unused_parameters=True)
            print(f"使用 DDP 策略进行 {num_gpus} 张 GPU 训练")
        else:
            ddp_strategy = "auto"
            print("使用单 GPU 训练")


        enable_checkpoint = bool(kwargs.get("enable_checkpoint", True))
        save_last = bool(kwargs.get("save_last", True))
        save_top_k = int(kwargs.get("save_top_k", 1))
        checkpoint_monitor = kwargs.get("checkpoint_monitor", "train/total_loss_epoch")
        checkpoint_mode = kwargs.get("checkpoint_mode", "min")
        checkpoint_every_n_epochs = int(kwargs.get("checkpoint_every_n_epochs", 1))
        checkpoint_every_n_train_steps = kwargs.get("checkpoint_every_n_train_steps", None)
        save_weights_only = bool(kwargs.get("save_weights_only", False))

        use_wandb = bool(kwargs.get("use_wandb", True))
        if use_wandb:
            callbacks = [
                WandbSampleLogger(log_batch_freq=log_interval),
                LearningRateMonitor(logging_interval="step"),
            ]
            trainer_logger = loggers.WandbLogger(
                project=wandb_project, name=training_signature, save_dir=save_ckpt_path
            )
        else:
            callbacks = []
            trainer_logger = False
        if enable_checkpoint:
            checkpoint_kwargs = dict(
                dirpath=save_ckpt_path,
                save_last=save_last,
                save_top_k=save_top_k,
                monitor=checkpoint_monitor if save_top_k != 0 else None,
                mode=checkpoint_mode,
                filename="best-{epoch:03d}-{step:08d}",
                save_weights_only=save_weights_only,
            )
            if checkpoint_every_n_train_steps:
                checkpoint_kwargs["every_n_train_steps"] = int(checkpoint_every_n_train_steps)
            else:
                checkpoint_kwargs["every_n_epochs"] = checkpoint_every_n_epochs
            callbacks.append(
                ModelCheckpoint(**checkpoint_kwargs)
            )

        trainer = Trainer(
            accelerator=kwargs.get("accelerator", "gpu"),
            devices=num_gpus,
            strategy=ddp_strategy,
            logger=trainer_logger,
            callbacks=callbacks,
            enable_checkpointing=enable_checkpoint,
            precision=kwargs.get("precision", "bf16-mixed"),
            max_epochs=max_epochs,
            limit_train_batches=kwargs.get("limit_train_batches", 1.0),
            limit_val_batches=kwargs.get("limit_val_batches", 1.0),
            num_sanity_val_steps=int(kwargs.get("num_sanity_val_steps", 2)),
            log_every_n_steps=int(kwargs.get("log_every_n_steps", 50)),
            check_val_every_n_epoch=1,
            deterministic=bool(kwargs.get("deterministic", False)),
            accumulate_grad_batches=int(kwargs.get("accumulate_grad_batches", 1)),
        )

        trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        if 'log_file' in locals():
            log_file.close()
            print(f"训练日志已保存到: {log_filepath}")


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    main(**config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
