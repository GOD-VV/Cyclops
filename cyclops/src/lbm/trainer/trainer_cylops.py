import importlib
import logging
import re
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from ..models.base.base_model import BaseModel
from .training_config_cylops import TrainingConfig

logging.basicConfig(level=logging.INFO)


class TrainingPipeline(pl.LightningModule):
    """
    Main Training Pipeline class

    Args:

        model (BaseModel): The model to train
        pipeline_config (TrainingConfig): The configuration for the training pipeline
        verbose (bool): Whether to print logs in the console. Default is False.
    """

    def __init__(
        self,
        model: BaseModel,
        pipeline_config: TrainingConfig,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.pipeline_config = pipeline_config
        self.log_samples_model_kwargs = pipeline_config.log_samples_model_kwargs

        # save hyperparameters.
        self.save_hyperparameters(ignore="model")
        self.save_hyperparameters({"model_config": model.config.to_dict()})

        # logger.
        self.verbose = verbose

        # setup logging.
        log_keys = pipeline_config.log_keys

        if isinstance(log_keys, str):
            log_keys = [log_keys]

        if log_keys is None:
            log_keys = []

        self.log_keys = log_keys

    def on_fit_start(self) -> None:
        self.model.on_fit_start(device=self.device)
        if self.global_rank == 0:
            self.timer = time.perf_counter()

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        if self.global_rank == 0:
            logging.debug("on_train_batch_end")
        self.model.on_train_batch_end(batch)

        average_time_frequency = 10
        if self.global_rank == 0 and batch_idx % average_time_frequency == 0:
            delta = time.perf_counter() - self.timer
            logging.info(
                f"Average time per batch {batch_idx} took {delta / (batch_idx + 1)} seconds"
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Setup optimizers and learning rate schedulers.
        """
        optimizers = []
        lr = self.pipeline_config.learning_rate
        param_list = []
        n_params = 0
        param_list_ = {"params": []}
        for name, param in self.model.named_parameters():
            for regex in self.pipeline_config.trainable_params:
                pattern = re.compile(regex)
                if re.match(pattern, name):
                    if param.requires_grad:
                        param_list_["params"].append(param)
                        n_params += param.numel()

        param_list.append(param_list_)

        logging.info(f"Number of trainable parameters: {n_params}")

        optimizer_cls = getattr(
            importlib.import_module("torch.optim"),
            self.pipeline_config.optimizer_name,
        )
        optimizer = optimizer_cls(
            param_list, lr=lr, **self.pipeline_config.optimizer_kwargs
        )
        optimizers.append(optimizer)

        self.optims = optimizers
        schedulers_config = self.configure_lr_schedulers()

        for name, param in self.model.named_parameters():
            set_grad_false = True
            for regex in self.pipeline_config.trainable_params:
                pattern = re.compile(regex)
                if re.match(pattern, name):
                    if param.requires_grad:
                        set_grad_false = False
            if set_grad_false:
                param.requires_grad = False

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logging.info(f"Number of trainable parameters: {num_trainable_params}")

        schedulers_config = self.configure_lr_schedulers()

        if schedulers_config is None:
            return optimizers

        return optimizers, [
            schedulers_config_ for schedulers_config_ in schedulers_config
        ]

    def configure_lr_schedulers(self):
        schedulers_config = []
        if self.pipeline_config.lr_scheduler_name is None:
            scheduler = None
            schedulers_config.append(scheduler)
        else:
            scheduler_cls = getattr(
                importlib.import_module("torch.optim.lr_scheduler"),
                self.pipeline_config.lr_scheduler_name,
            )
            scheduler = scheduler_cls(
                self.optims[0],
                **self.pipeline_config.lr_scheduler_kwargs,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": self.pipeline_config.lr_scheduler_interval,
                "monitor": "val_loss",
                "frequency": self.pipeline_config.lr_scheduler_frequency,
            }
            schedulers_config.append(lr_scheduler_config)

        if all([scheduler is None for scheduler in schedulers_config]):
            return None

        return schedulers_config

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> dict:

        model_output = self.model(
            train_batch, step=self.global_step, batch_idx=batch_idx
        )
        loss = model_output["loss"]

        temporal_w = getattr(self.pipeline_config, "temporal_loss_weight", 0.0)
        batch_temporal_loss = None
        if temporal_w > 0.0 and "predicted_hr" in model_output:
            pred = model_output["predicted_hr"]
            if isinstance(pred, torch.Tensor) and pred.dim() == 4 and pred.size(0) > 1:
                diff = pred[1:] - pred[:-1]
                batch_temporal_loss = torch.mean(torch.abs(diff))
                loss = loss + temporal_w * batch_temporal_loss

        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if "rl_loss" in model_output and model_output["rl_loss"] != 0:
            self.log(
                "train/rl_loss", model_output["rl_loss"], on_step=True, prog_bar=True
            )

        if "latent_recon_loss" in model_output:
            self.log(
                "train/latent_loss", model_output["latent_recon_loss"], on_step=True
            )
        if "color_loss" in model_output:
            self.log("train/color_loss", model_output["color_loss"], on_step=True)
        if "grad_loss" in model_output:
            self.log("train/grad_loss", model_output["grad_loss"], on_step=True)

        if batch_temporal_loss is not None:
            self.log("train/loss_batch_temporal", batch_temporal_loss, on_step=True)

        if batch_idx % 10 == 0:
            msg = f"Step {batch_idx} - Total Loss: {loss:.6f}"
            logging.info(msg)

        return {
            "loss": loss,
            "batch_idx": batch_idx,
        }

    def validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
        loss = self.model(val_batch, device=self.device)["loss"]

        metrics = self.model.compute_metrics(val_batch)

        return {"loss": loss, "metrics": metrics}

    def log_samples(self, batch: Dict[str, Any]):
        logging.debug("log_samples")
        logs = self.model.log_samples(
            batch,
            **self.log_samples_model_kwargs,
        )

        if logs is not None:
            N = min([logs[keys].shape[0] for keys in logs])
        else:
            N = 0

        # Log inputs
        if self.log_keys is not None:
            for key in self.log_keys:
                if key in batch:
                    if N > 0:
                        logs[key] = batch[key][:N]
                    else:
                        logs[key] = batch[key]

        return logs
