
import logging
import time
from typing import Any, Dict

from lbm.trainer import TrainingPipeline


class CyclopsTrainingPipeline(TrainingPipeline):
    metric_names = (
        "lbm_loss",
        "lpips_loss",
        "gradient_loss",
        "color_loss",
        "terminal_reward_penalty",
        "reward_fidelity",
        "reward_temporal",
        "teacher_forcing_probability",
    )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if getattr(self.model, "training_phase", None) == "phase2":
            self.model.teacher_forcing_total_steps = max(
                1, int(self.trainer.estimated_stepping_batches)
            )
        if self.global_rank == 0:
            self.timer = time.perf_counter()

    def _batch_size(self, batch) -> int:
        return int(batch[self.model.target_key].shape[0])

    def _log_output(self, prefix: str, output: Dict[str, Any], batch, prog_bar=False):
        self.log(
            f"{prefix}/total_loss",
            output["loss"],
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=True,
            batch_size=self._batch_size(batch),
        )
        for name in self.metric_names:
            if name in output:
                self.log(
                    f"{prefix}/{name}",
                    output[name],
                    on_step=prefix == "train",
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self._batch_size(batch),
                )

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int):
        output = self.model(train_batch, step=self.global_step, batch_idx=batch_idx)
        self._log_output("train", output, train_batch, prog_bar=True)
        if batch_idx % 10 == 0:
            logging.info("Step %d - Total Loss: %.6f", batch_idx, output["loss"].item())
        return {"loss": output["loss"], "batch_idx": batch_idx}

    def validation_step(self, val_batch: Dict[str, Any], batch_idx: int):
        output = self.model(val_batch, step=self.global_step, batch_idx=batch_idx)
        self._log_output("val", output, val_batch, prog_bar=True)
        return {"loss": output["loss"]}
