import os
from functools import wraps

from swift.callbacks import TrainerCallback, callbacks_map
from swift.utils.logger import get_logger


def _get_output_value(outputs, key):
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


class MaxLossLogCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        origin_compute_loss = trainer.compute_loss

        @wraps(origin_compute_loss)
        def compute_loss_with_metrics(model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = origin_compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )

            mode = "train" if trainer.model.training else "eval"
            metrics = trainer.custom_metrics[mode]
            for name in ("lm_loss", "reg_loss", "rollout_gt_mse", "rollout_pred_mse"):
                value = _get_output_value(outputs, name)
                if value is not None:
                    metrics[name].update(value)

            return (loss, outputs) if return_outputs else loss

        trainer.compute_loss = compute_loss_with_metrics


class MaxRolloutScheduleCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = self.trainer.model
        self.start_ratio = float(os.environ.get("WARMUP_SCHEDULED_SAMPLING_RATIO", 0.0))
        self.target_ratio = model.config.scheduled_sampling_ratio
        self.start_step = state.global_step
        self.warmup_steps = int(os.environ["WARMUP_SCHEDULED_SAMPLING_STEPS"])
        self.logger = get_logger()

    def on_step_begin(self, args, state, control, **kwargs):
        model = self.trainer.model
        steps_since_start = state.global_step - self.start_step
        progress = min(1.0, steps_since_start / self.warmup_steps)
        model.config.scheduled_sampling_ratio = (
            self.start_ratio
            + (self.target_ratio - self.start_ratio) * progress
        )
        self.logger.info(
            "Rollout warmup progress: %.1f%% | scheduled sampling ratio: %.4f",
            progress * 100,
            model.config.scheduled_sampling_ratio,
        )


callbacks_map["max_loss_log"] = MaxLossLogCallback
callbacks_map["max_rollout_schedule"] = MaxRolloutScheduleCallback
