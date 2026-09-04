import numpy as np

from models.max_v1.waypoint_text import parse_trajectory
from swift.metrics import eval_metrics_map
from swift.metrics.base import EvalMetrics
from swift.utils import Serializer


UNIAD_STEPS = [1, 3, 5]
PARSED_STATUSES = {"ok", "truncated"}


class NuScenesTrajMetrics(EvalMetrics):
    """L2 against the ground-truth trajectory text, in a full and a parsed-only view.

    ``full`` counts every sample, padding-filled ones included, matching the
    UniAD / VAD / SparseDrive convention. ``*_parsed`` keeps only the samples
    that needed no padding, matching GPT-Driver / OpenEMMA.
    """

    def compute_metrics(self, eval_prediction):
        pred_len = self.trainer.model.config.pred_len
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids

        trajectories, targets, statuses = [], [], []
        for i in range(preds.shape[0]):
            trajectory, status = parse_trajectory(
                Serializer.from_tensor(preds[i]), pred_len)
            trajectories.append(trajectory)
            targets.append(parse_trajectory(
                Serializer.from_tensor(labels[i]), pred_len)[0])
            statuses.append(status)

        l2 = np.sqrt(((np.stack(trajectories) - np.stack(targets)) ** 2).sum(-1))
        uniad_l2 = l2[:, UNIAD_STEPS]
        parsed = np.array([status in PARSED_STATUSES for status in statuses])
        return {
            "uniad_l2_avg": float(uniad_l2.mean()),
            # nan while nothing parses yet, which is the whole early-training phase.
            "uniad_l2_avg_parsed": float(uniad_l2[parsed].mean()) if parsed.any() else float("nan"),
            "parse_fail_rate": float(np.mean(np.array(statuses) == "fail")),
            "parse_partial_rate": float(np.mean(np.array(statuses) == "partial")),
        }


eval_metrics_map["max_nusc_traj"] = NuScenesTrajMetrics
