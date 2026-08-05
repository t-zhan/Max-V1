"""MM-free nuScenes planning evaluation aligned with UniDriveVLA.

Derived from the planning metrics, dataset helpers, and box utilities in
https://github.com/xiaomi-research/UniDriveVLA at revision a93c175af893.

Only OpenMMLab imports and logging are replaced. Metric constants, coordinate
operations, tensor operations, collision rules, and aggregation are retained.
Licensed under Apache-2.0; see ``LICENSES/Apache-2.0.txt`` and
``docs/THIRD_PARTY_NOTICES.md``.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion
from shapely.geometry import Polygon
from skimage.draw import polygon
from tqdm import tqdm

from tools.nuscenes.utils import select_samples


# Vendored from datasets/utils.py at UniDriveVLA commit a93c175af893.
def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().float().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [3, 4, 5]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, 6])
    rot_sin = np.sin(box3d[:, 6])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners


# Vendored from planning_eval.py at UniDriveVLA commit a93c175af893.
def check_collision(ego_box, boxes):
    """
    ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
    boxes: tensor with shape [N, 7]
    """
    if boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly = Polygon([(point[0], point[1]) for point in corners_box[i]])
        if ego_poly.intersects(box_poly):
            return True
    return False


def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0] + 1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]


class PlanningMetric:
    """Strict metric: 3D box Shapely collision, skip incomplete GT."""

    def __init__(self, n_future=6):
        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes, safe_incomplete=False):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        available_timesteps = len(fut_boxes)
        for t in range(n_future):
            if safe_incomplete and t >= available_timesteps:
                collision[t] = False
                continue
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes, safe_incomplete=False):
        batch_size, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert batch_size == 1, "only support bs=1"
        for i in range(batch_size):
            gt_box_coll = self.evaluate_single_coll(
                gt_trajs[i], fut_boxes, safe_incomplete
            )
            box_coll = self.evaluate_single_coll(
                trajs[i], fut_boxes, safe_incomplete
            )
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))

            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        return torch.sqrt(
            (
                ((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2)
                * gt_trajs_mask
            ).sum(dim=-1)
        )

    def update(
        self,
        trajs,
        gt_trajs,
        gt_trajs_mask,
        fut_boxes,
        safe_incomplete=False,
    ):
        assert trajs.shape == gt_trajs.shape
        trajs = trajs.clone()
        gt_trajs = gt_trajs.clone()
        trajs[..., 0] = -trajs[..., 0]
        gt_trajs[..., 0] = -gt_trajs[..., 0]
        l2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(
            trajs[:, :, :2],
            gt_trajs[:, :, :2],
            fut_boxes,
            safe_incomplete,
        )
        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += l2.sum(dim=0)
        self.total += len(trajs)

    def compute(self):
        return {
            "obj_col": self.obj_col / self.total,
            "obj_box_col": self.obj_box_col / self.total,
            "L2": self.L2 / self.total,
        }


def _gen_dx_bx(xbound, ybound):
    dx = torch.tensor([xbound[2], ybound[2]])
    bx = torch.tensor(
        [xbound[0] + xbound[2] / 2.0, ybound[0] + ybound[2] / 2.0]
    )
    nx = int((xbound[1] - xbound[0]) / xbound[2])
    ny = int((ybound[1] - ybound[0]) / ybound[2])
    return dx, bx, (nx, ny)


class PlanningMetricLoose:
    """BEV occupancy metric used by the UniDriveVLA ST-P3 report."""

    def __init__(self, n_future=6):
        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future

        self.dx, self.bx, (self.bev_h, self.bev_w) = _gen_dx_bx(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5]
        )
        self.bev_dimension = np.array([self.bev_h, self.bev_w])

        pts = np.array([
            [-self.H / 2.0 + 0.5, self.W / 2.0],
            [self.H / 2.0 + 0.5, self.W / 2.0],
            [self.H / 2.0 + 0.5, -self.W / 2.0],
            [-self.H / 2.0 + 0.5, -self.W / 2.0],
        ])
        pts = (pts - self.bx.numpy()) / self.dx.numpy()
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        self.rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, segmentation):
        n_future = traj.shape[0]
        trajs = traj.view(n_future, 1, 2).clone()
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
        trajs = trajs / self.dx.to(traj.device)
        trajs = trajs.cpu().numpy() + self.rc

        r = np.clip(
            trajs[:, :, 0].astype(np.int32), 0, self.bev_dimension[0] - 1
        )
        c = np.clip(
            trajs[:, :, 1].astype(np.int32), 0, self.bev_dimension[1] - 1
        )

        collision = np.full(n_future, False)
        for t in range(n_future):
            collision[t] = np.any(
                segmentation[t, r[t], c[t]].cpu().numpy()
            )

        return torch.from_numpy(collision)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        batch_size, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future)
        obj_box_coll_sum = torch.zeros(n_future)

        for i in range(batch_size):
            gt_box_coll = self.evaluate_single_coll(
                gt_trajs[i], segmentation[i]
            )

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).long()
            xi = ((xx - self.bx[1]) / self.dx[1]).long()
            m1 = (
                (yi >= 0)
                & (yi < self.bev_dimension[0])
                & (xi >= 0)
                & (xi < self.bev_dimension[1])
                & (~gt_box_coll)
            )
            ti = torch.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[
                i, ti[m1], yi[m1], xi[m1]
            ].long()

            m2 = ~gt_box_coll
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        return torch.sqrt(
            (
                ((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2)
                * gt_trajs_mask
            ).sum(dim=-1)
        )

    def update(self, trajs, gt_trajs, gt_trajs_mask, segmentation):
        assert trajs.shape == gt_trajs.shape
        trajs = trajs.clone()
        gt_trajs = gt_trajs.clone()

        l2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(
            trajs[:, :, :2], gt_trajs[:, :, :2], segmentation
        )

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += l2.sum(dim=0)
        self.total += len(trajs)

    def compute(self):
        return {
            "obj_col": self.obj_col / self.total,
            "obj_box_col": self.obj_box_col / self.total,
            "L2": self.L2 / self.total,
        }


def _print_log(message, logger=None):
    if logger is None:
        print(message)
    else:
        logger.info(message)


def print_uniad_format(planning_results, logger=None):
    """Print raw 1s/2s/3s values exactly as UniDriveVLA UniAD."""
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["metrics", "1s", "2s", "3s", "avg"]
    metric_dict = {}
    for key, tensor in planning_results.items():
        value = tensor.tolist()
        v1s, v2s, v3s = value[1], value[3], value[5]
        avg = (v1s + v2s + v3s) / 3.0
        metric_dict[key] = avg
        fmt = (
            (lambda v: "%.3f%%" % (v * 100))
            if "col" in key
            else (lambda v: "%.4f" % v)
        )
        table.add_row([key, fmt(v1s), fmt(v2s), fmt(v3s), fmt(avg)])
    _print_log(
        "\n--- Planning Metrics  [GPT-Driver — UniAD]  "
        "BEV-occupancy vehicle-only | raw value at 1s/2s/3s | all samples ---",
        logger,
    )
    _print_log("\n" + str(table), logger)
    return metric_dict


def print_stp3_format(planning_results, logger=None):
    """Print cumulative 1s/2s/3s values exactly as UniDriveVLA ST-P3."""
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["metrics", "1s", "2s", "3s", "avg"]
    metric_dict = {}
    for key, tensor in planning_results.items():
        value = tensor.tolist()
        v1s = float(np.mean(value[:2]))
        v2s = float(np.mean(value[:4]))
        v3s = float(np.mean(value[:6]))
        avg = (v1s + v2s + v3s) / 3.0
        metric_dict[key] = avg
        fmt = (
            (lambda v: "%.3f%%" % (v * 100))
            if "col" in key
            else (lambda v: "%.4f" % v)
        )
        table.add_row([key, fmt(v1s), fmt(v2s), fmt(v3s), fmt(avg)])
    _print_log(
        "\n--- Planning Metrics  [GPT-Driver — STP-3]  "
        "BEV-occupancy vehicle-only | cumul avg at 1s/2s/3s | all samples ---",
        logger,
    )
    _print_log("\n" + str(table), logger)
    return metric_dict


def print_strict_format(planning_results, logger=None):
    """Print all strict cumulative half-second results and their average."""
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = [
        "metrics",
        "0.5s",
        "1.0s",
        "1.5s",
        "2.0s",
        "2.5s",
        "3.0s",
        "avg",
    ]
    metric_dict = {}
    for key, tensor in planning_results.items():
        value = tensor.tolist()
        cumavg = [
            float(np.mean(value[: index + 1])) for index in range(len(value))
        ]
        avg = (cumavg[1] + cumavg[3] + cumavg[5]) / 3.0
        metric_dict[key] = avg
        fmt = (
            (lambda v: "%.3f%%" % (v * 100))
            if "col" in key
            else (lambda v: "%.4f" % v)
        )
        table.add_row([key] + [fmt(value) for value in cumavg] + [fmt(avg)])
    _print_log(
        "\n--- Planning Metrics  [SparseDrive — STRICT]  "
        "3D-box Shapely collision | skip incomplete GT ---",
        logger,
    )
    _print_log("\n" + str(table), logger)
    return metric_dict


# Vendored from nuscenes_3d_dataset.py at UniDriveVLA commit a93c175af893.
def get_T_global(info):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(
        info["lidar2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(
        info["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])
    return ego2global @ lidar2ego


# Vendored from NuScenes3DDataset.get_ann_info() at commit a93c175af893.
def _get_future_boxes(infos, index):
    info = infos[index]
    future_steps = int(info["gt_ego_fut_masks"].sum())
    future_boxes = []
    current_scene_token = info["scene_token"]
    current_T_global = get_T_global(info)
    for offset in range(1, future_steps + 1):
        future_info = infos[index + offset]
        if current_scene_token != future_info["scene_token"]:
            break

        mask = future_info["num_lidar_pts"] > 0
        future_gt_boxes = future_info["gt_boxes"][mask]
        future_T_global = get_T_global(future_info)
        T_future_to_current = (
            np.linalg.inv(current_T_global) @ future_T_global
        )

        center = (
            future_gt_boxes[:, :3] @ T_future_to_current[:3, :3].T
            + T_future_to_current[:3, 3]
        )
        yaw = np.stack(
            [
                np.cos(future_gt_boxes[:, 6]),
                np.sin(future_gt_boxes[:, 6]),
            ],
            axis=-1,
        )
        yaw = yaw @ T_future_to_current[:2, :2].T
        yaw = np.arctan2(yaw[..., 1], yaw[..., 0])

        future_gt_boxes[:, :3] = center
        future_gt_boxes[:, 6] = yaw
        future_boxes.append(
            torch.as_tensor(future_gt_boxes).unsqueeze(0)
        )
    return future_boxes


def _load_info_pkl(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return sorted(data["infos"], key=lambda info: info["timestamp"])


def load_predictions(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _prepare_predictions(raw_predictions, infos):
    info_tokens = {info["token"] for info in infos}
    missing_tokens = info_tokens - set(raw_predictions)
    if missing_tokens:
        raise ValueError(
            "Predictions are missing selected nuScenes samples "
            f"({len(info_tokens) - len(missing_tokens)}/{len(info_tokens)})"
        )

    predictions = {}
    for info in infos:
        token = info["token"]
        trajectory = raw_predictions[token]
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if trajectory.shape != (6, 2) or not np.isfinite(trajectory).all():
            raise ValueError(
                f"Invalid waypoint prediction for {token}: {trajectory.shape}"
            )
        predictions[token] = torch.from_numpy(trajectory).unsqueeze(0)
    return predictions


def _load_occupancy_map(path):
    if not path or not Path(path).exists():
        print(f"[WARNING] BEV seg pkl not found at {path}")
        return None
    with open(path, "rb") as file:
        occupancy_map = pickle.load(file)
    for token in occupancy_map:
        occupancy = occupancy_map[token]
        if not isinstance(occupancy, torch.Tensor):
            occupancy = torch.tensor(occupancy)
        occupancy_map[token] = torch.flip(occupancy, [-1])
    print(f"Loaded BEV seg maps from {path} ({len(occupancy_map)} tokens)")
    return occupancy_map


def _get_occupancy(occupancy_map, token):
    occupancy = occupancy_map.get(token)
    if occupancy is None:
        return None
    if occupancy.ndim == 4:
        occupancy = occupancy.squeeze(0)
    if occupancy.shape[0] % 2 == 1:
        occupancy = occupancy[1:]
    if occupancy.shape[0] < 6:
        return None
    return occupancy[:6].unsqueeze(0)


def planning_eval(
    predictions,
    info_pkl,
    seg_pkl,
    n_samples=None,
    seed=42,
    checkpoint=None,
    prediction_pkl=None,
    enable_thinking=None,
):
    infos = _load_info_pkl(info_pkl)
    prediction_tokens = set(predictions)
    info_tokens = {info["token"] for info in infos}
    unknown_tokens = prediction_tokens - info_tokens
    if unknown_tokens:
        raise ValueError(
            f"Predictions contain {len(unknown_tokens)} unknown nuScenes tokens"
        )

    indexed_infos = list(enumerate(infos))
    if n_samples is None:
        indexed_infos = [
            (index, info)
            for index, info in indexed_infos
            if info["token"] in prediction_tokens
        ]
        if not indexed_infos:
            raise ValueError("Prediction PKL contains no nuScenes samples")
    else:
        indexed_infos = select_samples(indexed_infos, n_samples, seed)

    selected_infos = [info for _, info in indexed_infos]
    predictions = _prepare_predictions(predictions, selected_infos)
    occupancy_map = _load_occupancy_map(seg_pkl)

    strict_metric = PlanningMetric()
    loose_metric = (
        PlanningMetricLoose() if occupancy_map is not None else None
    )

    for index, info in tqdm(indexed_infos, desc="Evaluating planning"):
        prediction = predictions[info["token"]]
        gt_trajectory = torch.as_tensor(
            info["gt_ego_fut_trajs"], dtype=torch.float32
        ).cumsum(dim=-2)[:6, :2].unsqueeze(0)
        gt_mask = torch.as_tensor(
            info["gt_ego_fut_masks"], dtype=torch.float32
        )[:6]
        gt_mask = gt_mask.unsqueeze(-1).repeat(1, 2).unsqueeze(0)

        if gt_mask.all():
            future_boxes = _get_future_boxes(infos, index)
            strict_metric.update(
                prediction.clone(),
                gt_trajectory.clone(),
                gt_mask.clone(),
                future_boxes,
            )

        if loose_metric is not None:
            segmentation = _get_occupancy(occupancy_map, info["token"])
            if segmentation is not None:
                loose_metric.update(
                    prediction.clone(),
                    gt_trajectory.clone(),
                    gt_mask.clone(),
                    segmentation,
                )

    strict_results = strict_metric.compute()
    strict_summary = print_strict_format(strict_results)
    metadata = {}
    if checkpoint is not None:
        metadata["checkpoint"] = checkpoint
    if prediction_pkl is not None:
        metadata["prediction_pkl"] = prediction_pkl
    if enable_thinking is not None:
        metadata["enable_thinking"] = enable_thinking
    metadata.update({
        "total_samples": len(infos),
        "evaluated_samples": len(indexed_infos),
        "seed": seed,
        "collision_unit": "fraction",
    })
    metrics = {
        "metadata": metadata,
        "strict": {
            "valid_samples": int(strict_metric.total.item()),
            "per_step": {
                key: value.tolist()
                for key, value in strict_results.items()
            },
            "summary": strict_summary,
        },
    }
    if loose_metric is None:
        return metrics

    loose_results = loose_metric.compute()
    uniad_summary = print_uniad_format(loose_results)
    stp3_summary = print_stp3_format(loose_results)
    metrics.update(
        {
            "uniad": {
                "valid_samples": int(loose_metric.total.item()),
                "per_step": {
                    key: value.tolist()
                    for key, value in loose_results.items()
                },
                "summary": uniad_summary,
            },
            "stp3": {
                "valid_samples": int(loose_metric.total.item()),
                "per_step": {
                    key: value.tolist()
                    for key, value in loose_results.items()
                },
                "summary": stp3_summary,
            },
        }
    )
    return metrics


def save_planning_metrics(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "planning_metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Saved planning metrics to {output_path}")
