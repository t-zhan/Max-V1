"""Max nuScenes inference used by :mod:`tools.nuscenes.test`."""

import json
import os
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.max_v1.prompt_template import NUSCENES_SYSTEM
from tools.nuscenes.utils import select_samples
from tools.nuscenes.utils import strip_ego_status


def _sample_path_key(path):
    path = str(path).replace("\\", "/")
    marker = "samples/"
    marker_index = path.find(marker)
    return path[marker_index:] if marker_index >= 0 else path.lstrip("./")


def _load_jsonl_by_front_image(traj_file):
    rows = {}
    with open(traj_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[_sample_path_key(row["image"][0])] = row
    return rows


def _load_records(info_pkl, traj_file, ego_status=True):
    with open(info_pkl, "rb") as file:
        infos = sorted(pickle.load(file)["infos"], key=lambda info: info["timestamp"])
    rows = _load_jsonl_by_front_image(traj_file)

    records = []
    for index, info in enumerate(infos):
        front_key = _sample_path_key(info["cams"]["CAM_FRONT"]["data_path"])
        row = rows[front_key]
        user_content = next(
            message["value"]
            for message in row["conversations"]
            if message["from"] == "human"
        )
        if not ego_status:
            user_content = strip_ego_status(user_content)
        gt_trajectory = torch.as_tensor(
            info["gt_ego_fut_trajs"], dtype=torch.float32
        ).cumsum(dim=-2)[:6, :2]
        gt_mask = torch.as_tensor(
            info["gt_ego_fut_masks"], dtype=torch.float32
        )[:6]
        gt_mask = gt_mask.unsqueeze(-1).repeat(1, 2)
        records.append({
            "index": index,
            "token": info["token"],
            "messages": [{"role": "user", "content": user_content}],
            "image_paths": row["image"],
            "gt_trajectory": gt_trajectory,
            "gt_mask": gt_mask,
        })
    return records


class _NuScenesInferenceDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = []
        for path in record["image_paths"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return (
            record["index"],
            record["token"],
            record["messages"],
            images,
            record["gt_trajectory"],
            record["gt_mask"],
        )


def _collate(batch):
    return batch


def _init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(hours=2),
        device_id=torch.device("cuda", local_rank),
    )
    return dist.get_rank(), dist.get_world_size()


def _build_metric_loggers(seg_pkl, sample_logger):
    from tools.nuscenes.utils.planning_eval import (
        PlanningMetricLoose,
        _get_occupancy,
        _load_occupancy_map,
    )

    occupancy_map = _load_occupancy_map(seg_pkl)
    metric = PlanningMetricLoose() if occupancy_map is not None else None
    uniad_steps = [1, 3, 5]
    accumulating_l2_sum = 0.0
    accumulating_l2_total = 0
    batch_obj_box_col_start = (
        metric.obj_box_col.clone() if metric is not None else None
    )
    batch_total_start = metric.total.clone() if metric is not None else None

    def log_sample(index, token, prediction, gt_trajectory, gt_mask):
        nonlocal accumulating_l2_sum, accumulating_l2_total
        prediction = torch.from_numpy(prediction).unsqueeze(0)
        gt_trajectory = gt_trajectory.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
        sample_l2 = torch.sqrt(
            (
                ((prediction[:, :, :2] - gt_trajectory[:, :, :2]) ** 2)
                * gt_mask
            ).sum(dim=-1)
        )
        payload = {}

        if metric is not None:
            segmentation = _get_occupancy(occupancy_map, token)
            if segmentation is not None:
                metric.update(
                    prediction,
                    gt_trajectory,
                    gt_mask,
                    segmentation,
                )
                payload["accumulating/uniad_obj_box_col"] = (
                    metric.obj_box_col[uniad_steps] / metric.total
                ).mean().item()

        if gt_mask[0, uniad_steps, 0].bool().all().item():
            sample_l2_avg = sample_l2[0, uniad_steps].mean().item()
            accumulating_l2_sum += sample_l2_avg
            accumulating_l2_total += 1
            payload.update(
                {
                    "sample_or_batch/uniad_l2_avg": sample_l2_avg,
                    "accumulating/uniad_l2_avg": (
                        accumulating_l2_sum / accumulating_l2_total
                    ),
                }
            )
        if payload:
            sample_logger(payload, step=index)

    def log_batch(batch_index):
        nonlocal batch_obj_box_col_start, batch_total_start
        assert metric is not None
        batch_total = metric.total - batch_total_start
        if batch_total.item():
            sample_logger(
                {
                    "sample_or_batch/uniad_obj_box_col": (
                        (
                            metric.obj_box_col[uniad_steps]
                            - batch_obj_box_col_start[uniad_steps]
                        )
                        / batch_total
                    ).mean().item()
                },
                step=batch_index,
            )
        batch_obj_box_col_start = metric.obj_box_col.clone()
        batch_total_start = metric.total.clone()

    return log_sample, log_batch if metric is not None else None


def _infer(
    model,
    loader,
    system_prompt,
    enable_thinking,
    rank,
    sample_metric_logger=None,
    batch_metric_logger=None,
):
    results = []
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"rank{rank}", position=rank, disable=rank != 0)
    ):
        samples = [
            {
                # StdTemplateInputs.from_dict only reads the system prompt from
                # messages[0]; a top-level "system" key is silently dropped.
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                "images": images,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            for _, _, messages, images, _, _ in batch
        ]
        pred_waypoints, generated_texts = model.generate_waypoints(samples)
        predictions = pred_waypoints.detach().cpu().float().numpy()
        for (
            index,
            token,
            _,
            _,
            gt_trajectory,
            gt_mask,
        ), prediction, generated_text in zip(
            batch,
            predictions,
            generated_texts,
            strict=True,
        ):
            if prediction.shape != (6, 2) or not np.isfinite(prediction).all():
                raise ValueError(
                    f"Invalid waypoint prediction for {token}: {prediction.shape}"
                )
            if sample_metric_logger is not None:
                sample_metric_logger(
                    index,
                    token,
                    prediction,
                    gt_trajectory,
                    gt_mask,
                )
            results.append((
                index,
                token,
                prediction.astype(np.float32, copy=False),
                generated_text,
            ))
        if batch_metric_logger is not None:
            batch_metric_logger(batch_index)
    return results


def _gather_results(results, world_size):
    gathered = [None] * world_size
    dist.all_gather_object(gathered, results)
    return sorted(
        (result for rank_results in gathered for result in rank_results),
        key=lambda result: result[0],
    )


def _build_predictions(results, records):
    expected_tokens = {record["token"] for record in records}
    prediction_tokens = {token for _, token, _, _ in results}
    if prediction_tokens != expected_tokens or len(results) != len(records):
        raise ValueError(
            "Prediction tokens do not match the selected nuScenes dataset "
            f"({len(prediction_tokens)}/{len(expected_tokens)})"
        )

    predictions = {
        token: {
            "trajectory": trajectory,
            "generated_text": generated_text,
        }
        for _, token, trajectory, generated_text in results
    }
    return predictions


def save_predictions(predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "max_results.pkl"
    with result_path.open("wb") as file:
        pickle.dump(predictions, file, protocol=2)

    print(f"Saved {len(predictions)} inference results to {result_path}")


def run_inference(
    model_path,
    traj_file,
    info_pkl,
    batch_size=1,
    num_workers=8,
    enable_thinking=False,
    ego_status=True,
    max_new_tokens=None,
    n_samples=None,
    seed=42,
    seg_pkl=None,
    sample_logger=None,
):
    from models.max_v1.max_carla import Max

    rank, world_size = _init_distributed()
    try:
        records = select_samples(
            _load_records(info_pkl, traj_file, ego_status),
            n_samples,
            seed,
        )
        rank_records = records[rank::world_size]
        sample_metric_logger = None
        batch_metric_logger = None
        if rank == 0 and sample_logger is not None:
            sample_metric_logger, batch_metric_logger = _build_metric_loggers(
                seg_pkl,
                sample_logger,
            )
        loader = DataLoader(
            _NuScenesInferenceDataset(rank_records),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
            shuffle=False,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

        model = Max.from_pretrained(model_path).eval().cuda()
        if max_new_tokens is not None:
            model.config.max_new_tokens = max_new_tokens
        if model.config.pred_len != 6:
            raise ValueError(
                "nuScenes evaluation requires pred_len=6, "
                f"got {model.config.pred_len}"
            )

        results = _infer(
            model,
            loader,
            NUSCENES_SYSTEM,
            enable_thinking,
            rank,
            sample_metric_logger,
            batch_metric_logger,
        )
        results = _gather_results(results, world_size)
        if rank == 0:
            return _build_predictions(results, records)
        return None
    finally:
        dist.destroy_process_group()
