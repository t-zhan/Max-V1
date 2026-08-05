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


def _load_records(info_pkl, traj_file):
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
        records.append({
            "index": index,
            "token": info["token"],
            "messages": [{"role": "user", "content": user_content}],
            "image_paths": row["image"],
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
        return record["index"], record["token"], record["messages"], images


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


def _infer(model, loader, system_prompt, enable_thinking, rank):
    results = []
    for batch in tqdm(loader, desc=f"rank{rank}", position=rank, disable=rank != 0):
        samples = [
            {
                "messages": messages,
                "system": system_prompt,
                "images": images,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            for _, _, messages, images in batch
        ]
        pred_waypoints, generated_texts = model.generate_waypoints(samples)
        predictions = pred_waypoints.detach().cpu().float().numpy()
        for (index, token, _, _), prediction, generated_text in zip(
            batch,
            predictions,
            generated_texts,
            strict=True,
        ):
            if prediction.shape != (6, 2) or not np.isfinite(prediction).all():
                raise ValueError(
                    f"Invalid waypoint prediction for {token}: {prediction.shape}"
                )
            results.append((
                index,
                token,
                prediction.astype(np.float32, copy=False),
                generated_text,
            ))
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
        token: trajectory
        for _, token, trajectory, _ in results
    }
    generated_texts = {
        token: generated_text
        for _, token, _, generated_text in results
    }
    return predictions, generated_texts


def save_predictions(predictions, generated_texts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "max_pred_trajs.pkl"
    with trajectory_path.open("wb") as file:
        pickle.dump(predictions, file, protocol=2)

    text_path = output_dir / "max_generated_texts.pkl"
    with text_path.open("wb") as file:
        pickle.dump(generated_texts, file, protocol=2)

    print(f"Saved {len(predictions)} trajectories to {trajectory_path}")
    print(f"Saved {len(generated_texts)} generated texts to {text_path}")


def run_inference(
    model_path,
    traj_file,
    info_pkl,
    batch_size=1,
    num_workers=8,
    enable_thinking=False,
    max_new_tokens=None,
    n_samples=None,
    seed=42,
):
    from models.max_v1.max_carla import Max

    rank, world_size = _init_distributed()
    try:
        records = select_samples(
            _load_records(info_pkl, traj_file),
            n_samples,
            seed,
        )
        rank_records = records[rank::world_size]
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
        )
        results = _gather_results(results, world_size)
        if rank == 0:
            return _build_predictions(results, records)
        return None
    finally:
        dist.destroy_process_group()
