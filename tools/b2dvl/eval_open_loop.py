#!/usr/bin/env python3
"""Open-loop L2 evaluation for Max-V1 using Bench2Drive raw data.

Usage:
  torchrun --nproc_per_node=4 tools/b2dvl/eval_open_loop.py \
    --model-path outputs/v32-.../checkpoint-1560 --split train --ratio 0.05 --batch-size 4
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<ip> \
    tools/b2dvl/eval_open_loop.py --model-path ... --split val --ratio 0.1
"""
import argparse
import gzip
import json
import os
import random
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.max_v1.max_carla import Max

WP_KEYS = ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "3.5s", "4.0s"]
CAM_DIRS = [
    "rgb_front_left", "rgb_front", "rgb_front_right",
    "rgb_back_right", "rgb_back", "rgb_back_left",
]
JUDGE_QUESTIONS = [
    ("traffic_signals",
     "Identify all traffic lights and signs affecting the ego vehicle and determine the required actions accordingly."),
    ("weather",
     "What is current time and weather? What should the ego vehicle do according to them?"),
    ("other_hazards",
     "Apart from vehicles on the road, visible pedestrians and the weather, what other factors in the current scenario could pose potential hazards? What strategies should the ego vehicle adopt to address them?"),
    ("speed_limit",
     "What is the current speed limit?"),
    ("brake",
     "Determine whether the ego vehicle needs to brake."),
    ("lane_change",
     "Determine whether the ego vehicle needs to change lane or deviate from the current lane."),
]
_COT_QA_RE = re.compile(
    r"Consider: \*\*(.*?)\*\*\s*Answer:\s*(.*?)(?=\nConsider:|\nConsider the final question:|</think>|$)",
    re.S,
)


def _collect_frames(vqa_dir):
    frames = []
    for sd in sorted(Path(vqa_dir).iterdir()):
        frames.extend((sd.name, vf.stem) for vf in sorted(sd.glob("*.json")))
    return frames


def _default_sft_path(split):
    return "data/sft/b2dvl_base.json" if split == "train" else "data/sft/b2dvl_full.json"


def _frame_key_from_sft_image(image_path):
    path = Path(image_path)
    return path.parent.name, path.stem.split("_", 1)[1]


def _load_cot_map(sft_path):
    samples = json.loads(Path(sft_path).read_text())
    cot_map = {}
    for sample in samples:
        messages = sample.get("messages") or []
        images = sample.get("images") or []
        cot_map[_frame_key_from_sft_image(images[0])] = messages[1]["content"]
    return cot_map


def _normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_cot_answers(cot):
    return {
        _normalize_text(question): _normalize_text(answer)
        for question, answer in _COT_QA_RE.findall(cot or "")
    }


def _cot_judge_items(gt_cot, pred_cot):
    gt_answers = _parse_cot_answers(gt_cot)
    pred_answers = _parse_cot_answers(pred_cot)
    return [
        {
            "name": name,
            "question": question,
            "reference_answer": gt_answers[question],
            "generated_answer": pred_answers.get(question, ""),
        }
        for name, question in JUDGE_QUESTIONS
    ]


def _deepseek_chat_json(args, messages):
    payload = {
        "model": args.deepseek_model,
        "messages": messages,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": args.deepseek_thinking},
        "reasoning_effort": args.deepseek_reasoning_effort,
    }
    request = urllib.request.Request(
        f"{args.deepseek_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.deepseek_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    return json.loads(data["choices"][0]["message"]["content"])


def _judge_cot_with_deepseek(args, frame_id, gt_cot, pred_cot):
    items = _cot_judge_items(gt_cot, pred_cot)
    user_prompt = (
        "Compare the generated autonomous-driving CoT against the reference CoT "
        "for one frame. Judge semantic driving correctness, not wording.\n\n"
        "Do not penalize different object ids or bounding-box coordinates if the "
        "object type, approximate location, and driving action are semantically consistent. "
        "Penalize wrong traffic light/sign state, missed hazards, wrong speed limit, "
        "wrong brake decision, or wrong lane-change/deviation decision. Ignore the final "
        "waypoint placeholder.\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "overall_score": 1-5,\n'
        '  "question_scores": {\n'
        '    "traffic_signals": 1-5,\n'
        '    "weather": 1-5,\n'
        '    "other_hazards": 1-5,\n'
        '    "speed_limit": 1-5,\n'
        '    "brake": 1-5,\n'
        '    "lane_change": 1-5\n'
        "  },\n"
        '  "action_consistency": {"brake": "same|different|unclear", "lane_change": "same|different|unclear"},\n'
        '  "critical_errors": ["..."],\n'
        '  "minor_differences": ["..."],\n'
        '  "summary": "..."\n'
        "}\n\n"
        "Scoring: 5 = semantically equivalent or better; 4 = mostly correct with minor omissions; "
        "3 = partially correct; 2 = major driving-relevant errors; 1 = mostly wrong.\n\n"
        f"Frame id: {frame_id}\n"
        f"Question-answer pairs:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )
    return _deepseek_chat_json(args, [
        {
            "role": "system",
            "content": "You are a strict but fair judge for autonomous-driving chain-of-thought explanations. Output JSON only.",
        },
        {"role": "user", "content": user_prompt},
    ])


class _EvalDataset(Dataset):
    def __init__(self, frames, vqa_dir, raw_dir, cot_map=None, require_cot=False):
        self._frames, self._vqa_dir, self._raw_dir = frames, vqa_dir, raw_dir
        self._cot_map = cot_map or {}
        self._require_cot = require_cot

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        scenario, stem = self._frames[idx]

        rgb = []
        for d in CAM_DIRS:
            bgr = cv2.imread(f"{self._raw_dir}/{scenario}/camera/{d}/{stem}.jpg")
            rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        vqa = json.loads((Path(self._vqa_dir) / scenario / f"{stem}.json").read_text())
        gt = None
        for q in vqa["QA"]["behaviour"]:
            if q["qid"] == 42:
                gt = np.array([json.loads(q["A"])[k] for k in WP_KEYS], dtype=np.float32)
                break

        with gzip.open(f"{self._raw_dir}/{scenario}/anno/{stem}.json.gz", "rt") as f:
            anno = json.load(f)

        cot = self._cot_map.get((scenario, stem))
        if self._require_cot and cot is None:
            raise KeyError(f"Missing CoT for {scenario}/{stem}")
        return rgb, gt, anno["speed"], anno.get("command_near", 0) or 0, scenario, stem, cot

    @staticmethod
    def collate(batch):
        rgbs, speeds, cmds, scenarios, stems, gts, cots = [], [], [], [], [], [], []
        for rgb, gt, speed, cmd, sc, st, cot in batch:
            rgbs.append(rgb)
            speeds.append(speed)
            cmds.append(cmd)
            scenarios.append(sc)
            stems.append(st)
            gts.append(gt)
            cots.append(cot)
        return rgbs, speeds, cmds, scenarios, stems, gts, cots


def _parse_args():
    parser = argparse.ArgumentParser(description="Max-V1 open-loop L2 evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", default="true", choices=["true", "false"])
    parser.add_argument("--sft-path", default="")
    parser.add_argument("--judge-cot", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--deepseek-api-key", default="")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--deepseek-thinking", default="enabled", choices=["enabled", "disabled"])
    parser.add_argument("--deepseek-reasoning-effort", default="max", choices=["high", "max"])
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _init_distributed():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world


def _data_dirs(split):
    vqa_dir = "data/Bench2Drive-VL-base" if split == "train" else "data/Bench2Drive-VL-Full"
    raw_dir = "data/Bench2Drive" if split == "train" else "data/Bench2Drive-Full"
    return vqa_dir, raw_dir


def _sample_frames(args, vqa_dir, rank, world):
    random.seed(args.seed)
    frames = _collect_frames(vqa_dir)
    if args.split == "val":
        base = {p.name for p in Path("data/Bench2Drive-VL-base").iterdir() if p.is_dir()}
        frames = [(s, f) for s, f in frames if s not in base]
    n = args.n_samples if args.n_samples > 0 else max(1, int(len(frames) * args.ratio))
    sampled = random.sample(frames, min(n, len(frames)))
    my_frames = [sampled[i] for i in range(rank, len(sampled), world)]
    return frames, sampled, my_frames


def _load_cot_map_for_mode(args, rank):
    cot_map = None
    if args.judge_cot:
        sft_path = args.sft_path or _default_sft_path(args.split)
        cot_map = _load_cot_map(sft_path)
        if rank == 0:
            print(f"Loaded GT CoT data: {len(cot_map)} frames from {sft_path}")
    return cot_map


def _build_loader(args, frames, vqa_dir, raw_dir, cot_map):
    ds = _EvalDataset(
        frames,
        vqa_dir,
        raw_dir,
        cot_map,
        require_cot=args.judge_cot,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=_EvalDataset.collate,
        num_workers=args.num_workers,
    )


def _load_model(args):
    return Max.from_pretrained(args.model_path).eval().cuda()


def _infer_batch(model, args, rgbs, speeds, cmds):
    enable_thinking = args.enable_thinking == "true"
    preds, cots, statuses = model.carla_generate(
        rgbs,
        speeds,
        cmds,
        enable_thinking=enable_thinking,
    )
    return preds, cots, statuses


def _format_batch_results(
    raw_dir,
    args,
    preds,
    cots,
    statuses,
    scenarios,
    stems,
    gts,
    gt_cots,
):
    results = []
    for i in range(len(scenarios)):
        pred_np = preds[i].reshape(8, 2)
        diffs = np.linalg.norm(pred_np - gts[i], axis=-1)
        result = {
            "frame_id": f"{scenarios[i]}/{stems[i]}",
            "images": [f"{raw_dir}/{scenarios[i]}/camera/{d}/{stems[i]}.jpg" for d in CAM_DIRS],
            "cot": cots[i],
            "gt_waypoints": gts[i].tolist(),
            "pred_waypoints": pred_np.tolist(),
            "L2_error": diffs.tolist(),
            "parse_status": statuses[i],
        }
        if args.judge_cot:
            result["gt_cot"] = gt_cots[i]
            try:
                result["cot_judge"] = _judge_cot_with_deepseek(
                    args, result["frame_id"], gt_cots[i], cots[i])
            except Exception:
                continue
        results.append(result)
    return results


def _run_eval_loop(model, loader, args, raw_dir, rank):
    results = []
    for rgbs, speeds, cmds, scenarios, stems, gts, gt_cots in tqdm(
            loader, desc=f"rank{rank}", position=rank, disable=(rank != 0)):
        preds, cots, statuses = _infer_batch(model, args, rgbs, speeds, cmds)
        results.extend(_format_batch_results(
            raw_dir, args, preds, cots, statuses, scenarios, stems, gts, gt_cots))
    return results


def _gather_results(results, world):
    all_results = [None] * world
    dist.all_gather_object(all_results, results)
    return [r for sub in all_results for r in sub]


def _summarize_results(flat, args):
    L2s = [r["L2_error"] for r in flat]
    ade = float(np.mean(L2s))
    fde = float(np.mean([errors[-1] for errors in L2s]))
    steps = np.mean(L2s, axis=0)
    print(f"ADE: {ade:.4f}  FDE: {fde:.4f}")
    print(f"Per-step: {[f'{s:.4f}' for s in steps]}")

    cot_judge_score = None
    if args.judge_cot:
        cot_judge_score = float(np.mean([r["cot_judge"]["overall_score"] for r in flat]))
        print(f"CoT judge score: {cot_judge_score:.4f}")

    return {
        "ade": ade,
        "fde": fde,
        "per_step": steps.tolist(),
        "n_frames": len(flat),
        "cot_judge_score": cot_judge_score,
        "details": flat,
    }


def _save_summary(summary, output):
    if output:
        json.dump(summary, open(output, "w"), indent=2, ensure_ascii=False)


def main():
    args = _parse_args()
    rank, world = _init_distributed()
    vqa_dir, raw_dir = _data_dirs(args.split)
    frames, sampled, my_frames = _sample_frames(args, vqa_dir, rank, world)

    if rank == 0:
        print(f"Split: {args.split}  Frames: {len(sampled)}/{len(frames)}  "
              f"Batch: {args.batch_size}  GPUs: {world}")

    cot_map = _load_cot_map_for_mode(args, rank)
    loader = _build_loader(args, my_frames, vqa_dir, raw_dir, cot_map)
    model = _load_model(args)
    results = _run_eval_loop(model, loader, args, raw_dir, rank)
    flat = _gather_results(results, world)

    if rank == 0:
        _save_summary(_summarize_results(flat, args), args.output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
