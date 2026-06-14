#!/usr/bin/env python3
"""
Bench2Drive-VL → Max SFT data pipeline.
All GoT and CoT functions are imported directly from the B2DVL submodule.

Data sources:
  Telkwevr/Bench2Drive-VL-base → data/Bench2Drive-VL-base/
  rethinklab/Bench2Drive       → data/Bench2Drive/

Usage (from project root):
  bash tools/get_data.sh
  python tools/prepare_data.py --data-dir data
"""
import argparse, json, multiprocessing
import re
from functools import partial
from pathlib import Path

from infer_configure import InferConfig
from qa_process import find_qdict_by_id
from dataset_converter import (
    find_context_for_question, get_anno_path,
    generate_sharegpt_CoT_unit,
)
from image_process import generate_concat_camera_images

FINAL_TRAJECTORY_NOTICE = (
    "The final trajectory will be predicted by the waypoint regression head."
)

# Pattern to match sequences of B2DVL tokenized waypoints.
_TOKEN_SEQ_PATTERN = (
    r'(?:<[xy](?:_(?:stay|fwd|back|right|left))?_[0-9a-fA-F]{2}>'
    r'|<(?:dir_(?:fwd|back|right|left)|spd)_[0-9a-fA-F]{2,3}>)+'
)
_FINAL_WAYPOINT_ANSWER_RE = re.compile(r'Final Answer:\s*' + _TOKEN_SEQ_PATTERN)
_ANSWER_WAYPOINT_TAG_RE = re.compile(
    r'\s*<answer>\s*' + _TOKEN_SEQ_PATTERN + r'\s*</answer>'
)

_got = InferConfig()
GOT_ORDER = _got.CHAIN["ORDER"]
GOT_PREV = _got.CHAIN["PREV"]
GOT_INHERIT = _got.CHAIN["INHERIT"]

# ── Constants ──────────────────────────────────────────────────────────────
VQA_DIRNAME = "Bench2Drive-VL-mini"
RAW_DIRNAME = "Bench2Drive-mini"

WP_KEYS = ['0.5s', '1.0s', '1.5s', '2.0s', '2.5s', '3.0s', '3.5s', '4.0s']

_CAMERA_MAP = {
    "rgb_front": "CAM_FRONT",
    "rgb_front_left": "CAM_FRONT_LEFT",
    "rgb_front_right": "CAM_FRONT_RIGHT",
    "rgb_back": "CAM_BACK",
    "rgb_back_left": "CAM_BACK_LEFT",
    "rgb_back_right": "CAM_BACK_RIGHT",
}


def _remove_waypoint_outputs(text):
    text = _FINAL_WAYPOINT_ANSWER_RE.sub(FINAL_TRAJECTORY_NOTICE, text)
    return _ANSWER_WAYPOINT_TAG_RE.sub('', text)


# ── Processing ─────────────────────────────────────────────────────────────

def extract_waypoints(qdict):
    wp_dict = json.loads(qdict["A"])
    return [wp_dict[k] for k in WP_KEYS]


def build_camera_dict(scenario_dir, frame_stem):
    return {
        cam_name: str(scenario_dir / "camera" / dir_name / f"{frame_stem}.jpg")
        for dir_name, cam_name in _CAMERA_MAP.items()
    }


def concat_images(cam_dict, save_dir, frame_stem):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    front_path = str(save_dir / f"front_{frame_stem}.jpg")
    back_path = str(save_dir / f"back_{frame_stem}.jpg")
    if Path(front_path).exists() and Path(back_path).exists():
        return front_path, back_path
    front_img, back_img = generate_concat_camera_images(cam_dict)
    if front_img is not None:
        front_img.save(front_path)
    if back_img is not None:
        back_img.save(back_path)
    return front_path, back_path


def build_cot_sample(vqa_json, prev_vqa_json, qdict_42, front_path, back_path, curr_anno):
    inherit_list, context_list = find_context_for_question(
        qid=42, prev_vqa=prev_vqa_json, vqa=vqa_json,
        prev=GOT_PREV, order=GOT_ORDER, inherit=GOT_INHERIT,
    )
    raw_dict = {
        "inherit": inherit_list,
        "context": context_list,
        "question": qdict_42,
        "images": {"CAM_FRONT_CONCAT": front_path, "CAM_BACK_CONCAT": back_path},
    }
    _, answer_str, final_dict = generate_sharegpt_CoT_unit(
        raw_dict, no_tags=False, curr_anno=curr_anno)
    answer_str = _remove_waypoint_outputs(answer_str)
    final_dict["messages"][1]["content"] = answer_str
    return final_dict


def _process_scenario(data_dir, scenario):
    raw_dir = data_dir / RAW_DIRNAME
    concat_dir = data_dir / "concat_images"
    scenario_dir = raw_dir / scenario
    save_dir = concat_dir / scenario
    vqa_sorted = sorted((data_dir / VQA_DIRNAME / scenario).glob("*.json"),
                        key=lambda p: int(p.stem))
    samples = []
    prev_vqa_json = None
    for vf in vqa_sorted:
        frame_stem = vf.stem
        with open(vf) as f:
            vqa_json = json.load(f)
        qdict_42 = find_qdict_by_id(42, vqa_json)[0]
        waypoints = extract_waypoints(qdict_42)
        cam_dict = build_camera_dict(scenario_dir, frame_stem)
        front_path, back_path = concat_images(cam_dict, save_dir, frame_stem)
        curr_anno = get_anno_path(str(raw_dir), scenario, int(frame_stem))
        sample = build_cot_sample(vqa_json, prev_vqa_json, qdict_42, front_path, back_path, curr_anno)
        sample["waypoints"] = waypoints
        samples.append(sample)
        prev_vqa_json = vqa_json
    return samples


def create_sft_dataset(data_dir, workers):
    data_dir = Path(data_dir)
    vqa_dir = data_dir / VQA_DIRNAME

    scenarios = sorted(p.name for p in vqa_dir.iterdir() if p.is_dir())

    print(f"Processing {len(scenarios)} scenarios with {workers} workers")
    process_scenario = partial(_process_scenario, data_dir)
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_scenario, scenarios)
    return [s for r in results for s in r]


def main():
    parser = argparse.ArgumentParser(description="Bench2Drive-VL → Max CoT SFT")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    samples = create_sft_dataset(data_dir, args.workers)
    out = data_dir / "sft" / "max_sft_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples → {out}")


if __name__ == "__main__":
    main()
