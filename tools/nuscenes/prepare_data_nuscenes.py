#!/usr/bin/env python3
"""
nuScenes 轨迹预测 + VQA → Max V1 CoT SFT 数据构造脚本。

数据来源:
  owl10/UniDriveVLA_Data  → nuscenes_traj_train.jsonl  (28,130 条, 轨迹预测)
  owl10/ReCogDrive_Pretraining → dataset_nuscenes_qa.jsonl (24,988 条, 多轮 VQA)
  nuScenes v1.0-trainval → 六路相机图片 / nuscenes_infos_train.pkl

匹配方式: 通过六路图片路径(samples/CAM_*/xxx.jpg)精确关联。
场景末尾不足 6 帧的样本直接过滤，不输出。

用法 (从项目根目录，数据路径已有默认值):
  python tools/nuscenes/prepare_data_nuscenes.py \
    --out data/sft/nuscenes_cot_train.json \
    --enable-thinking true \
    --ego-status true \
    --limit 100
"""

import argparse
import json
import pickle
import re
from pathlib import Path

from tqdm import tqdm

from models.max_v1.prompt_template import NUSCENES_SYSTEM
from tools.nuscenes.utils import strip_ego_status

_WAYPOINT_RE = re.compile(r"\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)")


def _build_sample(vqa: dict | None, traj: dict, root: Path,
                  enable_thinking: bool, ego_status: bool,
                  system_prompt: str, cot_notice: str) -> dict:
    """为单个匹配帧构造一条 CoT SFT 样本。vqa=None 时跳过 CoT。"""
    if not enable_thinking or vqa is None:
        cot = ""
    else:
        human_msgs = [c for c in vqa["conversations"] if c["from"] == "human"]
        gpt_msgs = [c for c in vqa["conversations"] if c["from"] == "gpt"]
        assert len(human_msgs) == len(gpt_msgs)

        pairs = []
        for i, (h, g) in enumerate(zip(human_msgs, gpt_msgs)):
            q = h["value"]
            if i == 0:
                idx = q.rfind("<image>")
                if idx != -1:
                    q = q[idx + len("<image>"):].lstrip("\n").strip()
                if not q:
                    continue
            pairs.append((q, g["value"]))

        cot_lines = [f"Consider: **{q}**\nAnswer: {a}" for q, a in pairs]
        cot_lines.append(cot_notice)
        cot = "<think>\n" + "\n".join(cot_lines) + "\n</think>"

    wp_matches = _WAYPOINT_RE.findall(traj["conversations"][1]["value"])
    assert len(wp_matches) == 6
    waypoints = [[float(x), float(y)] for x, y in wp_matches]

    if vqa is not None:
        images = [str((root / p).resolve()) for p in vqa["image"]]
    else:
        images = [str((root / p[p.find("samples/"):]).resolve())
                  for p in traj["image"]]

    user_content = traj["conversations"][0]["value"]
    if not ego_status:
        user_content = strip_ego_status(user_content)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": cot},
        ],
        "images": images,
        "waypoints": waypoints,
        "system": system_prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="nuScenes 轨迹预测 + VQA → Max V1 CoT SFT 数据构造"
    )
    parser.add_argument(
        "--traj-file",
        default="data/UniDriveVLA_Data/nuscenes_traj_train.jsonl",
    )
    parser.add_argument(
        "--vqa-file",
        default="data/ReCogDrive_Pretraining/Nuscenes-QA/dataset_nuscenes_qa.jsonl",
    )
    parser.add_argument(
        "--pkl-file",
        default="data/nuscenes/nuscenes_infos_train.pkl"
    )
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-thinking", required=True,
                        choices=["true", "false"])
    parser.add_argument("--ego-status", required=True,
                        choices=["true", "false"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    traj_file = Path(args.traj_file)
    vqa_file = Path(args.vqa_file)
    nuscenes_root = Path(args.nuscenes_root)
    enable_thinking = args.enable_thinking == "true"
    ego_status = args.ego_status == "true"

    cot_notice = "The final trajectory will be predicted by the waypoint regression head."

    # PKL: path→token + prev_token→next 一步构建
    print(f"加载 PKL: {args.pkl_file}")
    with open(args.pkl_file, "rb") as f:
        pkl_infos = pickle.load(f)["infos"]
    pkl_token_by_path = {}
    token_next = {}
    for info in pkl_infos:
        path = info["cams"]["CAM_FRONT"]["data_path"].lstrip("./")
        pkl_token_by_path[path] = info["token"]
        prev = info.get("prev_token")
        if prev:
            token_next[prev] = info["token"]
    print(f"  → {len(pkl_infos)} 条")

    # Phase 1: 加载 trajectory JSONL，过滤不足 6 帧的样本，构建索引
    traj_index = {}
    with open(traj_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="索引 trajectory"):
            if not line.strip():
                continue
            d = json.loads(line)
            token = pkl_token_by_path.get(d["image"][0])
            n_fut = 0
            cur = token
            while cur and token_next.get(cur, ""):
                n_fut += 1
                cur = token_next[cur]
            if n_fut < 6:
                continue
            key = tuple(p[p.find("samples/"):] for p in d["image"])
            traj_index[key] = d
    print(f"  → {len(traj_index)} 条")

    # Phase 2: 构造样本
    samples = []
    if enable_thinking:
        with open(vqa_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="匹配 VQA"):
                if not line.strip():
                    continue
                vqa = json.loads(line)
                bad = False
                for c in vqa["conversations"]:
                    if c["from"] == "gpt":
                        val = c["value"].strip()
                        if not val or "<image>" in val:
                            bad = True
                            break
                if bad:
                    continue
                key = tuple(p for p in vqa["image"])
                traj = traj_index.get(key)
                if traj is None:
                    continue
                samples.append(
                    _build_sample(vqa, traj, nuscenes_root,
                                  enable_thinking, ego_status,
                                  NUSCENES_SYSTEM, cot_notice))
                if args.limit and len(samples) >= args.limit:
                    break
        print(f"  → {len(samples)} 条")
    else:
        for traj in tqdm(traj_index.values(), desc="构造样本 (non-thinking)"):
            samples.append(
                _build_sample(None, traj, nuscenes_root,
                              enable_thinking, ego_status,
                              NUSCENES_SYSTEM, cot_notice))
            if args.limit and len(samples) >= args.limit:
                break
        print(f"  → {len(samples)} 条")

    # Phase 3: 输出
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"已保存 {len(samples)} 条 → {out}")


if __name__ == "__main__":
    main()
