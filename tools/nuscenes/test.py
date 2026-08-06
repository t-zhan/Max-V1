#!/usr/bin/env python3
"""Run Max nuScenes inference and UniDriveVLA-aligned planning evaluation."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Max nuScenes inference and planning evaluation"
    )
    parser.add_argument("--model-path")
    parser.add_argument(
        "--traj-file",
        default="data/UniDriveVLA_Data/nuscenes_traj_val.jsonl",
    )
    parser.add_argument(
        "--info-pkl",
        default="data/UniDriveVLA_Data/nuscenes_infos_val.pkl",
    )
    parser.add_argument(
        "--pred-pkl",
        help="Evaluate this existing prediction PKL and skip inference",
    )
    parser.add_argument(
        "--result-dir",
        default="results/",
        help="Base directory for timestamped prediction and metric results",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run STRICT, UniAD, and STP-3 planning evaluation",
    )
    parser.add_argument("--seg-pkl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-samples", "--n_samples", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used to select --n-samples",
    )
    parser.add_argument(
        "--enable-thinking",
        type=json.loads,
        choices=(True, False),
        default=None,
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--ego-status", choices=("true", "false"), default="true")
    args = parser.parse_args(argv)

    if args.pred_pkl:
        if not args.eval:
            parser.error("--pred-pkl requires --eval")
    else:
        if not args.model_path:
            parser.error("--model-path is required when --pred-pkl is omitted")
        if args.enable_thinking is None:
            args.enable_thinking = False
    return args


def main(argv=None):
    args = _parse_args(argv)

    if args.pred_pkl:
        from tools.nuscenes.utils.planning_eval import load_predictions

        predictions = load_predictions(args.pred_pkl)
    else:
        from tools.nuscenes.utils.inference import (
            run_inference,
            save_predictions,
        )

        outputs = run_inference(
            model_path=args.model_path,
            traj_file=args.traj_file,
            info_pkl=args.info_pkl,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            enable_thinking=args.enable_thinking,
            ego_status=args.ego_status == "true",
            max_new_tokens=args.max_new_tokens,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        if outputs is None:
            return
        predictions, generated_texts = outputs

    result_dir = Path(args.result_dir) / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    if not args.pred_pkl:
        save_predictions(predictions, generated_texts, result_dir)

    if args.eval:
        from tools.nuscenes.utils.planning_eval import (
            planning_eval,
            save_planning_metrics,
        )

        metrics = planning_eval(
            predictions,
            args.info_pkl,
            args.seg_pkl,
            n_samples=args.n_samples,
            seed=args.seed,
            checkpoint=None if args.pred_pkl else args.model_path,
            prediction_pkl=args.pred_pkl,
            enable_thinking=args.enable_thinking,
        )
        save_planning_metrics(metrics, result_dir)


if __name__ == "__main__":
    main()
