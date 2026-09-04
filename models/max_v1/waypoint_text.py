"""Trajectory text parsing shared by nuScenes data prep, inference, and evaluation."""

import re

import numpy as np


PT_BLOCK_RE = re.compile(r"\[PT\s*,(.*?)\]", re.S)
POINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def parse_trajectory(text, pred_len):
    """Parse ``[PT, (x, y), ...]`` into ``(pred_len, 2)`` points and a status.

    The status is ``ok`` when exactly ``pred_len`` points are found,
    ``truncated`` when more are found, ``partial`` when fewer are found (padded
    by repeating the last point), and ``fail`` when none are found (filled with
    a stationary trajectory). The returned array always has the full shape.
    """
    body = text.rpartition("</think>")[2]
    blocks = PT_BLOCK_RE.findall(body)
    points = POINT_RE.findall(blocks[-1] if blocks else body)

    if not points:
        return np.zeros((pred_len, 2), dtype=np.float32), "fail"

    trajectory = np.array(points, dtype=np.float32)
    if len(trajectory) > pred_len:
        return trajectory[:pred_len], "truncated"
    if len(trajectory) < pred_len:
        padding = np.repeat(trajectory[-1:], pred_len - len(trajectory), axis=0)
        return np.concatenate((trajectory, padding)), "partial"
    return trajectory, "ok"
