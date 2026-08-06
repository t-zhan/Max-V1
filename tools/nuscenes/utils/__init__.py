"""Internal nuScenes inference, evaluation, and map utilities."""

import random
import re


_EGO_RE_PATTERNS = (
    r'\n3\. Historical ego motion in 2D BEV frame:\n\s+\[PT_HIST[^\]]*\]',
    r'\n- Use \[PT[^\]]*\] to encapsulate the trajectory\.',
    r'\n2\. Active navigation command: \[[^\]]*\].',
)


def strip_ego_status(text: str) -> str:
    """Remove ego-status lines from a UniDriveVLA-style user prompt."""
    for pattern in _EGO_RE_PATTERNS:
        text = re.sub(pattern, '', text)
    return text


def select_samples(items, n_samples, seed):
    if n_samples is None:
        return list(items)
    if not 0 < n_samples <= len(items):
        raise ValueError(
            f"n_samples must be between 1 and {len(items)}, got {n_samples}"
        )
    indices = sorted(
        random.Random(seed).sample(range(len(items)), n_samples)
    )
    return [items[index] for index in indices]
