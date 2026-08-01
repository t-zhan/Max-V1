"""Internal nuScenes inference, evaluation, and map utilities."""

import random


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
