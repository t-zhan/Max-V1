from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F


def _gathered_parameters(parameters, modifier_rank=None):
    parameters = list(parameters)
    if not any(getattr(parameter, "ds_id", None) is not None for parameter in parameters):
        return nullcontext()

    import deepspeed

    return deepspeed.zero.GatheredParameters(parameters, modifier_rank=modifier_rank)


@torch.no_grad()
def mean_nonzero_embedding_l2(embedding_weight, eps=1e-12, chunk_size=4096):
    """Mean row-wise L2 norm over the non-zero text embedding vocabulary."""
    with _gathered_parameters([embedding_weight]):
        if embedding_weight.is_meta:
            raise RuntimeError("Cannot measure text embedding norms on a meta tensor.")
        if embedding_weight.ndim != 2:
            raise ValueError("Expected a 2D text embedding matrix.")

        total = torch.zeros((), device=embedding_weight.device, dtype=torch.float64)
        count = torch.zeros((), device=embedding_weight.device, dtype=torch.long)
        for chunk in embedding_weight.split(chunk_size, dim=0):
            norms = chunk.float().norm(p=2, dim=-1)
            nonzero = norms > eps
            total += norms[nonzero].double().sum()
            count += nonzero.sum()

        if count.item() == 0:
            raise ValueError("The text embedding matrix contains no non-zero rows.")
        return (total / count).item()


class GlobalWeightCompensatedLayerNorm(nn.LayerNorm):
    """LayerNorm with the backward-only global weight compensation from the paper."""

    def __init__(self, hidden_size, eps=1e-5, min_gain=1e-3, *, device=None, dtype=None):
        super().__init__(hidden_size, eps=eps, device=device, dtype=dtype)
        self.min_gain = min_gain

    @torch.no_grad()
    def initialize_from_target_l2(self, target_l2):
        if target_l2 <= 0:
            raise ValueError("target_l2 must be positive.")

        gain = float(target_l2) / math.sqrt(self.normalized_shape[0])
        with _gathered_parameters(self.parameters(), modifier_rank=0):
            self.weight.fill_(gain)
            self.bias.zero_()
        return gain

    def forward(self, inputs):
        normalized = F.layer_norm(
            inputs,
            self.normalized_shape,
            weight=None,
            bias=None,
            eps=self.eps,
        )
        if self.training and normalized.requires_grad:
            mean_gain = self.weight.detach().abs().mean().clamp_min(self.min_gain)
            normalized.register_hook(lambda grad: grad / mean_gain)
        return normalized * self.weight + self.bias
