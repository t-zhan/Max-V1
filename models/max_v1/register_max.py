import json
import os
from dataclasses import replace

import torch
from transformers import AutoProcessor

from swift.model import (
    Model,
    ModelGroup,
    ModelLoader,
    ModelMeta,
    MultiModelKeys,
    register_model,
    register_model_arch,
)
from swift.model.patcher import patch_get_input_embeddings
from swift.template import TEMPLATE_MAPPING, register_template
from swift.template.templates.qwen import Qwen2_5VLTemplate, Qwen3VLTemplate, Qwen3_5Template
from models.max_v1.config import MaxConfig
from models.max_v1.max_carla import Max
from models.max_v1.prompt_template import MAX_DEFAULT_SYSTEM


MAX_TEMPLATE_QWEN2_5 = "max_vl_qwen2_5"
MAX_TEMPLATE_QWEN3 = "max_vl_qwen3"
MAX_TEMPLATE_QWEN3_5 = "max_vl_qwen3_5"
MAX_ARCH = "max_vl"


def _is_max_checkpoint(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    return (
        config.get("model_type") == MaxConfig.model_type
        and "Max" in config.get("architectures", [])
        and isinstance(config.get("qwen_config"), dict)
        and isinstance(config.get("reg_head_config"), dict)
    )


class MaxLoader(ModelLoader):
    QWEN_MODEL_FAMILY = ""

    def get_config(self, model_dir):
        if _is_max_checkpoint(model_dir):
            config = MaxConfig.from_pretrained(model_dir)
        else:
            config = MaxConfig(
                qwen_model_dir=model_dir,
                qwen_model_family=self.QWEN_MODEL_FAMILY,
            )

        config.pred_len = int(
            self.model_kwargs.pop(
                "pred_len",
                config.pred_len,
            )
        )
        config.scheduled_sampling_ratio = float(
            self.model_kwargs.pop(
                "scheduled_sampling_ratio",
                config.scheduled_sampling_ratio,
            )
        )
        config.rollout_use_cache = self.model_kwargs.pop(
            "rollout_use_cache",
            config.rollout_use_cache,
        )
        return config

    def get_processor(self, model_dir, config):
        if _is_max_checkpoint(model_dir):
            return AutoProcessor.from_pretrained(
                model_dir,
                tokenizer_type=config.qwen_model_family,
                padding_side="right",
                backend="torchvision",
            )

        return AutoProcessor.from_pretrained(
            config.qwen_model_dir,
            padding_side="right",
            backend="torchvision",
        )

    def get_model(
        self,
        model_dir,
        config,
        processor,
        model_kwargs,
    ):
        if _is_max_checkpoint(model_dir):
            model = Max.from_pretrained(model_dir, config=config, **model_kwargs)
        else:
            model = Max(config, is_finetuned=False, **model_kwargs)

        patch_get_input_embeddings(model.backbone.model.visual, "patch_embed")
        return model


def _make_loader(qwen_model_family):
    class FamilyMaxLoader(MaxLoader):
        QWEN_MODEL_FAMILY = qwen_model_family
    return FamilyMaxLoader


class _MaxTemplateMixin:
    def _post_encode(self, model, inputs):
        encoded = super()._post_encode(model, inputs)
        encoded["waypoints"] = inputs["waypoints"]
        return encoded

    def _encode(self, inputs):
        waypoints = inputs.extra_kwargs["waypoints"]
        encoded = super()._encode(inputs)
        encoded["waypoints"] = torch.tensor(waypoints, dtype=torch.float32)
        return encoded

    def _data_collator(
        self,
        batch,
        *,
        padding_to=None,
    ):
        res = super()._data_collator(batch, padding_to=padding_to)
        waypoints = [b["waypoints"] for b in batch]
        res["waypoints"] = torch.stack(waypoints)
        return res


class MaxQwen2_5Template(_MaxTemplateMixin, Qwen2_5VLTemplate):
    pass


class MaxQwen3Template(_MaxTemplateMixin, Qwen3VLTemplate):
    pass


class MaxQwen3_5Template(_MaxTemplateMixin, Qwen3_5Template):
    pass


def _register_max_template(template_type, template_cls, base_template_type):
    base_meta = TEMPLATE_MAPPING[base_template_type]
    register_template(replace(
        base_meta,
        template_type=template_type,
        template_cls=template_cls,
        default_system=MAX_DEFAULT_SYSTEM,
    ))


def _register_max_model_type(
    model_type,
    model_id,
    template_type,
    loader_cls,
):
    register_model(
        ModelMeta(
            model_type=model_type,
            model_groups=[
                ModelGroup([
                    Model(ms_model_id=model_id, hf_model_id=model_id)
                ])
            ],
            loader=loader_cls,
            template=template_type,
            is_multimodal=True,
            model_arch=MAX_ARCH,
            architectures=["Max"],
            requires=["transformers", "torch"],
            tags=["vision", "autonomous-driving"],
            additional_saved_files=[],
        ))


register_model_arch(
    MultiModelKeys(
        MAX_ARCH,
        language_model=["backbone.model.language_model", "backbone.lm_head"],
        vision_tower=["backbone.model.visual"],
        aligner=[],
        generator=[],
    ))

_register_max_template(MAX_TEMPLATE_QWEN2_5, MaxQwen2_5Template, "qwen2_5_vl")
_register_max_template(MAX_TEMPLATE_QWEN3, MaxQwen3Template, "qwen3_vl")
_register_max_template(MAX_TEMPLATE_QWEN3_5, MaxQwen3_5Template, "qwen3_5")

_register_max_model_type(
    model_type="max_qwen2_5_vl",
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    template_type=MAX_TEMPLATE_QWEN2_5,
    loader_cls=_make_loader("qwen2_5_vl"),
)

_register_max_model_type(
    model_type="max_qwen3_vl",
    model_id="Qwen/Qwen3-VL-4B-Instruct",
    template_type=MAX_TEMPLATE_QWEN3,
    loader_cls=_make_loader("qwen3_vl"),
)

_register_max_model_type(
    model_type="max_qwen3_5",
    model_id="Qwen/Qwen3.5-0.8B",
    template_type=MAX_TEMPLATE_QWEN3_5,
    loader_cls=_make_loader("qwen3_5"),
)
