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
from swift.template import TemplateMeta, register_template
from swift.template.templates.qwen import Qwen2_5VLTemplate, Qwen3VLTemplate, Qwen3_5Template
from models.max_v1.config import MaxConfig
from models.max_v1.max_carla import Max
from models.max_v1.prompt_template import (
    MAX_CHAT_SEP,
    MAX_DEFAULT_SYSTEM,
    MAX_PROMPT,
    MAX_SUFFIX,
    MAX_SYSTEM_PREFIX,
)


MAX_TEMPLATE_QWEN2_5 = "max_vl_qwen2_5"
MAX_TEMPLATE_QWEN3 = "max_vl_qwen3"
MAX_TEMPLATE_QWEN3_5 = "max_vl_qwen3_5"
MAX_ARCH = "max_vl"


class MaxLoader(ModelLoader):
    QWEN_MODEL_FAMILY = ""

    def get_config(self, model_dir):
        return MaxConfig(
            qwen_model_dir=model_dir,
            qwen_model_family=self.QWEN_MODEL_FAMILY,
            pred_len=8,
            use_cache=False,
            scheduled_sampling_ratio=0.0,
            max_model_len=65536,
            max_new_tokens=512
        )

    def get_processor(self, model_dir, config):
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


def _register_max_template(template_type, template_cls):
    register_template(
        TemplateMeta(
            template_type=template_type,
            template_cls=template_cls,
            prefix=[],
            prompt=[MAX_PROMPT],
            chat_sep=[MAX_CHAT_SEP],
            suffix=[MAX_SUFFIX],
            system_prefix=[MAX_SYSTEM_PREFIX],
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

_register_max_template(MAX_TEMPLATE_QWEN2_5, MaxQwen2_5Template)
_register_max_template(MAX_TEMPLATE_QWEN3, MaxQwen3Template)
_register_max_template(MAX_TEMPLATE_QWEN3_5, MaxQwen3_5Template)

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
