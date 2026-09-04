import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, PreTrainedModel
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from models.max_v1.config import MaxConfig
from models.max_v1.norm_alignment import (
    GlobalWeightCompensatedLayerNorm,
    mean_nonzero_embedding_l2,
)
from models.max_v1.prompt_template import (
    B2DVL_IMAGE_DESC,
    B2DVL_WAYPOINT_QUESTION,
    COMMAND_TO_TEXT,
    MAX_DEFAULT_SYSTEM,
)
from models.max_v1.waypoint_text import parse_trajectory
from swift.model import ModelInfo, ModelMeta
from swift.template import TEMPLATE_MAPPING
from swift.template.template_inputs import TemplateInputs


_QWEN_FAMILIES = {
    "qwen2_5_vl": (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig),
    "qwen3_vl": (Qwen3VLForConditionalGeneration, Qwen3VLConfig),
    "qwen3_5": (Qwen3_5ForConditionalGeneration, Qwen3_5Config),
}


class Max(PreTrainedModel):
    config_class = MaxConfig

    def __init__(self, config, is_finetuned=False, **kwargs):
        super().__init__(config)
        qwen_model_cls, qwen_config_cls = _QWEN_FAMILIES[self.config.qwen_model_family]
        self.qwen_template_meta = TEMPLATE_MAPPING[self.config.qwen_model_family]

        if is_finetuned:
            self.backbone = qwen_model_cls(qwen_config_cls.from_dict(config.qwen_config))
        else:
            self.backbone = qwen_model_cls.from_pretrained(config.qwen_model_dir, **kwargs)

        self.config.hidden_size = self.backbone.config.text_config.hidden_size
        self.config.image_token_id = self.backbone.config.image_token_id

        backbone_embeddings = self.backbone.get_input_embeddings().weight
        if self.config.use_visual_norm_alignment:
            self.visual_token_norm = GlobalWeightCompensatedLayerNorm(
                self.config.hidden_size,
                device=backbone_embeddings.device,
                dtype=backbone_embeddings.dtype,
            )
        else:
            self.visual_token_norm = nn.Identity()
        if self.config.use_visual_norm_alignment:
            self.backbone.model.visual.register_forward_hook(
                self._apply_visual_token_norm,
            )
        self.post_init()
        if self.config.use_visual_norm_alignment and not is_finetuned:
            self._initialize_visual_token_norm(backbone_embeddings)

    @property
    def model(self):
        return self.backbone.model

    @property
    def visual(self):
        return self.backbone.model.visual

    def _initialize_visual_token_norm(self, embedding_weight=None):
        if not isinstance(self.visual_token_norm, GlobalWeightCompensatedLayerNorm):
            return None
        if embedding_weight is None:
            embedding_weight = self.backbone.get_input_embeddings().weight
        target_l2 = mean_nonzero_embedding_l2(embedding_weight)
        self.visual_token_norm.initialize_from_target_l2(target_l2)
        return target_l2

    def _apply_visual_token_norm(self, _module, _inputs, outputs):
        outputs.pooler_output = self.visual_token_norm(outputs.pooler_output)
        return outputs

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        return self.backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    @staticmethod
    def _concat_camera_images(rgb):
        front_concat = np.concatenate(rgb[:3], axis=1)
        back_concat = np.concatenate(rgb[3:], axis=1)
        return Image.fromarray(front_concat, mode="RGB"), Image.fromarray(back_concat, mode="RGB")

    def _build_inference_template(self):
        template = self.qwen_template_meta.template_cls(
            processor=self.processor,
            template_meta=self.qwen_template_meta,
            padding_side="left",
            use_chat_template=True,
            template_backend="swift",
        )
        template.set_mode("transformers")
        template.model = self.backbone
        return template

    @torch.inference_mode()
    def generate_trajectories(self, samples):
        encoded_list = [
            self.inference_template.encode(TemplateInputs.from_dict(sample))
            for sample in samples
        ]
        batch = self.inference_template.data_collator(encoded_list)
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)

        generated_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].to(device),
            mm_token_type_ids=batch["mm_token_type_ids"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            image_grid_thw=batch["image_grid_thw"].to(device),
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
        )[:, input_ids.shape[1]:]
        texts = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        parsed = [parse_trajectory(text, self.config.pred_len) for text in texts]
        trajectories = np.stack([trajectory for trajectory, _ in parsed])
        statuses = [status for _, status in parsed]
        return trajectories, texts, statuses

    @torch.inference_mode()
    def carla_generate(self, rgbs, ego_speeds, command_idxs, enable_thinking=True):
        samples = []
        for rgb, speed, cmd in zip(rgbs, ego_speeds, command_idxs):
            command_text = COMMAND_TO_TEXT[int(cmd)]
            front_concat, back_concat = self._concat_camera_images(rgb)
            user_content = (
                f"{B2DVL_IMAGE_DESC}<image><image>"
                "Use information above to answer:\n"
                f"The ego vehicle is driving at the speed of {float(speed):.1f} m/s, "
                f"and it wants to {command_text}. "
                f"{B2DVL_WAYPOINT_QUESTION}"
            )
            samples.append({
                # StdTemplateInputs.from_dict only reads the system prompt from
                # messages[0]; a top-level "system" key is silently dropped.
                "messages": [
                    {"role": "system", "content": MAX_DEFAULT_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "images": [front_concat, back_concat],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            })

        return self.generate_trajectories(samples)

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            model_name_or_path,
            *model_args,
            is_finetuned=True,
            **kwargs,
        )
        model.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            tokenizer_type=model.config.qwen_model_family,
            padding_side="left",
            backend="torchvision",
        )
        model.processor.tokenizer.padding_side = "left"
        model.backbone.generation_config.eos_token_id = (
            model.processor.tokenizer.eos_token_id
        )
        model.backbone.generation_config.pad_token_id = (
            model.processor.tokenizer.pad_token_id
        )

        model_type = f"max_{model.config.qwen_model_family}"
        model_info = ModelInfo(
            model_type=model_type,
            model_dir=str(model_name_or_path),
            torch_dtype=next(model.backbone.parameters()).dtype,
            max_model_len=model.config.max_model_len,
            quant_method=None,
            quant_bits=0,
            is_multimodal=True,
            config=model.backbone.config,
            task_type="causal_lm",
        )
        model_meta = ModelMeta(
            model_type=model_type,
            model_groups=[],
            is_multimodal=True,
            task_type="causal_lm",
        )
        model.processor.model_info = model_info
        model.processor.model_meta = model_meta
        model.inference_template = model._build_inference_template()
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        self.config.qwen_config = self.backbone.config.to_dict()
        super().save_pretrained(save_directory, *args, **kwargs)
