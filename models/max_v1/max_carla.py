from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import cross_entropy, mse_loss
from transformers import AutoProcessor, PreTrainedModel
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from models.max_v1.config import MaxConfig, RegHeadConfig
from models.max_v1.prompt_template import (
    B2DVL_IMAGE_DESC,
    B2DVL_WAYPOINT_QUESTION,
    COMMAND_TO_TEXT,
    MAX_DEFAULT_SYSTEM,
)
from swift.model import ModelInfo, ModelMeta
from swift.template import TEMPLATE_MAPPING
from swift.template.template_inputs import TemplateInputs


_QWEN_FAMILIES = {
    "qwen2_5_vl": (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig),
    "qwen3_vl": (Qwen3VLForConditionalGeneration, Qwen3VLConfig),
    "qwen3_5": (Qwen3_5ForConditionalGeneration, Qwen3_5Config),
}


class RegHead(PreTrainedModel):
    config_class = RegHeadConfig

    def __init__(self, config: RegHeadConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=config.reg_head_bias),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=config.reg_head_bias),
        )
        self.point_decoder = nn.Linear(config.hidden_size, 2, bias=config.reg_head_bias)

    def forward(self, features):
        output = self.dropout(features)
        residual = output
        output = self.norm(output)
        return self.point_decoder(self.ffn(output) + residual)


class Max(PreTrainedModel):
    config_class = MaxConfig

    def __init__(self, config, is_finetuned=False, **kwargs):
        super().__init__(config)
        qwen_model_cls, qwen_config_cls = _QWEN_FAMILIES[self.config.qwen_model_family]
        self.qwen_template_meta = TEMPLATE_MAPPING[self.config.qwen_model_family]

        if is_finetuned:
            self.backbone = qwen_model_cls(qwen_config_cls.from_dict(config.qwen_config))
            reg_head_config = RegHeadConfig.from_dict(config.reg_head_config)
        else:
            self.backbone = qwen_model_cls.from_pretrained(config.qwen_model_dir, **kwargs)
            reg_head_config = RegHeadConfig(
                hidden_size=self.backbone.config.text_config.hidden_size,
            )
        # MaxConfig is authoritative; the RegHead value is serialized metadata.
        reg_head_config.pred_len = self.config.pred_len

        self.get_rope_index = self.backbone.model.get_rope_index
        self.config.hidden_size = self.backbone.config.text_config.hidden_size
        self.config.image_token_id = self.backbone.config.image_token_id

        backbone_embeddings = self.backbone.get_input_embeddings().weight
        self.point_embed_layer = nn.Linear(
            2,
            self.config.hidden_size,
            bias=False,
            device=backbone_embeddings.device,
            dtype=backbone_embeddings.dtype,
        )
        self.reg_head = RegHead(reg_head_config).to(
            device=backbone_embeddings.device,
            dtype=backbone_embeddings.dtype,
        )
        self.post_init()

    @property
    def model(self):
        return self.backbone.model

    @property
    def visual(self):
        return self.backbone.model.visual

    @staticmethod
    def _extend_attention_mask(attention_mask, point_count):
        point_mask = attention_mask.new_ones((attention_mask.shape[0], point_count))
        return torch.cat((attention_mask, point_mask), dim=-1)

    @staticmethod
    def _extend_position_ids(position_ids, attention_mask, point_count):
        token_indices = torch.arange(
            attention_mask.shape[-1],
            device=attention_mask.device,
        ).expand_as(attention_mask)
        last_token_indices = token_indices.masked_fill(attention_mask == 0, -1).amax(dim=-1)
        gather_indices = last_token_indices[None, :, None].expand(position_ids.shape[0], -1, 1)
        last_position = position_ids.gather(-1, gather_indices)
        offsets = torch.arange(
            1,
            point_count + 1,
            device=position_ids.device,
            dtype=position_ids.dtype,
        )[None, None, :]
        return torch.cat((position_ids, last_position + offsets), dim=-1)

    @contextmanager
    def _suspend_gradient_checkpointing(self):
        enabled_modules = [
            module
            for module in self.backbone.modules()
            if getattr(module, "gradient_checkpointing", False)
        ]
        for module in enabled_modules:
            module.gradient_checkpointing = False
        try:
            yield
        finally:
            for module in enabled_modules:
                module.gradient_checkpointing = True

    def _decode_points(
        self,
        base_inputs_embeds,
        point_inputs,
        attention_mask,
        position_ids,
    ):
        point_embeds = self.point_embed_layer(point_inputs)
        return self.backbone(
            input_ids=None,
            inputs_embeds=torch.cat((base_inputs_embeds, point_embeds), dim=1),
            attention_mask=self._extend_attention_mask(attention_mask, point_inputs.shape[1]),
            position_ids=self._extend_position_ids(
                position_ids,
                attention_mask,
                point_inputs.shape[1],
            ),
            output_hidden_states=True,
            use_cache=False,
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        waypoints=None,
        labels=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        base_inputs_embeds = (
            inputs_embeds
            if inputs_embeds is not None
            else self._process_embeddings(input_ids, pixel_values, image_grid_thw)
        )

        point_weight = self.point_embed_layer.weight
        base_inputs_embeds = base_inputs_embeds.to(
            device=point_weight.device,
            dtype=point_weight.dtype,
        )
        if waypoints.shape[1] != self.config.pred_len:
            raise ValueError(
                f"Expected {self.config.pred_len} waypoints, "
                f"got {waypoints.shape[1]}"
            )
        waypoints = waypoints.to(device=point_weight.device, dtype=point_weight.dtype)
        attention_mask = attention_mask.to(point_weight.device)
        position_ids = position_ids.to(point_weight.device)

        batch_size, base_seq_len, _ = base_inputs_embeds.shape
        seq_len = waypoints.shape[1]
        start_point = point_weight.new_zeros(batch_size, 1, 2)

        scheduled_sampling_ratio = float(self.config.scheduled_sampling_ratio)
        rollout_use_cache = self.config.rollout_use_cache
        rollout_steps = 0
        rollout_points = None
        if scheduled_sampling_ratio > 0.0:
            rollout_steps = int(torch.distributions.Binomial(
                total_count=seq_len,
                probs=scheduled_sampling_ratio,
            ).sample().item())
        if rollout_steps > 0:
            teacher_steps = seq_len - rollout_steps

            if teacher_steps > 0:
                point_inputs = torch.cat((start_point, waypoints[:, :teacher_steps]), dim=1)
            else:
                point_inputs = start_point

            with self._suspend_gradient_checkpointing(), torch.no_grad():
                past_key_values = None
                next_point = None
                rollout_points = []
                for step in range(rollout_steps):
                    if not rollout_use_cache or past_key_values is None:
                        point_embeds = self.point_embed_layer(point_inputs)
                        rollout_outputs = self.backbone.model(
                            inputs_embeds=torch.cat((base_inputs_embeds, point_embeds), dim=1),
                            attention_mask=self._extend_attention_mask(
                                attention_mask,
                                point_inputs.shape[1],
                            ),
                            position_ids=self._extend_position_ids(
                                position_ids,
                                attention_mask,
                                point_inputs.shape[1],
                            ),
                            use_cache=rollout_use_cache,
                        )
                    else:
                        rollout_outputs = self.backbone.model(
                            inputs_embeds=self.point_embed_layer(next_point),
                            attention_mask=self._extend_attention_mask(
                                attention_mask,
                                point_inputs.shape[1],
                            ),
                            position_ids=self._extend_position_ids(
                                position_ids,
                                attention_mask,
                                point_inputs.shape[1],
                            )[..., -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    past_key_values = rollout_outputs.past_key_values
                    next_point = self.reg_head(rollout_outputs.last_hidden_state[:, -1:, :])
                    rollout_points.append(next_point)
                    if step < rollout_steps - 1:
                        point_inputs = torch.cat((point_inputs, next_point), dim=1)
                rollout_points = torch.cat(rollout_points, dim=1)
        else:
            point_inputs = torch.cat((start_point, waypoints[:, :-1]), dim=1)

        outputs = self._decode_points(
            base_inputs_embeds,
            point_inputs,
            attention_mask,
            position_ids,
        )
        predicted_points = self.reg_head(outputs.hidden_states[-1][:, base_seq_len:, :])

        outputs["pred_waypoints"] = predicted_points
        reg_loss = mse_loss(predicted_points, waypoints)
        outputs["reg_loss"] = reg_loss.detach()
        if rollout_points is not None:
            with torch.no_grad():
                outputs["rollout_gt_mse"] = mse_loss(
                    rollout_points,
                    waypoints[:, teacher_steps:],
                ).detach()
                outputs["rollout_pred_mse"] = mse_loss(
                    rollout_points,
                    predicted_points[:, teacher_steps:],
                ).detach()

        if labels is None:
            outputs["loss"] = reg_loss
            outputs["lm_loss"] = None
            return outputs

        shift_logits = outputs.logits[:, :base_seq_len - 1, :].contiguous()
        shift_labels = labels[:, 1:base_seq_len].to(shift_logits.device).contiguous()
        lm_loss = cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        outputs["loss"] = reg_loss + lm_loss
        outputs["lm_loss"] = lm_loss.detach()
        return outputs

    def _predict_waypoints(
        self,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        pixel_values,
        image_grid_thw,
    ):
        point_weight = self.point_embed_layer.weight
        base_inputs_embeds = self._process_embeddings(
            input_ids,
            pixel_values,
            image_grid_thw,
        ).to(device=point_weight.device, dtype=point_weight.dtype)

        batch_size = input_ids.shape[0]
        current_points = point_weight.new_zeros(batch_size, 1, 2)
        position_ids, _ = self.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        position_ids = position_ids.to(point_weight.device)
        past_key_values = None
        next_point = None

        for step in range(self.config.pred_len):
            if self.config.use_cache and step > 0:
                outputs = self.backbone.model(
                    inputs_embeds=self.point_embed_layer(next_point),
                    attention_mask=self._extend_attention_mask(attention_mask, step + 1),
                    position_ids=self._extend_position_ids(
                        position_ids,
                        attention_mask,
                        step + 1,
                    )[..., -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                point_embeds = self.point_embed_layer(current_points)
                outputs = self.backbone.model(
                    inputs_embeds=torch.cat((base_inputs_embeds, point_embeds), dim=1),
                    attention_mask=self._extend_attention_mask(
                        attention_mask,
                        current_points.shape[1],
                    ),
                    position_ids=self._extend_position_ids(
                        position_ids,
                        attention_mask,
                        current_points.shape[1],
                    ),
                    use_cache=self.config.use_cache,
                )

            past_key_values = outputs.past_key_values
            next_point = self.reg_head(outputs.last_hidden_state[:, -1:, :])
            current_points = torch.cat((current_points, next_point), dim=1)

        return current_points[:, 1:]

    @staticmethod
    def _generated_attention_mask(generated_ids, eos_token_ids):
        eos_token_ids = torch.as_tensor(
            eos_token_ids,
            device=generated_ids.device,
        ).flatten()
        is_eos = generated_ids.unsqueeze(-1).eq(eos_token_ids).any(dim=-1)
        is_eos = is_eos.to(torch.int64)
        ended_before = is_eos.cumsum(dim=-1) - is_eos
        return ended_before.eq(0)

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
    def carla_generate(self, rgbs, ego_speeds, command_idxs, enable_thinking=True):
        encoded_list = []
        command_texts = []
        for rgb, speed, cmd in zip(rgbs, ego_speeds, command_idxs):
            command_text = COMMAND_TO_TEXT[int(cmd)]
            command_texts.append(command_text)
            front_concat, back_concat = self._concat_camera_images(rgb)
            user_content = (
                f"{B2DVL_IMAGE_DESC}<image><image>"
                "Use information above to answer:\n"
                f"The ego vehicle is driving at the speed of {float(speed):.1f} m/s, "
                f"and it wants to {command_text}. "
                f"{B2DVL_WAYPOINT_QUESTION}"
            )
            sample = {
                "messages": [{"role": "user", "content": user_content}],
                "system": MAX_DEFAULT_SYSTEM,
                "images": [front_concat, back_concat],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            encoded_list.append(self.inference_template.encode(TemplateInputs.from_dict(sample)))

        batch = self.inference_template.data_collator(encoded_list)
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mm_token_type_ids = batch["mm_token_type_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch["image_grid_thw"].to(device)

        cot_input_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
        )
        generated_ids = cot_input_ids[:, input_ids.shape[1]:]
        cot_texts = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        generated_attention_mask = self._generated_attention_mask(
            generated_ids,
            self.backbone.generation_config.eos_token_id,
        ).to(attention_mask.dtype)
        cot_attention_mask = torch.cat((attention_mask, generated_attention_mask), dim=-1)
        cot_mm_token_type_ids = torch.cat(
            (mm_token_type_ids, torch.zeros_like(generated_ids)),
            dim=-1,
        )
        pred_waypoints = self._predict_waypoints(
            cot_input_ids,
            cot_attention_mask,
            cot_mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        )
        return pred_waypoints, command_texts, cot_texts

    def _process_embeddings(self, input_ids, pixel_values, image_grid_thw):
        inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids)
        image_outputs = self.backbone.model.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        image_mask, _ = self.backbone.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

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
        self.reg_head.config.pred_len = self.config.pred_len
        self.config.qwen_config = self.backbone.config.to_dict()
        self.config.reg_head_config = self.reg_head.config.to_dict()
        super().save_pretrained(save_directory, *args, **kwargs)
