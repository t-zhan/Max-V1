import torch
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy
from PIL import Image
import numpy as np

from transformers import AutoProcessor, PreTrainedModel
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.generation.utils import GenerationMixin
from models.max_v1.config import RegHeadConfig, MaxConfig
from models.max_v1.prompt_template import B2DVL_IMAGE_DESC, B2DVL_WAYPOINT_QUESTION, COMMAND_TO_TEXT, MAX_DEFAULT_SYSTEM
from swift.template import TEMPLATE_MAPPING
from swift.template.template_inputs import StdTemplateInputs


class RegHead(PreTrainedModel):
    config_class = RegHeadConfig

    def __init__(self, config: RegHeadConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(0.1)
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
        output = self.ffn(output) + residual
        output = self.point_decoder(output)
        return output


class Max(PreTrainedModel, GenerationMixin):
    config_class = MaxConfig

    @staticmethod
    def _resolve_qwen_family_and_classes(config: MaxConfig):
        model_family = config.qwen_model_family
        family_to_classes = {
            "qwen2_5_vl": (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig),
            "qwen3_vl": (Qwen3VLForConditionalGeneration, Qwen3VLConfig),
            "qwen3_5": (Qwen3_5ForConditionalGeneration, Qwen3_5Config),
        }
        qwen_model_cls, qwen_config_cls = family_to_classes[model_family]
        qwen_template_meta = TEMPLATE_MAPPING[model_family]
        return model_family, qwen_model_cls, qwen_config_cls, qwen_template_meta

    def __init__(self, config, is_finetuned=False, **kwargs):
        super().__init__(config)
        (self.qwen_model_family, qwen_model_cls, qwen_config_cls,
         self.qwen_template_meta) = self._resolve_qwen_family_and_classes(config)

        if not is_finetuned:
            self.backbone = qwen_model_cls.from_pretrained(config.qwen_model_dir)
            reg_head_config = RegHeadConfig(
                hidden_size=self.backbone.config.text_config.hidden_size,
                pred_len=config.pred_len,
            )
        else:
            qwen_config_dict = config.qwen_config
            qwen_config = qwen_config_cls.from_dict(qwen_config_dict)
            self.backbone = qwen_model_cls(qwen_config)
            reg_head_config = RegHeadConfig.from_dict(config.reg_head_config)

        self.get_rope_index = self.backbone.model.get_rope_index
        self.config.hidden_size = self.backbone.config.text_config.hidden_size
        self.config.image_token_id = self.backbone.config.image_token_id

        backbone_dtype = self.backbone.get_input_embeddings().weight.dtype
        self.point_embed_layer = nn.Linear(
            2,
            self.config.hidden_size,
            bias=False,
            device=self.backbone.device,
        ).to(dtype=backbone_dtype)
        self.reg_head = RegHead(reg_head_config).to(dtype=backbone_dtype)

        self.scheduled_sampling_ratio = 0.0

    @property
    def model(self):
        return self.backbone.model

    @property
    def visual(self):
        return self.backbone.model.visual

    def forward(self, 
                input_ids=None, 
                pixel_values=None, 
                image_grid_thw=None,
                waypoints=None, 
                labels=None, 
                inputs_embeds=None, 
                **kwargs):
        if inputs_embeds is not None:
            base_inputs_embeds = inputs_embeds
        else:
            base_inputs_embeds = self._process_embeddings(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        batch_size, base_seq_len, _ = base_inputs_embeds.shape
        seq_len = waypoints.shape[1]
        embed_dtype = self.point_embed_layer.weight.dtype
        embed_device = self.point_embed_layer.weight.device

        if self.scheduled_sampling_ratio > 0.0:
            start_point = torch.zeros(batch_size, 1, 2, device=embed_device, dtype=embed_dtype)
            current_points = start_point
            all_predicted_points = []

            for step in range(seq_len):
                use_teacher_forcing = torch.rand(1).item() > self.scheduled_sampling_ratio
                if step == 0:
                    input_points = current_points
                elif use_teacher_forcing:
                    input_points = torch.cat([start_point, waypoints[:, :step]], dim=1)
                else:
                    input_points = current_points

                point_embeds = self.point_embed_layer(input_points).to(embed_device)
                full_inputs_embeds = torch.cat([base_inputs_embeds, point_embeds], dim=1)

                outputs = self.backbone(
                    input_ids=None,
                    inputs_embeds=full_inputs_embeds,
                    output_hidden_states=True,
                )

                point_hidden_states = outputs.hidden_states[-1][:, -1:, :].to(embed_device)
                predicted_point = self.reg_head(point_hidden_states)
                all_predicted_points.append(predicted_point)
                current_points = torch.cat([current_points, predicted_point], dim=1)

            predicted_points = torch.cat(all_predicted_points, dim=1)
        else:
            # Teacher forcing for waypoint regression
            start_point = torch.zeros(batch_size, 1, 2, device=embed_device, dtype=embed_dtype)
            decoder_input_points = torch.cat([start_point, waypoints[:, :-1]], dim=1)
            point_embeds = self.point_embed_layer(decoder_input_points).to(embed_device)

            full_inputs_embeds = torch.cat([base_inputs_embeds, point_embeds], dim=1)

            outputs = self.backbone(
                input_ids=None,
                inputs_embeds=full_inputs_embeds,
                output_hidden_states=True,
            )

            point_hidden_states = outputs.hidden_states[-1].to(embed_device)
            predicted_points = self.reg_head(point_hidden_states)
            predicted_points = predicted_points[:, base_seq_len:]

        # --- MSE loss from RegHead (waypoint coordinates) ---
        reg_loss = mse_loss(predicted_points, waypoints)

        # --- LM loss on CoT text tokens ---
        if labels is not None:
            lm_logits = self.backbone.lm_head(outputs.hidden_states[-1][:, :base_seq_len, :])
            shift_logits = lm_logits[:, :-1, :].contiguous()
            # labels aligned with input_ids: [system, user, assistant(CoT)]
            shift_labels = labels[:, 1:base_seq_len].contiguous()
            lm_loss = cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            lm_loss = torch.tensor(0.0, device=embed_device)

        outputs["loss"] = reg_loss + lm_loss
        outputs["pred_waypoints"] = predicted_points
        outputs["lm_loss"] = lm_loss.detach()
        outputs["reg_loss"] = reg_loss.detach()
        return outputs

    def _get_inference_template(self):
        if getattr(self, '_inference_template', None) is not None:
            return self._inference_template
        self._inference_template = self.qwen_template_meta.template_cls(
            processor=self.processor,
            template_meta=self.qwen_template_meta,
            padding_side='left',
            use_chat_template=True,
            template_backend='swift',
        )
        self._inference_template.set_mode('transformers')
        return self._inference_template

    def generate(self, input_ids, pixel_values, image_grid_thw):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        current_points = torch.zeros(
            batch_size,
            1,
            2,
            device=device,
            dtype=self.point_embed_layer.weight.dtype,
        )
        next_point = None
        past_key_values = None

        base_inputs_embeds = self._process_embeddings(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        for step in range(self.config.pred_len):
            if not self.config.use_cache or step == 0:
                point_embeds = self.point_embed_layer(current_points)
                base_inputs_embeds = base_inputs_embeds.to(point_embeds.device)
                full_inputs_embeds = torch.cat([base_inputs_embeds, point_embeds], dim=1)
                outputs = self.backbone.model(inputs_embeds=full_inputs_embeds)
            else:
                point_embeds = self.point_embed_layer(next_point)
                outputs = self.backbone.model(
                    inputs_embeds=point_embeds,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            if self.config.use_cache:
                past_key_values = outputs.past_key_values

            point_hidden_states = outputs.last_hidden_state[:, -1:, :]
            next_point = self.reg_head(point_hidden_states)

            current_points = current_points.to(next_point.device)
            current_points = torch.cat([current_points, next_point], dim=1)

        return current_points[:, 1:, [1, 0]]

    @classmethod
    def _concat_camera_images(cls, rgb):
        front_concat = torch.cat(rgb[:3], dim=2)
        back_concat = torch.cat(rgb[3:], dim=2)
        front_concat = front_concat.permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
        back_concat = back_concat.permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
        return Image.fromarray(front_concat).convert("RGB"), Image.fromarray(back_concat).convert("RGB")

    def carla_generate(self, rgb, ego_vel, command):
        """CARLA inference: take RGB tensor + command, return waypoints.

        Returns a tuple matching the CARLA leaderboard agent interface:
        (waypoints, *steer_throttle_brake_etc, command_text).
        """
        cmd_idx = int(torch.argmax(command).item())
        command_text = COMMAND_TO_TEXT[cmd_idx]

        front_concat, back_concat = self._concat_camera_images(rgb)

        template = self._get_inference_template()
        st_inputs = StdTemplateInputs(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": B2DVL_IMAGE_DESC},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": (
                        "Use information above to answer:\n"
                        f"The ego vehicle is driving at the speed of {float(ego_vel.item()):.1f} m/s, "
                        f"and it wants to {command_text}. "
                        f"{B2DVL_WAYPOINT_QUESTION}"
                    )},
                ]
            }],
            system=MAX_DEFAULT_SYSTEM,
            images=[front_concat, back_concat],
        )
        encoded = template.encode(st_inputs)
        batch = template.data_collator([encoded])
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device)

        with torch.no_grad():
            cot_input_ids = self.backbone.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                do_sample=False,
            )
            pred_wp = self.generate(
                input_ids=cot_input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        # CARLA leaderboard interface expects 11 return values
        return pred_wp, None, None, None, None, None, None, None, None, None, command_text

    def _process_embeddings(self, input_ids, pixel_values, image_grid_thw):

        inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids)

        image_outputs = self.backbone.model.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.backbone.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(model_name_or_path, *model_args, is_finetuned=True, **kwargs)
        model.processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="left", backend="torchvision")
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        self.config.qwen_model_family = self.qwen_model_family
        self.config.qwen_config = self.backbone.config.to_dict()
        self.config.reg_head_config = self.reg_head.config.to_dict()
        super().save_pretrained(save_directory, *args, **kwargs)
