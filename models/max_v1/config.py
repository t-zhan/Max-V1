from transformers import PretrainedConfig


class MaxConfig(PretrainedConfig):
    model_type = "max"

    def __init__(self,
                 qwen_model_dir=None,
                 qwen_model_family=None,
                 pred_len=8,
                 use_cache=False,
                 max_model_len=65536,
                 max_new_tokens=65536,
                 use_visual_norm_alignment=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.qwen_model_dir = qwen_model_dir
        self.qwen_model_family = qwen_model_family
        self.pred_len = pred_len
        self.use_cache = use_cache
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.use_visual_norm_alignment = use_visual_norm_alignment
