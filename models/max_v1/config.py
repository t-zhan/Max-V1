from transformers import PretrainedConfig


class RegHeadConfig(PretrainedConfig):
    model_type = "reg_head"

    def __init__(self, 
                 hidden_size=1024, 
                 pred_len=8, 
                 reg_head_bias=False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.reg_head_bias = reg_head_bias


class MaxConfig(PretrainedConfig):
    model_type = "max"

    def __init__(self,
                 qwen_model_dir=None,
                 qwen_model_family=None,
                 pred_len=8,
                 use_cache=False,
                 scheduled_sampling_ratio=0.0,
                 max_model_len=65536,
                 max_new_tokens=512,
                 **kwargs):
        super().__init__(**kwargs)
        self.qwen_model_dir = qwen_model_dir
        self.qwen_model_family = qwen_model_family
        self.pred_len = pred_len
        self.use_cache = use_cache
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.scheduled_sampling_ratio = scheduled_sampling_ratio
