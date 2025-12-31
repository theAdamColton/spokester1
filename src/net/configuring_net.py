from dataclasses import dataclass, field
from typing import Literal

NORMALIZATION_MODE = Literal[
    "layernorm", "rmsnorm", "dyntanh", "derfnorm", "none", None
]


@dataclass
class TransformerConfig:
    hidden_size: int = 256
    head_dim: int = 64

    pre_norm: NORMALIZATION_MODE = "rmsnorm"

    qk_norm: NORMALIZATION_MODE = "none"
    use_qv_bias: bool = True
    use_k_bias: bool = False
    use_proj_bias: bool = True
    use_attention_gating: bool = False

    mlp_mode: Literal["gated", "vanilla"] = "vanilla"

    @property
    def num_attention_heads(self):
        return self.hidden_size // self.head_dim


@dataclass
class ViTDenoiserConfig:
    input_size: int = 256
    condition_input_size: int | None = None
    condition_input_norm: NORMALIZATION_MODE = "none"
    num_blocks: int = 2
    should_pin_adaln_projections: bool = True
    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig())
    norm_out: NORMALIZATION_MODE = "rmsnorm"

    @staticmethod
    def from_dict(d: dict):
        transformer_kwargs = d.pop("transformer", {})
        transformer = TransformerConfig(**transformer_kwargs)
        config = ViTDenoiserConfig(transformer=transformer, **d)
        return config
