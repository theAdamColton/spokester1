import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from src.net.configuring_net import (
    NORMALIZATION_MODE,
    TransformerConfig,
    ViTDenoiserConfig,
)
from src.net.torch_utils import unsqueeze_leading
from src.net.optim import override_parameter_needs_weight_decay_
from src.net.rotary_embeddings import Rope2DPositionEmbedding, apply_rotary_emb


def norm_layer_factory(
    hidden_size: int, norm_mode: NORMALIZATION_MODE, eps=1e-5, **kwargs
):
    if norm_mode == "dyntanh":
        return DynTanh(hidden_size, **kwargs)
    if norm_mode == "derfnorm":
        return DerfNorm(hidden_size, **kwargs)
    elif norm_mode == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps, **kwargs)
    elif norm_mode == "rmsnorm":
        return nn.RMSNorm(hidden_size, eps=eps, **kwargs)
    elif norm_mode == "none" or norm_mode is None:
        return nn.Identity()
    raise ValueError(norm_mode)


@torch.no_grad()
def simple_init_weights_(module, init_std=0.02, torch_rng=None):
    if isinstance(module, nn.Linear):
        init.trunc_normal_(module.weight, std=init_std, generator=torch_rng)
        if module.bias is not None:
            init.zeros_(module.bias)


@torch.no_grad()
def init_block_weights_(
    blocks, init_std=0.02, mup_width_multiplier=1.0, torch_rng=None
):
    """
    `mup_width_multiplier = width / base_width` where base_width is typically 256
    """

    for i, block in enumerate(blocks):
        assert isinstance(block, DiTTransformerBlock)

        ff1_std = init_std / math.sqrt(mup_width_multiplier)
        ff2_std = init_std / math.sqrt(2 * (i + 1) * mup_width_multiplier)

        init.zeros_(block.adaLN_modulation.weight)

        init.trunc_normal_(
            block.attention.proj_q.weight, std=ff1_std, generator=torch_rng
        )
        init.trunc_normal_(
            block.attention.proj_k.weight, std=ff1_std, generator=torch_rng
        )
        init.trunc_normal_(
            block.attention.proj_v.weight, std=ff1_std, generator=torch_rng
        )

        if block.attention.proj_g is not None:
            init.trunc_normal_(
                block.attention.proj_g.weight, std=ff1_std, generator=torch_rng
            )

        init.trunc_normal_(
            block.attention.proj_out.weight, std=ff2_std, generator=torch_rng
        )

        init.trunc_normal_(block.mlp.up_proj.weight, std=ff1_std, generator=torch_rng)

        if block.config.mlp_mode == "gated":
            init.trunc_normal_(
                block.mlp.up_proj_gate.weight, std=ff1_std, generator=torch_rng
            )

        init.trunc_normal_(block.mlp.down_proj.weight, std=ff2_std, generator=torch_rng)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a N-D Tensor of timesteps from 1 to 1000
        embedding_dim (int):
            the dimension of the output.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [... x dim] Tensor of positional embeddings.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim

    emb = torch.exp(exponent)

    # ..., d -> ... d
    timesteps = timesteps.unsqueeze(-1).float()
    emb = timesteps * unsqueeze_leading(emb, timesteps)

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class DynTanh(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True):
        super().__init__()

        self.alpha = nn.Parameter(torch.full((hidden_size,), 0.5))
        override_parameter_needs_weight_decay_(self.alpha)

        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = F.tanh(x * unsqueeze_leading(self.alpha, x))
        if self.elementwise_affine:
            x = x * unsqueeze_leading(self.gamma, x)
            x = x + unsqueeze_leading(self.beta, x)

        return x


class DerfNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        elementwise_affine: bool = True,
        alpha_init_value=0.5,
        shift_init_value=0.0,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)

    def forward(self, x):
        x = self.alpha * x + self.shift
        if self.elementwise_affine:
            return torch.erf(x) * self.weight + self.bias
        else:
            return torch.erf(x)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}, shift_init_value={self.shift_init_value}"


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int = 256, use_bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=use_bias)
        self.up_proj_gate = nn.Linear(hidden_size, 4 * hidden_size, bias=use_bias)
        self.activation_function = nn.SiLU()
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x = self.activation_function(self.up_proj_gate(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


class VanillaMLP(nn.Module):
    def __init__(self, hidden_size: int = 256, use_bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=use_bias)
        self.activation_function = nn.GELU()
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x = self.activation_function(self.up_proj(x))
        x = self.down_proj(x)
        return x


class ResidualMLP(nn.Module):
    def __init__(
        self, hidden_size: int = 256, use_bias: bool = False, num_blocks: int = 4
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(hidden_size), VanillaMLP(hidden_size, use_bias))
            for _ in range(num_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        x = self.norm_out(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        config: TransformerConfig = TransformerConfig(),
    ):
        super().__init__()
        self.config = config

        self.proj_q = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.use_qv_bias,
        )
        self.proj_k = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.use_k_bias,
        )
        self.proj_v = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.use_qv_bias,
        )

        self.proj_g = None
        if config.use_attention_gating:
            self.proj_g = nn.Linear(
                config.hidden_size, config.num_attention_heads * config.head_dim
            )

        self.q_norm = norm_layer_factory(config.head_dim, config.qk_norm)
        self.k_norm = norm_layer_factory(config.head_dim, config.qk_norm)

        self.proj_out = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.use_proj_bias,
        )

    def forward(
        self,
        features: torch.Tensor,
        cross_attention_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_attention_rotary_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_scale: float | None = None,
        kv_cache: torch.Tensor | None = None,
        kv_cache_length: int | None = None,
        is_causal: bool = False,
    ):
        config = self.config

        use_cross_attention = cross_attention_features is not None
        if not use_cross_attention:
            cross_attention_features = features
            cross_attention_rotary_embeds = rotary_embeds

        b, n, _ = features.shape

        # b l d -> b l (h dh)
        q = self.proj_q(features)

        g = None
        if self.proj_g is not None:
            g = self.proj_g(features)

        # b s d -> b s (h dh)
        k = self.proj_k(cross_attention_features)
        v = self.proj_v(cross_attention_features)

        # ... (h dh) -> ... h dh
        q, k, v = (
            t.reshape(b, -1, config.num_attention_heads, config.head_dim)
            for t in (q, k, v)
        )
        if g is not None:
            g = g.reshape(b, -1, config.num_attention_heads, config.head_dim)

        # b s h dh -> b h s dh
        q, k, v = (t.transpose(2, 1) for t in (q, k, v))
        if g is not None:
            g = g.transpose(2, 1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if rotary_embeds is not None:
            q = apply_rotary_emb(q, *rotary_embeds)

        if cross_attention_rotary_embeds is not None:
            k = apply_rotary_emb(k, *cross_attention_rotary_embeds)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                # b l s -> b 1 l s
                attention_mask = attention_mask[:, None, :, :]

        if kv_cache is not None:
            assert kv_cache_length is not None
            _, _, s, _ = k.shape

            # 2 b h s d_head
            kv_cache[:, :, :, kv_cache_length : kv_cache_length + s, :] = torch.stack(
                (k, v)
            )
            k, v = kv_cache.unbind(0)

            if kv_cache.requires_grad:
                # cloning is needed to allow inplace modification of kv cache with
                # grads enabled
                k = k.clone()
                v = v.clone()

        features = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            scale=attention_scale,
            is_causal=is_causal,
        )

        if g is not None:
            features = features * F.sigmoid(g)

        # b h l d -> b l h d -> b l (h d)
        features = features.transpose(1, 2).reshape(b, n, -1)
        return self.proj_out(features)


class DiTTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.config = config

        self.attention_pre_norm = norm_layer_factory(
            config.hidden_size, config.pre_norm
        )
        self.adaLN_modulation = nn.Linear(
            config.hidden_size, 6 * config.hidden_size, bias=config.use_proj_bias
        )

        self.attention = Attention(config)

        self.mlp_pre_norm = norm_layer_factory(config.hidden_size, config.pre_norm)
        if config.mlp_mode == "gated":
            self.mlp = GatedMLP(config.hidden_size, config.use_proj_bias)
        elif config.mlp_mode == "vanilla":
            self.mlp = VanillaMLP(config.hidden_size, config.use_proj_bias)
        else:
            raise ValueError(config.mlp_mode)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        projected_condition: torch.Tensor | None = None,
        cross_attention_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_scale: float | None = None,
        rotary_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_attention_rotary_embeds: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: torch.Tensor | None = None,
        kv_cache_length: int | None = None,
    ):
        if projected_condition is None:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(condition).chunk(6, dim=-1)
            )
        else:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                projected_condition
            )

        residual = hidden_states
        hidden_states = self.attention_pre_norm(hidden_states)
        hidden_states = modulate(hidden_states, shift_attn, scale_attn)
        hidden_states = self.attention(
            hidden_states,
            cross_attention_features=cross_attention_hidden_states,
            attention_mask=attention_mask,
            attention_scale=attention_scale,
            rotary_embeds=rotary_embeds,
            kv_cache=kv_cache,
            kv_cache_length=kv_cache_length,
            cross_attention_rotary_embeds=cross_attention_rotary_embeds,
        )
        hidden_states = gate_attn * hidden_states
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_pre_norm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gate_mlp * hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    freqs: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor):
        # ... -> ... d
        t_freq = get_timestep_embedding(t, embedding_dim=self.frequency_embedding_size)
        t_freq = t_freq.to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ViTDenoiserOutput(NamedTuple):
    prediction: torch.Tensor
    layer_hidden_states: torch.Tensor | None = None


class ViTDenoiser(nn.Module):
    def __init__(self, config: ViTDenoiserConfig = ViTDenoiserConfig()):
        super().__init__()
        self.config = config

        self.proj_in = nn.Linear(
            config.input_size,
            config.transformer.hidden_size,
            bias=config.transformer.use_proj_bias,
        )

        self.norm_in_condition = nn.Identity()
        self.proj_in_condition = nn.Identity()
        if config.condition_input_size is not None:
            self.norm_in_condition = norm_layer_factory(
                config.condition_input_size, config.condition_input_norm
            )
            self.proj_in_condition = nn.Linear(
                config.condition_input_size,
                config.transformer.hidden_size,
            )

        self.timestep_emedder = TimestepEmbedder(config.transformer.hidden_size)

        self.null_condition_embedding = nn.Parameter(
            torch.empty(config.transformer.hidden_size)
        )

        self.rotary_embeds = Rope2DPositionEmbedding(
            head_dim=config.transformer.head_dim
        )
        self.blocks = nn.ModuleList(
            DiTTransformerBlock(config.transformer) for _ in range(config.num_blocks)
        )

        self.norm_out = norm_layer_factory(
            config.transformer.hidden_size, config.norm_out
        )
        self.proj_out = nn.Linear(
            config.transformer.hidden_size,
            config.input_size,
            bias=config.transformer.use_proj_bias,
        )

        self.reset_weights_()

    def reset_weights_(
        self, init_std=0.02, mup_width_multiplier: float = 1.0, torch_rng=None
    ):
        self.apply(
            lambda m: simple_init_weights_(m, init_std=init_std, torch_rng=torch_rng)
        )
        init_block_weights_(
            self.blocks,
            init_std=init_std,
            mup_width_multiplier=mup_width_multiplier,
            torch_rng=torch_rng,
        )

        if self.config.should_pin_adaln_projections:
            adaln_modulation_0 = self.blocks[0].adaLN_modulation
            for block in self.blocks:
                block.adaLN_modulation = adaln_modulation_0

        init.zeros_(self.proj_out.weight)
        init.trunc_normal_(self.null_condition_embedding, std=init_std)

        return self

    def forward(
        self,
        patches: torch.Tensor,
        timesteps: torch.Tensor,
        patch_coords: torch.Tensor,
        patch_condition: torch.Tensor | None = None,
        patch_condition_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        kv_cache_length: int | None = None,
        return_layer_indices: list[int] | None = None,
        drop_rate: float = 0.0,
        drop_at_layer: int = 2,
        undrop_at_layer: int = -2,
    ):
        """
        --- Tensors ---
        patches: b l d_patch
        timesteps: b/1 l/1
        patch_coords: b/1 l 2
        patch_condition: b/1 l/1 d_cond
        patch_condition_mask: b/1 l/1
        attention_mask: b/1 l s
        kv_cache: n 2 b h s d_head
        """

        config = self.config
        b, s, d = patches.shape
        device, dtype = patches.device, patches.dtype

        # Resolve negative layer indices (e.g., -2 -> len - 2)
        if undrop_at_layer < 0:
            undrop_at_layer += len(self.blocks)

        # State variables for token dropout
        keep_indices = None
        # Stores (hidden, condition, rotary, mask, projected_condition) from before the drop
        original_inputs: tuple[torch.Tensor, ...] | None = None

        hidden_states = self.proj_in(patches)

        # patch-wise timesteps (b,s) or batch timesteps (b,1)
        condition = self.timestep_emedder(timesteps)

        if patch_condition is not None:
            # patch-wise conditions (b,s,d_cond) -> (b,s,d)
            patch_condition = self.norm_in_condition(patch_condition)
            patch_condition = self.proj_in_condition(patch_condition)

            if patch_condition_mask is not None:
                # Use the patch condition only where the patch_condition_mask is True
                # otherwise add the null condition embedding
                patch_condition = patch_condition * patch_condition_mask[:, :, None]
                patch_condition = (
                    patch_condition
                    + patch_condition_mask[:, :, None].logical_not()
                    * self.null_condition_embedding[None, None, :]
                )

            condition = condition + patch_condition

        else:
            # Assume fully unconditional
            condition = condition + self.null_condition_embedding[None, None, :]

        condition = F.silu(condition)

        projected_condition = None
        if config.should_pin_adaln_projections:
            projected_condition = (
                self.blocks[0].adaLN_modulation(condition).chunk(6, dim=-1)
            )

        rotary_embeds = self.rotary_embeds(patch_coords)

        layer_hidden_states = None
        if return_layer_indices is not None:
            layer_hidden_states = torch.empty(
                len(return_layer_indices),
                b,
                s,
                config.transformer.hidden_size,
                device=device,
                dtype=dtype,
            )

        for i, block in enumerate(self.blocks):
            if i == drop_at_layer and drop_rate > 0.0:
                # Determine how many tokens to keep
                keep_length = int(s * (1 - drop_rate))
                keep_length = round(keep_length / 16) * 16

                # Generate random scores and sort to get indices
                # We use argsort to select the top-k tokens to keep.
                # This ensures tensor shapes remain consistent across the batch (b, keep_len, d).
                drop_scores = torch.rand(b, s, device=device)
                ids_shuffle = torch.argsort(drop_scores, dim=1)
                keep_indices = ids_shuffle[:, :keep_length]

                # Save original state for restoration later
                original_inputs = (
                    hidden_states,
                    condition,
                    rotary_embeds,
                    attention_mask,
                    projected_condition,
                )

                hidden_states = hidden_states.take_along_dim(
                    keep_indices.unsqueeze(-1), 1
                )
                condition = condition.take_along_dim(keep_indices.unsqueeze(-1), 1)
                if projected_condition is not None:
                    projected_condition = tuple(
                        pc.take_along_dim(keep_indices[:, :, None], 1)
                        for pc in projected_condition
                    )
                rotary_embeds = tuple(
                    rot.take_along_dim(keep_indices[:, None, :, None], -2)
                    for rot in rotary_embeds
                )

                # Handle Attention Mask (b, 1, s, s) or (b, 1, l, s)
                # We need to slice both Query (dim 2) and Key (dim 3) axes if it's a square mask
                if attention_mask is not None:
                    # Expand batch dim if needed
                    if attention_mask.shape[0] == 1 and b > 1:
                        attention_mask = attention_mask.expand(
                            b, *attention_mask.shape[1:]
                        )

                    # Gather on Query axis (dim 2) -> (b, 1, keep_len, s)
                    attention_mask = attention_mask.take_along_dim(
                        keep_indices[:, :, None], 1
                    )

                    # Gather on Key axis (dim 3) -> (b, 1, keep_len, keep_len)
                    # Assuming self-attention square mask here
                    attention_mask = attention_mask.take_along_dim(
                        keep_indices[:, None, :], 2
                    )

            block_kv_cache = None
            if kv_cache is not None:
                block_kv_cache = kv_cache[i]

            hidden_states = block(
                hidden_states=hidden_states,
                condition=condition,
                projected_condition=projected_condition,
                attention_mask=attention_mask,
                rotary_embeds=rotary_embeds,
                kv_cache=block_kv_cache,
                kv_cache_length=kv_cache_length,
            )

            if i == undrop_at_layer and keep_indices is not None:
                # Retrieve originals
                (
                    orig_hidden,
                    orig_cond,
                    orig_rot,
                    orig_mask,
                    orig_projected_condition,
                ) = original_inputs

                # Create a canvas from the original hidden states (skip connection for dropped tokens)
                # We scatter the processed tokens back into their original positions.
                # Use clone to ensure we don't mess up if original_inputs are used elsewhere (rare)
                # Expand indices for scatter: (b, keep_len) -> (b, keep_len, d)
                scatter_ind = keep_indices.unsqueeze(-1).expand(
                    -1, -1, hidden_states.shape[-1]
                )
                orig_hidden.scatter(1, scatter_ind, hidden_states)

                # Restore state
                hidden_states = orig_hidden
                condition = orig_cond
                rotary_embeds = orig_rot
                attention_mask = orig_mask
                projected_condition = orig_projected_condition

                # Reset
                keep_indices = None
                original_inputs = None

            if return_layer_indices is not None:
                if i in return_layer_indices:
                    if keep_indices is not None:
                        # If we are currently in a "dropped" state, we must temporarily undrop
                        # to save the full-sized feature map.
                        orig_hidden = original_inputs[0]
                        scatter_ind = keep_indices.unsqueeze(-1).expand(
                            -1, -1, hidden_states.shape[-1]
                        )
                        orig_hidden.scatter(1, scatter_ind, hidden_states)

                        layer_hidden_states[return_layer_indices.index(i)] = orig_hidden
                    else:
                        layer_hidden_states[return_layer_indices.index(i)] = (
                            hidden_states
                        )

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        return ViTDenoiserOutput(hidden_states, layer_hidden_states)
