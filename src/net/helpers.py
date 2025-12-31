import math
from typing import NamedTuple
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from src.net.flow_matching import FlowMatchingHelper
from src.net.net import ViTDenoiser
from src.net.supplemental_net import (
    DepthExtractor,
    DinoEncoder,
    OpticalFlowExtractor,
    REPAProjector,
)
from src.net.torch_utils import unsqueeze_trailing


def _compute_h_lim_w_lim(height, width):
    # h_lim = math.sqrt(height / width)
    # w_lim = math.sqrt(width / height)
    h_lim, w_lim = 1 - 1 / height, 1 - 1 / width
    return h_lim, w_lim


def create_coordinate_grid_2d(
    height: int = 16,
    width: int = 16,
    device=torch.device("cpu"),
):
    dtype = torch.float32

    h_lim, w_lim = _compute_h_lim_w_lim(height, width)

    coordinates = torch.meshgrid(
        [
            torch.linspace(-h_lim, h_lim, height, device=device, dtype=dtype),
            torch.linspace(-w_lim, w_lim, width, device=device, dtype=dtype),
        ],
        indexing="ij",
    )
    coordinates = torch.stack(coordinates, -1)

    return coordinates


def _get_autocast_fn(device, dtype):
    return torch.autocast(device.type, dtype, enabled=dtype != torch.float32)


def extract_real_patch_condition(
    pixel_values: torch.Tensor,
    depth_extractor: DepthExtractor,
    flow_extractor: OpticalFlowExtractor,
    patch_duration: int,
    patch_side_length: int,
    should_augment: bool = False,
    torch_rng=None,
):
    device, dtype = pixel_values.device, pixel_values.dtype

    with _get_autocast_fn(device, dtype):
        # b n h w
        depth = depth_extractor(pixel_values)
        # b n h w 2
        flow = flow_extractor(pixel_values)

    # Simple heuristic statistics
    depth_mean = 2.0
    depth_std = 1.7

    depth = (depth - depth_mean) / depth_std

    flow_mean = 0.0
    flow_std = 2.5
    flow = (flow - flow_mean) / flow_std

    if should_augment:
        depth, flow = _augment_condition(depth, flow, torch_rng=torch_rng)

    condition = torch.cat((depth.unsqueeze(-1), flow), -1)

    patch_condition = rearrange(
        condition,
        "b (npn pn) (nph ph) (npw pw) c -> b npn (nph npw) (pn ph pw c)",
        pn=patch_duration,
        ph=patch_side_length,
        pw=patch_side_length,
    )

    return patch_condition


def extract_synth_patch_condition(
    depth: torch.Tensor,
    flow: torch.Tensor,
    patch_duration: int,
    patch_side_length: int,
    flow_mean: float = 0.0,
    flow_std: float = 0.05,
    depth_mean: float = 0.008,
    depth_std: float = 0.008,
    should_augment: bool = False,
    torch_rng=None,
):
    """
    depth: b n h w
    flow: b n h w uv

    depth is real coordinate-space depth
    flow is real coordinate-space optical flow
    """

    # Take only uv flow
    flow = flow[..., :2]

    depth = (depth - depth_mean) / depth_std
    flow = (flow - flow_mean) / flow_std

    if should_augment:
        depth, flow = _augment_condition(depth, flow, torch_rng=torch_rng)

    condition = torch.cat((depth.unsqueeze(-1), flow), -1)

    patch_condition = rearrange(
        condition,
        "b (npn pn) (nph ph) (npw pw) c -> b npn (nph npw) (pn ph pw c)",
        pn=patch_duration,
        ph=patch_side_length,
        pw=patch_side_length,
    )

    return patch_condition


def _blur_pixels(pixel_values, sigma=0.5, kernel_size=3):
    device, dtype = pixel_values.device, pixel_values.dtype
    channels = pixel_values.shape[1]

    x = torch.linspace(
        -(kernel_size // 2), kernel_size // 2, kernel_size, device=device, dtype=dtype
    )
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_final = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    pad_size = kernel_size // 2

    padded_values = F.pad(
        pixel_values, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )

    blurred = F.conv2d(padded_values, kernel_final, groups=channels, padding=0)

    return blurred


def _rand_a_b(a, b, torch_rng: torch.Generator | None = None, shape=(1,)):
    u = torch.rand(
        shape,
        generator=torch_rng,
        device=torch_rng.device if torch_rng is not None else torch.device("cpu"),
    )
    u = u * (b - a) + a
    return u


def _augment_condition(
    depth: torch.Tensor,
    flow: torch.Tensor,
    torch_rng: torch.Generator | None = None,
):
    b, n, h, w, _ = flow.shape

    device = depth.device

    u = torch.rand((1,), device=device, generator=torch_rng).cpu().item()
    pixelize_rate = 0.9
    should_pixelize_and_blur = u <= pixelize_rate
    if should_pixelize_and_blur:
        depth = repeat(depth, "b n h w -> (b n) c h w", c=1)
        flow = rearrange(flow, "b n h w uv -> (b n) uv h w")

        # Downsample
        min_scale, max_scale = 0.05, 0.5
        scale = _rand_a_b(min_scale, max_scale, torch_rng).cpu().item()
        depth = F.interpolate(depth, scale_factor=(scale, scale), mode="nearest-exact")
        scale = _rand_a_b(min_scale, max_scale, torch_rng).cpu().item()
        flow = F.interpolate(flow, scale_factor=(scale, scale), mode="nearest-exact")

        # Blur downsampled
        min_sigma, max_sigma = 0.5, 4.0
        sigma = _rand_a_b(min_sigma, max_sigma, torch_rng).cpu().item()
        depth = _blur_pixels(depth, sigma=sigma, kernel_size=3)

        sigma = _rand_a_b(min_sigma, max_sigma, torch_rng).cpu().item()
        flow = _blur_pixels(flow, sigma=sigma, kernel_size=3)

        # Upsample
        depth = F.interpolate(depth, size=(h, w), mode="nearest-exact")
        flow = F.interpolate(flow, size=(h, w), mode="nearest-exact")

        depth = rearrange(depth, "(b n) 1 h w -> b n h w", b=b)
        flow = rearrange(flow, "(b n) uv h w -> b n h w uv", b=b)

    u = torch.rand((1,), device=device, generator=torch_rng).cpu().item()
    posterize_rate = 0.9
    should_posterize = u <= posterize_rate
    if should_posterize:
        min_posterize_factor, max_posterize_factor = 5, 200
        posterize_factor = (
            _rand_a_b(min_posterize_factor, max_posterize_factor, torch_rng)
            .cpu()
            .item()
        )
        depth = depth.mul(posterize_factor).round().div(posterize_factor)

        posterize_factor = (
            _rand_a_b(min_posterize_factor, max_posterize_factor, torch_rng)
            .cpu()
            .item()
        )
        flow = flow.mul(posterize_factor).round().div(posterize_factor)

    # Add random noise
    base_noise_gamma = 1
    noise_std = torch.rand((b,), device=device, generator=torch_rng) * base_noise_gamma
    noise_std = unsqueeze_trailing(noise_std, depth)
    depth = depth + torch.randn_like(depth) * noise_std

    noise_std = torch.rand((b,), device=device, generator=torch_rng) * base_noise_gamma
    noise_std = unsqueeze_trailing(noise_std, flow)
    flow = flow + torch.randn_like(flow) * noise_std

    # Clip to avoid huge values
    min_clip_value, max_clip_value = 5, 20
    clip_value = _rand_a_b(min_clip_value, max_clip_value, torch_rng).cpu().item()
    depth = depth.clip(-clip_value, clip_value)
    clip_value = _rand_a_b(min_clip_value, max_clip_value, torch_rng).cpu().item()
    flow = flow.clip(-clip_value, clip_value)

    return depth, flow


def _get_frame_temporal_attention_mask(
    npn_q: int,
    npn_kv: int,
    num_spatial_tokens: int,
    start_q_temporal_idx: int = 0,
    start_kv_idx: int = 0,
    temporal_window_size: int | None = None,
    device=torch.device("cpu"),
):
    """Generates framelet-temporal attention mask"""

    temporal_ids_q = torch.arange(npn_q, device=device)
    temporal_ids_q = repeat(temporal_ids_q, "npn -> (npn spa)", spa=num_spatial_tokens)
    temporal_ids_q = temporal_ids_q.unsqueeze(1) + start_q_temporal_idx

    temporal_ids_kv = torch.arange(npn_kv, device=device)
    temporal_ids_kv = repeat(
        temporal_ids_kv, "npn -> (npn spa)", spa=num_spatial_tokens
    )
    temporal_ids_kv = temporal_ids_kv.unsqueeze(0) + start_kv_idx

    mask = temporal_ids_q >= temporal_ids_kv

    if temporal_window_size is not None:
        mask = mask & ((temporal_ids_q - temporal_ids_kv) <= temporal_window_size)

    mask = repeat(mask, "l s -> b l s", b=1)
    return mask


def compute_loss(
    *,
    pixel_flow_matching_helper: FlowMatchingHelper,
    pixel_denoiser: ViTDenoiser,
    pixel_denoiser_projector: REPAProjector,
    dino_encoder: DinoEncoder,
    depth_extractor: DepthExtractor,
    flow_extractor: OpticalFlowExtractor,
    pixel_values: torch.Tensor,
    uncondition_rate: float = 0.0,
    temporal_window_size: int | None = None,
    patch_duration: int = 2,
    patch_side_length: int = 16,
    repa_alignment_depth: int = 6,
    pixel_denoiser_token_drop_rate: float = 0.0,
    pixel_denoiser_drop_at_layer: int = 2,
    pixel_denoiser_undrop_at_layer: int = 2,
    repa_weight: float = 0.5,
    should_jitter_positions: bool = True,
    should_augment_patch_condition: bool = True,
    torch_rng: torch.Generator | None = None,
):
    b, n, h, w, c = pixel_values.shape

    device, dtype = pixel_values.device, pixel_values.dtype

    npn = n // patch_duration
    nph = h // patch_side_length
    npw = w // patch_side_length

    loss_dict = dict()
    extra_dict = dict()

    # b n h w c -> b npn nph_npw (pn ph pw)
    patch_condition = extract_real_patch_condition(
        pixel_values=pixel_values,
        depth_extractor=depth_extractor,
        flow_extractor=flow_extractor,
        patch_duration=patch_duration,
        patch_side_length=patch_side_length,
        should_augment=should_augment_patch_condition,
        torch_rng=torch_rng,
    )

    extra_dict["real_patch_condition"] = patch_condition

    real_patches = rearrange(
        pixel_values,
        "b (npn pn) (nph ph) (npw pw) c -> b npn (nph npw) (pn ph pw c)",
        pn=patch_duration,
        ph=patch_side_length,
        pw=patch_side_length,
    )

    # Add noise to patches
    timesteps, x_t, v_t, x_0 = pixel_flow_matching_helper.forward_noise(
        real_patches, timesteps_shape=(b, npn), torch_rng=torch_rng
    )

    denoiser_attention_mask = _get_frame_temporal_attention_mask(
        npn_q=npn,
        npn_kv=npn,
        num_spatial_tokens=nph * npw,
        device=device,
        temporal_window_size=temporal_window_size,
    )
    return_layer_indices = [repa_alignment_depth]

    # Prepare denoiser inputs
    x_t = rearrange(x_t, "b npn nph_npw d_patch -> b (npn nph_npw) d_patch")
    timesteps = repeat(timesteps, "b npn -> b (npn nph_npw)", nph_npw=nph * npw)
    patch_coords = create_coordinate_grid_2d(nph, npw, device)
    patch_coords = repeat(
        patch_coords, "nph npw nd -> b (npn nph npw) nd", b=b, npn=npn
    )

    if should_jitter_positions:
        jitter_min = 0.5
        jitter_max = 2.0
        jitter = (
            torch.rand(b, 1, 1, device=device) * (jitter_max - jitter_min) + jitter_min
        )
        patch_coords = patch_coords * jitter

    patch_condition = rearrange(
        patch_condition, "b npn nph_npw d_cond -> b (npn nph_npw) d_cond"
    )

    patch_condition_mask = (
        torch.rand(b, 1, device=device, dtype=dtype, generator=torch_rng)
        > uncondition_rate
    )

    with _get_autocast_fn(device, dtype):
        model_prediction, student_features = pixel_denoiser(
            patches=x_t,
            timesteps=timesteps,
            patch_coords=patch_coords,
            patch_condition_mask=patch_condition_mask,
            patch_condition=patch_condition,
            attention_mask=denoiser_attention_mask,
            return_layer_indices=return_layer_indices,
            drop_rate=pixel_denoiser_token_drop_rate,
            drop_at_layer=pixel_denoiser_drop_at_layer,
            undrop_at_layer=pixel_denoiser_undrop_at_layer,
        )

    # b n h w c -> b n nph_npw d
    with _get_autocast_fn(device, dtype):
        _, teacher_features = dino_encoder(pixel_values)

    # Spatial norm
    teacher_features = (teacher_features - teacher_features.mean(-2, keepdim=True)) / (
        teacher_features.std(-2, keepdim=True) + 1e-6
    )

    # Patchlet averaging
    teacher_features = reduce(
        teacher_features,
        "b (npn pn) nph_npw d_feat -> b npn nph_npw d_feat",
        "mean",
        npn=npn,
    )
    extra_dict["teacher_features"] = teacher_features

    # Project student features to obtain REPA loss
    student_features = student_features[0]
    student_features = rearrange(
        student_features,
        "b (npn nph npw) d_hidden -> b d_hidden npn nph npw",
        npn=npn,
        nph=nph,
        npw=npw,
    )
    student_features = pixel_denoiser_projector(student_features)
    student_features = rearrange(
        student_features, "b d_feat npn nph npw -> b npn (nph npw) d_feat"
    )
    loss_dict["pixel_denoiser_repa_loss"] = -F.cosine_similarity(
        student_features, teacher_features, dim=-1
    ).mean()

    v_t_hat = pixel_flow_matching_helper.compute_velocity(
        x_t, timesteps, model_prediction=model_prediction
    )

    loss_dict["real_denoising_loss"] = F.mse_loss(
        v_t_hat.view(-1).float(), v_t.view(-1).float()
    )

    loss_dict["total_loss"] = (
        loss_dict["pixel_denoiser_repa_loss"] * repa_weight
        + loss_dict["real_denoising_loss"]
    )

    return loss_dict, extra_dict


class GenerationOutput(NamedTuple):
    sample: torch.Tensor
    kv_cache: torch.Tensor
    kv_cache_length: int


def generate_autoregressive(
    *,
    sample_shape: tuple[int, int, int, int, int] = (1, 16, 256, 256, 3),
    generation_chunk_size: int = 1,
    pixel_flow_matching_helper: FlowMatchingHelper,
    pixel_denoiser: ViTDenoiser,
    patch_condition: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    patch_duration: int = 2,
    patch_side_length: int = 16,
    temporal_window_size: int = 16,
    num_denoising_steps: int = 50,
    cache_at_step: int | None = None,
    torch_rng: torch.Generator | None = None,
    reshape_out: bool = True,
    dtype=torch.float32,
):
    b, n, h, w, c = sample_shape

    npn = n // patch_duration
    nph = h // patch_side_length
    npw = w // patch_side_length

    num_ar_steps = math.ceil(npn / generation_chunk_size)

    framelets = []
    kv_cache = None
    kv_cache_length = 0
    for i in range(num_ar_steps):
        npn_idx = i * generation_chunk_size
        query_length_npn = min(npn - npn_idx, generation_chunk_size)

        patch_condition_q = None
        if patch_condition is not None:
            patch_condition_q = patch_condition[:, npn_idx : npn_idx + query_length_npn]

        noise_q = None
        if noise is not None:
            noise_q = noise[:, npn_idx : npn_idx + query_length_npn]

        framelet, kv_cache, kv_cache_length = generate(
            sample_shape=(b, query_length_npn * patch_duration, h, w, c),
            pixel_flow_matching_helper=pixel_flow_matching_helper,
            pixel_denoiser=pixel_denoiser,
            patch_condition=patch_condition_q,
            noise=noise_q,
            patch_duration=patch_duration,
            patch_side_length=patch_side_length,
            temporal_window_size=temporal_window_size,
            num_denoising_steps=num_denoising_steps,
            cache_at_step=cache_at_step,
            reshape_out=False,
            kv_cache=kv_cache,
            kv_cache_length=kv_cache_length,
            dtype=dtype,
            torch_rng=torch_rng,
        )

        framelets.append(framelet)

        i += generation_chunk_size

    framelets = torch.cat(framelets, 1)

    if reshape_out:
        framelets = rearrange(
            framelets,
            "b npn (nph npw) (pn ph pw c) -> b (npn pn) (nph ph) (npw pw) c",
            nph=nph,
            npw=npw,
            pn=patch_duration,
            ph=patch_side_length,
            pw=patch_side_length,
        )

    return GenerationOutput(framelets, kv_cache, kv_cache_length)


def generate(
    *,
    sample_shape: tuple[int, int, int, int, int] = (1, 16, 256, 256, 3),
    pixel_flow_matching_helper: FlowMatchingHelper,
    pixel_denoiser: ViTDenoiser,
    noise: torch.Tensor | None = None,
    patch_condition: torch.Tensor | None = None,
    patch_duration: int = 2,
    patch_side_length: int = 16,
    temporal_window_size: int = 16,
    num_denoising_steps: int = 50,
    torch_rng: torch.Generator | None = None,
    kv_cache: torch.Tensor | None = None,
    kv_cache_length: int = 0,
    cache_at_step: int | None = None,
    reshape_out: bool = True,
    dtype=torch.float32,
):
    """
    Generate some pixel values,
    conditioned on patch_condition

    pass previous kv_cache and kv_cache_length returned by this function
    """
    device = next(iter(pixel_denoiser.parameters())).device

    b, n, h, w, c = sample_shape
    npn_q = n // patch_duration
    nph = h // patch_side_length
    npw = w // patch_side_length
    num_spatial_tokens = nph * npw
    patch_dim = patch_duration * patch_side_length**2 * c

    if patch_condition is not None:
        assert patch_condition.shape == (
            b,
            npn_q,
            num_spatial_tokens,
            pixel_denoiser.config.condition_input_size,
        )

    # Prepare noise
    noise_shape = (
        b,
        npn_q,
        nph * npw,
        patch_dim,
    )
    if noise is None:
        noise = torch.randn(
            noise_shape,
            device=device,
            dtype=dtype,
            generator=torch_rng,
        )
    assert noise.shape == noise_shape

    patch_coords = create_coordinate_grid_2d(nph, npw, device)
    patch_coords = repeat(
        patch_coords, "nph npw nd -> b (npn nph npw) nd", b=1, npn=npn_q
    )

    num_query_tokens = npn_q * num_spatial_tokens

    # Prepare KV cache
    num_kv_tokens = max(temporal_window_size * num_spatial_tokens, num_query_tokens)
    kv_cache_shape = (
        pixel_denoiser.config.num_blocks,
        2,
        b,
        pixel_denoiser.config.transformer.num_attention_heads,
        num_kv_tokens,
        pixel_denoiser.config.transformer.head_dim,
    )
    if kv_cache is None:
        kv_cache = torch.zeros(
            kv_cache_shape,
            device=device,
            dtype=dtype,
        )
    assert kv_cache.shape == kv_cache_shape

    # Prepare KV offset
    if kv_cache_length + num_query_tokens > num_kv_tokens:
        # If the new queries do not fit in the kv cache
        # roll back the KV cache so that the new query fits
        num_tokens_to_rollback = num_query_tokens + kv_cache_length - num_kv_tokens
        kv_cache = kv_cache.roll(-num_tokens_to_rollback, -2)
        kv_cache_length -= num_tokens_to_rollback

    start_q_temporal_idx = kv_cache_length // num_spatial_tokens

    # (npn_q nph npw) (npn_kv nph npw)
    denoiser_attention_mask = _get_frame_temporal_attention_mask(
        npn_q=npn_q,
        npn_kv=num_kv_tokens // num_spatial_tokens,
        num_spatial_tokens=num_spatial_tokens,
        start_q_temporal_idx=start_q_temporal_idx,
        temporal_window_size=temporal_window_size,
        start_kv_idx=0,
        device=device,
    )

    if patch_condition is not None:
        patch_condition = rearrange(
            patch_condition, "b npn nph_npw d_cond -> b (npn nph_npw) d_cond"
        )

    kv_cache_intermediate = None

    # Denoise, with gradients disabled except for the final
    # denoising step
    _denoising_step = 0

    def denoise(x_t, timestep):
        # hacky way to track the denoising step
        nonlocal _denoising_step
        is_last_timestep = _denoising_step == num_denoising_steps - 1

        x_t = rearrange(x_t, "b npn nph_npw d_patch -> b (npn nph_npw) d_patch")
        timesteps = repeat(
            timestep, "-> b (npn nph_npw)", b=1, npn=npn_q, nph_npw=num_spatial_tokens
        )

        with torch.set_grad_enabled(is_last_timestep):
            with _get_autocast_fn(device, dtype):
                model_prediction, _ = pixel_denoiser(
                    patches=x_t,
                    timesteps=timesteps,
                    patch_coords=patch_coords,
                    patch_condition=patch_condition,
                    attention_mask=denoiser_attention_mask,
                    kv_cache=kv_cache,
                    kv_cache_length=kv_cache_length,
                )

        if cache_at_step is not None:
            if cache_at_step > num_denoising_steps:
                raise ValueError(
                    f"cache_at_step {cache_at_step} must be < num_denoising_steps {num_denoising_steps}"
                )
            if _denoising_step == cache_at_step:
                nonlocal kv_cache_intermediate
                kv_cache_intermediate = kv_cache[
                    ..., kv_cache_length : kv_cache_length + num_query_tokens, :
                ].clone()

        model_prediction = rearrange(
            model_prediction,
            "b (npn nph_npw) d_patch -> b npn nph_npw d_patch",
            npn=npn_q,
        )

        _denoising_step += 1

        return model_prediction

    sample = pixel_flow_matching_helper.sample_euler(
        noise=noise, denoiser=denoise, num_steps=num_denoising_steps
    )

    if kv_cache_intermediate is not None:
        kv_cache[..., kv_cache_length : kv_cache_length + num_query_tokens, :] = (
            kv_cache_intermediate
        )

    if reshape_out:
        sample = rearrange(
            sample,
            "b npn (nph npw) (pn ph pw c) -> b (npn pn) (nph ph) (npw pw) c",
            nph=nph,
            npw=npw,
            pn=patch_duration,
            ph=patch_side_length,
            pw=patch_side_length,
        )

    # Update the kv_cache_length
    kv_cache_length += npn_q * num_spatial_tokens

    return GenerationOutput(sample, kv_cache, kv_cache_length)
