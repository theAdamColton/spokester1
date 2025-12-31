import math
from dataclasses import dataclass
import random
from pathlib import Path

from einops import rearrange, repeat
from tqdm import tqdm
import numpy as np
import jsonargparse
import torch
import torch.nn.functional as F
import av

from src.net.helpers import generate
from src.net.flow_matching import FlowMatchingHelper
from src.net.model_factory import make_model
from src.net.net import ViTDenoiser
from src.game.configuring_game_engine import (
    CameraConfig,
    GameEngineConfig,
    RenderConfig,
)
from src.game.auto_play import auto_play
from src.game.game_engine import GameEngine


def _blur_pixels(pixel_values, sigma=0.5, kernel_size=3):
    device, dtype = pixel_values.device, pixel_values.dtype

    _, channels, _, _ = pixel_values.shape

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


def rand_a_b(a=0.0, b=1.0):
    return random.random() * (b - a) + a


def rand_logu_a_b(a=0.0, b=1.0):
    a = math.log(a)
    b = math.log(b)
    u = rand_a_b(a, b)
    return math.exp(u)


@dataclass
class GenConfig:
    temporal_window_size: int = 4
    time_shift_dim: int = 32768
    num_denoising_steps: int = 15
    cache_at_step: int | None = 10
    condition_clip_value: float = 12.0

    condition_depth_mean: float = 0.008
    condition_depth_std: float = 0.008
    condition_flow_mean: int = 0
    condition_flow_std: float = 0.04

    condition_depth_gamma: float = 1.0
    condition_flow_gamma: float = 1.0

    condition_depth_rescale_factor: float = 1.0
    condition_flow_rescale_factor: float = 1.0

    condition_depth_blur_sigma: float = 1.0
    condition_flow_blur_sigma: float = 1.0

    condition_depth_noise_std: float = 0.25
    condition_flow_noise_std: float = 0.25

    camera_fov: int = 45
    camera_height: float = 1.7
    camera_distance: float = 4.7


def _generate(
    *,
    c: GenConfig = GenConfig(),
    model: ViTDenoiser,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    save_path=None,
    height=320,
    width=320,
    fps=24,
    max_duration=256,
    patch_duration=2,
    patch_side_length=16,
    seed=42,
):
    output_container = av.open(save_path, "w")
    output_video_stream = output_container.add_stream(
        "libx264", rate=24, options={"crf": "18"}
    )
    output_video_stream.height = height * 2
    output_video_stream.width = width * 2

    def write_frames(frames):
        # frames: n h w c, np uint8
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, channel_last=True)
            packet = output_video_stream.encode(frame)
            output_container.mux(packet)

    pixel_flow_matching_helper = FlowMatchingHelper(time_shift_dim=c.time_shift_dim)
    torch_rng = torch.Generator(device).manual_seed(seed)
    kv_cache = None
    kv_cache_length = 0

    def generate_patchlet(depth, flow):
        depth = torch.from_numpy(depth).to(device, non_blocking=True)
        flow = torch.from_numpy(flow).to(device, non_blocking=True)

        # n h w ... -> b n h w ...
        depth = depth.unsqueeze(0)
        flow = flow.unsqueeze(0)

        # Take only uv flow
        flow = flow[..., :2]

        # Normalize
        depth = depth.sub_(c.condition_depth_mean).div_(c.condition_depth_std)
        flow = flow.sub_(c.condition_flow_mean).div_(c.condition_flow_std)

        # Downscale
        depth = repeat(depth, "b n h w -> (b n) c h w", c=1)
        flow = rearrange(flow, "b n h w c -> (b n) c h w")
        depth = F.interpolate(
            depth, scale_factor=c.condition_depth_rescale_factor, mode="nearest-exact"
        )
        flow = F.interpolate(
            flow, scale_factor=c.condition_flow_rescale_factor, mode="nearest-exact"
        )

        # Blur
        depth = _blur_pixels(depth, c.condition_depth_blur_sigma)
        flow = _blur_pixels(flow, c.condition_flow_blur_sigma)

        # Upscale
        depth = F.interpolate(depth, size=(height, width), mode="nearest-exact")
        flow = F.interpolate(flow, size=(height, width), mode="nearest-exact")

        depth = rearrange(depth, "(b n) 1 h w -> b n h w", n=patch_duration)
        flow = rearrange(flow, "(b n) c h w -> b n h w c", n=patch_duration)

        # Power transform
        depth = depth.sign() * depth.abs() ** c.condition_depth_gamma
        flow = flow.sign() * flow.abs() ** c.condition_flow_gamma

        # Add noise
        depth = depth + torch.rand_like(depth) * c.condition_depth_noise_std
        flow = flow + torch.rand_like(flow) * c.condition_flow_noise_std

        condition = torch.cat((depth.unsqueeze(-1), flow), -1)

        condition = condition.clip(-c.condition_clip_value, c.condition_clip_value)
        condition = condition.to(dtype)

        patch_condition = rearrange(
            condition,
            "b (npn pn) (nph ph) (npw pw) c -> b npn (nph npw) (pn ph pw c)",
            pn=patch_duration,
            ph=patch_side_length,
            pw=patch_side_length,
        )

        nonlocal kv_cache
        nonlocal kv_cache_length
        sample, kv_cache, kv_cache_length = generate(
            sample_shape=(1, patch_duration, height, width, 3),
            pixel_flow_matching_helper=pixel_flow_matching_helper,
            pixel_denoiser=model,
            patch_condition=patch_condition,
            patch_duration=patch_duration,
            patch_side_length=patch_side_length,
            temporal_window_size=c.temporal_window_size,
            num_denoising_steps=c.num_denoising_steps,
            torch_rng=torch_rng,
            kv_cache=kv_cache,
            kv_cache_length=kv_cache_length,
            cache_at_step=c.cache_at_step,
            dtype=dtype,
        )

        sample = sample.squeeze(0)
        sample = sample.add_(1).div_(2).clip_(0, 1).mul_(255).round_().to(torch.uint8)
        sample = sample.cpu().numpy()

        # n h w c
        return sample, depth, flow

    # Setup game engine
    game_config = GameEngineConfig(
        render=RenderConfig(IS_HEADLESS=True, RENDER_HEIGHT=height, RENDER_WIDTH=width),
        camera=CameraConfig(
            FOV=c.camera_fov,
            FOLLOW_HEIGHT=c.camera_height,
            FOLLOW_DISTANCE=c.camera_distance,
        ),
    )
    game = GameEngine(config=game_config)
    game.reset_game(seed)

    prog_bar = tqdm(total=max_duration, desc="generating...")
    depth_frames = []
    seg_frames = []
    flow_frames = []

    def game_observation_callback(depth, seg, flow):
        depth_frames.append(depth)
        seg_frames.append(seg)
        flow_frames.append(flow)

        if len(depth_frames) >= patch_duration:
            depth = np.stack(depth_frames)
            seg = np.stack(seg_frames)
            flow = np.stack(flow_frames)
            depth_frames.clear()
            seg_frames.clear()
            flow_frames.clear()

            sample, depth, flow = generate_patchlet(depth, flow)

            # Make a 2x2 grid of videos
            # with
            #  -------------
            # | sample, seg |
            # | depth, flow |
            #  -------------
            depth = depth * 0.4 + 0.63
            depth = (
                depth.squeeze(0)
                .clip(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            depth = repeat(depth, "n h w -> n h w c", c=3)

            flow = flow * 1.0 + 0.5
            flow = (
                flow.squeeze(0)
                .clip(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            flow = np.concat(
                [flow, np.zeros((*flow.shape[:-1], 1), dtype=np.uint8)], axis=-1
            )

            vis = np.stack((sample, seg, depth, flow), axis=0)
            vis = rearrange(vis, "(nh nw) n h w c -> n (nh h) (nw w) c", nh=2, nw=2)
            write_frames(vis)

        prog_bar.update(1)

    # Run game and autoregressively generate frames
    auto_play(
        game,
        max_trajectory_length=max_duration,
        fps=fps,
        callback_fn=game_observation_callback,
    )

    prog_bar.close()

    # Cleanup output stream
    out_packet = output_video_stream.encode(None)
    output_container.mux(out_packet)
    output_container.close()

    print("Wrote", save_path)


@torch.inference_mode()
def main(
    model_path_or_url: str = "adams-story/spokester1-vit-base",
    output_path: Path = Path("runs") / "samples",
    height: int = 320,
    width: int = 320,
    fps: float = 24.0,
    max_duration: int = 512,
    device_str: str = "cuda",
    dtype_str: str = "bfloat16",
    should_compile: bool = False,
    seed: int = 42,
):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)

    model, run_config_dict = make_model(model_path_or_url, device, dtype)
    patch_duration = run_config_dict["patch_duration"]
    patch_side_length = run_config_dict["patch_side_length"]

    if should_compile:
        model = torch.compile(model)

    output_path.mkdir(parents=True, exist_ok=True)

    save_path = output_path / "sample.mp4"
    c = GenConfig(
        time_shift_dim=height * width * patch_duration * 3,
    )
    _generate(
        c=c,
        model=model,
        device=device,
        dtype=dtype,
        save_path=save_path,
        height=height,
        width=width,
        fps=fps,
        max_duration=max_duration,
        patch_duration=patch_duration,
        patch_side_length=patch_side_length,
        seed=seed,
    )


if __name__ == "__main__":
    jsonargparse.CLI(main)
