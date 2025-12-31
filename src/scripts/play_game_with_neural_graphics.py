"""
Play Spokester1 with neural graphics from a
pretrained checkpoint
"""

from dataclasses import dataclass
import time

import pygame
from einops import rearrange, repeat
import numpy as np
import jsonargparse
import torch.nn.functional as F
import torch

from src.net.helpers import generate
from src.net.flow_matching import FlowMatchingHelper
from src.net.model_factory import make_model
from src.game.configuring_game_engine import (
    CameraConfig,
    GameEngineConfig,
    PhysicsConfig,
    RenderConfig,
)
from src.game.game_engine import GameEngine


@dataclass
class GenConfig:
    # The height/width for the denoiser
    height: int = 320
    width: int = 320

    # Upsample before displaying to the pygame window resolution
    view_resolution_mult: int = 2

    # Number of patchlets to keep in the rolling KV cache
    temporal_window_size: int = 3
    # Timestep shifting. This should be a value over 4096
    # Larger values adhere more to the game's depth/flow
    # but are more fuzzy.
    time_shift_dim: int = 65536
    # Number of denoising steps, limited improvement past 20
    num_denoising_steps: int = 12
    # Cache the KV tensors after this denoising step
    # smaller numbers greatly mitigate long-term degredation
    cache_at_step: int | None = 5

    # Small precomputed-mean and std to roughly normalize
    # the game's depth
    # Smaller mean makes everything seem closer
    # Smaller depth std accentuates differences in distance
    # but can cause artifacting and instability
    condition_depth_mean: float = 0.012
    condition_depth_std: float = 0.0075
    # Depth condition gamma scaling:
    # Make this greater than one to increase the sensitivity
    # to distant objects.
    condition_depth_gamma: float = 1.02
    # Small precomputed-mean and std to roughly normalize
    # the game's optical flow
    condition_flow_mean: int = 0
    condition_flow_std: float = 0.1
    # Optical Flow gamma scaling:
    # Make this greater than one to increase the sensitivity to
    # sudden movements. I haven't observed any improvement by tweaking
    # this.
    condition_flow_gamma: float = 1.0

    # Clip the condition input tensors to a low number
    # to prevent overflow and crazy artifacts when cars
    # get too close to the screen
    condition_depth_clip_value: float = 10.0
    condition_flow_clip_value: float = 10.0

    # Add this amount of random noise
    # to the condition tensors
    condition_depth_noise_std: float = 0.5
    condition_flow_noise_std: float = 0.5


def main(
    model_path_or_url: str = "adams-story/spokester1-vit-base",
    gen_conf: GenConfig = GenConfig(),
    device_str: str = "cuda",
    dtype_str: str = "bfloat16",
    # Compilation gives ~20% FPS boost
    should_compile: bool = False,
):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)

    model, run_config_dict = make_model(model_path_or_url, device, dtype)
    patch_duration = run_config_dict["patch_duration"]
    patch_side_length = run_config_dict["patch_side_length"]

    pixel_flow_matching_helper = FlowMatchingHelper(
        time_shift_dim=gen_conf.time_shift_dim
    )

    if should_compile:
        model = torch.compile(model, fullgraph=True, dynamic=False)

    def _run_dummy():
        dummy_patch_condition = torch.zeros(
            1,
            1,
            (gen_conf.height // patch_side_length)
            * (gen_conf.width // patch_side_length),
            patch_duration * patch_side_length**2 * 3,
            device=device,
            dtype=dtype,
        )
        # torch.compile compiles a seperate
        # graph for each q_idx in the temporal window
        kv_cache = None
        kv_cache_length = 0
        for _ in range(gen_conf.temporal_window_size):
            _, kv_cache, kv_cache_length = generate(
                sample_shape=(1, patch_duration, gen_conf.height, gen_conf.width, 3),
                pixel_flow_matching_helper=pixel_flow_matching_helper,
                pixel_denoiser=model,
                patch_condition=dummy_patch_condition,
                patch_duration=patch_duration,
                patch_side_length=patch_side_length,
                temporal_window_size=gen_conf.temporal_window_size,
                num_denoising_steps=gen_conf.num_denoising_steps,
                kv_cache=kv_cache,
                kv_cache_length=kv_cache_length,
                cache_at_step=gen_conf.cache_at_step,
                dtype=dtype,
            )

    # Run the model a couple times to warmup
    # and estimate the FPS

    print("Warming up model...")
    num_warmup_steps = 2
    for _ in range(num_warmup_steps):
        _run_dummy()
    num_est_steps = 4
    start_time = time.time()
    for _ in range(num_est_steps):
        _run_dummy()
    fps = patch_duration * num_est_steps / (time.time() - start_time)

    print("initial FPS", fps)

    # change physics to make the game easier
    game_config = GameEngineConfig(
        render=RenderConfig(
            RENDER_WIDTH=gen_conf.width * gen_conf.view_resolution_mult,
            RENDER_HEIGHT=gen_conf.height * gen_conf.view_resolution_mult,
            IS_HEADLESS=False,
            IS_VIS_MODE=False,
        ),
        camera=CameraConfig(
            FOV=55.0,
        ),
        physics=PhysicsConfig(
            INITIAL_WORLD_SPEED=5.0,
            WORLD_ACCELERATION=0.2,
            PLAYER_TURN_SPEED=5.0,
            PLAYER_COUNTERSTEER_FACTOR=3.5,
        ),
    )
    game = GameEngine(config=game_config)

    running = True
    game.reset_game()
    kv_cache = None
    kv_cache_length = 0

    frame_buffer = []

    while running:
        # Step the game for (patch_duration) times.
        #
        # Display (patch_duration) x frames from the frame_buffer,
        # sleep between concecutive frames
        # based on the estimated FPS
        #
        # For example, if patch_duration==2,
        # this will update the game twice and draw twice with a (1/fps)
        # second sleep between the first and the
        # second draw
        done = False
        buffered_depth = []
        buffered_flow = []
        for i in range(patch_duration):
            dt = 1 / fps

            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 2
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            (depth, seg, flow), reward, done = game.step(action, dt)
            buffered_depth.append(depth)
            buffered_flow.append(flow)
            if frame_buffer:
                frame = frame_buffer.pop(0)
                game.display_to_screen(frame)
                if i + 1 < patch_duration:
                    time.sleep(1 / fps)

        if done:
            print(f"Game Over! Score: {game.score}")
            kv_cache = None
            kv_cache_length = 0
            frame_buffer = []
            game.reset_game()

        if len(buffered_depth) < patch_duration:
            continue

        # Generate a patchlet
        # and save the frames
        # to the frame buffer

        start_time = time.time()

        depth = np.stack(buffered_depth)
        flow = np.stack(buffered_flow)

        depth = torch.from_numpy(depth).to(device, non_blocking=True)
        flow = torch.from_numpy(flow).to(device, non_blocking=True)

        depth = repeat(depth, "n h w -> n c h w", c=1)
        depth = F.interpolate(depth, size=(gen_conf.height, gen_conf.width))
        depth = rearrange(depth, "n 1 h w -> n h w")

        flow = rearrange(flow, "n h w c -> n c h w")
        flow = F.interpolate(flow, size=(gen_conf.height, gen_conf.width))
        flow = rearrange(flow, "n c h w -> n h w c")

        # n h w ... -> b n h w ...
        depth = depth.unsqueeze(0)
        flow = flow.unsqueeze(0)

        # Take only uv flow
        flow = flow[..., :2]

        # Normalize
        depth = depth.sub_(gen_conf.condition_depth_mean).div_(
            gen_conf.condition_depth_std
        )
        flow = flow.sub_(gen_conf.condition_flow_mean).div_(gen_conf.condition_flow_std)
        depth = depth.sign() * depth.abs().pow(gen_conf.condition_depth_gamma)
        flow = flow.sign() * flow.abs().pow(gen_conf.condition_flow_gamma)

        # Clip
        depth = depth.clip(
            -gen_conf.condition_depth_clip_value, gen_conf.condition_depth_clip_value
        )
        flow = flow.clip(
            -gen_conf.condition_flow_clip_value, gen_conf.condition_flow_clip_value
        )

        # Add noise
        depth = depth + torch.rand_like(depth) * gen_conf.condition_depth_noise_std
        flow = flow + torch.rand_like(flow) * gen_conf.condition_flow_noise_std

        condition = torch.cat((depth.unsqueeze(-1), flow), -1)

        condition = condition.to(dtype)
        patch_condition = rearrange(
            condition,
            "b (npn pn) (nph ph) (npw pw) c -> b npn (nph npw) (pn ph pw c)",
            pn=patch_duration,
            ph=patch_side_length,
            pw=patch_side_length,
        )

        # Generate a framelet and update the kv_cache
        # and kv_cache_length
        sample, kv_cache, kv_cache_length = generate(
            sample_shape=(1, patch_duration, gen_conf.height, gen_conf.width, 3),
            pixel_flow_matching_helper=pixel_flow_matching_helper,
            pixel_denoiser=model,
            patch_condition=patch_condition,
            patch_duration=patch_duration,
            patch_side_length=patch_side_length,
            temporal_window_size=gen_conf.temporal_window_size,
            num_denoising_steps=gen_conf.num_denoising_steps,
            kv_cache=kv_cache,
            kv_cache_length=kv_cache_length,
            cache_at_step=gen_conf.cache_at_step,
            dtype=dtype,
        )

        sample = rearrange(sample, "1 n h w c -> n c h w")
        sample = F.interpolate(
            sample,
            scale_factor=gen_conf.view_resolution_mult,
            mode="bilinear",
            antialias=True,
        )
        sample = rearrange(sample, "n c h w -> n h w c")

        sample = sample.add_(1).div_(2).clip_(0, 1).mul_(255).round_().to(torch.uint8)
        sample = sample.cpu().numpy()

        # Save the framelet to the frame buffer
        frame_buffer.extend(list(sample))

        # Update the estimated FPS
        frame_time = (time.time() - start_time) / patch_duration
        beta = 0.9
        fps = (1 - beta) * (1 / frame_time) + beta * fps

        print("FPS", fps, "FRAME TIME", frame_time)


if __name__ == "__main__":
    jsonargparse.CLI(main)
