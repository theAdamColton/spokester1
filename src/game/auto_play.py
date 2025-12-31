"""
Automatically play the game with a simple agent that attempts to dodge cars
"""

from typing import Callable

import numpy as np
import pygame

from src.game.configuring_game_engine import GameEngineConfig, RenderConfig
from src.game.game_engine import GameEngine


def auto_play(
    game: GameEngine,
    max_trajectory_length: int = 100000,
    fps: float = 1 / 24,
    callback_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], None] | None = None,
):
    is_headless = game.config.render.IS_HEADLESS

    clock = None
    if not is_headless:
        clock = pygame.time.Clock()

    for step in range(max_trajectory_length):
        dt = 1 / fps
        if clock is not None:
            dt = clock.tick(fps) / 1000.0

        # The agent looks at the closest upcoming obstacle by z distance.
        # It picks any lane that takes it away from the obstacle.
        closest_z = 200.0
        closest_obstacle = None

        for obstacle in game.obstacles:
            obs_z = obstacle["z"]

            if 0.0 < obs_z < closest_z:
                closest_z = obs_z
                closest_obstacle = obstacle

        action = 0
        if closest_obstacle is not None:
            obs_x = closest_obstacle["x"]
            lane_x = game.lanes[game.player_lane_idx]
            if abs(obs_x - lane_x) < game.config.physics.PLAYER_COLLISION_THRESH:
                other_lanes = [
                    i for i in range(len(game.lanes)) if i != game.player_lane_idx
                ]

                best_lane_idx = -1
                for target_idx in other_lanes:
                    target_x = game.lanes[target_idx]

                    if abs(obs_x - target_x) >= game.config.physics.LANE_WIDTH:
                        best_lane_idx = target_idx

                if best_lane_idx != -1:
                    if best_lane_idx < game.player_lane_idx:
                        action = 1  # Move Left
                    elif best_lane_idx > game.player_lane_idx:
                        action = 2  # Move Right

        (obs, reward, done) = game.step(action, dt)
        depth, seg, flow = obs

        if not is_headless:
            game.display_to_screen(depth, seg, flow)

        if callback_fn is not None:
            callback_fn(depth, seg, flow)

        if done:
            return


def auto_playthrough(
    seed=None,
    height: int = 320,
    width: int = 320,
    max_length: int = 512,
    fps: float = 24.0,
):
    game_config = GameEngineConfig(
        render=RenderConfig(IS_HEADLESS=True, RENDER_HEIGHT=height, RENDER_WIDTH=width)
    )
    game = GameEngine(config=game_config)
    game.reset_game(seed)

    depths = []
    segs = []
    flows = []

    def callback(depth_np: np.ndarray, seg_np: np.ndarray, flow_np: np.ndarray):
        depths.append(depth_np)
        segs.append(seg_np)
        flows.append(flow_np)

    auto_play(
        game,
        max_trajectory_length=max_length,
        fps=fps,
        callback_fn=callback,
    )

    depths = np.stack(depths)
    segs = np.stack(segs)
    flows = np.stack(flows)

    return depths, segs, flows
