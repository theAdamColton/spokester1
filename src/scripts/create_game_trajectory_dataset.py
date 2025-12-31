import jsonargparse
import pygame

from src.game.configuring_game_engine import GameEngineConfig, RenderConfig
from src.game.game_engine import GameEngine
from src.game.auto_play import auto_play


def main(
    game_config: GameEngineConfig = GameEngineConfig(),
    height: int = 256,
    width: int = 256,
    max_trajectory_length: int = 100000,
    fps: int = 24,
):
    game_config = GameEngineConfig(
        render=RenderConfig(IS_HEADLESS=False, RENDER_HEIGHT=height, RENDER_WIDTH=width)
    )
    game = GameEngine(config=game_config)

    print("Controls: Escape to quit")

    while True:
        game.reset_game()

        def callback(*args, **kwargs):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                elif event.key == pygame.K_ESCAPE:
                    break

        auto_play(
            game,
            max_trajectory_length=max_trajectory_length,
            fps=fps,
            callback_fn=callback,
        )

    pygame.quit()


if __name__ == "__main__":
    jsonargparse.CLI(main)
