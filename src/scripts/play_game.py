import pygame

from src.game.configuring_game_engine import GameEngineConfig
from src.game.game_engine import GameEngine

if __name__ == "__main__":
    game_config = GameEngineConfig()
    game = GameEngine(config=game_config)

    clock = pygame.time.Clock()
    running = True

    game.reset_game()

    print("Controls: Left/Right Arrows")

    while running:
        dt_milliseconds = clock.tick(60)
        dt = dt_milliseconds / 1000.0

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

        if done:
            print(f"Game Over! Score: {game.score}")
            game.reset_game()

        game.display_to_screen(depth, seg, flow)

    pygame.quit()
