import gymnasium as gym
import numpy as np

from game import Game


class PlatformerEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 60,
    }

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps = 0

        self.action_space = gym.spaces.Discrete(6)

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -2.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    2.0,
                    2.0,
                    1.0,
                    2.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.game = Game(
            headless=(render_mode != "human")
        )

        self.previous_x = 0


    def _get_observation(self):

        player = self.game.player.sprite
        enemy = self.game.enemies.sprites()[0]
        hazard = self.game.hazards.sprites()[0]
        goal = self.game.goal.sprite

        observation = np.array(
            [
                player.rect.x / self.game.WIDTH,
                player.rect.y / self.game.HEIGHT,

                player.direction.x,
                player.direction.y / 20.0,

                float(player.on_ground),

                enemy.rect.x / self.game.WIDTH,
                enemy.rect.y / self.game.HEIGHT,

                hazard.rect.x / self.game.WIDTH,
                hazard.rect.y / self.game.HEIGHT,

                goal.rect.x / self.game.WIDTH,
                goal.rect.y / self.game.HEIGHT,
            ],
            dtype=np.float32,
        )

        return observation


    def _get_info(self):

        player = self.game.player.sprite
        goal = self.game.goal.sprite

        return {
            "player_x": player.rect.x,
            "player_y": player.rect.y,
            "goal_x": goal.rect.x,
            "distance_to_goal": goal.rect.x - player.rect.x,
            "won": self.game.won,
            "steps": self.steps,
        }


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if seed is None:
            seed = 0

        self.game.reset(seed=seed)

        self.steps = 0

        player = self.game.player.sprite
        self.previous_x = player.rect.x

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.game.draw()

        return observation, info


    def step(self, action):

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. "
                "Action must be an integer from 0 to 5."
            )

        self.game.take_action(action)
        self.game.update()

        self.steps += 1

        player = self.game.player.sprite

        progress = player.rect.x - self.previous_x

        reward = progress * 0.01

        terminated = False

        if not self.game.running:

            terminated = True

            if self.game.won:
                reward += 10.0
            else:
                reward -= 10.0

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        self.previous_x = player.rect.x

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.game.draw()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    def render(self):

        if self.render_mode == "human":
            self.game.draw()


    def close(self):

        import pygame
        pygame.quit()