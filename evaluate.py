import argparse
import json
import time
from pathlib import Path

import numpy as np
import pygame

from stable_baselines3 import PPO

from environment import PlatformerEnv


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_platformer_best.zip"
)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config.json"
)


def load_config(config_path):
    """
    Load experiment configuration.
    """
    with open(config_path, "r") as file:
        return json.load(file)


def evaluate_agent(
    env,
    model=None,
    episodes=100,
    seed=42,
    random_policy=False,
):
    """
    Evaluate PPO or a random policy.
    """

    episode_rewards = []
    episode_lengths = []
    furthest_positions = []
    successes = []

    rng = np.random.default_rng(seed)

    for episode in range(episodes):

        episode_seed = seed + episode

        observation, info = env.reset(
            seed=episode_seed
        )

        total_reward = 0.0
        steps = 0
        furthest_x = info["player_x"]

        terminated = False
        truncated = False

        while not terminated and not truncated:

            if random_policy:

                action = int(
                    rng.integers(
                        env.action_space.n
                    )
                )

            else:

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                action = int(action)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward
            steps += 1

            furthest_x = max(
                furthest_x,
                info["player_x"],
            )

        episode_rewards.append(
            total_reward
        )

        episode_lengths.append(
            steps
        )

        furthest_positions.append(
            furthest_x
        )

        successes.append(
            1 if info["won"] else 0
        )

    return {
        "mean_reward": float(
            np.mean(episode_rewards)
        ),

        "std_reward": float(
            np.std(episode_rewards)
        ),

        "success_rate": float(
            np.mean(successes)
        ),

        "mean_furthest_x": float(
            np.mean(furthest_positions)
        ),

        "mean_episode_length": float(
            np.mean(episode_lengths)
        ),
    }


def play_agent(
    model,
    max_episode_steps,
    seed,
):
    """
    Render one episode using the trained PPO policy.
    """

    env = PlatformerEnv(
        render_mode="human",
        max_steps=max_episode_steps,
    )

    observation, info = env.reset(
        seed=seed
    )

    terminated = False
    truncated = False

    total_reward = 0.0

    while not terminated and not truncated:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                env.close()
                return

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(int(action))

        total_reward += reward

        time.sleep(1 / 60)

    print("\nRendered episode finished")
    print("Won:", info["won"])
    print("Final X:", info["player_x"])
    print(
        "Total reward:",
        round(total_reward, 3),
    )
    print("Steps:", info["steps"])

    waiting = True

    while waiting:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                waiting = False

        # Avoid using unnecessary CPU while waiting.
        time.sleep(0.01)

    # Shut down Pygame after the window is closed.
    env.close()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained PPO platformer agent."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to PPO model.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Number of evaluation episodes. "
            "Overrides config.json."
        ),
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Render one episode after evaluation."
        ),
    )

    args = parser.parse_args()

    config = load_config(
        Path(args.config)
    )

    seed = config["seed"]

    max_episode_steps = config[
        "training"
    ]["max_episode_steps"]

    if args.episodes is None:

        episodes = config[
            "final_evaluation"
        ]["episodes"]

    else:

        episodes = args.episodes

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train.py first or provide "
            "--model PATH."
        )

    print("=" * 60)
    print("PPO PLATFORMER EVALUATION")
    print("=" * 60)

    print("Model:", model_path)
    print("Seed:", seed)
    print("Episodes:", episodes)

    print("\nLoading model...")

    model = PPO.load(
        str(model_path)
    )

    # Random baseline

    random_env = PlatformerEnv(
        max_steps=max_episode_steps
    )

    random_results = evaluate_agent(
        env=random_env,
        episodes=episodes,
        seed=seed,
        random_policy=True,
    )

    random_env.close()

    # PPO

    ppo_env = PlatformerEnv(
        max_steps=max_episode_steps
    )

    ppo_results = evaluate_agent(
        env=ppo_env,
        model=model,
        episodes=episodes,
        seed=seed,
    )

    ppo_env.close()

    print("\nRESULTS")
    print("-" * 60)

    print(
        f"{'Metric':<25}"
        f"{'Random':>15}"
        f"{'PPO':>15}"
    )

    print("-" * 55)

    print(
        f"{'Mean reward':<25}"
        f"{random_results['mean_reward']:>15.2f}"
        f"{ppo_results['mean_reward']:>15.2f}"
    )

    print(
        f"{'Success rate':<25}"
        f"{random_results['success_rate'] * 100:>14.1f}%"
        f"{ppo_results['success_rate'] * 100:>14.1f}%"
    )

    print(
        f"{'Mean furthest X':<25}"
        f"{random_results['mean_furthest_x']:>15.1f}"
        f"{ppo_results['mean_furthest_x']:>15.1f}"
    )

    print(
        f"{'Episode length':<25}"
        f"{random_results['mean_episode_length']:>15.1f}"
        f"{ppo_results['mean_episode_length']:>15.1f}"
    )

    if args.render:

        print(
            "\nStarting rendered episode..."
        )

        play_agent(
            model=model,
            max_episode_steps=(
                max_episode_steps
            ),
            seed=seed,
        )


if __name__ == "__main__":
    main()