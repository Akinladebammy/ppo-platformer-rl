import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from stable_baselines3 import PPO

from environment import PlatformerEnv


PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_config(config_path):
    """
    Load experiment configuration from a JSON file.
    """
    with open(config_path, "r") as file:
        return json.load(file)


def set_global_seeds(seed):
    """
    Seed the random-number generators used by Python,
    NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_agent(
    env,
    model=None,
    episodes=100,
    seed=42,
    random_policy=False,
):
    """
    Evaluate either a PPO policy or a random baseline.

    Returns aggregate statistics across evaluation episodes.
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
                    rng.integers(env.action_space.n)
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

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        furthest_positions.append(furthest_x)

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


def save_learning_curves(
    timesteps_history,
    reward_history,
    success_history,
    furthest_x_history,
    random_results,
):
    """
    Save the three learning-curve figures used in the
    experiment.
    """

    # Reward curve

    plt.figure(figsize=(9, 5))

    plt.plot(
        timesteps_history,
        reward_history,
        marker="o",
    )

    plt.axhline(
        y=random_results["mean_reward"],
        linestyle="--",
        label="Random baseline",
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.title(
        "PPO Learning Curve — Evaluation Reward"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "reward_curve.png"
    )

    plt.close()

    # Completion curve

    success_percent = [
        rate * 100
        for rate in success_history
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        timesteps_history,
        success_percent,
        marker="o",
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Completion rate (%)")
    plt.title(
        "PPO Learning Curve — Level Completion"
    )

    plt.ylim(-5, 105)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "completion_curve.png"
    )

    plt.close()

    # Progress curve

    plt.figure(figsize=(9, 5))

    plt.plot(
        timesteps_history,
        furthest_x_history,
        marker="o",
    )

    plt.axhline(
        y=random_results["mean_furthest_x"],
        linestyle="--",
        label="Random baseline",
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean furthest X position")
    plt.title(
        "PPO Learning Curve — Level Progress"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "progress_curve.png"
    )

    plt.close()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train PPO on the custom platformer."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to experiment configuration.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)

    seed = config["seed"]

    training_config = config["training"]
    ppo_config = config["ppo"]
    final_eval_config = config[
        "final_evaluation"
    ]

    total_timesteps = training_config[
        "total_timesteps"
    ]

    evaluation_interval = training_config[
        "evaluation_interval"
    ]

    evaluation_episodes = training_config[
        "evaluation_episodes"
    ]

    max_episode_steps = training_config[
        "max_episode_steps"
    ]

    final_evaluation_episodes = (
        final_eval_config["episodes"]
    )

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    set_global_seeds(seed)

    print("=" * 60)
    print("PPO PLATFORMER TRAINING")
    print("=" * 60)

    print("Seed:", seed)
    print(
        "Training budget:",
        total_timesteps,
    )
    print(
        "Evaluation interval:",
        evaluation_interval,
    )

    # --------------------------------------------------
    # RANDOM BASELINE
    # --------------------------------------------------

    print("\nEvaluating random baseline...")

    random_env = PlatformerEnv(
        max_steps=max_episode_steps
    )

    random_results = evaluate_agent(
        env=random_env,
        episodes=final_evaluation_episodes,
        seed=seed,
        random_policy=True,
    )

    random_env.close()

    with open(
        RESULTS_DIR / "random_baseline.json",
        "w",
    ) as file:
        json.dump(
            random_results,
            file,
            indent=4,
        )

    print(
        "Random mean reward:",
        round(
            random_results["mean_reward"],
            3,
        ),
    )

    print(
        "Random success rate:",
        f"{random_results['success_rate'] * 100:.1f}%",
    )

    print(
        "Random mean furthest X:",
        round(
            random_results["mean_furthest_x"],
            1,
        ),
    )

    # --------------------------------------------------
    # PPO MODEL
    # --------------------------------------------------

    print("\nCreating PPO model...")

    train_env = PlatformerEnv(
        max_steps=max_episode_steps
    )

    train_env.reset(seed=seed)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=ppo_config[
            "learning_rate"
        ],

        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],

        gamma=ppo_config["gamma"],

        gae_lambda=ppo_config[
            "gae_lambda"
        ],

        clip_range=ppo_config[
            "clip_range"
        ],

        ent_coef=ppo_config["ent_coef"],

        seed=seed,

        verbose=0,

        tensorboard_log=str(LOGS_DIR),
    )

    # --------------------------------------------------
    # LEARNING CURVE STORAGE
    # --------------------------------------------------

    timesteps_history = []
    reward_history = []
    success_history = []
    furthest_x_history = []
    episode_length_history = []

    best_success_rate = -1.0
    best_mean_reward = float("-inf")
    best_timesteps = 0

    best_model_path = (
        MODELS_DIR / "ppo_platformer_best"
    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    print("\nStarting training...\n")

    training_start_time = time.perf_counter()

    while (
        model.num_timesteps
        < total_timesteps
    ):

        remaining_steps = (
            total_timesteps
            - model.num_timesteps
        )

        training_chunk = min(
            evaluation_interval,
            remaining_steps,
        )

        model.learn(
            total_timesteps=training_chunk,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        actual_steps = model.num_timesteps

        eval_env = PlatformerEnv(
            max_steps=max_episode_steps
        )

        results = evaluate_agent(
            env=eval_env,
            model=model,
            episodes=evaluation_episodes,
            seed=seed,
        )

        eval_env.close()

        timesteps_history.append(
            actual_steps
        )

        reward_history.append(
            results["mean_reward"]
        )

        success_history.append(
            results["success_rate"]
        )

        furthest_x_history.append(
            results["mean_furthest_x"]
        )

        episode_length_history.append(
            results["mean_episode_length"]
        )

        success = results[
            "success_rate"
        ]

        reward = results["mean_reward"]

        is_better = (
            success > best_success_rate
            or (
                success
                == best_success_rate
                and reward
                > best_mean_reward
            )
        )

        marker = ""

        if is_better:

            best_success_rate = success
            best_mean_reward = reward
            best_timesteps = actual_steps

            model.save(
                str(best_model_path)
            )

            marker = " <-- SAVED BEST"

        print(
            f"Steps: {actual_steps:>6} | "
            f"Reward: {reward:>7.2f} | "
            f"Success: "
            f"{success * 100:>5.1f}% | "
            f"Furthest X: "
            f"{results['mean_furthest_x']:>6.1f}"
            f"{marker}"
        )

    training_seconds = (
        time.perf_counter()
        - training_start_time
    )

    train_env.close()

    # --------------------------------------------------
    # SAVE LEARNING-CURVE DATA
    # --------------------------------------------------

    learning_curve_data = []

    for i in range(
        len(timesteps_history)
    ):

        learning_curve_data.append({
            "timesteps":
                timesteps_history[i],

            "mean_reward":
                reward_history[i],

            "success_rate":
                success_history[i],

            "mean_furthest_x":
                furthest_x_history[i],

            "mean_episode_length":
                episode_length_history[i],
        })

    with open(
        RESULTS_DIR / "learning_curve.json",
        "w",
    ) as file:

        json.dump(
            learning_curve_data,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------

    save_learning_curves(
        timesteps_history,
        reward_history,
        success_history,
        furthest_x_history,
        random_results,
    )

    # --------------------------------------------------
    # LOAD BEST MODEL FROM DISK
    # --------------------------------------------------

    print("\nLoading best saved model...")

    best_model = PPO.load(
        str(best_model_path)
    )

    final_eval_env = PlatformerEnv(
        max_steps=max_episode_steps
    )

    ppo_results = evaluate_agent(
        env=final_eval_env,
        model=best_model,
        episodes=final_evaluation_episodes,
        seed=seed,
    )

    final_eval_env.close()

    # --------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------

    comparison = {
        "random": random_results,
        "ppo": ppo_results,
        "ppo_checkpoint_timesteps":
            best_timesteps,
    }

    with open(
        RESULTS_DIR / "comparison.json",
        "w",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # HEADLINE RESULTS
    # --------------------------------------------------

    headline_results = {
        "seed": seed,

        "training_budget":
            total_timesteps,

        "best_checkpoint_timesteps":
            best_timesteps,

        "training_seconds":
            training_seconds,

        "random_baseline":
            random_results,

        "best_ppo":
            ppo_results,

        "selection_rule": (
            "Highest evaluation success rate; "
            "mean reward used as tie-breaker."
        ),

        "evaluation_note": (
            "Environment and evaluation policy "
            "are deterministic. Repeated "
            "evaluation episodes therefore "
            "reproduce the same trajectory for "
            "a fixed checkpoint."
        ),
    }

    with open(
        RESULTS_DIR / "headline_results.json",
        "w",
    ) as file:

        json.dump(
            headline_results,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best checkpoint:",
        best_timesteps,
    )

    print(
        "Training time:",
        f"{training_seconds:.2f} seconds",
    )

    print("\nFINAL COMPARISON")
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

    print(
        "\nBest model saved to:",
        best_model_path.with_suffix(".zip"),
    )

    print(
        "Results saved to:",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()