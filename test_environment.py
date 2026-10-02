import numpy as np

from stable_baselines3.common.env_checker import check_env

from environment import PlatformerEnv


def test_reset():

    print("\n--- Testing reset() ---")

    env = PlatformerEnv()

    observation, info = env.reset(seed=0)

    print("Observation:")
    print(observation)

    print("\nInfo:")
    print(info)

    print("\nObservation shape:")
    print(observation.shape)

    print("\nObservation valid:")
    print(env.observation_space.contains(observation))

    env.close()


def test_random_actions():

    print("\n--- Testing random actions ---")

    env = PlatformerEnv()

    observation, info = env.reset(seed=0)

    for step_number in range(20):

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Step: {step_number}, "
            f"Action: {action}, "
            f"Reward: {reward:.3f}, "
            f"Terminated: {terminated}, "
            f"Truncated: {truncated}"
        )

        if terminated or truncated:

            print("Episode finished.")

            observation, info = env.reset(seed=0)

    env.close()


def test_deterministic_reset():

    print("\n--- Testing deterministic reset ---")

    env = PlatformerEnv()

    observation_1, info_1 = env.reset(seed=42)

    observation_2, info_2 = env.reset(seed=42)

    same = np.array_equal(
        observation_1,
        observation_2,
    )

    print("First observation:")
    print(observation_1)

    print("\nSecond observation:")
    print(observation_2)

    print("\nObservations identical:")
    print(same)

    assert same

    env.close()


def test_deterministic_trajectory():

    print("\n--- Testing deterministic trajectory ---")

    actions = [
        2, 2, 2, 2, 2,
        5,
        2, 2, 2, 2,
        0, 0,
        2, 2, 2,
    ]

    env_1 = PlatformerEnv()
    env_2 = PlatformerEnv()

    observation_1, _ = env_1.reset(seed=42)
    observation_2, _ = env_2.reset(seed=42)

    assert np.array_equal(
        observation_1,
        observation_2,
    )

    for action in actions:

        result_1 = env_1.step(action)
        result_2 = env_2.step(action)

        observation_1 = result_1[0]
        observation_2 = result_2[0]

        reward_1 = result_1[1]
        reward_2 = result_2[1]

        terminated_1 = result_1[2]
        terminated_2 = result_2[2]

        truncated_1 = result_1[3]
        truncated_2 = result_2[3]

        assert np.array_equal(
            observation_1,
            observation_2,
        )

        assert reward_1 == reward_2
        assert terminated_1 == terminated_2
        assert truncated_1 == truncated_2

    print("Same seed + same actions produced identical results.")

    env_1.close()
    env_2.close()


def test_sb3_checker():

    print("\n--- Running Stable-Baselines3 check_env() ---")

    env = PlatformerEnv()

    check_env(
        env,
        warn=True,
    )

    print("Environment passed check_env().")

    env.close()


if __name__ == "__main__":

    test_reset()

    test_random_actions()

    test_deterministic_reset()

    test_deterministic_trajectory()

    test_sb3_checker()

    print("\nAll environment tests completed.")