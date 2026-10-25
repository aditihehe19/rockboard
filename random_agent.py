"""Step 2: watch it fail.

Opens the game window and sends random actions. The rocket tumbles and
crashes, then restarts. Prints the eight observation numbers as it goes.
"""

import argparse

import gymnasium as gym

LABELS = [
    "left/right of pad",
    "height above pad",
    "sideways speed",
    "up/down speed",
    "tilt (rad)",
    "spin speed",
    "left leg down",
    "right leg down",
]
ACTIONS = ["nothing", "left engine", "main engine", "right engine"]


def print_observation(obs):
    for label, value in zip(LABELS, obs):
        print(f"  {label:<18} {value:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=20, help="frames between printouts")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3", render_mode="human")
    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)
        print(f"\nEpisode {episode}: start")
        print_observation(obs)
        score, frame = 0.0, 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            frame += 1
            if frame % args.print_every == 0:
                print(f"frame {frame}: action={ACTIONS[action]}, reward={reward:+.2f}")
                print_observation(obs)
            if terminated or truncated:
                outcome = "landed" if reward == 100 else "crashed" if reward == -100 else "timed out"
                print(f"Episode {episode}: {outcome} after {frame} frames, score {score:.1f}")
                break
    env.close()


if __name__ == "__main__":
    main()
