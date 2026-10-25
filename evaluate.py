"""How you know it works: average score over many attempts, with your wrapper off.

    python evaluate.py models/builtin_seed0.zip models/builtin_seed1.zip

Every model flies the plain LunarLander-v3 game, so a bonus from your own
reward rules can't flatter it. 200 or more on average counts as solved.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate(model, episodes, seed):
    env = gym.make("LunarLander-v3")
    scores, lengths, outcomes = [], [], []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        score, frames = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            frames += 1
            if terminated or truncated:
                break
        scores.append(score)
        lengths.append(frames)
        outcomes.append("landed" if reward == 100 else "crashed" if reward == -100 else "timed out")
    env.close()
    return {
        "episodes": episodes,
        "mean_score": round(float(np.mean(scores)), 1),
        "std_score": round(float(np.std(scores)), 1),
        "min_score": round(float(np.min(scores)), 1),
        "mean_frames": round(float(np.mean(lengths)), 1),
        "landed": outcomes.count("landed"),
        "crashed": outcomes.count("crashed"),
        "timed_out": outcomes.count("timed out"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000, help="seeds the training never used")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    for path in args.models:
        name = Path(path).stem
        result = evaluate(PPO.load(path, device="cpu"), args.episodes, args.seed)
        verdict = "SOLVED" if result["mean_score"] >= 200 else "not solved"
        print(
            f"{name:<22} {result['mean_score']:7.1f} +/- {result['std_score']:5.1f}  "
            f"landed {result['landed']}/{args.episodes}, crashed {result['crashed']}, "
            f"timed out {result['timed_out']}, avg {result['mean_frames']:.0f} frames  [{verdict}]"
        )
        with open(f"results/eval_{name}.json", "w") as f:
            json.dump({"model": name, **result}, f, indent=2)


if __name__ == "__main__":
    main()
