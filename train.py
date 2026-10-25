"""Steps 3, 5 and 6: train PPO on Lunar Lander and save the model.

    python train.py                        # built-in reward
    python train.py --reward time_fee      # step 5
    python train.py --reward hover_bonus   # step 6, the broken rule
    python train.py --reward calm_progress # step 6, the fix

While it runs, rollout/ep_rew_mean is the average score PPO is paid per
attempt over the last 100 (built-in plus your own rules), and
rollout/builtin_score_mean is the plain Lunar Lander score for the same attempts.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from rewards import EXPERIMENTS, make_env


class LogBuiltinScore(BaseCallback):
    """Prints the plain Lunar Lander score next to ep_rew_mean."""

    def _on_rollout_end(self):
        scores = [ep["builtin_score"] for ep in self.model.ep_info_buffer]
        if scores:
            self.logger.record("rollout/builtin_score_mean", float(np.mean(scores)))

    def _on_step(self):
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=EXPERIMENTS, default="builtin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000, help="one step is one frame")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run = f"{args.reward}_seed{args.seed}"

    # 16 copies of the game played side by side, each logged by a Monitor
    # wrapped around the reward rules, so its score is the one PPO sees.
    env = make_vec_env(
        make_env,
        n_envs=16,
        seed=args.seed,
        env_kwargs={"experiment": args.reward},
        monitor_dir=f"logs/{run}",
        monitor_kwargs={"info_keywords": ("builtin_score",)},
    )

    # Settings from the Hugging Face Deep RL Course, Unit 1.
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        seed=args.seed,
        device="cpu",
        verbose=1,
    )

    start = time.time()
    model.learn(total_timesteps=args.steps, callback=LogBuiltinScore())
    minutes = (time.time() - start) / 60

    Path("models").mkdir(exist_ok=True)
    model.save(f"models/{run}")
    Path("results").mkdir(exist_ok=True)
    with open(f"results/train_{run}.json", "w") as f:
        json.dump({"run": run, "steps": args.steps, "minutes": round(minutes, 1)}, f, indent=2)
    print(f"Saved models/{run}.zip after {minutes:.1f} minutes")


if __name__ == "__main__":
    main()
