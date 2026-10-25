"""Step 4: watch a trained model fly and record it.

    python record.py --model models/builtin_seed0.zip --name landing
    python record.py --model random --name random

Writes one MP4 per attempt with Gymnasium's RecordVideo, plus a smaller GIF
of the first attempt for the README.
"""

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy import VideoFileClip
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help='path to a saved model, or "random"')
    parser.add_argument("--name", required=True, help="file name prefix for the clips")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100, help="seed of the first attempt")
    parser.add_argument("--gif-seconds", type=float, default=12, help="cut the GIF after this long")
    args = parser.parse_args()

    model = None if args.model == "random" else PPO.load(args.model, device="cpu")

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", name_prefix=args.name, episode_trigger=lambda e: True)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        score, frames = 0.0, 0
        while True:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            frames += 1
            if terminated or truncated:
                outcome = "landed" if reward == 100 else "crashed" if reward == -100 else "timed out"
                print(f"{args.name} episode {episode}: {outcome} after {frames} frames, score {score:.1f}")
                break
    env.close()

    # Half-size, half-frame-rate GIF of the first attempt, for the README.
    mp4 = Path("videos") / f"{args.name}-episode-0.mp4"
    clip = VideoFileClip(str(mp4))
    clip = clip.subclipped(0, min(clip.duration, args.gif_seconds)).resized(0.5)
    clip.write_gif(str(mp4.with_suffix(".gif")), fps=25, logger=None)
    clip.close()
    print(f"Wrote {mp4} and {mp4.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
