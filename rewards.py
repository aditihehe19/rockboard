"""Reward wrappers for Lunar Lander.

Each wrapper sits between the game and PPO. On every step it takes the
built-in reward, adds or subtracts its own terms, and passes the total on.
The game's code is never touched.
"""

import gymnasium as gym
import numpy as np

# "Upright and slow", in the units of the observation vector.
MAX_TILT = 0.2   # radians, about 11 degrees either way
MAX_SPEED = 0.2  # length of the (sideways speed, up/down speed) vector


def is_calm(obs):
    """True when the rocket is upright and slow."""
    tilt = obs[4]
    speed = np.hypot(obs[2], obs[3])
    return abs(tilt) < MAX_TILT and speed < MAX_SPEED


class TrackBuiltinScore(gym.Wrapper):
    """Keeps a running total of the game's own reward for the episode.

    The total goes into info["builtin_score"] so the training logs can show
    the real Lunar Lander score next to the shaped one PPO is trained on.
    This sits innermost, so the rewards it sees are the untouched ones.
    """

    def reset(self, **kwargs):
        self.builtin_score = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.builtin_score += reward
        info["builtin_score"] = self.builtin_score
        return obs, reward, terminated, truncated, info


class TimeFee(gym.Wrapper):
    """Step 5: a tiny charge for every frame, so taking too long costs something.

    The built-in reward has no time cost at all, only fuel.
    """

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


class HoverBonus(gym.Wrapper):
    """Step 6, the broken rule: +1 for every frame the rocket is upright and slow.

    An attempt is cut off at 1,000 frames, so a rocket that just hangs in the
    air can collect up to 1,000 points from this rule and pays at most 300 in
    fuel. Landing ends the attempt and the payments stop. Hovering wins.
    """

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if is_calm(obs):
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class CalmProgress(gym.Wrapper):
    """Step 6, the fix: pay for becoming upright and slow, not for staying that way.

    Like the built-in score, this pays the change between frames: +1 on the
    frame the rocket becomes calm, -1 on the frame it stops being calm, and
    nothing while it stays calm. Over a whole attempt the payments add up to
    at most +1, so there is nothing to farm by hovering.
    """

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.was_calm = is_calm(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        calm = is_calm(obs)
        reward += self.bonus * (float(calm) - float(self.was_calm))
        self.was_calm = calm
        return obs, reward, terminated, truncated, info


# Each experiment is the list of wrappers stacked on top of the game, inner first.
EXPERIMENTS = {
    "builtin": [],
    "time_fee": [TimeFee],
    "hover_bonus": [HoverBonus],
    "calm_progress": [CalmProgress, TimeFee],
}


def make_env(experiment="builtin", render_mode=None):
    """Lunar Lander with the given experiment's reward rules on top."""
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env = TrackBuiltinScore(env)
    for wrapper in EXPERIMENTS[experiment]:
        env = wrapper(env)
    return env
