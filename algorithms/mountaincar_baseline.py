"""
Hand-Crafted Policy Baseline for MountainCar-v0

A simple rule-based agent demonstrating how domain knowledge can solve MountainCar.
This serves as a baseline for comparing learned policies.

Features:
- Hand-crafted decision boundaries based on position and velocity
- No learning required - pure domain knowledge
- Useful as a baseline or demonstration
"""

import gymnasium as gym
import numpy as np
import signal
import sys


class Config:
    def __init__(self):
        self.env_name = "MountainCar-v0"
        self.seed = 42
        self.test_episodes = 10


class RuleBasedAgent:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(config.env_name, render_mode="rgb_array")

        print(f"Environment: {config.env_name}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")

    def select_action(self, observation: np.ndarray) -> int:
        position, velocity = observation
        lb = min(
            -0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008
        )
        ub = -0.07 * (position + 0.38) ** 2 + 0.07

        if lb < velocity < ub:
            return 2
        else:
            return 0

    def run_episode(self, render: bool = False) -> float:
        if render:
            env = gym.make(self.cfg.env_name, render_mode="human")
        else:
            env = self.env

        state, _ = env.reset(seed=self.cfg.seed)
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = self.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        if render:
            env.close()

        return episode_reward, steps

    def eval(self, num_episodes: int = 10) -> list:
        print(f"\nEvaluating for {num_episodes} episodes...")
        rewards = []
        steps_list = []

        for episode in range(num_episodes):
            reward, steps = self.run_episode(render=False)
            rewards.append(reward)
            steps_list.append(steps)
            print(f"  Episode {episode + 1}: Reward = {reward:.0f}, Steps = {steps}")

        print(
            f"Evaluation: Mean Reward = {np.mean(rewards):.1f}, Mean Steps = {np.mean(steps_list):.1f}"
        )
        return rewards

    def test(self):
        self.eval(num_episodes=self.cfg.test_episodes)

        print("\nStarting visual test...")
        reward, steps = self.run_episode(render=True)
        print(f"Visual Test: Reward = {reward:.0f}, Steps = {steps}")


if __name__ == "__main__":
    config = Config()
    agent = RuleBasedAgent(config)

    def signal_handler(signum, frame):
        print("\n\nInterrupted.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    agent.test()
