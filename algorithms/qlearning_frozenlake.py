"""
Q-Learning for FrozenLake-v1

Tabular Q-Learning with epsilon-greedy exploration on the 4x4 FrozenLake grid.
Reference: "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018)

Features:
- Q-table with numpy array
- Epsilon-greedy with exponential decay
- Custom reward shaping for faster learning
"""

import gymnasium as gym
import numpy as np
import signal
import sys
import math


class Config:
    def __init__(self):
        self.env_name = "FrozenLake-v1"
        self.map_name = "4x4"
        self.is_slippery = True
        self.seed = 42
        self.max_episodes = 500
        self.max_steps = 100
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.use_reward_shaping = True


class QLearningTrainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.env = gym.make(
            config.env_name,
            map_name=config.map_name,
            is_slippery=config.is_slippery,
            render_mode="rgb_array",
        )
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

        self.epsilon = config.epsilon_start
        self.sample_count = 0
        self.episode_rewards = []

        print(f"Environment: {config.env_name}")
        print(f"States: {self.n_states}, Actions: {self.n_actions}")

    def _get_epsilon(self) -> float:
        self.sample_count += 1
        self.epsilon = self.cfg.epsilon_end + (
            self.cfg.epsilon_start - self.cfg.epsilon_end
        ) * math.exp(-1.0 * self.sample_count / self.cfg.epsilon_decay)
        return self.epsilon

    def _shape_reward(
        self, state: int, next_state: int, reward: float, done: bool
    ) -> float:
        if not self.cfg.use_reward_shaping:
            return reward

        holes = {5, 7, 11, 12}
        goal = 15

        if next_state in holes:
            return -10.0
        elif next_state == goal:
            return 100.0
        elif state == next_state:
            return -5.0
        else:
            return -1.0

    def select_action(self, state: int, deterministic: bool = False) -> int:
        if not deterministic and np.random.random() < self._get_epsilon():
            return np.random.randint(0, self.n_actions)
        return int(np.argmax(self.Q[state, :]))

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ):
        predict = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.cfg.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.cfg.lr * (target - predict)

    def train(self):
        print("Starting training...")

        for episode in range(self.cfg.max_episodes):
            state, _ = self.env.reset(seed=self.cfg.seed if episode == 0 else None)
            episode_reward = 0.0

            for step in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                shaped_reward = self._shape_reward(state, next_state, reward, done)
                self.update(state, action, shaped_reward, next_state, done)

                state = next_state
                episode_reward += shaped_reward

                if done:
                    break

            self.episode_rewards.append(episode_reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(
                    f"Episode {episode + 1}/{self.cfg.max_episodes} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Avg(50): {avg_reward:.1f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        print("Training completed!")
        self.env.close()

    def eval(self, num_episodes: int = 20) -> list:
        print(f"\nEvaluating for {num_episodes} episodes...")
        rewards = []
        successes = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if terminated and reward > 0:
                    successes += 1

            rewards.append(episode_reward)

        success_rate = successes / num_episodes * 100
        print(f"Evaluation: Success Rate = {success_rate:.1f}%")
        return rewards

    def test(self):
        self.eval(num_episodes=20)

        print("\nStarting visual test...")
        env = gym.make(
            self.cfg.env_name,
            map_name=self.cfg.map_name,
            is_slippery=self.cfg.is_slippery,
            render_mode="human",
        )
        state, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = self.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        result = "SUCCESS" if reward > 0 else "FAIL"
        print(f"Visual Test: {result} in {steps} steps")
        env.close()


if __name__ == "__main__":
    config = Config()
    trainer = QLearningTrainer(config)

    def signal_handler(signum, frame):
        print("\n\nTraining interrupted. Starting test...")
        trainer.test()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    trainer.test()
