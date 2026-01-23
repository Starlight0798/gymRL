import gymnasium as gym
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.algo_name = 'DDQN + PER + DUELING'
        self.render_mode = 'rgb_array'
        self.train_eps = 100
        self.test_eps = 5
        self.max_steps = 2000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 600
        self.lr = 1e-3
        self.gamma = 0.9
        self.seed = random.randint(0, 100)
        self.batch_size = 256
        self.memory_capacity = 20000
        self.hidden_dim = 256
        self.target_update = 20
        self.alpha = 0.6
        self.beta = 0.4
        self.error_max = 1.0
        self.eps = 1e-6
        self.beta_increment_per_sampling = 0.001
        self.n_states = None
        self.n_actions = None
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)


class VAnet(nn.Module):
    def __init__(self, cfg):
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.hidden_dim)
        self.fc_a = nn.Linear(cfg.hidden_dim, cfg.n_actions)
        self.fc_v = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        a = self.fc_a(x)
        v = self.fc_v(x)
        q = v + a - a.mean(dim=1).view(-1, 1)
        return q


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.data_pointer = 0

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += change

    def add(self, priority, data):
        index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.size < self.capacity:
            self.size += 1

    def get_leaf(self, v):
        pa_idx = 0
        while True:
            lc_idx = pa_idx * 2 + 1
            rc_idx = lc_idx + 1
            if lc_idx >= len(self.tree):
                leaf_idx = pa_idx
                break
            else:
                if v <= self.tree[lc_idx]:
                    pa_idx = lc_idx
                else:
                    v -= self.tree[lc_idx]
                    pa_idx = rc_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

class ReplayBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tree = SumTree(self.cfg.memory_capacity)

    def push(self, transition):
        max_priority = np.max(self.tree.tree[-self.cfg.memory_capacity:])
        max_priority = max_priority if max_priority != 0 else 1
        self.tree.add(max_priority, transition)

    def sample(self):
        batch, idxs = [], []
        segment = self.tree.total_priority() / self.cfg.batch_size
        self.cfg.beta = np.min([1., self.cfg.beta + self.cfg.beta_increment_per_sampling])
        priorities = []
        for i in range(self.cfg.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(v)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.cfg.beta)
        is_weight /= is_weight.max()
        batchs = map(lambda x: torch.tensor(np.array(x), device=self.cfg.device,
                                             dtype=torch.float32), zip(*batch))
        is_weight = torch.tensor(np.array(is_weight), device=self.cfg.device,
                                 dtype=torch.float32)
        return batchs, np.array(idxs), is_weight

    def update(self, idx, error):
        error += self.cfg.eps
        clipped_error = np.minimum(error, self.cfg.error_max)
        ps = np.power(clipped_error, self.cfg.alpha)
        for i, p in zip(idx, ps):
            self.tree.update(i, p)

    def size(self):
        return self.tree.size


class PER_DUEL_DDQN:
    def __init__(self, cfg):
        self.sample_count = 0
        self.learn_count = 0
        self.memory = ReplayBuffer(cfg)
        self.policy_net = VAnet(cfg).to(cfg.device)
        self.target_net = VAnet(cfg).to(cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                       np.exp(-1. * self.sample_count / self.cfg.epsilon_decay)
        if random.uniform(0, 1) > self.epsilon:
            state = torch.tensor(np.array([state]), device=self.cfg.device, dtype=torch.float32)
            action = self.policy_net(state).argmax(dim=1).item()
        else:
            action = random.randrange(self.cfg.n_actions)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(np.array([state]), device=self.cfg.device, dtype=torch.float32)
        action = self.policy_net(state).argmax(dim=1).item()
        return action

    def update(self):
        if self.memory.size() < self.cfg.batch_size:
            return 0
        (state_batch, action_batch, reward_batch, next_state_batch,
            done_batch), idxs_batch, is_weight_batch = self.memory.sample()
        action_batch = action_batch.type(torch.long).view(-1, 1)
        q_value = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_q_value = self.policy_net(next_state_batch)
        next_target_value = self.target_net(next_state_batch)
        next_q_value = next_target_value.gather(1, next_q_value.argmax(dim=1).unsqueeze(1)).squeeze(1)
        expect_q_value = reward_batch + self.cfg.gamma * next_q_value * (1 - done_batch)
        loss = (q_value - expect_q_value.detach()).pow(2) * is_weight_batch
        prios = loss + self.cfg.eps
        loss = torch.mean(loss)
        self.memory.update(idxs_batch, prios.cpu().detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.learn_count % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_count += 1

        return loss.item()


def env_agent_config(cfg):
    env = gym.make(cfg.env_name, render_mode = cfg.render_mode).unwrapped
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    cfg.n_states = env.observation_space.shape[0]
    cfg.n_actions = env.action_space.n
    agent = PER_DUEL_DDQN(cfg)
    return env, agent


def train(env, agent, cfg):
    print('开始训练!')
    cfg.show()
    rewards, steps = [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        loss = 0.0
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            loss_ = agent.update()
            loss += loss_
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_step)
        print(f'回合:{i + 1}/{cfg.train_eps}  奖励:{ep_reward:.0f}  步数:{ep_step:.0f}  '
              f'epsilon:{agent.epsilon:.4f}  Loss:{loss/ep_step:.4f}')
    print('完成训练!')
    env.close()
    return rewards, steps


def test(agent, cfg):
    print('开始测试!')
    rewards, steps = [], []
    env = gym.make(cfg.env_name, render_mode='human')
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    env.close()
    return rewards, steps

if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    train_rewards, train_steps = train(env, agent, cfg)
    test_rewards, test_steps = test(agent, cfg)