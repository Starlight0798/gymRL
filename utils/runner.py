import gymnasium as gym
import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import AtariPreprocessing

class BasicConfig:
    def __init__(self):
        self.render_mode = 'rgb_array'
        self.train_eps = 500
        self.test_eps = 3
        self.eval_freq = 10
        self.max_steps = 500
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.param_update_freq = 5
        self.gamma = 0.99
        self.lamda = 0.95
        self.seed = random.randint(0, 100)
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.use_atari = False
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)


class PyTorchFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.rollaxis(observation, 2)


def make_env(env_name, render_mode='human', use_atari=False, unwrapped=False):
    env = gym.make(env_name, render_mode=render_mode)
    s = env.observation_space.shape
    use_rgb = len(s) == 3 and s[2] in [1, 3]
    frame_skip = 4 if 'NoFrameskip' in env_name else 1
    if use_atari:
        env = AtariPreprocessing(env, grayscale_obs=False, terminal_on_life_loss=True,
                                 scale_obs=True, frame_skip=frame_skip)
    if use_rgb:
        env = PyTorchFrame(env)
    if unwrapped:
        env = env.unwrapped
    return env


def make_env_agent(cfg, Algorithm):
    env = make_env(cfg.env_name, render_mode=cfg.render_mode, use_atari=cfg.use_atari)
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    cfg.n_states = env.observation_space.shape[0]
    env_continuous = isinstance(env.action_space, gym.spaces.Box)
    if env_continuous:
        cfg.action_bound = env.action_space.high[0]
        cfg.n_actions = env.action_space.shape[0]
    else:
        cfg.n_actions = env.action_space.n
    cfg.max_steps = env.spec.max_episode_steps or cfg.max_steps
    agent = Algorithm(cfg)
    return env, agent, cfg


def train(env, agent, cfg):
    print('开始训练!')
    use_rnn = hasattr(agent.net, 'reset_hidden')
    use_reward_scale = hasattr(agent, 'reward_scaling')
    use_state_norm = hasattr(agent, 'state_norm')
    use_acttion_fix = hasattr(agent, 'fix_action')
    cfg.show()
    writer = SummaryWriter(f'./exp/{cfg.algo_name}_{cfg.env_name.replace("/", "-")}')
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        if use_reward_scale:
            agent.reward_scaling.reset()
        if use_state_norm:
            state = agent.state_norm(state)
        if use_rnn:
            agent.net.reset_hidden()
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.choose_action(state)
            if use_acttion_fix:
                input_action = agent.fix_action(action)
            else:
                input_action = action
            next_state, reward, terminated, truncated, info = env.step(input_action)
            if use_state_norm:
                next_state = agent.state_norm(next_state)
            ep_reward += reward
            if use_reward_scale:
                reward = agent.reward_scaling(reward)[0]
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            if not use_rnn:
                monitors = agent.update()
                for key, value in monitors.items():
                    if not np.isnan(value):
                        writer.add_scalar(f'train/{key}', value, global_step=agent.learn_step)
                writer.add_scalar('train/lr', agent.lr, global_step=agent.learn_step)
            if done:
                break
        if use_rnn:
            monitors = agent.update()
            for key, value in monitors.items():
                if not np.isnan(value):
                    writer.add_scalar(f'train/{key}', value, global_step=agent.learn_step)
            writer.add_scalar('train/lr', agent.lr, global_step=agent.learn_step)
        writer.add_scalar('train/reward', ep_reward, global_step=i)
        writer.add_scalar('train/step', ep_step, global_step=i)
        print(f'回合:{i + 1}/{cfg.train_eps}  奖励:{ep_reward:.0f}  步数:{ep_step:.0f}')
        if (i + 1) % cfg.eval_freq == 0:
            evaluate(env, agent, cfg, writer)
    print('完成训练!')
    env.close()
    writer.close()


def evaluate(env, agent, cfg, writer):
    ep_reward, ep_step = 0.0, 0
    state, _ = env.reset(seed=cfg.seed)
    use_rnn = hasattr(agent.net, 'reset_hidden')
    use_state_norm = hasattr(agent, 'state_norm')
    use_acttion_fix = hasattr(agent, 'fix_action')
    if use_state_norm:
        state = agent.state_norm(state, update=False)
    if use_rnn:
        agent.net.reset_hidden()
    for _ in range(cfg.max_steps):
        ep_step += 1
        action = agent.evaluate(state)
        if use_acttion_fix:
            input_action = agent.fix_action(action)
        else:
            input_action = action
        next_state, reward, terminated, truncated, _ = env.step(input_action)
        if use_state_norm:
            next_state = agent.state_norm(next_state, update=False)
        state = next_state
        ep_reward += reward
        if terminated or truncated:
            break
    writer.add_scalar('eval/reward', ep_reward, global_step=agent.learn_step)
    writer.add_scalar('eval/step', ep_step, global_step=agent.learn_step)



def test(env, agent, cfg):
    print('开始测试!')
    use_rnn = hasattr(agent.net, 'reset_hidden')
    use_state_norm = hasattr(agent, 'state_norm')
    use_acttion_fix = hasattr(agent, 'fix_action')
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        if use_state_norm:
            state = agent.state_norm(state, update=False)
        if use_rnn:
            agent.net.reset_hidden()
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.evaluate(state)
            if use_acttion_fix:
                input_action = agent.fix_action(action)
            else:
                input_action = action
            next_state, reward, terminated, truncated, _ = env.step(input_action)
            if use_state_norm:
                next_state = agent.state_norm(next_state, update=False)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    env.close()
