import gymnasium as gym
import numpy as np
import torch
import random
import time
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import AtariPreprocessing
from utils.normalization import Normalization, RewardScaling
from utils.env_wrappers import PyTorchFrame

class BasicConfig:
    def __init__(self):
        self.render_mode = 'rgb_array'
        self.train_eps = 500
        self.test_eps = 3
        self.eval_freq = 10
        self.max_steps = 500
        self.lr_start = 1e-3
        self.lr_end = 1e-5
        self.gamma = 0.99
        self.lamda = 0.95
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.use_atari = False
        self.unwrapped = False
        self.load_model = False
        self.save_freq = 100
        self.state_replay = True
        self.state_storage_prob = 0.4 
        self.reward_diff = 0.1
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

    
    
def log_monitors(writer, monitors, agent, phase, step):
    for key, value in monitors.items():
        if not np.isnan(value):
            writer.add_scalar(f'{phase}/{key}', value, global_step=step)


def make_env(cfg):
    env = gym.make(cfg.env_name, render_mode=cfg.render_mode)
    s = env.observation_space.shape
    use_rgb = len(s) == 3 and s[2] in [1, 3]
    if cfg.use_atari:
        frame_skip = 4 if 'NoFrameskip' in cfg.env_name else 1
        env = AtariPreprocessing(env, grayscale_obs=False, terminal_on_life_loss=True,
                                 scale_obs=True, frame_skip=frame_skip)
    if use_rgb:
        env = PyTorchFrame(env)

    if cfg.unwrapped:
        env = env.unwrapped
        
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    
    cfg.n_states = int(env.observation_space.shape[0])
    env_continuous = isinstance(env.action_space, gym.spaces.Box)
    if env_continuous:
        cfg.action_bound = env.action_space.high[0]
        cfg.n_actions = int(env.action_space.shape[0])
    else:
        cfg.n_actions = int(env.action_space.n)
    cfg.max_steps = int(env.spec.max_episode_steps or cfg.max_steps)
    return env


def train(env, agent, cfg):
    print('开始训练!')
    if cfg.load_model:
        agent.load_model()
    
    if not hasattr(agent, "state_norm"):
        agent.state_norm = Normalization(shape=env.observation_space.shape)
        
    reward_scaler = RewardScaling(shape=1, gamma=cfg.gamma)
    use_rnn = hasattr(agent.net, 'reset_hidden')
    cfg.show()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'./exp/{cfg.algo_name}_{cfg.env_name.replace("/", "-")}_{timestamp}')
    
    for i in range(cfg.train_eps):
        last_reward, ep_reward, ep_step, state = 0.0, 0.0, 0, None
        steps = cfg.max_steps
        reward_scaler.reset()
        
        if use_rnn:
            agent.net.reset_hidden()
            if cfg.state_replay and agent.state_buffer.size() > (cfg.max_steps // 2) and np.random.rand() < cfg.state_storage_prob:
                state, agent.net.rnn_h, ep_reward, ep_step, env, reward_scaler = agent.load_state()
                steps -= ep_step
                
        if state is None:
            state, _ = env.reset(seed=random.randint(1, 2**31 - 1))  
            state = agent.state_norm(state)
            
        for _ in range(steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            ep_step += 1
            next_state = agent.state_norm(next_state)
            reward = reward_scaler(reward)[0] 
            
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            
            if not use_rnn:
                monitors = agent.update()
                log_monitors(writer, monitors, agent, 'train', agent.learn_step)
                
            if done:
                break
            
            if cfg.state_replay and use_rnn and abs(reward - last_reward) > cfg.reward_diff:
                agent.save_state(state, agent.net.rnn_h, ep_reward, ep_step, env, reward_scaler)
            last_reward = reward
            
        if use_rnn:
            monitors = agent.update()
            log_monitors(writer, monitors, agent, 'train', agent.learn_step)
            
        log_monitors(writer, {'reward': ep_reward, 'step': ep_step}, agent, 'train', i)
        print(f'回合:{i + 1}/{cfg.train_eps}  奖励:{ep_reward:.0f}  步数:{ep_step:.0f}')
        
        if (i + 1) % cfg.eval_freq == 0:
            tools = {'writer': writer}
            evaluate(env, agent, cfg, tools)
            
        if (i + 1) % cfg.save_freq == 0:
            agent.save_model()    
            
    print('完成训练!')
    agent.save_model()
    env.close()
    writer.close()


def evaluate(env, agent, cfg, tools):
    ep_reward, ep_step = 0.0, 0
    state, _ = env.reset(seed=random.randint(1, 2**31 - 1))
    use_rnn = hasattr(agent.net, 'reset_hidden')
    writer = tools['writer']
    state = agent.state_norm(state, update=False)
    if use_rnn:
        agent.net.reset_hidden()
    for _ in range(cfg.max_steps):
        ep_step += 1
        action = agent.evaluate(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.state_norm(next_state, update=False)
        state = next_state
        ep_reward += reward
        if terminated or truncated:
            break
    log_monitors(writer, {'reward': ep_reward, 'step': ep_step}, agent, 'eval', agent.learn_step)



def test(env, agent, cfg):
    print('开始测试!')
    agent.load_model()
    use_rnn = hasattr(agent.net, 'reset_hidden')
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=random.randint(1, 2**31 - 1))
        state = agent.state_norm(state, update=False)
        if use_rnn:
            agent.net.reset_hidden()
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.evaluate(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.state_norm(next_state, update=False)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    env.close()
    
    

