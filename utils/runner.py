import gymnasium as gym
import numpy as np
import torch
import time, sys
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import AtariPreprocessing
from utils.normalization import Normalization, RewardScaling
from utils.env_wrappers import PyTorchFrame
from utils.buffer import *
from loguru import logger

np.random.seed(int(time.time()))
logger.remove()
logger.add(sys.stdout, level='INFO', format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>")

class BasicConfig:
    def __init__(self):
        self.render_mode = 'rgb_array'
        self.train_eps = 500
        self.test_eps = 3
        self.eval_freq = 10
        self.max_steps = 20000
        self.lr = 1e-4
        self.gamma = 0.99
        self.lamda = 0.95
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.use_atari = False
        self.unwrapped = False
        self.load_model = False
        self.save_freq = 50
        self.use_rnn = None
        self.on_policy = None
        self.save_path = './checkpoints/model.pth'
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

    def show(self):
        print('-' * 30 + 'Parameters' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

    
def log_monitors(writer, monitors, agent, phase, step):
    for key, value in monitors.items():
        if not np.isnan(value):
            writer.add_scalar(f'{phase}/{key}', value, global_step=step)


def make_env(cfg, **kwargs):
    env = gym.make(cfg.env_name, render_mode=cfg.render_mode, **kwargs)
    s = env.observation_space.shape
    use_rgb = (len(s) == 3 and s[2] in [1, 3]) if s is not None else False
    if cfg.use_atari:
        frame_skip = 4 if 'NoFrameskip' in cfg.env_name else 1
        env = AtariPreprocessing(env, grayscale_obs=False, terminal_on_life_loss=True,
                                 scale_obs=True, frame_skip=frame_skip)
    if use_rgb:
        env = PyTorchFrame(env)

    if cfg.unwrapped:
        env = env.unwrapped
        
    logger.info(f'Observation Space = {env.observation_space}')
    logger.info(f'Action Space = {env.action_space}')
    
    cfg.state_shape = env.observation_space.shape
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
    logger.info('Start training!')
    
    if cfg.load_model:
        agent.load_model()
    
    if not hasattr(agent, "state_norm"):
        agent.state_norm = Normalization(shape=env.observation_space.shape)
        logger.debug('Add state normalization!')
    if not hasattr(agent, "reward_scaler"):
        agent.reward_scaler = RewardScaling(shape=1, gamma=cfg.gamma)
        logger.debug('Add reward scaling!')

    cfg.on_policy = (
        isinstance(agent.memory, ReplayBuffer_on_policy) or 
        isinstance(agent.memory, ReplayBuffer_on_policy_v2) or
        isinstance(agent.memory, list) and isinstance(agent.memory[0], ReplayBuffer_on_policy)
    )
    cfg.use_rnn = hasattr(agent.net, 'reset_hidden')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'./exp/{cfg.algo_name}_{cfg.env_name.replace("/", "-")}_{timestamp}')
    cfg.show()
    
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        agent.reward_scaler.reset()
        
        if cfg.use_rnn:
            agent.net.reset_hidden()

        state, _ = env.reset(seed=np.random.randint(1, 2**31 - 1))  
        state = agent.state_norm(state)
        
        if cfg.on_policy:
            action, log_prob, value = agent.choose_action(state)
        else:
            action = agent.choose_action(state)

        for _ in range(cfg.max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_step += 1

            reward = agent.reward_scaler(reward)[0] 
            next_state = agent.state_norm(next_state)
            
            if cfg.on_policy:
                _action, _log_prob, _value = agent.choose_action(next_state)
                transitions = (state, action, reward, done, terminated, log_prob, value, _value)
                if cfg.use_rnn:
                    agent.memory[i % cfg.batch_size].store(transitions)
                else:
                    agent.memory.store(transitions)
                action, log_prob, value = _action, _log_prob, _value
            else:
                agent.memory.store((state, action, reward, next_state, done))
                action = agent.choose_action(next_state)
                
            state = next_state
            
            if not cfg.use_rnn and agent.memory.size() >= cfg.batch_size:
                monitors = agent.update()
                log_monitors(writer, monitors, agent, 'train', agent.learn_step)
                
            if done:
                break
        
        if cfg.use_rnn and i % cfg.batch_size == 0 and i > 0:
            monitors = agent.update()
            log_monitors(writer, monitors, agent, 'train', agent.learn_step)

        log_monitors(writer, {'reward': ep_reward, 'step': ep_step}, agent, 'train', i)
        logger.info(f'Episode:{i + 1}/{cfg.train_eps}  Reward:{ep_reward:.0f}  Step:{ep_step:.0f}')
        
        if (i + 1) % cfg.eval_freq == 0:
            tools = {'writer': writer}
            evaluate(env, agent, cfg, tools)
        
        if (i + 1) % cfg.save_freq == 0:
            agent.save_model()    
            
    logger.info('Finish training!')
    agent.save_model()
    env.close()
    writer.close()


def evaluate(env, agent, cfg, tools):
    ep_reward, ep_step, done = 0.0, 0, False
    state, _ = env.reset(seed=np.random.randint(1, 2**31 - 1))
    writer = tools['writer']
    state = agent.state_norm(state, update=False)
    if cfg.use_rnn:
        agent.net.reset_hidden()
    while not done:
        ep_step += 1
        action = agent.evaluate(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.state_norm(next_state, update=False)
        state = next_state
        ep_reward += reward
        done = terminated or truncated
    log_monitors(writer, {'reward': ep_reward, 'step': ep_step}, agent, 'eval', agent.learn_step)


def test(env, agent, cfg):
    logger.info('Start test!')
    agent.load_model()
    for i in range(cfg.test_eps):
        ep_reward, ep_step, done = 0.0, 0, False
        state, _ = env.reset(seed=np.random.randint(1, 2**31 - 1))
        state = agent.state_norm(state, update=False)
        if cfg.use_rnn:
            agent.net.reset_hidden()
        while not done:
            ep_step += 1
            action = agent.evaluate(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.state_norm(next_state, update=False)
            state = next_state
            ep_reward += reward
            done = terminated or truncated
        logger.info(f'Episode:{i + 1}/{cfg.train_eps}  Reward:{ep_reward:.0f}  Step:{ep_step:.0f}')
    logger.info('Finish test!')
    env.close()
    
    
class BenchMark:
    
    @logger.catch(reraise=True)
    @staticmethod
    def train(algo, config):
        cfg = config()
        env = make_env(cfg)     # must be called before agent is created
        agent = algo(cfg)
        train(env, agent, cfg)
    
    @logger.catch(reraise=True) 
    @staticmethod
    def test(algo, config):
        cfg = config()
        cfg.render_mode = 'human'
        env = make_env(cfg)
        agent = algo(cfg)
        test(env, agent, cfg)
        
    
    

