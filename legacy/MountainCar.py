import gym

env = gym.make('MountainCar-v0', render_mode = "human")
print(f'观测空间 = {env.observation_space}')
print(f'动作空间 = {env.action_space}')

class BespokeAgent:
    def __init__(self, env):
        pass

    @staticmethod
    def decide(observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):  # 学习
        pass

agent = BespokeAgent(env)

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0.  # 记录回合总奖励，初始化为0
    observation, info = env.reset()  # 重置游戏环境，开始新回合
    while True:  # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面，图形界面可用鼠标拖动
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward)  # 学习
        if terminated or truncated:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward  # 返回回合总奖励

episode_reward = play_montecarlo(env, agent, render=True)
print(f'回合奖励 = {episode_reward:.0f}')
env.close()