import gym
import time

env = gym.make('Acrobot-v1') # CartPole-v0 # LunarLander-v2 # Acrobot-v1
observation = env.reset()
done = False
fitness = 0
s = 0

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    fitness += reward

    if done:
        time.sleep(3)
    env.render()
    time.sleep(0.1)

    print("Step", s)
    print("Observation vector", observation)
    print("Action vector", action)
    print("Reward", reward)
    print('Info', info)
    print(done)
    print("Fitness", fitness, "\n")

    s +=1
env.close()

print("Final fitness", fitness)
