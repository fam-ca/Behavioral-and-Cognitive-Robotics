import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import numpy as np

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)

# evaluate random untrained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model.learn(total_timesteps=10000)
model.save("ppo_cartpole")

images = []

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    img = model.env.render(mode='rgb_array')
    images.append(img)
    if done:
      obs = env.reset()

env.close()

# imageio.mimsave('PPO_CartPole.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

import cv2
# import glob
 
# img_array = []
# for filename in glob.glob('C:/New folder/Images/*.jpg'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
 
height, width, layers = img.shape
size = (width,height)
 
out = cv2.VideoWriter('ppo_cartpole.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(images)):
    out.write(images[i])
out.release()