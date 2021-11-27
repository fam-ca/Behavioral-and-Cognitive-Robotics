import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import cv2

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)

# evaluate random untrained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model.learn(total_timesteps=10000)
model.save("ppo_cartpole")

img_array = []

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    img = model.env.render(mode='rgb_array')
    img_array.append(img)

    if done:
      obs = env.reset()

env.close()

# save a video of trained model
height, width, layers = img.shape
size = (width,height)
 
out = cv2.VideoWriter('ppo_cartpole.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()