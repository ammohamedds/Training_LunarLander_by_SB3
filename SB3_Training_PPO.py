import gym
from stable_baselines3 import PPO
import os


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v2')
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
for i in range(100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO") #Note the reset_num_timesteps=False. This allows us to see the actual total number of timesteps for the model rather than resetting every iteration or we can set total timesteps very high without iteration.
    model.save(f"{models_dir}/{TIMESTEPS*i}") # We should see every ~10,000 timesteps a model will be saved



# Now, while the model trains, we can view the results over time by opening a new terminal and doing: tensorboard --logdir=logs