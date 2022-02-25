import gym
import numpy as np

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


# "Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. " \
# "Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine. "

# We chose the MlpPolicy because input of Lunar Lander is a feature vector, not images.
#
# The type of action to use (discrete/continuous) will be automatically deduced from the environment action space

# model = DQN('MlpPolicy', 'LunarLander-v2', verbose=1, exploration_final_eps=0.1, target_update_interval=250) #We load a helper function to evaluate the agent:


# We still don't use the trained one
# Create the environment




################################### Training#############

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2


env.reset()

model2 = A2C('MlpPolicy', env, verbose=1)
# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model2.learn(total_timesteps=100000)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model2.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)

env.close()

########################################################## Evulation#########################3

# Separate env for evaluation
model = A2C('MlpPolicy', 'LunarLander-v2', verbose=1) #We load a helper function to evaluate the agent:
eval_env = gym.make('LunarLander-v2')

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward (Random Agent, before training)={mean_reward:.2f} +/- {std_reward}")

mean_reward, std_reward = evaluate_policy(model2, env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward (Trained Agent, After training)={mean_reward:.2f} +/- {std_reward}")