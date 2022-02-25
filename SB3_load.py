import gym
from stable_baselines3 import PPO,A2C

# models_dir = "models/PPO"
models_dir = "models/A2C"

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

# model_path = f"{models_dir}/940000.zip" #PPO best
model_path = f"{models_dir}/720000.zip" # A2C Best

# model = PPO.load(model_path, env=env)
model = A2C.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)