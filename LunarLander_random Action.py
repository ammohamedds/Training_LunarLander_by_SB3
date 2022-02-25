import gym

# Create the environment
env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2

# required before you can step the environment

# Action space (Discrete)
#
#     0- Do nothing
#     1- Fire left engine
#     2- Fire down engine
#     3- Fire right engine


env.reset()


for step in range(100):
    env.render()
    print("sample action:", env.action_space.sample())
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs,reward, done,info)
	# take random action


# sample action:
print("sample action:", env.action_space.sample())

# observation space shape:
print("observation space shape:", env.observation_space.shape)

# sample observation:
print("sample observation:", env.observation_space.sample())

env.close()

# # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L447:

# env: The environment
# 	s (list): The state. Attributes:
# 				s[0] is the horizontal coordinate
# 				s[1] is the vertical coordinate
# 				s[2] is the horizontal speed
# 				s[3] is the vertical speed
# 				s[4] is the angle
# 				s[5] is the angular speed
# 				s[6] 1 if first leg has contact, else 0
# 				s[7] 1 if second leg has contact, else 0

episodes = 10

# for eps in range(episodes):
#   obs = env.reset()
#   done =False
#   while not done:
#     # env.render()
#     obs, reward, done, info = env.step(env.action_space.sample())
#     print(reward, done)
#
# env.close()