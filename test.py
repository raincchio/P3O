import gym
env = gym.make("Walker2d-v2")
# env = gym.make("Ant-v2")
env.reset()
for _ in range(20):
   # env.render()
   action = env.action_space.sample()  # User-defined policy function
   observation, reward, done, info = env.step(action)
   print(info, reward)

   if done:
      env.reset()
env.close()