import gym
env = gym.make("Ant-v2")
env.reset()
for _ in range(1000):
   # env.render()
   action = env.action_space.sample()  # User-defined policy function
   observation, reward, done, info = env.step(action)
   print(info)

   if done:
      env.reset()
env.close()