import gymnasium as gym
import matplotlib.pyplot as plt
import json

env = gym.make("CliffWalking-v0", render_mode = "human")
state, info = env.reset()

print(state)

policy = {
0:2,1:2,2:2,3:2,
4:2,5:2,6:2,7:2,
25:0,26:0,27:0,28:0,
29:0,30:0,31:0,32:0,
33:0,34:0, 37:0,38:0,
39:0,40:0,41:0,42:0,
43:0,44:0,45:0,46:0,
36:0, 24:0, 12:1, 8:2,
9:2,10:2,11:2,
13:1,14:1,15:1,16:1,
17:1,18:1,19:1,20:1,
21:1,22:1,23:2,35:2
}

file = open("Reinforcement Learning Implementation\\improved_policy.json", "r")
policy = json.load(file)
file.close()
policy = {int(k):int(v) for k,v in policy.items()}

# #Information about states and actions
# print(env.action_space)
# print(env.observation_space)
# print(env.unwrapped.P)


# #Visulazing the state

# state_image = env.render()
# plt.imshow(state_image)
# plt.show()

# #Performing actions
# action = 1
# state, reward, terminated, truncated, info = env.step(action)

# state_image = env.render()
# plt.imshow(state_image)
# plt.show()

#visualize the policy
done = False
while not done:
    action = policy[state]
    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    if done:
        done = not done
        state, info = env.reset()