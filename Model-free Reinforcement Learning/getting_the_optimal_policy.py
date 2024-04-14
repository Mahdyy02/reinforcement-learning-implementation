import gymnasium as gym
import numpy as np
import ast
import json

env = gym.make("FrozenLake", render_mode = "human", is_slippery = True)
num_states = env.observation_space.n
num_actions = env.action_space.n
state, info = env.reset()

with open(r'C:\Users\ASUS TUF\Desktop\Study\ENIT\1A Télécom\PFA\Reinforcement Learning Implementation\Model-free Reinforcement Learning\sarsa_improved_q_value.json', 'r') as json_file:
    json_data = json.load(json_file)

Q = np.zeros((num_states, num_actions)) 

for i in range(len(json_data)):   
    for key in json_data[i]:
        row, col = ast.literal_eval(key)
        Q[row, col] = json_data[i][key]

def get_policy():
    policy = {state: np.argmax(Q[state]) for state in range(num_states)}
    return policy

number_of_episodes = 1000
number_of_winning_times = 0
i = 0

policy = get_policy()
done = False
while i < number_of_episodes:
    action = policy[state]
    state, reward, terminated, truncated, info = env.step(action)

    if state == 8:
        number_of_winning_times+=1

    done = terminated or truncated
    if done:
        i+=1
        state, info = env.reset()

print(f"Average winning time: {(number_of_winning_times/number_of_episodes)*100}")