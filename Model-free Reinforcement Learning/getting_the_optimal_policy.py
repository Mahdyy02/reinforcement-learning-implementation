import gymnasium as gym
import numpy as np
import ast
import json

env = gym.make("FrozenLake", render_mode = "rgb_array", is_slippery = True)
num_states = env.observation_space.n
num_actions = env.action_space.n
state, info = env.reset()

training_algo_file_name = "q_learning_epsilon_greedy_improved_q_value"
with open(fr'C:\Users\ASUS TUF\Desktop\Study\ENIT\1A Télécom\PFA\Reinforcement Learning Implementation\Model-free Reinforcement Learning\{training_algo_file_name}.json', 'r') as json_file:
    json_data = json.load(json_file)

Q = np.zeros((num_states, num_actions)) 

for i in range(len(json_data)):   
    for key in json_data[i]:
        row, col = ast.literal_eval(key)
        Q[row, col] = json_data[i][key]

def get_policy():
    policy = {state: np.argmax(Q[state]) for state in range(num_states)}
    return policy

number_of_episodes = 10000
number_of_winning_times = 0

policy = get_policy()

for _ in range(number_of_episodes):
    state, info = env.reset()
    done = False

    while not done:

        # action = policy[state]
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        if state == 15:
            number_of_winning_times+=1
            break

        done = terminated or truncated


print(f"Average winning time: {(number_of_winning_times/number_of_episodes)*100}%")