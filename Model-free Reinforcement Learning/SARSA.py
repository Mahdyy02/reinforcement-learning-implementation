import gymnasium as gym
import numpy as np
import json

sarsa_improved_q_value_file = open("sarsa_improved_q_value.json", "w")

def update_q_table(state, action, reward, next_state, next_action):
    old_value = Q[state, action]
    next_value = Q[next_state, next_action]
    Q[state, action] = (1-alpha)*old_value + alpha*(reward+gamma*next_value)

def sarsa_loop(num_episodes):
    for _ in range(num_episodes):
        state, info = env.reset()
        action = env.action_space.sample()
        done = False

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = env.action_space.sample()
            update_q_table(state, action, reward, next_state, next_action)
            state, action = next_state, next_action

            done = terminated or truncated

env = gym.make("FrozenLake", render_mode="rgb_array", is_slippery = True)
num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))
alpha = 0.1
gamma = 1
num_episodes = 5000000

sarsa_loop(num_episodes)
json_data = [{f'({row},{col})': Q[row][col] for col in range(num_actions)} for row in range(num_states)]
json.dump(json_data, sarsa_improved_q_value_file, indent=4)
sarsa_improved_q_value_file.close()
