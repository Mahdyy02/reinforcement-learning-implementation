import gymnasium as gym
import numpy as np
import json

env = gym.make("FrozenLake", render_mode = "rgb_array", is_slippery = True)
num_states = env.observation_space.n
num_actions = env.action_space.n
num_episodes = 1000000

first_visit_mc_q_value_file = open("first_visit_mc_q_value.json", "w")
every_visit_mc_q_value_file = open("every_visit_mc_q_value.json", "w")

def generate_episode():
    episode = []
    state, info = env.reset()
    done = False 

    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state,action,reward))
        state = next_state

        done = terminated or truncated
    
    return episode

def first_visit_mc(num_episodes):

    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))

    for i in range(num_episodes):
        epsiode = generate_episode()
        visited_states_actions = set()

        for j, (state,action, reward) in enumerate(epsiode):
            if(state, action) not in visited_states_actions:
                returns_sum[state, action] += sum(x[2] for x in epsiode[j:])
                returns_count[state,action]+=1
                visited_states_actions.add((state,action))

    nonzero_counts = returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts]/returns_count[nonzero_counts]
    json_data = [{f'({row},{col})': Q[row][col] for col in range(num_actions)} for row in range(num_states)]
    json.dump(json_data, first_visit_mc_q_value_file, indent=4)
    first_visit_mc_q_value_file.close()
    return Q

def every_visit_mc(num_episodes):
    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))

    for i in range(num_episodes):
        episode = generate_episode()
        for j, (state,action, reward) in enumerate(episode):
            returns_sum[state, action]+= sum(x[2] for x in episode[j:])
            returns_count[state,action]+=1
    
    nonzero_counts = returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts]/returns_count[nonzero_counts]
    json_data = [{f'({row},{col})': Q[row][col] for col in range(num_actions)} for row in range(num_states)]
    json.dump(json_data, every_visit_mc_q_value_file, indent=4)
    every_visit_mc_q_value_file.close()
    return Q

first_visit_mc(num_episodes)
every_visit_mc(num_episodes)
