import gymnasium as gym 
import numpy as np
import json
import timeit 

def update_q_table(state, action, reward, next_state):
    i = np.random.randint(2)
    best_next_action = np.argmax(Q[i][next_state])
    Q[i][state, action] = (1-alpha)*Q[i][state,action] + alpha*(reward + gamma*Q[1-i][next_state, best_next_action])

def double_q_learning_loop(num_episodes):
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            update_q_table(state, action, reward, next_state)
            state = next_state

            done = terminated or truncated

double_q_learning_improved_q_value_file = open("double_q_learning_improved_q_value.json", "w")
env = gym.make("FrozenLake", render_mode="rgb_array", is_slippery = True)
num_states = env.observation_space.n
num_actions = env.action_space.n

Q = [np.zeros((num_states, num_actions))]*2
alpha = 0.1
gamma = 1
num_episodes = 1000000


function_call = lambda: double_q_learning_loop(num_episodes)

# Execute the function call 10 times and calculate the average time taken
time_taken = timeit.timeit(function_call, number=10)
average_time = time_taken / 10
print("Average time taken:", average_time)

final_Q = (Q[0] + Q[1])/2
json_data = [{f'({row},{col})': final_Q[row][col] for col in range(num_actions)} for row in range(num_states)]
json.dump(json_data, double_q_learning_improved_q_value_file, indent=4)
double_q_learning_improved_q_value_file.close()
