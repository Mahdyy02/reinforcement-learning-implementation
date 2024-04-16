import gymnasium as gym 
import numpy as np
import json 
import matplotlib.pyplot as plt

def update_q_table(Q, state, action, reward, next_state):
    old_value = Q[state, action]
    next_max = max(Q[next_state])
    Q[state, action] = (1-alpha)*old_value + alpha*(reward + gamma*next_max)

def update_q_tables(Q, state, action, reward, next_state):
    i = np.random.randint(2)
    best_next_action = np.argmax(Q[i][next_state])
    Q[i][state, action] = (1-alpha)*Q[i][state,action] + alpha*(reward + gamma*Q[1-i][next_state, best_next_action])

def double_q_learning_loop(num_episodes):
    for _ in range(num_episodes):
        episode_reward = 0
        state, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            update_q_tables(Q, state, action, reward, next_state)
            state = next_state
            episode_reward+=reward
            done = terminated or truncated
        rewards_double_q_learning.append(episode_reward)
        avg_double_q_learning.append(np.mean(rewards_double_q_learning))

double_q_learning_improved_q_value_file = open("double_q_learning_improved_q_value.json", "w")

def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else: 
        action = np.argmax(Q_learning_epsilon_greedy[state,:])
    return action

def q_learning_epsilon_greedy_loop(num_episodes):

    global epsilon

    for _ in range(num_episodes):
        episode_reward = 0
        state, info = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            update_q_table(Q_learning_epsilon_greedy, state, action, reward, next_state)
            state = next_state
            episode_reward+=reward
            done = terminated or truncated
        epsilon = max(min_epsilon, epsilon*epsilon_decay)
        rewards_epsilon_greedy.append(episode_reward)
        avg_eps_greedy.append(np.mean(rewards_epsilon_greedy))

def q_learning_loop(num_episodes):
    for _ in range(num_episodes):
        episode_reward = 0
        state, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            update_q_table(Q_learning, state, action, reward, next_state)
            state = next_state
            episode_reward+=reward
            done = terminated or truncated
        rewards_decay_eps_greedy.append(episode_reward)
        avg_decay.append(np.mean(rewards_decay_eps_greedy))

double_q_learning_improved_q_value_file = open("double_q_learning_improved_q_value.json", "w")
q_learning_improved_q_value_file = open("q_learning_improved_q_value.json", "w")
q_learning_epsilon_greedy_improved_q_value_file = open("q_learning_epsilon_greedy_improved_q_value.json", "w")
chatgpt = open("chatgpt.json", "w")

env = gym.make("FrozenLake", render_mode="rgb_array", is_slippery = True)
num_states = env.observation_space.n
num_actions = env.action_space.n

Q = [np.zeros((num_states, num_actions))]*2
Q_learning = np.zeros((num_states, num_actions))
Q_learning_epsilon_greedy = np.zeros((num_states, num_actions))
alpha = 0.1
gamma = 1
epsilon = 1
epsilon_decay = 0.999
min_epsilon = 0.01
num_episodes = 5000

rewards_decay_eps_greedy = []
rewards_epsilon_greedy = []
rewards_double_q_learning = []
avg_eps_greedy = []
avg_decay = []
avg_double_q_learning = []

q_learning_epsilon_greedy_loop(num_episodes)
q_learning_loop(num_episodes)
double_q_learning_loop(num_episodes)

Q_learning_double = (Q[0] + Q[1])/2

json_data = [{f'({row},{col})': Q_learning[row][col] for col in range(num_actions)} for row in range(num_states)]
json.dump(json_data, q_learning_epsilon_greedy_improved_q_value_file, indent=4)
q_learning_epsilon_greedy_improved_q_value_file.close()
json_data = [{f'({row},{col})': Q_learning[row][col] for col in range(num_actions)} for row in range(num_states)]
json.dump(json_data, q_learning_improved_q_value_file, indent=4)
q_learning_improved_q_value_file.close()
json_data = [{f'({row},{col})': Q_learning_double[row][col] for col in range(num_actions)} for row in range(num_states)]
json.dump(json_data, double_q_learning_improved_q_value_file, indent=4)
double_q_learning_improved_q_value_file.close()

episode_numbers = np.arange(len(avg_eps_greedy))

smooth_rewards_epsilon_greedy = np.interp(episode_numbers, episode_numbers, avg_eps_greedy)
smooth_rewards_decay_eps_greedy = np.interp(episode_numbers, episode_numbers, avg_decay)
smooth_rewards_double_q_learning = np.interp(episode_numbers, episode_numbers, avg_double_q_learning)


plt.plot(episode_numbers, smooth_rewards_epsilon_greedy, label='Epsilon Greedy')
plt.plot(episode_numbers, smooth_rewards_decay_eps_greedy, label='Decayed Epsilon Greedy')
plt.plot(episode_numbers, avg_double_q_learning, label='Double Q-learning')

plt.title("Average Reward per Episode")
plt.xlabel("Episode Number")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)
plt.show()
