import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image

env = gym.make("Taxi-v3", render_mode='rgb_array')

def update_q_table(state, action, reward, next_state):
    old_value = q_table[state, action]
    next_max = max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])
    return action

max_actions = 100
num_episodes = 2000
alpha = 0.1  
gamma = 1  
epsilon = 0.9  
epsilon_decay = 0.999  
min_epsilon = 0.01

num_states, num_actions = env.observation_space.n, env.action_space.n
q_table = np.zeros((num_states, num_actions))
episode_returns = []

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    time_steps = 0
    
    while not done:
        
        if time_steps > max_actions:
            break
        
        action = epsilon_greedy(state)  
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        update_q_table(state, action, reward, next_state)
        episode_reward += reward
        state = next_state
        time_steps += 1
        
    episode_returns.append(episode_reward)
    epsilon = max(min_epsilon, epsilon*epsilon_decay)

policy = {state: np.argmax(q_table[state]) for state in range(num_states)}

state, info = env.reset()
frames = []
episode_total_reward = 0
time_steps = 0
done = False

for _ in range(100):

    while not done:
        
        frames.append(env.render())

        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_total_reward += reward
        state = next_state
        time_steps += 1

        done = terminated or truncated



imageio.mimsave('taxi_agent_behavior.gif', frames, fps=5)

gif_path = "taxi_agent_behavior.gif" 
Image(gif_path) 