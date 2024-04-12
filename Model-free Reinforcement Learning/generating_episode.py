import gymnasium as gym

env = gym.make("FrozenLake", render_mode = "human", is_slippery = True)

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

episode = generate_episode()
print(episode)