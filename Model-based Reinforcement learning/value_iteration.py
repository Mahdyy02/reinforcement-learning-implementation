import gymnasium as gym
import json

env = gym.make("CliffWalking-v0", render_mode="human")
terminal_state = 47
gamma = 0.9

def compute_q_value(state, action, v):
    if state == terminal_state:
        return None

    probability, next_state, reward, info = env.unwrapped.P[state][action][0]
    return reward + gamma*v[next_state]

def get_max_action_and_value(state,v):
    Q_values = [compute_q_value(state,action,v) for action in range(env.action_space.n)]
    max_action = max(range(env.action_space.n), key = lambda action: Q_values[action])
    max_q_value = Q_values[max_action]
    return max_action, max_q_value

def value_iteration():
    v = {state:0 for state in range(env.observation_space.n)}
    policy = {state:0 for state in range(env.observation_space.n-1)}
    threshold = 0.001

    value_iteration_improved_policy_file = open("value_iteration_improved_policy.json", "w")
    value_iteration_improved_value_function_file = open("value_iteration_improved_value_function.json","w")

    while True:

        new_v = {state:0 for state in range(env.observation_space.n)}

        for state in range(env.observation_space.n-1):
            max_action, max_q_value= get_max_action_and_value(state,v)
            new_v[state] = max_q_value
            policy[state] = max_action

        if all(abs(new_v[state] - v[state]) < threshold for state in range(env.observation_space.n)):
            break
        v = new_v
    
    json.dump(policy, value_iteration_improved_policy_file, indent=4)
    json.dump(v, value_iteration_improved_value_function_file, indent=4)
    return policy, v

value_iteration()