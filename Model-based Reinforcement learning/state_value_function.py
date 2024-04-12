import gymnasium as gym
import json

env = gym.make("CliffWalking-v0", render_mode = "human")
state, info = env.reset()

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

gamma = 0.9

file = open("state_value.json", "w")

# computing state_values
def compute_state_value(state):
    "This function estimates the state's worth"
    if state == 47:
        return  0
    action = policy[state]
    for state_tuple in env.unwrapped.P[state][action]:
        probability, next_state, reward, info = state_tuple
        return reward + gamma*compute_state_value(next_state)


def policy_evaluation():
    v = {state: compute_state_value(state) for state in range(env.observation_space.n)}
    return v


json.dump(policy_evaluation(), file, indent=4)