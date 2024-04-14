import gymnasium as gym
import json
import timeit


env = gym.make("CliffWalking-v0", render_mode = "human")
file = open("q_values.json", "r")
Q = json.load(file)
Q = {tuple(eval(k)):float(v) for k,v in Q.items()}

gamma = 0.9

terminal_state = 47

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

# computing state_values
def compute_state_value(state, policy):
    "This function estimates the state's worth"
    if state == 47:
        return  0
    action = policy[state]
    for state_tuple in env.unwrapped.P[state][action]:
        probability, next_state, reward, info = state_tuple
        return reward + gamma*compute_state_value(next_state, policy)


def policy_evaluation(policy):
    policy_evaluation_file = open("policy_evaluation.json", "w")
    v = {state: compute_state_value(state, policy) for state in range(env.observation_space.n)}
    json.dump(v, policy_evaluation_file, indent=4)
    return v

def compute_q_value(state, action, policy):
    "Calculates Q-value function also known as action-value function"

    if state == terminal_state:
        return 0
    
    probabilty, next_state, reward, info = env.unwrapped.P[state][action][0]
    return reward + gamma*compute_state_value(next_state, policy)

def policy_improvement(policy):

    improved_policy_file = open("improved_policy.json", "w")

    improved_policy ={state:0 for state in range(env.observation_space.n)}
    Q = {(state,action): compute_q_value(state,action,policy) for state in range(env.observation_space.n) for action in range(env.action_space.n)}
    
    for state in range(env.observation_space.n):
        max_action = max(range(env.action_space.n), key=lambda action:Q[(state,action)])
        improved_policy[state] = max_action

    json.dump(improved_policy, improved_policy_file, indent=4)    
    return improved_policy

def policy_iteration():

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

    while True:
        v = policy_evaluation(policy)
        improved_policy = policy_improvement(policy)

        if improved_policy == policy:
            break
        else:
            policy = improved_policy
    
    return policy, v


time_taken = timeit.timeit(policy_iteration, number=10)  # Execute the function 10 times
average_time = time_taken / 10  # Calculate the average time taken per function call
print("Average time taken:", average_time)