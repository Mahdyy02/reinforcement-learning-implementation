import matplotlib.pyplot as plt
import numpy as np

def epsilon_greedy():
    print(epsilon)
    r = np.random.rand()
    if  r < epsilon:
        arm = np.random.randint(n_bandits)
    else:
        arm = np.argmax(values)
    return arm

n_bandits = 5
true_bandit_probs = np.random.rand(n_bandits)
n_iterations = 50000
epsilon = 0.99
min_epsilon = 0.01
epsilon_decay = 0.999

counts = np.zeros(n_bandits)
values = np.zeros(n_bandits)
rewards = np.zeros(n_iterations)
selected_arms = np.zeros(n_iterations, dtype=int)

for i in range(n_iterations):
    arm = epsilon_greedy()
    print(arm)
    reward = np.random.rand() < true_bandit_probs[arm]
    rewards[i] = reward
    selected_arms[i] = arm
    counts[arm]+=1
    values[arm]+=(reward-values[arm])/counts[arm]
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


for i, prob in enumerate(true_bandit_probs, 1):
    print(f"Bandit#{i} -> {prob:.2f}")

selections_percentage = np.zeros((n_iterations, n_bandits))
for i in range(n_iterations):
    selections_percentage[i, selected_arms[i]] = 1
selections_percentage = np.cumsum(selections_percentage, axis = 0)/np.arange(1, n_iterations+1).reshape(-1,1)
for arm in range(n_bandits):
    plt.plot(selections_percentage[:,arm], label=f"Bandits#{arm+1}")

plt.xscale("log")
plt.xlabel("Episode Number")
plt.ylabel("Percentage of Bandit Selections (%)")
plt.legend()
plt.show()