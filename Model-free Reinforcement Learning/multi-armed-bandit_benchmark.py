import timeit

setup_code = """
import numpy as np

def epsilon_greedy():
    r = np.random.rand()
    if r < epsilon:
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
selected_arms = np.zeros(n_iterations, dtype=int)
"""

benchmark_code = """
for i in range(n_iterations):
    arm = epsilon_greedy()
    selected_arms[i] = arm
    counts[arm] += 1
    values[arm] += (np.random.rand() < true_bandit_probs[arm] - values[arm]) / counts[arm]
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
"""

runtime = timeit.timeit(stmt=benchmark_code, setup=setup_code, number=10)
average_runtime = runtime / 10  # Calculate average runtime per iteration

print("Average runtime:", average_runtime, "seconds")
