import numpy as np
from matplotlib import pyplot as plt
import h5py
from models import MODEL_CONFIGS

def load_h5_iterations(filename):
    results = {
        'estimations': [],
        'produced_utterances': [],
        'speakers': [],
        'listeners': [],
        'priors': [],
        'correct_guesses': []
    }
    
    with h5py.File(filename, 'r') as f:
        # Iterate through the groups (numbered iterations)
        for i in sorted(f.keys(), key=int):
            group = f[i]
            
            # Load each dataset for the iteration
            results['estimations'].append(group['estimations'][()])
            results['produced_utterances'].append(group['produced_utterances'][()])
            results['speakers'].append(group['speakers'][()])
            results['listeners'].append(group['listeners'][()])
            results['priors'].append(group['priors'][()])
            results['correct_guesses'].append(group['correct_guesses'][()])
    
    return results



# Create a figure and axis
fig, ax = plt.subplots()

# Plot the results for argmax depth 0 CRSA
ax.plot(
    range(1, argmax_depth_0_CRSA_results.shape[2]+1), 
    (np.mean(argmax_depth_0_CRSA_results[:, 0, :], axis=0) + np.mean(argmax_depth_0_CRSA_results[:, 1, :], axis=0))/2,
    label='Argmax depth 0 CRSA',
)

# Plot the results for classic depth 0 CRSA
ax.plot(
    range(1, classic_depth_0_CRSA_results.shape[2]+1), 
    (np.mean(classic_depth_0_CRSA_results[:, 0, :], axis=0) + np.mean(classic_depth_0_CRSA_results[:, 1, :], axis=0))/2,
    label='Classic depth 0 CRSA',
)

# Plot the results for argmax depth 1 CRSA
ax.plot(
    range(1, argmax_depth_1_CRSA_results.shape[2]+1),
    (np.mean(argmax_depth_1_CRSA_results[:, 0, :], axis=0) + np.mean(argmax_depth_1_CRSA_results[:, 1, :], axis=0))/2,
    label='Argmax depth 1 CRSA',
)

# Plot the results for classic depth 1 CRSA
ax.plot(
    range(1, classic_depth_1_CRSA_results.shape[2]+1),
    (np.mean(classic_depth_1_CRSA_results[:, 0, :], axis=0) + np.mean(classic_depth_1_CRSA_results[:, 1, :], axis=0))/2,
    label='Classic depth 1 CRSA',
)

# Plot the results for argmax depth 2 CRSA
ax.plot(
    range(1, argmax_depth_2_CRSA_results.shape[2]+1),
    (np.mean(argmax_depth_2_CRSA_results[:, 0, :], axis=0) + np.mean(argmax_depth_2_CRSA_results[:, 1, :], axis=0))/2,
    label='Argmax depth 2 CRSA',
)

# Plot the results for classic depth 2 CRSA
ax.plot(
    range(1, classic_depth_2_CRSA_results.shape[2]+1),
    (np.mean(classic_depth_2_CRSA_results[:, 0, :], axis=0) + np.mean(classic_depth_2_CRSA_results[:, 1, :], axis=0))/2,
    label='Classic depth 2 CRSA',
)

# Plot the results for classic multi classic RSA
ax.plot(
    range(1, classic_multi_classic_RSA_results.shape[2]+1), 
    (np.mean(classic_multi_classic_RSA_results[:, 0, :], axis=0) + np.mean(classic_multi_classic_RSA_results[:, 1, :], axis=0))/2,
    label='Classic multi classic RSA',
)

# Plot the results for argmax multi classic RSA
ax.plot(
    range(1, argmax_multi_classic_RSA_results.shape[2]+1), 
    (np.mean(argmax_multi_classic_RSA_results[:, 0, :], axis=0) + np.mean(argmax_multi_classic_RSA_results[:, 1, :], axis=0))/2,
    label='Argmax multi classic RSA',
)

'''
# Plot the results for the random model
ax.plot(
    range(1, random_results.shape[2]+1), 
    (np.mean(random_results[:, 0, :], axis=0) + np.mean(random_results[:, 1, :], axis=0))/2,
    label='Random Model',
)

# Plot the results for the greedy model
ax.plot(
    range(1, greedy_results.shape[2]+1), 
    (np.mean(greedy_results[:, 0, :], axis=0) + np.mean(greedy_results[:, 1, :], axis=0))/2,
    label='Greedy Model',
)
'''

# Set the labels and title
ax.set_xlabel('Round')
ax.set_title(f'Percentage of correct guesses for alpha={alpha}')

# Set the y-axis ticks and labels
ax.set_yticks([0, 1])
ax.set_yticklabels(['0%', '100%'])

# Add a legend
ax.legend()

# Save the plot as a file
os.makedirs(f'plots_planning_inference/{date_string}', exist_ok=True)
plt.savefig(f'plots_planning_inference/{date_string}/alpha_{alpha}_average.png', bbox_inches='tight')
