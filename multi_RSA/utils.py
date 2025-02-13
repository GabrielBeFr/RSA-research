import torch
import random
import numpy as np
import logging
from matplotlib import pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def sample(X: torch.tensor, sampling: str):
    '''Sample from a distribution.
    input:
    * X, a torch tensor
    * sampling, a string, the sampling method
    
    output:
    * index, an int, the index of the maximum value in X
    '''
    if X.sum() == 0:
        X += 1/len(X)
    if sampling == 'argmax':
        index = argmax(X)
    elif sampling == 'softmax':
        index = softmax_sample(X)
    elif sampling == 'classic':
        index = classic_sample(X)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")

    return index

def argmax(X: torch.tensor, K: int = 1):
    '''Reimplement the torch.argmax function with equal probabilities of being output for equal values of X.
    input:
    * X, a torch tensor
    * K, an int, the number of argmaxes to return (default=1)
    
    output:
    * argmax, a torch tensor, the argmax of X along the given dimension
    '''
    # Compute all argmax indices
    argmaxes = torch.where(X == torch.max(X).item())[0]

    # Randomly select one of the argmax indices
    argmax = random.choice(argmaxes).item()

    return argmax

def softmax_sample(X: torch.tensor, temperature: float = 0.1):
    '''Reimplement the torch.softmax function with equal probabilities of being output for equal values of X.
    input:
    * X, a torch tensor
    * temperature, a float, the temperature of the softmax (default=1)
    
    output:
    * softmax_sample, a torch tensor, the softmax of X along the given dimension
    '''
    # Compute the softmax
    X = torch.exp(X/temperature)
    softmax = X/(X.sum() + 1e-10)

    # Sample an index based on the probabilities
    index = torch.multinomial(softmax, 1).item()

    return index

def classic_sample(X: torch.tensor):
    '''Simple torch.multinomial function to sample from a distribution.
    input:
    * X, a torch tensor
    
    output:
    * index, an int, the index of the maximum value in X
    '''
    # Sample an index based on the probabilities
    index = torch.multinomial(X, 1).item()

    return index

def test_variables_coherency(
        n_agents: int,
        initial_lexica: list,
        n_meanings: list,
        n_utterances: list,
        number_of_rounds: int=2,
):
    '''Test the variables for coherency.
    '''
    try:
        assert n_agents == 2
    except AssertionError:
        logging.error("The number of agents must be 2.")
        return False
    for i in range(n_agents):
        try:
            assert initial_lexica[i].shape[0] == n_meanings[i]
        except AssertionError:
            logging.error(f"The number of meanings of Agent {i} in the initial lexicon does not match the number of meanings in the prior.")
            return False
        try:
            assert initial_lexica[i].shape[1] == n_utterances[i]
        except AssertionError:
            logging.error(f"The number of utterances of Agent {i} in the initial lexicon does not match the number of utterances in the prior.")
            return False
    return True

def plot_scenario(id_scenario_to_plot, df_games):
    # scenarios = {'0,0,circle,cyan,A,1,0/0,1,rect,yellow,C,2,0/...', ...}
    scenarios = {}
    numero = 0
    for gameid_roundNum in df_games['gameid_roundNum'].unique():
        scenario_key = ""
        for row in df_games[df_games['gameid_roundNum'] == gameid_roundNum].iterrows():
            if scenario_key != "":
                scenario_key += "/"
            scenario_key += str(row[1]['pos_x']) + "," + str(row[1]['pos_y']) + "," + str(row[1]['shape']) + "," + str(row[1]['color']) + "," + str(row[1]['char']) + "," + str(row[1]['num']) + "," + str(row[1]['goal'])
        if scenario_key not in scenarios:
            scenarios[scenario_key] = {}
            numero += 1

    # Define the colors and shapes
    colors = {'cyan': 'c', 'Chartreuse': 'lime', 'yellow': 'gold'}
    shapes = {'rect': 's', 'circle': 'o', 'diamond': 'D'}

    # Define the positions
    positions = {
        '0,0': (0, 0),
        '0,1': (0, 1),
        '0,2': (0, 2),
        '1,0': (1, 0),
        '1,1': (1, 1),
        '1,2': (1, 2),
        '2,0': (2, 0),
        '2,1': (2, 1),
        '2,2': (2, 2)
    }

    # Define the underlined red text style
    underline_red = dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1')

    # Plot the scenarios
    fig, ax = plt.subplots(figsize=(10, 8))  # Set the figure size here
    

    scenario = list(scenarios.keys())[id_scenario_to_plot]
    objects = scenario.split('/')
    for obj in objects:
        x, y, shape, color, char, num, goal = obj.split(',')
        pos = positions[x + ',' + y]
        if goal == '1':
            ax.text(pos[0], - pos[1], char + num, color='black', size=20, fontweight='bold', ha='center', va='center', bbox=underline_red)
        else:
            ax.text(pos[0], - pos[1], char + num, color='black', size=20, ha='center', va='center')
        ax.plot(pos[0], -pos[1], marker=shapes[shape], color=colors[color], markersize=40)  # Flip the y-axis
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-2.5, 0.5])  # Adjust the y-axis limits
    ax.set_xticks(range(3))
    ax.set_yticks(range(-2, 1))  # Adjust the y-axis ticks
    ax.set_xticklabels(['0', '1', '2'])
    ax.set_yticklabels(['2', '1', '0'])  # Adjust the y-axis tick labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.xaxis.set_label_position('top')
    ax.set_title('Scenario ' + str(id_scenario_to_plot))
    ax.grid(True)
    plt.tight_layout()
    plt.show()