import torch
import random
import numpy as np
import logging

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
        A_utterances: list=None,
        B_utterances: list=None,
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
    if A_utterances is not None:
        try:
            assert len(A_utterances) == number_of_rounds
        except AssertionError:
            logging.error(f"The number of utterances of Agent A does not match the number of rounds.")
            return False
    if B_utterances is not None:
        try:
            assert len(B_utterances) == number_of_rounds
        except AssertionError:
            logging.error(f"The number of utterances of Agent B does not match the number of rounds.")
            return False
    return True