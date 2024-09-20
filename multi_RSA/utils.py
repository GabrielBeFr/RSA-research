import torch
import random

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