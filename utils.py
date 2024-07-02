import numpy as np
from pdb import set_trace

def compute_xlogy(x,y):
    '''Compute the x*log(y) of x, even in the case x=0=y.
    input:
    * x, an array
    * y, an array

    output: 
    * xlogy, an array, the value of x*log(y)
    '''
    xlogy = x * np.log(y, where=(x!=0))
    return xlogy

def compute_shannon_conditional_entropy(priors, probabilities):
    '''Compute the Shannon entropy of the probabilities.
    input:
    * probabilities, a 2D array of float, the probability of each meaning given each utterance

    output: 
    * entropy, a float, the entropy of the probabilities
    '''
    entropy = -np.sum(compute_xlogy(probabilities, probabilities), axis=0)
    entropy = np.sum(entropy * priors)
    return entropy

def compute_listener_value(alpha, priors, listener_probabilities, speaker_probabilities):
    '''Compute the value of the listener.
    input:
    * alpha, a float, the speaker pragmatism parameter
    * listener_probabilities, a 2D array of float, the probability of each meaning given each utterance
    * speaker_probabilities, a 2D array of float, the probability given by the previous speaker to each meaning conditionned to each utterance
    * world, a dictionary with the following
        - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list

    output: 
    * value, a float, the value of the listener
    '''
    value = np.sum(compute_xlogy(speaker_probabilities, listener_probabilities), axis=0)
    value = alpha * np.sum(value * priors)
    return value

def compute_proportionality_factor(priors, listener_probabilities, speaker_probabilities):
    '''Deprecated. Compute the proportionality factor that should exist between two expressions dependent
    on the optimal speaker and listener couple.
    input:
    * listener_probabilities, a 2D array of float, the probability of each meaning given each utterance
    * speaker_probabilities, a 2D array of float, the probability given by the previous speaker to each meaning conditionned to each utterance
    
    output:
    * res, a 2D array of float, the matrix equality
    '''
    res = np.divide(listener_probabilities,speaker_probabilities, where=speaker_probabilities!=0)
    res = priors*np.log(res, where=(speaker_probabilities!=0))
    print(res.shape)
    return np.divide(speaker_probabilities, res, where=(speaker_probabilities!=0))