import numpy as np
from scipy.special import xlogy
from pdb import set_trace

def compute_shannon_conditional_entropy(priors, probabilities):
    '''Compute the Shannon entropy of the probabilities.
    input:
    * probabilities, a 2D array of float, the probability of each meaning given each utterance

    output: 
    * entropy, a float, the entropy of the probabilities
    '''
    entropy = -np.sum(xlogy(probabilities, probabilities), axis=0)
    entropy = np.sum(entropy * priors)
    return entropy

def compute_listener_value(priors, listener_probabilities, speaker_probabilities):
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
    value = np.sum(xlogy(speaker_probabilities, listener_probabilities), axis=0)
    value = np.sum(value * priors)
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

def not_converged(losses):
    '''Check if the losses have converged.
    input:
    * losses, a list of float, the losses

    output:
    * boolean, True if the losses have not converged, False otherwise
    '''
    if len(losses) < 2:
        return True
    return abs(losses[-1] - losses[-2]) > 1e-9

def compute_KL_div(proba1, proba2):
    '''Compute the KL divergence between two distributions.
    input:
    * proba1, a 3D array of float, a list of the first joint distributions
    * proba2, a 3D array of float, a list of the second joint distributions
    
    output:
    * kl_div, a float, a list of the KL divergences between the two distributions
    '''
    kl_div = []  
    for k in range(len(proba1)):
        proba1_k = np.clip(proba1[k], 1e-7, 1)
        proba2_k = np.clip(proba2[k], 1e-7, 1)
        kl_div.append(np.sum(np.sum(proba1_k * np.log(np.divide(proba1_k, proba2_k)), axis=0)))
    return kl_div
