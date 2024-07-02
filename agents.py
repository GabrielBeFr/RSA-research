import numpy as np
from sklearn import preprocessing
from pdb import set_trace

def litteral_listener(world):
    '''The litteral listener is a simple listener that interprets the utterance.
    input: 
    * world, a dictionary with the following
        - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list

    output:
    * probabilities, a 2D array of float, the probability of each meaning given each utterance
    '''
    probabilities = np.copy(world['lexicon'])
    probabilities = probabilities * world['priors']
    probabilities = preprocessing.normalize(probabilities, norm='l1', axis=1)
    return probabilities

def pragmatic_speaker(world, speaker_probabilities, listener_probabilities, alpha=1, version="RSA"):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input:
    * world, a dictionary with the following
        - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list
    * speaker_probabilities, a 2D array of float, the probability given by the previous 
    speaker to each meaning conditionned to each utterance (used only in "RD-RSA" version)
    * listener_probabilities, a 2D array of float, the probability of each meaning given each utterance
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D array of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = np.copy(listener_probabilities)
    probabilities = np.power(probabilities, alpha)
    probabilities = np.reshape(np.exp(-alpha * world['costs']),(1,-1)).T * probabilities
    if version == "RD-RSA":
        S_u = np.dot(speaker_probabilities,world["priors"])
        probabilities = np.reshape(S_u,(1,-1)).T*probabilities
    probabilities = preprocessing.normalize(probabilities, norm='l1', axis=0)
    return probabilities

def pragmatic_listener(world, speaker_probabilities):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input: 
    * world, a dictionary with the following
        - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list
    * speaker_probabilities, a 2D array of float, the probability of each meaning given each utterance
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D array of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = np.copy(speaker_probabilities)
    probabilities = probabilities * world['priors']
    probabilities = preprocessing.normalize(probabilities, norm='l1', axis=1)
    return probabilities


################## LEXICAL UNCERTAINTY ##################


def litteral_listener_lexical_uncertainty(world):
    '''The litteral listener is a simple listener that interprets the utterance.
    input: 
    * world, a dictionary with the following
        - world['lexica']: a 3D array of boolean, the lexica contains multiple lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list

    output:
    * probabilities, a 2D array of float, the probability of each meaning given each utterance
    '''
    probabilities = np.copy(world["lexica"])
    probabilities = probabilities * world["priors"]
    for i in range(len(probabilities)):
        probabilities[i] = preprocessing.normalize(probabilities[i], norm='l1', axis=1)
    return probabilities

def pragmatic_speaker_lexical_uncertainty(world, listener_probabilities, alpha=1):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input: 
    * world, a dictionary with the following
        - world['lexica']: a 3D array of boolean, the lexica contains multiple lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list
    * listener_probabilities, a 2D array of float, the probability of each meaning given each utterance
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D array of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = np.copy(listener_probabilities)
    probabilities = np.power(probabilities, alpha)
    for i in range(len(probabilities)):
        probabilities[i] = (probabilities[i].T * np.exp(-alpha * world['costs'])).T
        probabilities[i] = preprocessing.normalize(probabilities[i], norm='l1', axis=0)
    return probabilities

def pragmatic_listener_lexical_uncertainty(world, speaker_probabilities):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input: 
    * world, a dictionary with the following
        - world['lexica']: a 3D array of boolean, the lexica contains multiple lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D array of float, the cost of each utterance
        - world['priors']: a 1D array of float, the priors on the meaning list
    * speaker_probabilities, a 2D array of float, the probability of each meaning given each utterance
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D array of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = np.copy(speaker_probabilities)
    probabilities = probabilities * world['priors']
    sum_probabilities = np.sum(probabilities, axis=0) # TODO: add a specific probability for each lexicon
    probabilities = preprocessing.normalize(sum_probabilities, norm='l1', axis=1)
    return probabilities