import torch
import torch.nn.functional as F

def literal_listener(lexicon, priors):
    '''The literal listener is a simple listener that interprets the utterance.
    input: 
    * lexicon, a 2D tensor of boolean, the lexicon, where lexicon[i,j]=True if meaning i maps to utterance j and False otherwise
    * priors, a 1D tensor of float, the priors on the meaning list
    output:
    '''
    probabilities = torch.clone(lexicon)
    probabilities = probabilities * priors.unsqueeze(1)
    probabilities = F.normalize(probabilities, p=1, dim=0)
    return probabilities

def pragmatic_speaker(listener_probabilities, priors, costs=None, speaker_probabilities=None, alpha=1, version="RSA"):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input:
    * listener_probabilities, a 2D tensor of float, the probability of each meaning given each utterance (row: meaning, column: utterance)
    * priors, a 1D tensor of float, the priors on the meaning list
    * costs, a 1D tensor of float, the cost of each utterance
    * speaker_probabilities, a 2D tensor of float, the probability given by the previous 
    speaker to each meaning conditionned to each utterance (used only in "RD-RSA" version) (default=None)
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
    * version, a string, the version of the RSA model (default="RSA")

    output: 
    * probabilities, a 2D tensor of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = torch.clone(listener_probabilities)
    probabilities = torch.pow(probabilities, alpha)
    if costs is not None:
        probabilities = torch.exp(-alpha * costs).unsqueeze(1) * probabilities
    if version == "RD-RSA":
        S_u = torch.matmul(speaker_probabilities, priors)
        probabilities = S_u.unsqueeze(1) * probabilities
    probabilities = F.normalize(probabilities, p=1, dim=1)
    return probabilities

def pragmatic_listener(speaker_probabilities, priors):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input: 
    * speaker_probabilities, a 2D tensor of float, the probability of each utterance given each meaning (row: meaning, column: utterance)   
    * priors, a 1D tensor of float, the priors on the meaning list
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D tensor of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = torch.clone(speaker_probabilities)
    probabilities = probabilities * priors.unsqueeze(1)
    probabilities = F.normalize(probabilities, p=1, dim=0)
    return probabilities


################## LEXICAL UNCERTAINTY ##################


def literal_listener_lexical_uncertainty(world):
    '''The literal listener is a simple listener that interprets the utterance.
    input: 
    * world, a dictionary with the following
        - world['lexica']: a 3D tensor of boolean, the lexica contains multiple lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D tensor of float, the cost of each utterance
        - world['priors']: a 1D tensor of float, the priors on the meaning list

    output:
    * probabilities, a 2D tensor of float, the probability of each meaning given each utterance
    '''
    probabilities = torch.clone(world["lexica"])
    probabilities = probabilities * world["priors"]
    for i in range(len(probabilities)):
        probabilities[i] = F.normalize(probabilities[i], p=1, dim=1)
    return probabilities

def pragmatic_speaker_lexical_uncertainty(world, listener_probabilities, alpha=1):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input: 
    * world, a dictionary with the following
        - world['lexica']: a 3D tensor of boolean, the lexica contains multiple lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D tensor of float, the cost of each utterance
        - world['priors']: a 1D tensor of float, the priors on the meaning list
    * listener_probabilities, a 2D tensor of float, the probability of each meaning given each utterance
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D tensor of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = torch.clone(listener_probabilities)
    probabilities = torch.pow(probabilities, alpha)
    for i in range(len(probabilities)):
        probabilities[i] = (probabilities[i].T * torch.exp(-alpha * world['costs'])).T
        probabilities[i] = F.normalize(probabilities[i], p=1, dim=0)
    return probabilities

def pragmatic_listener_lexical_uncertainty(world, speaker_probabilities):
    '''The pragmatic listener is a listener that interprets the utterance, taking into account the speaker's bias.
    input: 
    * world, a dictionary with the following
        - world['lexica']: a 3D tensor of boolean, the lexica contains multiple lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['costs']: a 1D tensor of float, the cost of each utterance
        - world['priors']: a 1D tensor of float, the priors on the meaning list
    * speaker_probabilities, a 2D tensor of float, the probability of each meaning given each utterance
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)

    output: 
    * probabilities, a 2D tensor of float, the probability of each meaning given each utterance, after speaker inference
    '''
    probabilities = torch.clone(speaker_probabilities)
    probabilities = probabilities * world['priors']
    sum_probabilities = torch.sum(probabilities, dim=0) # TODO: add a specific probability for each lexicon
    probabilities = F.normalize(sum_probabilities, p=1, dim=1)
    return probabilities