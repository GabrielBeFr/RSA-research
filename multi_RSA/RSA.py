import torch
from torch import nn
from torch.nn import functional as F
from RSA_agents import literal_listener, pragmatic_speaker, pragmatic_listener
import logging
from utils import bcolors, sample, test_variables_coherency


def run_RSA(lexicon, priors, alpha=1, depth=1, costs=None, version="RSA", verbose=False):
    '''Run the RSA model for the specified number of iterations.
    input:
    * lexicon, a 2D tensor of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
    * costs, a 1D tensor of float, the cost of each utterance
    * priors, a 1D tensor of float, the priors on the meaning list
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
    * depth, an integer, the number of iterations of the RSA model (default=1)
    * verbose, a boolean, whether to print the probabilities at each step (default=False)

    output: 
    * listeners, a list of 2D tensor of float, the probability of each meaning given each utterance
    * speakers, a list of 2D tensor of float, the probability of each meaning given each utterance
    '''

    listeners = []
    speakers = []

    listener_probabilities = literal_listener(lexicon=lexicon, priors=priors)
    listeners.append(listener_probabilities)

    speaker_probabilities = F.normalize(lexicon, p=1, dim=1)
    speakers.append(speaker_probabilities)

    for step in range(1, depth+1):
        speaker_probabilities = pragmatic_speaker(listener_probabilities=listener_probabilities, priors=priors, alpha=alpha, costs=costs, version=version)
        speakers.append(speaker_probabilities)

        listener_probabilities = pragmatic_listener(speaker_probabilities=speaker_probabilities, priors=priors)
        listeners.append(listener_probabilities)

    return listeners, speakers

def multi_classic_RSA(
        initial_lexica: list, 
        initial_prior: torch.FloatTensor, 
        game_model: dict, 
        A_meaning: int=None, 
        B_meaning: int=None, 
        A_utterances: list=None, 
        B_utterances: list=None, 
        alpha: int=1, 
        number_of_rounds: int=2, 
        RSA_depth: int=1, 
        sampling: str="classic",
        device: torch.device="cpu", 
        logging: logging.Logger=None,
        verbose: bool=False,
        ):
    '''Run the RSA model for the specified number of iterations.
    input:
    * initial_lexica, a list of 2D tensor of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
    * initial_prior, a 1D tensor of float, the priors on the meaning list
    * game_model, a dictionary with the following
        - game_model['costs']: a 1D tensor of float, the cost of each utterance
        - game_model['priors']: a 1D tensor of float, the priors on the meaning list
    * A_meaning, an integer, the meaning of the A agent
    * B_meaning, an integer, the meaning of the B agent
    * A_utterances, a list of integers, the utterances of the A agent
    * B_utterances, a list of integers, the utterances of the B agent
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
    * number_of_rounds, an integer, the number of iterations of the RSA model (default=2)
    * RSA_depth, an integer, the number of iterations of the RSA model (default=1)
    * sampling, a string, the sampling method to use (default="classic")
    * device, a torch device, the device to use (default="cpu")
    * logging, a logging object, the logger to use (default=None)
    * verbose, a boolean, whether to print the probabilities at each step (default=False)

    output:
    * estimations, a list of 2D tensor of float, the probability of each meaning given each utterance
    * produced_utterances, a list of 2D tensor of float, the probability of each meaning given each utterance
    * literal_listeners, a list of 2D tensor of float, the probability of each meaning given each utterance
    * pragmatic_speakers, a list of 2D tensor of float, the probability of each meaning given each utterance
    * pragmatic_listeners, a list of 2D tensor of float, the probability of each meaning given each utterance
    '''
    # Define useful constants
    n_agents = len(initial_lexica)
    n_utterances = [initial_lexica[i].shape[1] for i in range(n_agents)]
    n_meanings = [initial_prior.shape[i] for i in range(n_agents)]
    prior = initial_prior

    # Check the variables for coherency
    if not test_variables_coherency(
        n_agents=n_agents, 
        initial_lexica=initial_lexica, 
        n_meanings=n_meanings, 
        n_utterances=n_utterances, 
        A_utterances=A_utterances, 
        B_utterances=B_utterances, 
        number_of_rounds=number_of_rounds,
        ):
        return None
    
    # Print the initial conditions
    if A_meaning is not None:
        logging.info(f"Agent A observes: {game_model['mA'][A_meaning]} / id={A_meaning}.")
        if verbose:
            print("Agent A observes: " + bcolors.OKBLUE + f"{game_model['mA'][A_meaning]} / id={A_meaning}." + bcolors.ENDC)
    else:
        logging.info("Agent A's meaning is not known.")
        if verbose:
            print("Agent A's meaning is not known.")
    if B_meaning is not None:
        logging.info(f"Agent B observes: {game_model['mB'][B_meaning]} / id={B_meaning}.")
        if verbose:
            print("Agent B observes: " + bcolors.OKBLUE + f"{game_model['mB'][B_meaning]} / id={B_meaning}." + bcolors.ENDC)
    else:
        logging.info("Agent B's meaning is not known.")
        if verbose:
            print("Agent B's meaning is not known.")

    # Initialize the variables
    last_round = False
    estimations = [[],[]]
    produced_utterances = [[],[]]
    list_speakers = [[],[]]
    list_listeners = [[],[]]

    for round in range(number_of_rounds):
        # Round agent A
        listeners, speakers = run_RSA(
            lexicon=initial_lexica[0], 
            priors=(prior.sum(dim=2)).sum(dim=1),
            alpha=alpha, 
            depth=RSA_depth, 
            verbose=verbose,
            ) # (prior.sum(dim=2)).sum(dim=1) for having only P(m_A) in the prior instead of P(mA,mB,y)
        
        list_listeners[1].append(listeners)
        list_speakers[0].append(speakers)

        produced_utterance = sample(speakers[-1][A_meaning,:], sampling)
        produced_utterances[0].append(produced_utterance)
        logging.info(f"Utterance of Agent A: {game_model['u'][produced_utterance]}")
        if verbose:
            print(f"Utterance of Agent A: " + bcolors.OKGREEN + f"{game_model['u'][produced_utterance]}" + bcolors.ENDC)

        estimation_meaning_A = sample(listeners[-1][:,produced_utterance], sampling)
        estimation = sample(initial_prior[estimation_meaning_A,B_meaning,:], sampling)
        estimations[1].append(estimation)
        logging.info(f"Estimation of Agent B: {game_model['y'][estimation]}")
        if verbose:
            print(f"Estimation of Agent B: " + bcolors.OKGREEN + f"{game_model['y'][estimation]}" + bcolors.ENDC)

        # Round agent B
        listeners, speakers = run_RSA(
            lexicon=initial_lexica[1], 
            priors=(prior.sum(dim=2)).sum(dim=0), 
            alpha=alpha, 
            depth=RSA_depth,
            verbose=verbose,
            ) # (prior.sum(dim=2)).sum(dim=0) for having only P(m_B) in the prior instead of P(mA,mB,y)
        
        list_listeners[0].append(listeners)
        list_speakers[1].append(speakers)

        produced_utterance = sample(speakers[-1][B_meaning,:], sampling)
        produced_utterances[1].append(produced_utterance)
        logging.info(f"Utterance of Agent B: {game_model['v'][produced_utterance]}")
        if verbose:
            print(f"Utterance of Agent B: " + bcolors.OKGREEN + f"{game_model['v'][produced_utterance]}" + bcolors.ENDC)

        estimation_meaning_B = sample(listeners[-1][:,produced_utterance], sampling)
        estimation = sample(initial_prior[A_meaning,estimation_meaning_B,:], sampling)
        logging.info(f"Estimation of Agent A: {game_model['y'][estimation]}")
        if verbose:
            print(f"Estimation of Agent A: " + bcolors.OKGREEN + f"{game_model['y'][estimation]}" + bcolors.ENDC)

        estimations[0].append(estimation)

    return estimations, produced_utterances, list_speakers, list_listeners, [prior]
