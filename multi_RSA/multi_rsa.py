import torch
import logging
from agents import pragmatic_speaker_A, pragmatic_speaker_B, pragmatic_listener_A, pragmatic_listener_B
from utils import bcolors, argmax

def multi_rsa(initial_lexica: list, initial_prior: torch.FloatTensor, meanings: list, game_model: dict, alpha: int=1, number_of_rounds: int=2, RSA_depth: int=1, device: torch.device="cpu", verbose: bool=False):
    '''Run the RSA model for the specified number of iterations. 
    NOTA BENE: 
    - The model is currently only implemented for 2 agents.
    - The depth of RSA is fixed to 1.
    input:
    * initial_lexica, a list of 2D torch tensors of float, each gives a correspondance between i-th agent's meanings and i-th agent's utterances
    * initial_prior, a (n_agent+1)D torch tensor of float, the prior on the meaning of the agent 1 (dimension 0), ... the meaning of the agent n_agent (dimension n_agent-1), the concepts y (dimension n_agent)
    * meanings, a list of int, the meanings of each agent
    * game_model, a dict, the model of the game of the form {"mA": meanings_A, "mB": meanings_B, "u": utterances_A, "v": utterances_B, "y": Y}
    * alpha, a float, the alpha pragmatism parameter of RSA (default=1)
    * number_of_rounds, an int, the number of rounds to run the model for (default=2)
    * RSA_depth, an int, the depth of the RSA model (default=1)
    * device, a torch device, the device to run the model on (default="cpu")
    * verbose, a boolean, whether to print the probabilities at each step (default=False)
    '''

    n_agents = len(initial_lexica)
    n_worlds = initial_prior.shape[-1]
    n_utterances = [initial_lexica[i].shape[1] for i in range(n_agents)]
    n_meanings = [initial_prior.shape[i] for i in range(n_agents)]
    try:
        assert n_agents == 2
    except AssertionError:
        logging.error("The number of agents must be 2.")
        return
    for i in range(n_agents):
        try:
            assert initial_lexica[i].shape[0] == n_meanings[i]
        except AssertionError:
            logging.error(f"The number of meanings of Agent {i} in the initial lexicon does not match the number of meanings in the prior.")
            return
        try:
            assert initial_lexica[i].shape[1] == n_utterances[i]
        except AssertionError:
            logging.error(f"The number of utterances of Agent {i} in the initial lexicon does not match the number of utterances in the prior.")
            return

    prior = initial_prior
    last_round = False

    logging.info("Agent A observes: " + bcolors.OKBLUE + f"{game_model['mA'][meanings[0]]} / id={meanings[0]} " + bcolors.ENDC + "and Agent B observes: " + bcolors.OKBLUE + f"{game_model['mB'][meanings[1]]} / id={meanings[1]}" + bcolors.ENDC)

    for round in range(number_of_rounds):
        if verbose:
            logging.info(f"Round: {round}")

        if round == number_of_rounds - 1:
            last_round = True

        #### Agent A speaking ####
        # Compute the pragmatic speaker A
        pragmatic_speaker = pragmatic_speaker_A(initial_lexica[0], prior, alpha, RSA_depth, verbose)

        # Compute the utterance
        utterance = argmax(pragmatic_speaker[meanings[0]])
        if verbose:
            logging.info(f"Utterance of Agent A: " + bcolors.OKGREEN + f"{game_model['u'][utterance]}" + bcolors.ENDC)

        # Update the prior
        prior = pragmatic_speaker[:, utterance].view(n_meanings[0], 1, 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

        # Compute the pragmatic listener B
        pragmatic_listener = pragmatic_listener_B(pragmatic_speaker, prior, alpha, RSA_depth, verbose)
        if last_round and verbose:
            logging.info(f"Pragmatic Listener Agent B: {pragmatic_listener}")

        # Compute the estimated meanings by agent B
        estimated_y = argmax(pragmatic_listener[meanings[1], :, utterance])
        if verbose or last_round:
            logging.info(f"Estimated meaning by Agent B: " + bcolors.OKCYAN + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)


        #### Agent B speaking ####
        # Compute the pragmatic speaker B
        pragmatic_speaker = pragmatic_speaker_B(initial_lexica[1], prior, alpha, RSA_depth, verbose)

        # Compute the utterance
        utterance = argmax(pragmatic_speaker[meanings[1]])
        if verbose:
            logging.info(f"Utterance of Agent B: " + bcolors.HEADER + f"{game_model['v'][utterance]}" + bcolors.ENDC)

        # Update the prior
        prior = pragmatic_speaker[:, utterance].view(1, n_meanings[1], 1) * prior # S(v2|mA, u1, v1, u2) x P(mA, mB, y, u1, v1, u2) -> P(mA, mB, y, u1, v1, u2, v2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

        # Compute the pragmatic listener A
        pragmatic_listener = pragmatic_listener_A(pragmatic_speaker, prior, alpha, RSA_depth, verbose)
        if last_round and verbose:
            logging.info(f"Pragmatic Listener Agent A: {pragmatic_listener}")

        # Compute the estimated meanings by agent A
        estimated_y = argmax(pragmatic_listener[meanings[0], :, utterance])
        if verbose or last_round:
            logging.info(f"Estimated meaning by Agent A: " + bcolors.FAIL + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        
