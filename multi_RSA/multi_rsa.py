import torch
import logging
from agents import pragmatic_speaker_A, pragmatic_speaker_B, pragmatic_listener_A, pragmatic_listener_B
from utils import bcolors, argmax

def multi_rsa(
        initial_lexica: list, 
        initial_prior: torch.FloatTensor, 
        game_model: dict, 
        A_meaning: int=None, 
        B_meaning=None, 
        A_utterances: list=None, 
        B_utterances: list=None, 
        alpha: int=1, 
        number_of_rounds: int=2, 
        RSA_depth: int=1, 
        device: torch.device="cpu", 
        logging: logging.Logger=None,
        verbose: bool=False,
        ):
    '''Run the RSA model for the specified number of iterations. 
    NOTA BENE: 
    - The model is currently only implemented for 2 agents.
    - The depth of RSA is fixed to 1.
    input:
    * initial_lexica, a list of 2D torch tensors of float, each gives a correspondance between i-th agent's meanings and i-th agent's utterances
    * initial_prior, a (n_agent+1)D torch tensor of float, the prior on the meaning of the agent 1 (dimension 0), ... the meaning of the agent n_agent (dimension n_agent-1), the concepts y (dimension n_agent)
    * game_model, a dict, the model of the game of the form {"mA": meanings_A, "mB": meanings_B, "u": utterances_A, "v": utterances_B, "y": Y}
    * A_meaning, an int, the meaning agent A
    * B_meaning, an int, the meaning agent B
    * A_utterances, a list of str, the utterances of agent A if known (default=None)
    * B_utterances, a list of str, the utterances of agent B if known (default=None)
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
    if A_utterances is not None:
        try:
            assert len(A_utterances) == number_of_rounds
        except AssertionError:
            logging.error(f"The number of utterances of Agent A does not match the number of rounds.")
            return
    if B_utterances is not None:
        try:
            assert len(B_utterances) == number_of_rounds
        except AssertionError:
            logging.error(f"The number of utterances of Agent B does not match the number of rounds.")
            return

    prior = initial_prior
    last_round = False

    if A_meaning is not None:
        logging.info(f"Agent A observes: {game_model['mA'][A_meaning]} / id={A_meaning}.")
        print("Agent A observes: " + bcolors.OKBLUE + f"{game_model['mA'][A_meaning]} / id={A_meaning}." + bcolors.ENDC)
    else:
        logging.info("Agent A's meaning is not known.")
        print("Agent A's meaning is not known.")
    if B_meaning is not None:
        logging.info(f"Agent B observes: {game_model['mB'][B_meaning]} / id={B_meaning}.")
        print("Agent B observes: " + bcolors.OKBLUE + f"{game_model['mB'][B_meaning]} / id={B_meaning}." + bcolors.ENDC)
    else:
        logging.info("Agent B's meaning is not known.")
        print("Agent B's meaning is not known.")

    estimations = [[],[]]
    produced_utterances = [[],[]]

    for round in range(number_of_rounds):
        logging.info(f"Round: {round}")
        if verbose:
            print(f"Round: {round}")

        if round == number_of_rounds - 1:
            last_round = True

        #### Agent A speaking ####
        # Compute the pragmatic speaker A
        pragmatic_speaker = pragmatic_speaker_A(initial_lexica[0], prior, alpha, RSA_depth, verbose)

        # Compute the utterance
        if A_utterances is None:
            utterance = argmax(pragmatic_speaker[A_meaning])
        else:
            utterance = A_utterances[round]
        produced_utterances[0].append(utterance)
        logging.info(f"Utterance of Agent A: {game_model['u'][utterance]}")
        if verbose:
            print(f"Utterance of Agent A: " + bcolors.OKGREEN + f"{game_model['u'][utterance]}" + bcolors.ENDC)

        # Update the prior
        prior = pragmatic_speaker[:, utterance].view(n_meanings[0], 1, 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

        if B_meaning is not None:
            # Compute the pragmatic listener B
            pragmatic_listener = pragmatic_listener_B(pragmatic_speaker, prior, alpha, RSA_depth, verbose)
            # if last_round and verbose:
            #     logging.info(f"Pragmatic Listener Agent B: {pragmatic_listener}")

            # Compute the estimated truths by agent B
            estimated_y = argmax(pragmatic_listener[B_meaning, :, utterance])
            estimations[1].append(estimated_y)
            logging.info(f"Estimated truth by Agent B: {game_model['y'][estimated_y]}")
            if verbose or last_round:
                print(f"Estimated truth by Agent B: " + bcolors.OKCYAN + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        elif verbose:
            logging.info("Agent B's meaning is not known.")
            print("Agent B's meaning is not known.")
        else:
            logging.info("Agent B's meaning is not known.")


        #### Agent B speaking ####
        # Compute the pragmatic speaker B
        pragmatic_speaker = pragmatic_speaker_B(initial_lexica[1], prior, alpha, RSA_depth, verbose)
        logging.debug(f"Pragmatic Speaker Agent B:")
        logging.debug(f"For meaning {B_meaning}, the probability is {pragmatic_speaker[B_meaning]}.")

        # Compute the utterance
        if B_utterances is None:
            utterance = argmax(pragmatic_speaker[B_meaning])
        else:
            utterance = B_utterances[round]
        produced_utterances[1].append(utterance)
        logging.info(f"Utterance of Agent B: {game_model['v'][utterance]}")
        if verbose:
            print(f"Utterance of Agent B: " + bcolors.HEADER + f"{game_model['v'][utterance]}" + bcolors.ENDC)

        # Update the prior
        prior = pragmatic_speaker[:, utterance].view(1, n_meanings[1], 1) * prior # S(v2|mA, u1, v1, u2) x P(mA, mB, y, u1, v1, u2) -> P(mA, mB, y, u1, v1, u2, v2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

        # Compute the pragmatic listener A
        if A_meaning is not None:
            pragmatic_listener = pragmatic_listener_A(pragmatic_speaker, prior, alpha, RSA_depth, verbose)
            # if last_round and verbose:
            #     logging.info(f"Pragmatic Listener Agent A: {pragmatic_listener}")

            # Compute the estimated truths by agent A
            estimated_y = argmax(pragmatic_listener[A_meaning, :, utterance])
            estimations[0].append(estimated_y)
            logging.info(f"Estimated truth by Agent A: {game_model['y'][estimated_y]}")
            if verbose or last_round:
                print(f"Estimated truth by Agent A: " + bcolors.FAIL + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        elif verbose:
            print("Agent A's meaning is unknown.")
        else:
            logging.info("Agent A's meaning is unknown.")

    return estimations, produced_utterances
        
