import torch
import logging
from CRSA_agents import literal_listener_A, literal_listener_B, pragmatic_speaker_A, pragmatic_speaker_B, pragmatic_listener_A, pragmatic_listener_B
from utils import bcolors, sample, test_variables_coherency

def CRSA(
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
    * choice_method, a string, what sampling method to use for utterance and estimation choices (default="classic_sample")
    * device, a torch device, the device to run the model on (default="cpu")
    * logging, a logging.Logger, the logger to log the results (default=None)
    * verbose, a boolean, whether to print the probabilities at each step (default=False)
    '''

    # Define useful constants
    n_agents = len(initial_lexica)
    n_utterances = [initial_lexica[i].shape[1] for i in range(n_agents)]
    n_meanings = [initial_prior.shape[i] for i in range(n_agents)]
    prior = initial_prior
    
    # Check the variables for coherency
    if not test_variables_coherency(n_agents, initial_lexica, n_meanings, n_utterances, A_utterances, B_utterances, number_of_rounds):
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
    speakers = [[],[]]
    listeners = [[],[]]
    for i in range(RSA_depth+1):
        listeners[0].append([])
        listeners[1].append([])
        speakers[0].append([])
        speakers[1].append([])

    # Run the CRSA for the specified number of rounds
    for round in range(number_of_rounds):
        logging.info(f"Round: {round}")
        if verbose:
            print(f"Round: {round}")

        if round == number_of_rounds - 1:
            last_round = True

        #### Agent A speaking ####
        # Compute the literal listener B
        listener = literal_listener_B(initial_lexica[0], prior, verbose)
        # Save the literal listener probabilities
        listeners[1][0].append(listener[B_meaning, :, :])

        i = 0
        while True:
            # Compute the pragmatic speaker A
            speaker = pragmatic_speaker_A(listener, prior, alpha, RSA_depth, verbose)
            # Save the pragmatic speaker probabilities
            speakers[0][i].append(speaker[A_meaning])

            i += 1
            if i > RSA_depth:
                break

            # Compute the pragmatic listener B
            listener = pragmatic_listener_B(speaker, prior, alpha, RSA_depth, verbose)
            # Save the pragmatic listener probabilities
            listeners[1][i].append(listener[B_meaning, :, :])

        if A_utterances is None:
            try:
                utterance = sample(speaker[A_meaning], sampling) # Compute the utterance if not known
            except Exception as e:
                print(f"Line 104 and: {speaker[A_meaning]}")
        else:
            utterance = A_utterances[round] # Retrieve the utterance if known
        produced_utterances[0].append(utterance)
        logging.info(f"Utterance of Agent A: {game_model['u'][utterance]}")
        if verbose:
            print(f"Utterance of Agent A: " + bcolors.OKGREEN + f"{game_model['u'][utterance]}" + bcolors.ENDC)

        # Compute the estimated truths by agent B
        if B_meaning is not None:
            try:
                estimated_y = sample(listener[B_meaning, :, utterance], sampling)
            except Exception as e:
                print(f"Line 130 and: {listener[B_meaning, :, utterance]}")
            estimations[1].append(estimated_y)
            logging.info(f"Estimated truth by a Agent B: {game_model['y'][estimated_y]}")
            if verbose:
                print(f"Estimated truth by a Agent B: " + bcolors.OKCYAN + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        else:
            if verbose: print("Agent B's meaning is unknown.")
            logging.info("Agent B's meaning is unknown.")

        # Update the prior
        prior = speaker[:, utterance].view(n_meanings[0], 1, 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

        #### Agent B speaking ####
        # Compute the literal listener A
        listener = literal_listener_A(initial_lexica[1], prior, verbose)
        # Save the literal listener probabilities
        listeners[0][0].append(listener[A_meaning, :, :])

        i = 0
        while True:
            # Compute the pragmatic speaker B
            speaker = pragmatic_speaker_B(listener, prior, alpha, RSA_depth, verbose)
            # Save the pragmatic speaker probabilities
            speakers[1][i].append(speaker[B_meaning])

            i += 1
            if i > RSA_depth:
                break

            # Compute the pragmatic listener A
            listener = pragmatic_listener_A(speaker, prior, alpha, RSA_depth, verbose)
            # Save the pragmatic listener probabilities
            listeners[0][i].append(listener[A_meaning, :, :])
        

        # Compute the utterance
        if B_utterances is None:
            try:
                utterance = sample(speaker[B_meaning], sampling)
            except Exception as e:
                print(f"Line 170 and: {speaker[B_meaning]}")
        else:
            utterance = B_utterances[round]
        produced_utterances[1].append(utterance)
        logging.info(f"Utterance of Agent B: {game_model['v'][utterance]}")
        if verbose:
            print(f"Utterance of Agent B: " + bcolors.HEADER + f"{game_model['v'][utterance]}" + bcolors.ENDC)


        # Compute the pragmatic listener A
        if A_meaning is not None:
            try:
                estimated_y = sample(listener[A_meaning, :, utterance], sampling)
            except Exception as e:
                print(f"Line 202 and: {listener[A_meaning, :, utterance]}")
            estimations[0].append(estimated_y)
            logging.info(f"Estimated truth by a pragmatic Agent A: {game_model['y'][estimated_y]}")
            if verbose:
                print(f"Estimated truth by a pragmatic Agent A: " + bcolors.FAIL + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        else:
            if verbose: print("Agent A's meaning is unknown.")
            logging.info("Agent A's meaning is unknown.")

    return estimations, produced_utterances, speakers, listeners






######################






def multi_RSA(
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
    * choice_method, a string, what sampling method to use for utterance and estimation choices (default="classic_sample")
    * device, a torch device, the device to run the model on (default="cpu")
    * logging, a logging.Logger, the logger to log the results (default=None)
    * verbose, a boolean, whether to print the probabilities at each step (default=False)
    '''

    # Define useful constants
    n_agents = len(initial_lexica)
    n_utterances = [initial_lexica[i].shape[1] for i in range(n_agents)]
    n_meanings = [initial_prior.shape[i] for i in range(n_agents)]
    prior = initial_prior
    
    # Check the variables for coherency
    if not test_variables_coherency(n_agents, initial_lexica, n_meanings, n_utterances, A_utterances, B_utterances, number_of_rounds):
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
    speakers = [[],[]]
    listeners = [[],[]]
    for i in range(RSA_depth+1):
        listeners[0].append([])
        listeners[1].append([])
        speakers[0].append([])
        speakers[1].append([])

    # Run the CRSA for the specified number of rounds
    for round in range(number_of_rounds):
        logging.info(f"Round: {round}")
        if verbose:
            print(f"Round: {round}")

        if round == number_of_rounds - 1:
            last_round = True

        #### Agent A speaking ####
        # Compute the literal listener B
        listener = literal_listener_B(initial_lexica[0], prior, verbose)
        # Save the literal listener probabilities
        listeners[1][0].append(listener[B_meaning, :, :])

        # Compute the pragmatic speaker A
        speaker = pragmatic_speaker_A(listener, prior, alpha, RSA_depth, verbose)
        # Save the pragmatic speaker probabilities
        speakers[0][i].append(speaker[A_meaning])

        if A_utterances is None:
            try:
                utterance = sample(speaker[A_meaning], sampling) # Compute the utterance if not known
            except Exception as e:
                print(f"Line 104 and: {speaker[A_meaning]}")
        else:
            utterance = A_utterances[round] # Retrieve the utterance if known
        produced_utterances[0].append(utterance)
        logging.info(f"Utterance of Agent A: {game_model['u'][utterance]}")
        if verbose:
            print(f"Utterance of Agent A: " + bcolors.OKGREEN + f"{game_model['u'][utterance]}" + bcolors.ENDC)

        # Update the prior
        prior = speaker[:, utterance].view(n_meanings[0], 1, 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

         # Compute the pragmatic listener B
        listener = pragmatic_listener_B(speaker, prior, alpha, RSA_depth, verbose)
        # Save the pragmatic listener probabilities
        listeners[1][i].append(listener[B_meaning, :, :])

        # Compute the estimated truths by agent B
        if B_meaning is not None:
            try:
                estimated_y = sample(listener[B_meaning, :, utterance], sampling)
            except Exception as e:
                print(f"Line 130 and: {listener[B_meaning, :, utterance]}")
            estimations[1].append(estimated_y)
            logging.info(f"Estimated truth by a Agent B: {game_model['y'][estimated_y]}")
            if verbose:
                print(f"Estimated truth by a Agent B: " + bcolors.OKCYAN + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        else:
            if verbose: print("Agent B's meaning is unknown.")
            logging.info("Agent B's meaning is unknown.")


        #### Agent B speaking ####
        # Compute the literal listener A
        listener = literal_listener_A(initial_lexica[1], prior, verbose)
        # Save the literal listener probabilities
        listeners[0][0].append(listener[A_meaning, :, :])

        # Compute the pragmatic speaker B
        speaker = pragmatic_speaker_B(listener, prior, alpha, RSA_depth, verbose)
        # Save the pragmatic speaker probabilities
        speakers[1][i].append(speaker[B_meaning])

        # Compute the utterance
        if B_utterances is None:
            try:
                utterance = sample(speaker[B_meaning], sampling)
            except Exception as e:
                print(f"Line 170 and: {speaker[B_meaning]}")
        else:
            utterance = B_utterances[round]
        produced_utterances[1].append(utterance)
        logging.info(f"Utterance of Agent B: {game_model['v'][utterance]}")
        if verbose:
            print(f"Utterance of Agent B: " + bcolors.HEADER + f"{game_model['v'][utterance]}" + bcolors.ENDC)

        # Update the prior
        prior = speaker[:, utterance].view(n_meanings[0], 1, 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
        prior = prior/prior.sum() # Normalize the prior to avoid decreasing probabilities

        # Compute the pragmatic listener A
        listener = pragmatic_listener_A(speaker, prior, alpha, RSA_depth, verbose)
        # Save the pragmatic listener probabilities
        listeners[0][i].append(listener[A_meaning, :, :])

        # Compute the pragmatic listener A
        if A_meaning is not None:
            try:
                estimated_y = sample(listener[A_meaning, :, utterance], sampling)
            except Exception as e:
                print(f"Line 202 and: {listener[A_meaning, :, utterance]}")
            estimations[0].append(estimated_y)
            logging.info(f"Estimated truth by a pragmatic Agent A: {game_model['y'][estimated_y]}")
            if verbose:
                print(f"Estimated truth by a pragmatic Agent A: " + bcolors.FAIL + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        else:
            if verbose: print("Agent A's meaning is unknown.")
            logging.info("Agent A's meaning is unknown.")

    return estimations, produced_utterances, speakers, listeners
        
