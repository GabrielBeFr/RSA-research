import torch
import logging
from CRSA_agents import literal_listener_A, literal_listener_B, pragmatic_speaker_A, pragmatic_speaker_B, pragmatic_listener_A, pragmatic_listener_B
from utils import bcolors, sample, test_variables_coherency

def greedy_model(
        initial_lexica: list, 
        initial_prior: torch.FloatTensor, 
        game_model: dict, 
        A_meaning: int=None, 
        B_meaning: int=None, 
        A_utterances: list=None, 
        B_utterances: list=None, 
        alpha: float=1,
        number_of_rounds: int=2, 
        RSA_depth: int=1,
        sampling: str="classic",
        device: torch.device="cpu", 
        logging: logging.Logger=None,
        verbose: bool=False,
        ):
    '''Run the greedy model for the specified number of iterations. 
    NOTA BENE: 
    - The model is currently only implemented for 2 agents.
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
    list_speakers = [[],[]]
    list_listeners = [[],[]]
    list_prior = [initial_prior]

    # Run the CRSA for the specified number of rounds
    for round in range(number_of_rounds):
        logging.info(f"Round: {round}")
        if verbose:
            print(f"Round: {round}")

        if round == number_of_rounds - 1:
            last_round = True

        #### Agent A speaking ####
        if A_utterances is None:
            # Find the actions that are coherent with the meaning of the speaker
            coherent_utterances = initial_lexica[0][A_meaning, :]

            # Compute the amount of semantically valid states with respect to the past utterances
            valid_states = torch.ones(n_meanings[0], device=device)
            for i in range(round):
                valid_states *= initial_lexica[0][:, produced_utterances[0][i]]

            # Compute the amount of semantically valid states adding a potential new utterance
            score = torch.zeros(n_utterances[0], device=device)
            for utterance_id in range(n_utterances[0]):
                if coherent_utterances[utterance_id]!=0:
                    score[utterance_id] = valid_states.sum() - (valid_states*initial_lexica[0][:, utterance_id]).sum()
            
            # Sample the produced utterance
            if score.sum() == 0: score = coherent_utterances
            utterance = sample(score/score.sum(), sampling="classic")
            list_speakers[0].append(score)
        else:
            utterance = A_utterances[round]

        produced_utterances[0].append(utterance)
        logging.info(f"Utterance of greedy Agent A: {game_model['u'][utterance]}")
        if verbose:
            print(f"Utterance of greedy Agent A: " + bcolors.OKGREEN + f"{game_model['u'][utterance]}" + bcolors.ENDC)

        # Compute the CRSA listener
        
        if B_meaning is not None:
            # Compute the literal listener
            literal_listener = literal_listener_B(initial_lexica[0], prior, verbose)
            if RSA_depth > 0:
                # Compute the pragmatic speaker
                pragmatic_speaker = pragmatic_speaker_A(literal_listener, prior, alpha, RSA_depth, verbose)
                
                # Update the prior
                prior = pragmatic_speaker[:, utterance].view(n_meanings[0], 1, 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
                prior = prior/prior.sum()
                list_prior.append(prior)
                
                # Compute the pragmatic listener B
                pragmatic_listener = pragmatic_listener_B(pragmatic_speaker, prior, alpha, RSA_depth, verbose)
                list_listeners[1].append(pragmatic_listener)

                # Sample the estimation
                try:
                    estimated_y = sample(pragmatic_listener[B_meaning, :, utterance], sampling)
                except:
                    print("Heyo")

            elif RSA_depth == 0:
                estimated_y = sample(literal_listener[B_meaning, :, utterance], sampling)

            estimations[1].append(estimated_y)
            logging.info(f"Estimated truth by a pragmatic Agent B: {game_model['y'][estimated_y]}")
            if verbose:
                print(f"Estimated truth by a pragmatic Agent B: " + bcolors.OKCYAN + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        elif verbose:
            logging.info("Agent B's meaning is not known.")
            print("Agent B's meaning is not known.")
        else:
            logging.info("Agent B's meaning is not known.")


        #### Agent B speaking ####
        if B_utterances is None:
            # Find the actions that are coherent with the meaning of the speaker
            coherent_utterances = initial_lexica[1][B_meaning, :]

            # Compute the amount of semantically valid states with respect to the past utterances
            valid_states = torch.ones(n_meanings[1], device=device)
            for i in range(round):
                valid_states *= initial_lexica[1][:, produced_utterances[1][i]]

            # Compute the amount of semantically valid states adding a potential new utterance
            score = torch.zeros(n_utterances[1], device=device)
            for utterance_id in range(n_utterances[1]):
                if coherent_utterances[utterance_id]!=0:
                    score[utterance_id] = valid_states.sum() - (valid_states*initial_lexica[1][:, utterance_id]).sum()
            
            # Sample the produced utterance
            if score.sum() == 0: score = coherent_utterances
            utterance = sample(score/score.sum(), sampling="classic")
            list_speakers[1].append(score)
        else:
            utterance = B_utterances[round]

        produced_utterances[1].append(utterance)
        logging.info(f"Utterance of greedy Agent B: {game_model['v'][utterance]}")
        if verbose:
            print(f"Utterance of greedy Agent B: " + bcolors.OKGREEN + f"{game_model['v'][utterance]}" + bcolors.ENDC)

        # Compute the CRSA listener
        
        if A_meaning is not None:
            # Compute the literal listener
            literal_listener = literal_listener_A(initial_lexica[1], prior, verbose)
            if RSA_depth > 0:
                # Compute the pragmatic speaker
                pragmatic_speaker = pragmatic_speaker_B(literal_listener, prior, alpha, RSA_depth, verbose)
                
                # Update the prior
                prior = pragmatic_speaker[:, utterance].view(1, n_meanings[1], 1) * prior # S(u2|mA, u1, v1) x P(mA, mB, y, u1, v1) -> P(mA, mB, y, u1, v1, u2). The utterances are fixed and thus do not account as dimensions. However, the mmeanings are not fixed as each agent ignore the meaning of the other agent.
                prior = prior/prior.sum()
                list_prior.append(prior)
                
                # Compute the pragmatic listener A
                pragmatic_listener = pragmatic_listener_A(pragmatic_speaker, prior, alpha, RSA_depth, verbose)
                list_listeners[0].append(pragmatic_listener)

                # Sample the estimation
                try:
                    estimated_y = sample(pragmatic_listener[A_meaning, :, utterance], sampling)
                except:
                    print("Heyo")

            elif RSA_depth == 0:
                estimated_y = sample(literal_listener[A_meaning, :, utterance], sampling)

            estimations[0].append(estimated_y)
            logging.info(f"Estimated truth by a pragmatic Agent A: {game_model['y'][estimated_y]}")
            if verbose:
                print(f"Estimated truth by a pragmatic Agent A: " + bcolors.OKCYAN + f"{game_model['y'][estimated_y]}" + bcolors.ENDC)
        elif verbose:
            logging.info("Agent A's meaning is not known.")
            print("Agent A's meaning is not known.")
        else:
            logging.info("Agent A's meaning is not known.")

    return estimations, produced_utterances, list_speakers, list_listeners, prior
        
