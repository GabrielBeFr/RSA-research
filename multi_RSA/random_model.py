import torch
import logging
from utils import bcolors, sample, test_variables_coherency

def random_model(
        initial_lexica: list, 
        initial_prior: torch.FloatTensor, 
        game_model: dict, 
        A_meaning: int=None, 
        B_meaning: int=None, 
        A_utterances: list=None, 
        B_utterances: list=None, 
        number_of_rounds: int=2, 
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
    * number_of_rounds, an int, the number of rounds to run the model for (default=2)
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

    # Run the CRSA for the specified number of rounds
    for round in range(number_of_rounds):
        logging.info(f"Round: {round}")
        if verbose:
            print(f"Round: {round}")

        if round == number_of_rounds - 1:
            last_round = True

        #### Agent A speaking ####
        if A_utterances is None:
            # Find the actions that are coherent with the context
            if A_meaning is not None:
                coherent_actions = initial_lexica[0][A_meaning, :]
            else:
                coherent_actions = torch.ones(n_utterances[0], device=device)
            # Give uniform probability to the actions that are coherent with the context
            utterance = sample(coherent_actions/coherent_actions.sum(), sampling="classic")
        else:
            utterance = A_utterances[round]

        produced_utterances[0].append(utterance)
        logging.info(f"Utterance of Agent A: {game_model['u'][utterance]}")
        if verbose:
            print(f"Utterance of Agent A: " + bcolors.OKGREEN + f"{game_model['u'][utterance]}" + bcolors.ENDC)

        if B_meaning is not None:
            coherent_estimations = prior[:,B_meaning,:].sum(dim=0)!=0
        else:
            coherent_estimations = torch.ones(n_meanings[1], device=device)
        estimation = sample(coherent_estimations/coherent_estimations.sum(), sampling="classic")

        estimations[1].append(estimation)
        logging.info(f"Estimated truth by a random Agent B: {game_model['y'][estimation]}")
        if verbose:
            print(f"Estimated truth by a random Agent B: " + bcolors.OKCYAN + f"{game_model['y'][estimation]}" + bcolors.ENDC)


        #### Agent B speaking ####
        if B_utterances is None:
            # Find the actions that are coherent with the context
            if B_meaning is not None:
                coherent_actions = initial_lexica[1][B_meaning, :]
            else:
                coherent_actions = torch.ones(n_utterances[1], device=device)
            # Give uniform probability to the actions that are coherent with the context
            utterance = sample(coherent_actions/coherent_actions.sum(), sampling="classic")
        else:
            utterance = B_utterances[round]

        produced_utterances[1].append(utterance)
        logging.info(f"Utterance of Agent B: {game_model['v'][utterance]}")
        if verbose:
            print(f"Utterance of Agent B: " + bcolors.OKGREEN + f"{game_model['v'][utterance]}" + bcolors.ENDC)

        if A_meaning is not None:
            coherent_estimations = prior[A_meaning,:,:].sum(dim=0)!=0
        else:
            coherent_estimations = torch.ones(n_meanings[0], device=device)
        estimation = sample(coherent_estimations/coherent_estimations.sum(), sampling="classic")

        estimations[0].append(estimation)
        logging.info(f"Estimated truth by a random Agent A: {game_model['y'][estimation]}")
        if verbose:
            print(f"Estimated truth by a random Agent A: " + bcolors.OKCYAN + f"{game_model['y'][estimation]}" + bcolors.ENDC)

    return estimations, produced_utterances
        
