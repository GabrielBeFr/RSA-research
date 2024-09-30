from multi_rsa import multi_rsa
import torch
import random
import logging
import datetime

if __name__ == "__main__":

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model of the game
    meanings_A = ["BAA", "ABA", "ABB", "BAB", "BBA"]
    meanings_B = ["112", "221", "212", "122"]
    utterances_A = ["1st position", "2nd position", "3rd position"]
    utterances_B = ["1st position", "2nd position", "3rd position"]
    Y = ["There is no (A,1) pair", "1st position", "2nd position", "3rd position"] # the question we try to answer is what is the position of the A,1 pair.
    game_model = {"mA": meanings_A, "mB": meanings_B, "u": utterances_A, "v": utterances_B, "y": Y}

    # Define the prior
    prior = torch.tensor([
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
        [[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]],
        ], device=device, dtype=torch.float32)

    initial_prior = prior * 1/len(meanings_A) * 1/len(meanings_B)

    # Define the lexica
    lexicon_A = torch.tensor([
        [0, 1, 1], 
        [1, 0, 1], 
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        ], device=device, dtype=torch.float32)

    lexicon_B = torch.tensor([
        [1, 1, 0], 
        [0, 0, 1], 
        [0, 1, 0],
        [1, 0, 0],
        ], device=device, dtype=torch.float32)

    initial_lexica = [lexicon_A, lexicon_B]

    # Fix the meanings observed by the agents
    A_meaning = random.randint(0, len(meanings_A)-1)
    B_meaning = random.randint(0, len(meanings_B)-1)
    # A_meaning = 1
    # B_meaning = 2

    # Fix the utterances observed by the agents
    # A_utterances = [1, 1, 1, 1]
    A_utterances = None
    B_utterances = None

    # Define the parameters
    alpha = 1
    number_of_rounds = 10
    RSA_depth = 1
    verbose = False

    # Define the logging
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run the Interactive RSA
    multi_rsa(initial_lexica, initial_prior, game_model, A_meaning, B_meaning, A_utterances, B_utterances, alpha, number_of_rounds, RSA_depth, device, logging, verbose)