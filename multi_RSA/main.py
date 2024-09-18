import logging
from multi_rsa import multi_rsa
import torch
import torch
import datetime

def setup_logging():
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"logs/app_{current_datetime}.log"
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG
        format='%(levelname)s - %(message)s',  # Update the format argument
        handlers=[
            logging.FileHandler(log_file_name)
        ]
    )
    

def run_multi_rsa(initial_lexica, initial_prior, game_model, A_meaning=None, B_meaning=None, A_utterances=None, B_utterances=None, alpha=1, number_of_rounds=2, RSA_depth=1, device=None, verbose=False):
    setup_logging()

    # Run the RSA model
    multi_rsa(initial_lexica, initial_prior, game_model, A_meaning, B_meaning, A_utterances, B_utterances, alpha, number_of_rounds, RSA_depth, device, logging, verbose)

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

    # Define the prior
    prior = torch.tensor([
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0]],
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
    meanings = [2, 1]

    # Define the parameters
    alpha = 1
    number_of_rounds = 10
    RSA_depth = 1
    verbose = True 

    run_multi_rsa(initial_lexica, initial_prior, meanings, alpha, number_of_rounds, RSA_depth, device, verbose)
