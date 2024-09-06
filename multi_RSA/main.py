import logging
from multi_rsa import multi_rsa
import torch
import torch

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def main(initial_lexica, initial_prior, meanings, game_model, alpha, number_of_rounds, RSA_depth, device, verbose):
    setup_logging()

    # Run the RSA model
    multi_rsa(initial_lexica, initial_prior, meanings, game_model, alpha, number_of_rounds, RSA_depth, device, verbose)
    

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

    main(initial_lexica, initial_prior, meanings, alpha, number_of_rounds, RSA_depth, device, verbose)
