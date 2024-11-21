import logging
from CRSA import CRSA
from RSA import multi_classic_RSA
from random_model import random_model
from greedy_model import greedy_model
import torch
import datetime
from tqdm import tqdm
from itertools import combinations, product
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import random

if __name__ == "__main__":

    # Read the CSV file
    df_message = pd.read_csv('planning_inference/data/message_cleaned.csv')
    df_clickedObj = pd.read_csv('planning_inference/data/clickedObj_cleaned.csv')
    df_games = pd.read_csv('planning_inference/data/game_properties_cleaned.csv')


    # scenarios = {'0,0,circle,cyan,A,1,0/0,1,rect,yellow,C,2,0/...', ...}
    scenarios = {}
    numero = 0
    for gameid_roundNum in df_games['gameid_roundNum'].unique():
        scenario_key = ""
        for row in df_games[df_games['gameid_roundNum'] == gameid_roundNum].iterrows():
            if scenario_key != "":
                scenario_key += "/"
            scenario_key += str(row[1]['pos_x']) + "," + str(row[1]['pos_y']) + "," + str(row[1]['shape']) + "," + str(row[1]['color']) + "," + str(row[1]['char']) + "," + str(row[1]['num']) + "," + str(row[1]['goal'])
        if scenario_key not in scenarios:
            scenarios[scenario_key] = numero
            numero += 1


    letters_combinations = list(product('ABC', repeat=6))
    numbers_combinations = list(product('123', repeat=6))

    Y = ["There is no (A,1) pair", "The (A,1) pair is in 1st position", "The (A,1) pair is in 2nd position", "The (A,1) pair is in 3rd position"] + [f"The (A,1) pair is in {x}th position" for x in range(4,7)]  # the objective will be different from all scenarios. They will be the same as the original scenario they are derived from (eg scenarios derived from scenario 0 will have as objective to find a A3 pair)

    meanings_A = []
    meanings_B = []

    for letters_combination in letters_combinations:
        for scenario in scenarios:
            A_meaning = ""
            objects = scenario.split('/')
            for i, obj in enumerate(objects):
                if A_meaning != "":
                    A_meaning += "/"
                x, y, shape, color, char, num, goal = obj.split(',')
                A_meaning += x + ',' + y + ',' + shape + ',' + color + ',' + letters_combination[i]
            if A_meaning in meanings_A:
                print(f"Warning: A_meaning already in meanings_A - {letters_combination} - {scenario}")
            meanings_A.append(A_meaning)

    for numbers_combination in numbers_combinations:
        for scenario in scenarios:
            B_meaning = ""
            objects = scenario.split('/')
            for i, obj in enumerate(objects):
                if B_meaning != "":
                    B_meaning += "/"
                x, y, shape, color, char, num, goal = obj.split(',')
                B_meaning += x + ',' + y + ',' + shape + ',' + color + ',' + numbers_combination[i]
            if B_meaning in meanings_B:
                print(f"Warning: B_meaning already in meanings_B - {numbers_combination} - {scenario}")
            meanings_B.append(B_meaning)


    ids_scenario = {scenario: [] for scenario in list(scenarios.keys())}

    for scenario in list(scenarios.keys()):
        objects = scenario.split('/')
        scenario_to_meaning_A = ""
        scenario_to_meaning_B = ""
        for obj in objects:
            if scenario_to_meaning_A != "":
                scenario_to_meaning_A += "/"
                scenario_to_meaning_B += "/"
            x, y, shape, color, char, num, goal = obj.split(',')
            scenario_to_meaning_A += x + ',' + y + ',' + shape + ',' + color + ',' + char
            scenario_to_meaning_B += x + ',' + y + ',' + shape + ',' + color + ',' + num
        for meaning in meanings_A:
            if meaning == scenario_to_meaning_A:
                ids_scenario[scenario].append(meanings_A.index(meaning))
        for meaning in meanings_B:
            if meaning == scenario_to_meaning_B:
                ids_scenario[scenario].append(meanings_B.index(meaning))       

    scenario = list(scenarios.keys())[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists('planning_inference/data/prior.pt') and torch.cuda.is_available():
        initial_prior = torch.load('planning_inference/data/prior.pt')
    elif os.path.exists('planning_inference/data/prior_cpu.pt') and not torch.cuda.is_available():
        initial_prior = torch.load('planning_inference/data/prior_cpu.pt')
    else:
        initial_prior = torch.zeros((len(meanings_A), len(meanings_B), 7), device=device, dtype=torch.float32)
        for A_meaning in tqdm(meanings_A):
            for B_meaning in meanings_B:
                A_objects = A_meaning.split('/')
                B_objects = B_meaning.split('/')
                one_objective_pos = False
                for i, A_obj in enumerate(A_objects):
                    B_obj = B_objects[i]
                    char = A_obj[-1]
                    num = B_obj[-1]
                    if char == 'A' and num == '1':
                        if not one_objective_pos:
                            initial_prior[meanings_A.index(A_meaning), meanings_B.index(B_meaning), i+1] = 1
                            one_objective_pos = True
                        else: # if there is more than one (A,1) pair, the prior is 0 because m_A and m_B are thus not compatible
                            initial_prior[meanings_A.index(A_meaning), meanings_B.index(B_meaning), :] = 0
                            break
                if not one_objective_pos:
                    initial_prior[meanings_A.index(A_meaning), meanings_B.index(B_meaning), 0] = 1
        initial_prior = initial_prior / initial_prior.sum()
        if torch.cuda.is_available():
            torch.save(initial_prior, 'planning_inference/data/prior.pt')
        else:
            torch.save(initial_prior, 'planning_inference/data/prior_cpu.pt')
    

    # Listing the used words
    words = df_message['contents'].str.split()
    words_flat = [word.lower() for sublist in words for word in sublist] + ['']
    used_words = set(words_flat)
    print("Used words:", used_words)

    # Creating the utterance set
    utterances = list(combinations(used_words, 2))
    print("Utterance set:", utterances)



    def word_meaning_correspondancy(word, meaning):
        '''
        Gives the semantic correspondancy between a word and a meaning.
        The list of words chosen are the 12 most used words in the experiments led by Khani et al in "Planing, Inference and Pragmatics in Sequential Language Games". 
        The word 'not' is not present here as it is never alone and always negates the meaning of the word that follows it.
        '''
        objects = meaning.split('/')
        res = torch.zeros((len(objects)), dtype=torch.bool)
        if word == '':
            return res + 1
        if word == 'green':
            for i, obj in enumerate(objects):
                if ('Chartreuse' in obj):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'yellow':
            for i, obj in enumerate(objects):
                if ('yellow' in obj):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'blue':
            for i, obj in enumerate(objects):
                if ('cyan' in obj):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'circle':
            for i, obj in enumerate(objects):
                if ('circle' in obj):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'square':
            for i, obj in enumerate(objects):
                if ('rect' in obj):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'diamond':
            for i, obj in enumerate(objects):
                if ('diamond' in obj):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'left':
            for i, obj in enumerate(objects):
                if ('0' == obj[0]):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'middle':
            for i, obj in enumerate(objects):
                if (('1' == obj[0] and any('2' == obj_bis[0] for obj_bis in objects)) or ('1' == obj[2] and any('2' == obj_bis[2] for obj_bis in objects))):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'right':
            for i, obj in enumerate(objects):
                if (('2' == obj[0]) or (not any('2' == obj_bis[0] for obj_bis in objects) and '1' == obj[0])):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'top':
            for i, obj in enumerate(objects):
                if ('0' == obj[2]):
                    res[i] = 1
                else:
                    res[i] = 0
        elif word == 'bottom':
            for i, obj in enumerate(objects):
                if (('2' == obj[2]) or (not any('2' == obj_bis[2] for obj_bis in objects) and '1' == obj[2])):
                    res[i] = 1
                else:
                    res[i] = 0
        return res



    def utterance_meaning_correspondancy(utterance, meaning):
        '''
        Gives the semantic correspondancy between a meaning and a message (composed of two words among the 12 most used in the experiments led by Khani et al in "Planing, Inference and Pragmatics in Sequential Language Games".
        '''
        colors = {'blue', 'green', 'yellow'}
        horizontal = {'left', 'middle', 'right'}
        vertical = {'top', 'middle', 'bottom'}
        shapes = {'diamond', 'square', 'circle'}

        objects = meaning.split('/')
        target_vector = torch.zeros((len(objects)), dtype=torch.bool)
        for i, obj in enumerate(objects):
            target_vector[i] = (obj[-1] in {'1','A'})

        if 'not' in utterance: # if 'not' in the message, the other word is negated
            not_id = utterance.index('not')
            return torch.all((target_vector * (word_meaning_correspondancy(utterance[(not_id + 1) % 2], meaning) + 1) % 2)[target_vector==1]).item()
        
        elif any(utterance[0] in same_group and utterance[1] in same_group for same_group in [colors, horizontal, vertical, shapes]): # if both words are mutually exclusive, we interpret the message as "word 1" or "word 2"
            return torch.any(target_vector * torch.max(word_meaning_correspondancy(utterance[0], meaning), word_meaning_correspondancy(utterance[1], meaning))).item()
        
        else: # if the words are not mutually exclusive, we interpret the message as "word 1" and "word 2"
            return torch.any(target_vector * word_meaning_correspondancy(utterance[0], meaning) * word_meaning_correspondancy(utterance[1], meaning)).item()

    if os.path.exists('planning_inference/data/lexicon_A.pt') and os.path.exists('planning_inference/data/lexicon_B.pt') and torch.cuda.is_available():
        lexicon_A = torch.load('planning_inference/data/lexicon_A.pt')
        lexicon_B = torch.load('planning_inference/data/lexicon_B.pt') 
    elif os.path.exists('planning_inference/data/lexicon_A_cpu.pt') and os.path.exists('planning_inference/data/lexicon_B_cpu.pt') and not torch.cuda.is_available():
        lexicon_A = torch.load('planning_inference/data/lexicon_A_cpu.pt')
        lexicon_B = torch.load('planning_inference/data/lexicon_B_cpu.pt')
    else:
        lexicon_A = torch.zeros((len(meanings_A), len(utterances)), device=device, dtype=torch.float32)
        lexicon_B = torch.zeros((len(meanings_B), len(utterances)), device=device, dtype=torch.float32)

        for j, utterance in tqdm(enumerate(utterances)):
            for i, meaning in enumerate(meanings_A):
                lexicon_A[i, j] = utterance_meaning_correspondancy(utterance, meaning)
            for i, meaning in enumerate(meanings_B):
                if i == 2251 and j in [54, 50, 59, 58, 33]:
                    print(utterance, meaning, utterance_meaning_correspondancy(utterance, meaning))
                    print("----")
                lexicon_B[i, j] = utterance_meaning_correspondancy(utterance, meaning)

        if torch.cuda.is_available():
            torch.save(lexicon_A, 'planning_inference/data/lexicon_A.pt')
            torch.save(lexicon_B, 'planning_inference/data/lexicon_B.pt')
        else:
            torch.save(lexicon_A, 'planning_inference/data/lexicon_A_cpu.pt')
            torch.save(lexicon_B, 'planning_inference/data/lexicon_B_cpu.pt')

    # Model of the game
    game_model = {"mA": meanings_A, "mB": meanings_B, "u": utterances, "v": utterances, "y": Y}

    initial_lexica = [lexicon_A, lexicon_B]

    # Prepare data collection and analysis
    ground_truth = np.array([1, 5, 4, 2, 5, 6, 1, 4, 3, 3])
    results_json = {}

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for alpha in [0.1,1,5,10]:

            softmax_depth_0_results = []
            softmax_depth_1_results = []
            softmax_depth_2_results = []
            softmax_rsa_results = []

            for number in tqdm(range(1000)):
                random_id_scenario = random.randint(0,len(scenarios)-1)
                scenario = list(scenarios.keys())[random_id_scenario]
                A_meaning = ids_scenario[scenario][0]
                B_meaning = ids_scenario[scenario][1]

                # Compute the ground truth ie the position of the A,1 pair
                ground_truth = 0
                for i, str_meaning_A in enumerate(meanings_A[A_meaning].split('/')):
                    str_meaning_B = meanings_B[B_meaning].split('/')[i]
                    if str_meaning_A[-1] == 'A' and str_meaning_B[-1] == '1':
                        ground_truth = i+1
                
                # Run the RSA model with a literal Listener and a pragmatic Speaker
                softmax_depth_0_estimations, _, _, _ = CRSA(
                    initial_lexica = initial_lexica, 
                    initial_prior = initial_prior, 
                    game_model = game_model, 
                    A_meaning = A_meaning, 
                    B_meaning = B_meaning, 
                    A_utterances = None, 
                    B_utterances = None, 
                    alpha = alpha, 
                    number_of_rounds = 10, 
                    RSA_depth = 0, 
                    sampling = 'softmax', 
                    device = device, 
                    logging = logging, 
                    verbose = False,
                    )
                # Run the RSA model with depth 1
                softmax_depth_1_estimations, _, _, _ = CRSA(
                    initial_lexica = initial_lexica, 
                    initial_prior = initial_prior, 
                    game_model = game_model, 
                    A_meaning = A_meaning, 
                    B_meaning = B_meaning, 
                    A_utterances = None, 
                    B_utterances = None, 
                    alpha = alpha, 
                    number_of_rounds = 10, 
                    RSA_depth = 1, 
                    sampling = 'softmax', 
                    device = device, 
                    logging = logging, 
                    verbose = False,
                    )
                # Run the RSA model with depth 2
                softmax_depth_2_estimations, _, _, _ = CRSA(
                    initial_lexica = initial_lexica, 
                    initial_prior = initial_prior, 
                    game_model = game_model, 
                    A_meaning = A_meaning, 
                    B_meaning = B_meaning, 
                    A_utterances = None, 
                    B_utterances = None, 
                    alpha = alpha, 
                    number_of_rounds = 10, 
                    RSA_depth = 2, 
                    sampling = 'softmax', 
                    device = device, 
                    logging = logging, 
                    verbose = False,
                    )
                # Run classic RSA on multiple rounds
                softmax_rsa_estimations, _, _, _ = multi_classic_RSA(
                    initial_lexica = initial_lexica, 
                    initial_prior = initial_prior, 
                    game_model = game_model, 
                    A_meaning = A_meaning, 
                    B_meaning = B_meaning, 
                    A_utterances = None, 
                    B_utterances = None, 
                    alpha = alpha, 
                    number_of_rounds = 10, 
                    RSA_depth = 1, 
                    sampling = 'softmax', 
                    device = device, 
                    logging = logging, 
                    verbose = False,
                    )

                softmax_depth_0_estimations = np.array(softmax_depth_0_estimations)
                softmax_depth_0_correct_guesses = (softmax_depth_0_estimations == ground_truth) * 1

                softmax_depth_1_estimations = np.array(softmax_depth_1_estimations)
                softmax_depth_1_correct_guesses = (softmax_depth_1_estimations == ground_truth) * 1

                softmax_depth_2_estimations = np.array(softmax_depth_2_estimations)
                softmax_depth_2_correct_guesses = (softmax_depth_2_estimations == ground_truth) * 1

                softmax_rsa_estimations = np.array(softmax_rsa_estimations)
                softmax_rsa_correct_guesses = (softmax_rsa_estimations == ground_truth) * 1

                # Store the results
                softmax_depth_0_results.append(softmax_depth_0_correct_guesses)
                softmax_depth_1_results.append(softmax_depth_1_correct_guesses)
                softmax_depth_2_results.append(softmax_depth_2_correct_guesses)
                softmax_rsa_results.append(softmax_rsa_correct_guesses)
                
            softmax_depth_0_results = np.array(softmax_depth_0_results)
            softmax_depth_1_results = np.array(softmax_depth_1_results)
            softmax_depth_2_results = np.array(softmax_depth_2_results)
            softmax_rsa_results = np.array(softmax_rsa_results)
            
            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the results for CRSA depth 0 with softmax sampling
            ax.plot(
                range(1, softmax_depth_0_results.shape[2]+1), 
                (np.mean(softmax_depth_0_results[:, 0, :], axis=0) + np.mean(softmax_depth_0_results[:, 1, :], axis=0))/2,
                label='CRSA Depth 0 softmax',
                )

            # Plot the results for CRSA depth 1 with softmax sampling
            ax.plot(
                range(1, softmax_depth_1_results.shape[2]+1), 
                (np.mean(softmax_depth_1_results[:, 0, :], axis=0) + np.mean(softmax_depth_1_results[:, 1, :], axis=0))/2, 
                label='CRSA Depth 1 softmax', 
                )

            # Plot the results for CRSA depth 2 with softmax sampling
            ax.plot(
                range(1, softmax_depth_2_results.shape[2]+1), 
                (np.mean(softmax_depth_2_results[:, 0, :], axis=0) + np.mean(softmax_depth_2_results[:, 1, :], axis=0))/2, 
                label='CRSA Depth 2 softmax', 
                )
            
            # Plot the results for multi-turns softmax RSA
            ax.plot(
                range(1, softmax_rsa_results.shape[2]+1), 
                (np.mean(softmax_rsa_results[:, 0, :], axis=0) + np.mean(softmax_rsa_results[:, 1, :], axis=0))/2, 
                label='Softmax RSA', 
                )
            
            # Set the labels and title
            ax.set_xlabel('Round')
            ax.set_title(f'Percentage of correct guesses for alpha={alpha}')

            # Set the y-axis ticks and labels
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0%', '100%'])

            # Add a legend
            ax.legend()

            # Save the plot as a file
            os.makedirs(f'plots_planning_inference/{date_string}', exist_ok=True)
            plt.savefig(f'plots_planning_inference/{date_string}/alpha_{alpha}_average.png', bbox_inches='tight')

            # Save results in an npz file
            os.makedirs(f'results_planning_inference/{date_string}', exist_ok=True)
            np.savez(f'results_planning_inference/{date_string}/alpha_{alpha}_results.npz', 
                softmax_depth_0_results=softmax_depth_0_results, 
                softmax_depth_1_results=softmax_depth_1_results, 
                softmax_depth_2_results=softmax_depth_2_results, 
                softmax_rsa_results=softmax_rsa_results, 
                )