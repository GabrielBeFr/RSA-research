import logging
from multi_rsa import multi_rsa
import torch
import datetime
from tqdm import tqdm
from itertools import combinations, product
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json

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

    prior = torch.load('planning_inference/data/prior.pt')

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



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    # Model of the game
    game_model = {"mA": meanings_A, "mB": meanings_B, "u": utterances, "v": utterances, "y": Y}

    initial_lexica = [lexicon_A, lexicon_B]

    # Prepare data collection and analysis
    ground_truth = np.array([1, 5, 4, 2, 5, 6, 1, 4, 3, 3])
    results = {}

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for alpha in [0.1, 0.5, 1, 2, 5, 10]:
        logging.info(f"\n\n\n---Alpha {alpha}---\n")

        results[alpha] = {}

        correct_guesses_all = []
        classed_results_all = []

        for number in range(10):
            logging.info(f"\n\n\n---Scenario {number}---\n")

            results[alpha][number] = {}

            scenario = list(scenarios.keys())[number]
            A_meaning = ids_scenario[scenario][0]
            B_meaning = ids_scenario[scenario][1]
            
            # Run the RSA model
            estimations, produced_utterances = multi_rsa(initial_lexica, prior, game_model, A_meaning, B_meaning, None, None, alpha, 10, 2, device, logging, True)

            logging.info(f'estimations: {estimations}')
            results[alpha][number]['estimations'] = estimations
            results[alpha][number]['produced_utterances'] = produced_utterances

    # Save results to a JSON file
    with open('results.json', 'w') as file:
        json.dump(results, file)
