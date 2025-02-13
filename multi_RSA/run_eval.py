import logging
import torch
import datetime
from tqdm import tqdm
import os
import numpy as np
import h5py
from models import MODEL_CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
with h5py.File('data.h5', 'r') as f:
    # Load prior tensor
    prior = torch.tensor(f['prior'][:])
    
    # Load lexica
    lexica = []
    lexica_group = f['lexica']
    for pair_name in lexica_group:
        pair_group = lexica_group[pair_name]
        tensor1 = torch.tensor(pair_group['tensor1'][:])
        tensor2 = torch.tensor(pair_group['tensor2'][:])
        lexica.append((tensor1.to(device), tensor2.to(device)))
    
    # Load scenarios
    #scenarios = {key: {k: value[k][:] for k in value} for key, value in f['scenarios'].items()}
    scenarios = {}
    for key, value in f['scenarios'].items():
        scenarios[key] = {}
        for k in value:
            scenarios[key][k] = f['scenarios'][key][k][()]
    
    # Load game_model
    game_model = {key: f['game_model'][key][:].tolist() for key in f['game_model']}

    # Load meanings_A and meanings_B
    meanings_A = f['meanings_A'][:]
    meanings_B = f['meanings_B'][:]

    # Load utterances
    utterances = f['utterances'][:]

initial_prior = prior.to(device)

# Prepare data collection and analysis
ground_truth_array = np.array([1, 5, 4, 2, 5, 6, 1, 4, 3, 3])
results = {}
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

for model_name in MODEL_CONFIGS.keys():
    print("Running model: ", model_name)
    for id_scenario in tqdm(range(10)):
        scenario = list(scenarios.keys())[id_scenario]
        initial_lexica = lexica[scenarios[scenario]['corresponding_lexica']]
        A_meaning = scenarios[scenario]['corresponding_meaning_A']
        B_meaning = scenarios[scenario]['corresponding_meaning_B']
        ground_truth = ground_truth_array[id_scenario]
        for invert_agents in MODEL_CONFIGS[model_name]['invert_agents']:
            if invert_agents: 
                current_A_meaning = B_meaning; current_B_meaning = A_meaning
                current_lexica = (initial_lexica[1], initial_lexica[0])
            else: 
                current_A_meaning = A_meaning; current_B_meaning = B_meaning
                current_lexica = initial_lexica
            for alpha in MODEL_CONFIGS[model_name]['alphas']:
                for sampling in MODEL_CONFIGS[model_name]['samplings']:
                    if sampling == 'classic':
                        iter = 20
                    elif sampling == 'argmax':
                        iter = 1
                    for RSA_depth in MODEL_CONFIGS[model_name]['RSA_depths']:
                        res_estimations = []
                        res_produced_utterances = []
                        res_speakers = []
                        res_listeners = []
                        res_correct_guesses = []
                        res_priors = []

                        for i in range(iter):
                            # Run the model
                            estimations, produced_utterances, speakers, listeners, priors = MODEL_CONFIGS[model_name]['model'](
                                initial_lexica = current_lexica, 
                                initial_prior = initial_prior, 
                                game_model = game_model, 
                                A_meaning = current_A_meaning, 
                                B_meaning = current_B_meaning, 
                                A_utterances = None, 
                                B_utterances = None, 
                                alpha = alpha, 
                                number_of_rounds = 10, 
                                RSA_depth = RSA_depth,
                                sampling = sampling,
                                device = device, 
                                logging = logging, 
                                verbose = False,
                            )    
                        
                            # Compute the correct guesses
                            estimations_arr = np.array(estimations)
                            correct_guesses = (estimations_arr == ground_truth) * 1
                            
                            res_estimations.append(estimations) 
                            res_produced_utterances.append(produced_utterances)
                            res_speakers.append(speakers)
                            res_listeners.append(listeners)
                            res_priors.append(priors)
                            res_correct_guesses.append(correct_guesses)
                    
                    # Save the results in an h5py file in a repo with the name of the date in a repo with the name "results". The name of the file is the name of the model
                    if not os.path.exists('results/'+date_string):
                        os.makedirs('results/'+date_string)
                    with h5py.File(
                        'results/'+date_string+'/'+model_name+'_'+str(alpha)+'_'+str(sampling)+'_'+str(RSA_depth)+'_'+str(id_scenario)+'_'+str(invert_agents)+'.h5', 'w'
                        ) as f:
                        for i in range(iter):
                            group = f.create_group(str(i))
                            group.create_dataset('estimations', data=res_estimations[i])
                            group.create_dataset('produced_utterances', data=res_produced_utterances[i])
                            group.create_dataset('speakers', data=res_speakers[i])
                            group.create_dataset('listeners', data=res_listeners[i])
                            group.create_dataset('priors', data=res_priors[i])
                            group.create_dataset('correct_guesses', data=res_correct_guesses[i])