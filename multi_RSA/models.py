from CRSA import CRSA
from RSA import multi_classic_RSA
from random_model import random_model
from greedy_model import greedy_model

MODEL_CONFIGS = {
    'multi_classic_RSA': {
        'model': multi_classic_RSA,
        'samplings': ['argmax', 'classic'],
        'RSA_depths': [1],
        'invert_agents': [False],
        'alphas': [0.1, 1, 2, 5, 10],
    },
    'CRSA': {
        'model': CRSA,
        'samplings': ['argmax', 'classic'],
        'RSA_depths': [0, 1, 2],
        'invert_agents': [False, True],
        'alphas': [0.1, 1, 2, 5, 10],
    },
    'random_model': {
        'model': random_model,
        'samplings': ['classic'],
        'RSA_depths': [1],
        'invert_agents': [False],
        'alphas': [1],
    },
    'greedy_model': {
        'model': greedy_model,
        'samplings': ['argmax','classic'],
        'RSA_depths': [1],
        'invert_agents': [False, True],
        'alphas': [0.1,1,2,5,10],
    }
}