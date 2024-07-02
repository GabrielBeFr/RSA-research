import numpy as np

world_bergen_2016_fig1 = {
    'file_name': 'Bergen2016Fig1.txt',
    'surname': 'Classic RSA working: some/all',
    'utterances': ['Some', 'All'],
    'meanings': ['Some but not all', 'All'],
    'lexicon': np.array([[1, 1], [0, 1]]),
    'costs': np.array([0,0]),
    'priors': np.array([1/2, 1/2])
}

world_bergen_2016_fig3 = {
    'file_name': 'Bergen2016Fig3.txt',
    'surname': 'Failure of classic RSA for M-implicature',
    'utterances': ['SHORT', 'long'],
    'meanings': ['FREQ', 'rare'],
    'lexicon': np.array([[1, 1], [1, 1]]),
    'costs': np.array([1,2]),
    'priors': np.array([2/3, 1/3])
}

world_degen_2023_fig1a = {
    'file_name': 'Degen2023Fig1a.txt',
    'surname': 'Classic RSA working: cookies',
    'utterances': ['None', 'Some', 'All'],
    'meanings': ['0', '1', '2', '3', '4'],
    'lexicon': np.array([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1]]),
    'costs': np.array([0, 0, 0]),
    'priors': np.array([1/5, 1/5, 1/5, 1/5, 1/5])
}

world_degen_2023_fig1e = {
    'file_name': 'Degen2023Fig1e.txt',
    'surname': 'Influence of priors on classic RSA: cookies',
    'utterances': ['None', 'Some', 'All'],
    'meanings': ['0', '1', '2', '3', '4'],
    'lexicon': np.array([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1]]),
    'costs': np.array([0, 0, 0]),
    'priors': np.array([1/20, 1/20, 1/20, 1/20, 4/5])
}

world_bergen_2016_fig6 = {
    'file_name': 'Bergen2016Fig6.txt',
    'surname': 'Deriving M-implicature with lexical uncertainty RSA',
    'utterances': ['Nothing', 'long', 'SHORT'],
    'meanings': ['FREQ', 'rare'],
    'lexicon': np.array([[1, 1], [1, 1], [1, 1]]),
    'costs': np.array([5, 2, 1]),
    'priors': np.array([2/3, 1/3])
}

world_rate_distortion = {
    'file_name': 'rate_distortion.txt',
    'surname': 'Showing difference between RSA and RSA-RD',
    'utterances': ['hat', 'glasses', 'mustache', 'friend'],
    'meanings': ['only mustache', 'mustache and glasses', 'glasses and hat'],
    'lexicon': np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]),
    'costs': np.array([0, 0, 0, 0]),
    'priors': np.array([1/3,1/3,1/3])
}

world_playground = {
    'file_name': 'playground.txt',
    'surname': 'Testing',
    'utterances': ['u1', 'u2', 'u3', 'u4', 'u5'],
    'meanings': ['m1', 'm2', 'm3', 'm4', 'm5'],
    'lexicon': np.array([[0, 1, 1, 0, 1], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]),
    'costs': np.array([0,0,0,0,0]),
    'priors': np.array([1/5,1/5,1/5,1/5,1/5])
}