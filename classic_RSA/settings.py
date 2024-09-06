import numpy as np
from rsa import lexical_uncertainty_RSA, classic_RSA
from worlds import world_bergen_2016_fig1, world_rate_distortion, world_bergen_2016_fig3, world_degen_2023_fig1a, world_degen_2023_fig1e, world_bergen_2016_fig6, world_playground

worlds = {
    'bergen_2016_fig1': world_bergen_2016_fig1,
    'bergen_2016_fig3': world_bergen_2016_fig3,
    'degen_2023_fig1a': world_degen_2023_fig1a,
    'degen_2023_fig1e': world_degen_2023_fig1e,
    'bergen_2016_fig6': world_bergen_2016_fig6,
    'rate_distortion': world_rate_distortion,
    'playground': world_playground,
}

rsa_models = {
    'classic_RSA': classic_RSA,
    'lexical_uncertainty_RSA': lexical_uncertainty_RSA
}