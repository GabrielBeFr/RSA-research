from utils import compute_listener_value, compute_shannon_conditional_entropy
import matplotlib.pyplot as plt
import numpy as np

def asymptotic_analysis_alphas(alphas_list, rsa_model, world, version, depth, verbose):
    entropies_list = []
    values_list = []
    gains_list = []
    last_speakers = []

    for new_alpha in alphas_list:
        # Initialize the RSA model with verbose mode
        rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=new_alpha, depth=depth)
        # Run the RSA model
        world, listeners, speakers = rsa.run(verbose)

        last_speakers.append(speakers[-1])

        entropies = [compute_shannon_conditional_entropy(world["priors"],speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        values = [compute_listener_value(new_alpha, world["priors"], listeners[(i+1)//2], speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        gains = [values[i] + entropies[i] for i in range((len(speakers)-1)*2)]

        entropies_list.append(entropies)
        values_list.append(values)
        gains_list.append(gains)

    fig, axs = plt.subplots(len(alphas_list)+1, 4, figsize=(15, 25))
    fig.suptitle("No Structural Zeros")

    axs[0,0].remove()
    im = axs[0,1].imshow(np.round(world["lexicon"],3), cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(im, ax=axs[0,1])
    axs[0,1].set_title("Initial Lexicon")
    plt.text(2.5, 1, world["surname"], fontsize=20, ha='left', va='top', wrap=True, transform=axs[0,1].transAxes)
    axs[0,2].remove()
    axs[0,3].remove()

    axs[1,0].set_title("Conditional entropy")
    axs[1,1].set_title("Listener value")
    axs[1,2].set_title("Gain function")
    axs[1,3].set_title("Speaker")        

    for i in range(len(alphas_list)):
        axs[1+i,0].set_ylabel(f"Alpha={alphas_list[i]}")

        axs[1+i,0].plot(entropies_list[i])

        axs[1+i,1].plot(values_list[i])

        axs[1+i,2].plot(gains_list[i])

        # Add lines to the gain function plot
        axs[1+i,2].axhline(max((1 - alphas_list[i]),0) * np.log(len(world["meanings"])), linestyle=':', color='red')

        im = axs[1+i,3].imshow(np.round(last_speakers[i],3), cmap='viridis', interpolation='nearest')
        cbar = fig.colorbar(im, ax=axs[1+i,3])

    plt.show()