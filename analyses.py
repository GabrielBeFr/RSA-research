from utils import compute_listener_value, compute_shannon_conditional_entropy
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import torch
from NN_models import RSA_KL_model

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
        values = [new_alpha * compute_listener_value(world["priors"], listeners[(i+1)//2], speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        gains = [values[i] + entropies[i] for i in range((len(speakers)-1)*2)]

        entropies_list.append(entropies)
        values_list.append(values)
        gains_list.append(gains)

    fig, axs = plt.subplots(len(alphas_list)+1, 4, figsize=(6*4, 4*(len(alphas_list)+1)))

    axs[0,0].remove()
    im = axs[0,1].imshow(np.round(world["lexicon"],3), cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(im, ax=axs[0,1])
    for (j, k), value in np.ndenumerate(np.round(world["lexicon"],3)):
            axs[0,1].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')
    axs[0,1].set_title("Initial Lexicon")
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

        im = axs[1+i,3].imshow(np.round(last_speakers[i],3), cmap='viridis', interpolation='nearest')
        cbar = fig.colorbar(im, ax=axs[1+i,3])
        for (j, k), value in np.ndenumerate(np.round(last_speakers[i],3)):
            axs[i+1,3].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    plt.show()


def asymptotic_analysis_lexica(alpha, rsa_model, worlds, version, depth, verbose):
    entropies_list = []
    values_list = []
    gains_list = []
    last_speakers = []

    for world in worlds:
        # Initialize the RSA model with verbose mode
        rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=alpha, depth=depth)
        # Run the RSA model
        world, listeners, speakers = rsa.run(verbose)

        last_speakers.append(speakers[-1])

        entropies = [compute_shannon_conditional_entropy(world["priors"],speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        values = [compute_listener_value(world["priors"], listeners[(i+1)//2], speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        gains = [alpha * values[i] + entropies[i] for i in range((len(speakers)-1)*2)]

        entropies_list.append(entropies)
        values_list.append(values)
        gains_list.append(gains)

    fig, axs = plt.subplots(len(worlds), 5, figsize=(6*5, 4*len(worlds)))
    fig.suptitle(f"{world['surname']} with alpha={alpha}")

    axs[0,0].set_title("Initial lexicon")
    axs[0,1].set_title("Conditional entropy")
    axs[0,2].set_title("Listener value")
    axs[0,3].set_title("Gain function")
    axs[0,4].set_title("Speaker")        

    for i in range(len(worlds)):
        axs[i,0].set_ylabel(f"World number {i}")

        im = axs[i,0].imshow(np.round(worlds[i]["lexicon"],3), cmap='viridis', interpolation='nearest')
        cbar = fig.colorbar(im, ax=axs[i,0])
        for (j, k), value in np.ndenumerate(np.round(worlds[i]["lexicon"],3)):
            axs[i,0].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

        axs[i,1].plot(entropies_list[i])

        axs[i,2].plot(values_list[i])

        axs[i,3].plot(gains_list[i])

        im = axs[i,4].imshow(np.round(last_speakers[i],3), cmap='viridis', interpolation='nearest')
        cbar = fig.colorbar(im, ax=axs[i,4])
        for (j, k), value in np.ndenumerate(np.round(last_speakers[i],3)):
            axs[i,4].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    plt.show()


def compare_GD_KL_RSA_with_classic_RSA(alpha, gamma, worlds, rsa_model, depth=20, version='RSA', max_iter=3000,learning_rate=0.001, verbose=False):
    '''Compare the Neural Network implementation of RSA with the classic RSA implementation.
    input:
    * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
    * gamma, a float, the KL divergence weight
    * world, a dictionary with the following
        - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
        - world['priors']: a 1D array of float, the priors on the meaning list
    * max_iter, an int, the number of iterations
    * learning_rate, a float, the learning rate
    '''
    GD_loss_values = []
    GD_last_speakers = []
    GD_gain_values = []
    RSA_gains_list = []
    RSA_last_speakers = []

    for world in worlds:
        # Run NN model
        lexicon = torch.tensor(world["lexicon"], dtype=torch.float32)
        priors = torch.tensor(world["priors"], dtype=torch.float32)
        model = RSA_KL_model(alpha, gamma, priors, lexicon)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        losses = []
        for i in range(max_iter):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())           

        GD_last_speakers.append(torch.nn.functional.normalize(model.speaker, p=1, dim=0).detach().numpy())
        GD_gain_values.append(model.compute_speaker_entropy().item() + alpha * model.compute_listener_value().item())
        GD_loss_values.append(losses)

        # Run the RSA model
        rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=alpha, depth=depth)
        world, listeners, speakers = rsa.run(verbose)

        RSA_last_speakers.append(speakers[-1])
        entropies = [compute_shannon_conditional_entropy(world["priors"],speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        values = [compute_listener_value(world["priors"], listeners[(i+1)//2], speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        gains = [alpha * values[i] + entropies[i] for i in range((len(speakers)-1)*2)]
        RSA_gains_list.append(gains)

    fig, axs = plt.subplots(len(worlds), 5, figsize=(6*5, 4*len(worlds)))
    fig.suptitle(f"{world['surname']} with alpha={alpha}")

    axs[0,0].set_title("Initial lexicon")
    axs[0,1].set_title("RSA's last speaker")
    axs[0,2].set_title("GD's last speaker")
    axs[0,3].set_title("Gain function")
    axs[0,4].set_title("GD's loss")        

    for i in range(len(worlds)):
        axs[i,0].set_ylabel(f"World number {i}")

        im = axs[i,0].imshow(np.round(worlds[i]["lexicon"],3), cmap='viridis', interpolation='nearest')
        for (j, k), value in np.ndenumerate(np.round(worlds[i]["lexicon"],3)):
            axs[i,0].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

        im = axs[i,1].imshow(np.round(RSA_last_speakers[i],3), cmap='viridis', interpolation='nearest')
        for (j, k), value in np.ndenumerate(np.round(RSA_last_speakers[i],3)):
            axs[i,1].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

        im = axs[i,2].imshow(np.round(GD_last_speakers[i],3), cmap='viridis', interpolation='nearest')
        for (j, k), value in np.ndenumerate(np.round(GD_last_speakers[i],3)):
            axs[i,2].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

        axs[i,3].plot(RSA_gains_list[i])
        axs[i,3].axhline(y=GD_gain_values[i], color='red', linestyle='dotted')
        axs[i,3].annotate(f'GD gain={GD_gain_values[i]:.2f}', xy=(0, GD_gain_values[i]), xytext=(0, GD_gain_values[i]+0.05), arrowprops=dict(facecolor='red', shrink=0.005))

        axs[i,4].plot(GD_loss_values[i])        

    plt.show()

    return