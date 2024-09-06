from utils import compute_listener_value, compute_shannon_conditional_entropy, not_converged, compute_KL_div
import matplotlib.pyplot as plt
import numpy as np
import torch
from NN_models import RSA_KL_model, RSA_Optimal_Transport_model
from tqdm import tqdm
from pdb import set_trace
from IPython.display import display, clear_output

def asymptotic_analysis_alphas(alphas_list, rsa_model, world, version, depth, verbose):
    entropies_list = []
    values_list = []
    gains_list = []
    last_speakers = []
    coop_index_list = []

    for new_alpha in alphas_list:
        # Initialize the RSA model with verbose mode
        rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=new_alpha, depth=depth)
        # Run the RSA model
        world, listeners, speakers = rsa.run(verbose)

        last_speakers.append(speakers[-1])

        entropies = [compute_shannon_conditional_entropy(world["priors"],speakers[i]) for i in range(len(speakers))]
        values = [new_alpha * compute_listener_value(world["priors"], listeners[i], speakers[i]) for i in range(len(speakers))]
        gains = [values[i] + entropies[i] for i in range(len(speakers))]

        coop_index = [np.sum(np.multiply(speakers[i],listeners[i]))/len(listeners[i]) for i in range(len(speakers))]

        entropies_list.append(entropies)
        values_list.append(values)
        gains_list.append(gains)
        coop_index_list.append(coop_index)

    fig, axs = plt.subplots(len(alphas_list)+1, 4, figsize=(6*4, 4*(len(alphas_list)+1)))

    fig_bis, axs_bis = plt.subplots(1, 1+len(alphas_list), figsize=((1+len(alphas_list))*4, 4))

    fig_CI, axs_CI = plt.subplots(1, len(alphas_list), figsize=(len(alphas_list)*4, 3))

    axs[0,0].remove()
    im = axs[0,1].imshow(np.round(world["lexicon"],3), cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(im, ax=axs[0,1])
    for (j, k), value in np.ndenumerate(np.round(world["lexicon"],3)):
            axs[0,1].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    axs_bis[0].imshow(np.round(world["lexicon"],3), cmap='viridis', interpolation='nearest')
    axs_bis[0].set_title(f"Initial Lexicon")
    for (j, k), value in np.ndenumerate(np.round(world["lexicon"],3)):
            axs_bis[0].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

    axs[0,1].set_title("Initial Lexicon")
    axs[0,2].remove()
    axs[0,3].remove()

    axs[1,0].set_title("Conditional entropy")
    axs[1,1].set_title("Listener value")
    axs[1,2].set_title("Gain function")
    axs[1,3].set_title("Speaker")        

    for i in range(len(alphas_list)):
        axs_CI[i].plot(coop_index_list[i])
        axs_CI[i].set_title(f"Cooperation index for Alpha={alphas_list[i]}")

        axs[1+i,0].set_ylabel(f"Alpha={alphas_list[i]}")
        axs[1+i,0].plot(entropies_list[i])

        axs_bis[i+1].imshow(np.round(last_speakers[i],3), cmap='viridis', interpolation='nearest')
        axs_bis[i+1].set_title(f"Final Speaker for Alpha={alphas_list[i]}")
        for (j, k), value in np.ndenumerate(np.round(last_speakers[i],3)):
            axs_bis[i+1].text(k, j, f'{value:.2f}', ha='center', va='center', color='black')

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
        i=0
        while i <= max_iter and not_converged(losses):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            losses.append(loss.item()) 
            i+=1      

        final_speaker = model.get_speaker().detach().numpy()
        GD_last_speakers.append(final_speaker)
        GD_gain_values.append(model.get_RSA_gain().item())
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
        axs[i,3].annotate(f'GD gain={GD_gain_values[i]:.2f}', xy=(0, GD_gain_values[i]), xytext=(0, GD_gain_values[i]), color='red')
        
        axs[i,4].plot(GD_loss_values[i])        

    plt.show()


def large_comparison_GD_KL_RSA_with_classic_RSA(alphas, gammas, rsa_model, num_measures=100, depth=100, version='RSA', max_iter=40000, learning_rate=0.01, x=4, y=4, verbose=False):
    '''Compare the Neural Network implementation of RSA with the classic RSA implementation.
    input:
    * alphas, a list of floats, the speaker pragmatism parameters (default=1, the higher the more pragmatic)
    * gamma, a list of floats, the KL divergence weights
    * max_depth, an int, the maximum depth of the RSA model
    * version, a string, the version of the RSA model
    * max_iter, an int, the number of iterations
    * learning_rate, a float, the learning rate
    * verbose, a boolean, the verbose mode
    '''

    GD_last_speakers = np.zeros((len(alphas), len(gammas), num_measures, x, y))
    GD_gain_list = np.zeros((len(alphas), len(gammas), num_measures))
    RSA_last_speakers = np.zeros((len(alphas), len(gammas), num_measures, x, y))
    RSA_gain_list = np.zeros((len(alphas), len(gammas), num_measures))

    for alpha in alphas:
        print(f"--- Alpha = {alpha} ---")
        for gamma in gammas:
            print(f"    * gamma = {gamma}:")
            for i in tqdm(range(num_measures)):
                # Initialize the world
                world = {
                    'file_name': 'comparison_GD_KL_RSA.txt',
                    'surname': 'Comparison between classic RSA and gradient descent RSA',
                    'lexicon': np.random.random((x,y)),
                    'costs': np.zeros(x),
                    'priors': np.ones(y)*(1/y)
                }
                # Run NN model
                lexicon = torch.tensor(world["lexicon"], dtype=torch.float32)
                priors = torch.tensor(world["priors"], dtype=torch.float32)
                model = RSA_KL_model(alpha, gamma, priors, lexicon)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                losses = []
                j=0  
                while j <= max_iter and not_converged(losses):
                    optimizer.zero_grad()
                    loss = model()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item()) 
                    j+=1      

                final_speaker = model.get_speaker().detach().numpy()
                GD_last_speakers[alphas.index(alpha), gammas.index(gamma), i] = final_speaker
                GD_gain_list[alphas.index(alpha), gammas.index(gamma), i] = model.get_RSA_gain().item()

                # Run the RSA model
                rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=alpha, depth=depth)
                world, listeners, speakers = rsa.run(verbose)

                RSA_last_speakers[alphas.index(alpha), gammas.index(gamma), i] = speakers[-1]
                RSA_entropy = compute_shannon_conditional_entropy(world["priors"],speakers[-1])
                RSA_value = compute_listener_value(world["priors"], listeners[-1], speakers[-1])
                RSA_gain_list[alphas.index(alpha), gammas.index(gamma), i] = alpha * RSA_value + RSA_entropy

    fig, axs = plt.subplots(len(alphas), 2, figsize=(10*2, 5*len(alphas)))
    fig.suptitle(f"{world['surname']} for {num_measures} random initial lexica")

    axs[0,0].set_title(f"KL divergence between RSA and GD-RSA")
    axs[0,1].set_title(f"Difference between RSA and GD-RSA gains")  
    
    for i_alpha in range(len(alphas)):
        print(GD_last_speakers[i_alpha])
        print(RSA_last_speakers[i_alpha])
        axs[i_alpha,0].set_ylabel(f"Alpha = {alphas[i_alpha]}")

        kl_divergence = np.array([compute_KL_div(RSA_last_speakers[i_alpha, i_gamma], GD_last_speakers[i_alpha, i_gamma]) for i_gamma in range(len(gammas))])
        print(kl_divergence)

        axs[i_alpha,0].boxplot(kl_divergence.T, labels=gammas)
        axs[i_alpha,0].set_xlabel('gamma')

        gains_difference = RSA_gain_list[i_alpha] - GD_gain_list[i_alpha]
        print(gains_difference)
        
        axs[i_alpha,1].boxplot(gains_difference.T, labels=gammas)
        axs[i_alpha,1].set_xlabel('gamma')

    plt.show()


def compare_GD_Opti_Transport_RSA_with_classic_RSA(beta, gamma, lamb, worlds, rsa_model, depth=20, version='RSA', max_iter=3000,learning_rate=0.001, verbose=False):
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
    GD_KL = []
    GD_positivity_constraint = []
    GD_m_constraint = []
    GD_u_constraint = []
    GD_listener_value = []
    GD_speaker_entropy = []
    RSA_gains_list = []
    RSA_last_speakers = []

    for k_world, world in enumerate(worlds):
        # Run NN model
        lexicon = torch.tensor(world["lexicon"], dtype=torch.float32)
        priors = torch.tensor(world["priors"], dtype=torch.float32)
        model = RSA_Optimal_Transport_model(beta, gamma, lamb, priors, lexicon)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        losses = []
        i=0
        while i <= max_iter and not_converged(losses):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            losses.append(loss.item()) 
            if i % 100 == 0:
                clear_output(wait=True)
                loss_display = display(f'Loss for world number {k_world}: {loss.item()}', display_id=2, clear=False)
            i+=1      

        final_speaker = model.get_speaker().detach().numpy()
        GD_last_speakers.append(final_speaker)
        GD_gain_values.append(model.get_RSA_gain().item())
        GD_loss_values.append(losses)
        GD_KL.append(model.compute_KL_distance().item())
        GD_positivity_constraint.append(model.compute_positivy_constraint().item())
        GD_m_constraint.append(model.compute_norm_m_constraint().item())
        GD_u_constraint.append(model.compute_norm_u_constraint().item())
        GD_listener_value.append(model.compute_listener_value().item())
        GD_speaker_entropy.append(model.compute_speaker_entropy().item())

        # Run the RSA model
        rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=1, depth=depth)
        world, listeners, speakers = rsa.run(verbose)

        RSA_last_speakers.append(speakers[-1])
        entropies = [compute_shannon_conditional_entropy(world["priors"],speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        values = [compute_listener_value(world["priors"], listeners[(i+1)//2], speakers[1 + i//2]) for i in range((len(speakers)-1)*2)]
        gains = [values[i] + entropies[i] for i in range((len(speakers)-1)*2)]
        RSA_gains_list.append(gains)

    fig, axs = plt.subplots(len(worlds), 5, figsize=(6*5, 4*len(worlds)))
    fig.suptitle(f"{world['surname']}")

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
        axs[i,3].annotate(f'GD gain={GD_gain_values[i]:.2f}', xy=(0, GD_gain_values[i]), xytext=(0, GD_gain_values[i]), color='red')
        
        axs[i,4].plot(GD_loss_values[i])     

    plt.show()


def large_comparison_GD_Opti_Transport_RSA_with_classic_RSA(betas, gammas, lamb, rsa_model, num_measures=100, depth=100, version='RSA', max_iter=40000, learning_rate=0.01, x=4, y=4, verbose=False):
    '''Compare the Neural Network implementation of RSA with the classic RSA implementation.
    input:
    * betas, a list of floats, the speaker pragmatism parameters (default=1, the higher the more pragmatic)
    * gamma, a list of floats, the KL divergence weights
    * max_depth, an int, the maximum depth of the RSA model
    * version, a string, the version of the RSA model
    * max_iter, an int, the number of iterations
    * learning_rate, a float, the learning rate
    * verbose, a boolean, the verbose mode
    '''

    GD_last_speakers = np.zeros((len(betas), len(gammas), num_measures, x, y))
    GD_gain_list = np.zeros((len(betas), len(gammas), num_measures))
    RSA_last_speakers = np.zeros((len(betas), len(gammas), num_measures, x, y))
    RSA_gain_list = np.zeros((len(betas), len(gammas), num_measures))

    for beta in betas:
        print(f"--- Alpha = {beta} ---")
        for gamma in gammas:
            print(f"    * gamma = {gamma}:")
            for i in tqdm(range(num_measures)):
                # Initialize the world
                world = {
                    'file_name': 'comparison_GD_Opti_Transport_RSA.txt',
                    'surname': 'Comparison between classic RSA and gradient descent RSA',
                    'lexicon': np.random.random((x,y)),
                    'costs': np.zeros(x),
                    'priors': np.ones(y)*(1/y)
                }
                # Run NN model
                lexicon = torch.tensor(world["lexicon"], dtype=torch.float32)
                priors = torch.tensor(world["priors"], dtype=torch.float32)
                model = RSA_Optimal_Transport_model(beta, gamma, lamb, priors, lexicon)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                losses = []
                j=0  
                while j <= max_iter and not_converged(losses):
                    optimizer.zero_grad()
                    loss = model()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item()) 
                    j+=1      

                final_speaker = model.get_speaker().detach().numpy()
                GD_last_speakers[betas.index(beta), gammas.index(gamma), i] = final_speaker
                GD_gain_list[betas.index(beta), gammas.index(gamma), i] = model.get_RSA_gain().item()

                # Run the RSA model
                rsa = rsa_model(world, save_directory='papers_experiments/',version=version, alpha=beta, depth=depth)
                world, listeners, speakers = rsa.run(verbose)

                RSA_last_speakers[betas.index(beta), gammas.index(gamma), i] = speakers[-1]
                RSA_entropy = compute_shannon_conditional_entropy(world["priors"],speakers[-1])
                RSA_value = compute_listener_value(world["priors"], listeners[-1], speakers[-1])
                RSA_gain_list[betas.index(beta), gammas.index(gamma), i] = beta * RSA_value + RSA_entropy

    fig, axs = plt.subplots(len(betas), 2, figsize=(10*2, 5*len(betas)))
    fig.suptitle(f"{world['surname']} for {num_measures} random initial lexica")

    axs[0,0].set_title(f"KL divergence between RSA and GD-RSA")
    axs[0,1].set_title(f"Difference between RSA and GD-RSA gains")  
    
    for i_beta in range(len(betas)):
        print(GD_last_speakers[i_beta])
        print(RSA_last_speakers[i_beta])
        axs[i_beta,0].set_ylabel(f"Beta = {betas[i_beta]}")

        kl_divergence = np.array([compute_KL_div(RSA_last_speakers[i_beta, i_gamma], GD_last_speakers[i_beta, i_gamma]) for i_gamma in range(len(gammas))])
        print(kl_divergence)

        axs[i_beta,0].boxplot(kl_divergence.T, labels=gammas)
        axs[i_beta,0].set_xlabel('gamma')

        gains_difference = RSA_gain_list[i_beta] - GD_gain_list[i_beta]
        print(gains_difference)
        
        axs[i_beta,1].boxplot(gains_difference.T, labels=gammas)
        axs[i_beta,1].set_xlabel('gamma')

    plt.show()