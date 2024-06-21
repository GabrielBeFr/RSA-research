from agents import litteral_listener, pragmatic_speaker, pragmatic_listener
from agents import litteral_listener_lexical_uncertainty, pragmatic_speaker_lexical_uncertainty, pragmatic_listener_lexical_uncertainty
import numpy as np


################## VANILLA RSA ##################


class classic_RSA():
    def __init__(self, world, save_directory, alpha=1, depth=1):
        '''Initialize the RSA model.
        input:
        * world, a dictionary with the following
            - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
            - world['costs']: a 1D array of float, the cost of each utterance
            - world['priors']: a 1D array of float, the priors on the meaning list
        * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
        * depth, an integer, the number of iterations of the RSA model (default=1)
        '''
        self.world = world
        self.alpha = alpha
        self.depth = depth
        self.path = save_directory + world['file_name']

    def run(self, verbose=False):
        '''Run the RSA model for the specified number of iterations.
        input:
        * verbose, a boolean, whether to print the probabilities at each step (default=False)

        output: 
        * probabilities, a 2D array of float, the probability of each meaning given each utterance
        '''
        with open(self.path, 'w') as f:
            f.write(f'Topic: {self.world["surname"]}\n\n')

            probabilities = litteral_listener(self.world)
            if verbose:
                print('--- Litteral listener: ---')
                print(np.round(probabilities,3))
            f.write(f'--- Litteral listener: ---\n{np.round(probabilities,3)}\n\n')

            for step in range(1,self.depth+1):
                probabilities = pragmatic_speaker(self.world, probabilities, self.alpha)
                if verbose:
                    print(f'--- Pragmatic speaker {step}: ---')
                    print(np.round(probabilities,3))
                f.write(f'--- Pragmatic speaker {step}: ---\n{np.round(probabilities,3)}\n\n')

                probabilities = pragmatic_listener(self.world, probabilities)
                if verbose:
                    print(f'--- Pragmatic listener {step}: ---')
                    print(np.round(probabilities,3))
                f.write(f'--- Pragmatic listener {step}: ---\n{np.round(probabilities,3)}\n\n')

        return probabilities
    

################## LEXICAL UNCERTAINTY ##################

    
class lexical_uncertainty_RSA():
    def __init__(self, world, save_directory, alpha=1, depth=1):
        '''Initialize the RSA model.
        input:
        * world, a dictionary with the following
            - world['lexicon']: a 2D array of boolean, the lexicon, where lexicon[i,j]=True if utterance i maps to meaning j and False otherwise
            - world['costs']: a 1D array of float, the cost of each utterance
            - world['priors']: a 1D array of float, the priors on the meaning list
        * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
        * depth, an integer, the number of iterations of the RSA model (default=1)
        '''
        self.world = world
        self.alpha = alpha
        self.depth = depth
        self.path = save_directory + world['file_name']

    def generate_meanings(self, utterance, offset=0):
        '''Generate the meanings from the world's lexicon.
        input:
        * utterance, an integer, the utterance to generate from

        output: 
        * generated_meanings, a 2D array of boolean, the generated meanings
        '''
        generated_meanings = []
        if np.any(utterance):
            generated_meanings.append(utterance)
            for i in range(offset,len(utterance)):
                meaning = utterance[i]
                if meaning:
                    new_utterance = utterance.copy()
                    new_utterance[i] = 0
                    generated_meanings.extend(self.generate_meanings(new_utterance, offset+i))
        return generated_meanings
    
    def generate_lexica(self, huge_lexicon, incomplete_lexicon=[], offset=0):
        '''Generate the lexica from the world's lexicon based on the newly generated meanings.
        input:
        * huge_lexicon, a 2D array of boolean, the lexicon to generate from
        * incomplete_lexicon, a 1D array of boolean, the current lexicon
        * offset, an integer, the current offset

        output:
        * generated_lexica, a 3D array of boolean, the generated lexica
        '''
        generated_lexica = []
        current_utterance = huge_lexicon[offset]
        current_lexicon = incomplete_lexicon.copy()
        for j, meaning in enumerate(current_utterance):
            if j == 0:
                current_lexicon.append(meaning)
            else:
                current_lexicon[-1] = meaning

            if offset < len(huge_lexicon)-1:
                generated_lexica.extend(self.generate_lexica(huge_lexicon, current_lexicon, offset+1))
            else:
                generated_lexica.append(current_lexicon)
                
        return generated_lexica

    def initiate_lexica(self, lexicon):
        '''Generate the lexica from the world's lexicon.
        input:
        * lexicon, a 2D array of boolean, the lexicon to generate from

        output: 
        * generated_lexica, a 3D array of boolean, the generated lexica
        '''
        huge_lexicon = []
        for i, utterance in enumerate(lexicon):
            huge_lexicon.append(self.generate_meanings(utterance))
        
        generated_lexica = self.generate_lexica(huge_lexicon)

        return np.array(generated_lexica)

    def run(self, verbose=False):
        '''Run the RSA model for the specified number of iterations.
        input:
        * verbose, a boolean, whether to print the probabilities at each step (default=False)

        output: 
        * probabilities, a 2D array of float, the probability of each meaning given each utterance
        '''
        self.world["lexica"] = self.initiate_lexica(self.world['lexicon'])

        with open(self.path, 'w') as f:
            f.write(f'RSA model with lexical uncertainty\n')
            f.write(f'Topic: {self.world["surname"]}\n\n')

            probabilities = litteral_listener_lexical_uncertainty(self.world)
            if verbose:
                print('--- Litteral listener: ---')
                print(np.round(probabilities,3))
            f.write(f'--- Litteral listener: ---\n{np.round(probabilities,3)}\n\n')

            step = 1
            probabilities = pragmatic_speaker_lexical_uncertainty(self.world, probabilities, self.alpha)
            if verbose:
                print(f'--- Pragmatic speaker {step}: ---')
                print(np.round(probabilities,3))
            f.write(f'--- Pragmatic speaker {step}: ---\n{np.round(probabilities,3)}\n\n')

            probabilities = pragmatic_listener_lexical_uncertainty(self.world, probabilities)
            if verbose:
                print(f'--- Pragmatic listener {step}: ---')
                print(np.round(probabilities,3))
            f.write(f'--- Pragmatic listener {step}: ---\n{np.round(probabilities,3)}\n\n')

            for step in range(2,self.depth+1):
                probabilities = pragmatic_speaker(self.world, probabilities, self.alpha)
                if verbose:
                    print(f'--- Pragmatic speaker {step}: ---')
                    print(np.round(probabilities,3))
                f.write(f'--- Pragmatic speaker {step}: ---\n{np.round(probabilities,3)}\n\n')

                probabilities = pragmatic_listener(self.world, probabilities)
                if verbose:
                    print(f'--- Pragmatic listener {step}: ---')
                    print(np.round(probabilities,3))
                f.write(f'--- Pragmatic listener {step}: ---\n{np.round(probabilities,3)}\n\n')

        return probabilities