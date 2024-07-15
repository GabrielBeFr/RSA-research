import torch
import torch.nn as nn
from sklearn import preprocessing

class RSA_KL_model(nn.Module):
    ''' A speaker model whose loss function is the RSA's one to which a KL divergence term is added.
        The KL divergence term is the distance between the initial lexicon and the speaker's truth table.
        The Listener is computed as the bayesian inference of the speaker's truth table according to the priors.
    '''

    def __init__(self, alpha, gamma, priors, lexicon):
        ''' Initialize the model.
        input:
        * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
        * gamma, a float, the KL divergence weight
        * priors, a 1D array of float, the priors on the meaning list
        * lexicon, a 2D array of float, the initial lexicon
        '''
        super(RSA_KL_model, self).__init__()
        self.speaker = nn.Parameter(torch.clone(lexicon), requires_grad=True)
        self.alpha = alpha
        self.gamma = gamma
        self.priors = priors
        self.lexicon = self.proba_from_lexicon(lexicon)
    
    def proba_from_lexicon(self, lexicon):
        ''' Compute the joint probability according to the current model's parameters of each meaning
            given each utterance.
        input:
        * lexicon, a 2D array of float, the lexicon
        
        output:
        * joint_proba, a 2D array of float, the joint probability of each meaning given each utterance
        '''
        speaker = torch.nn.functional.normalize(lexicon, p=1, dim=0)
        joint_proba = torch.matmul(speaker, torch.diag(self.priors))
        return joint_proba
    
    def compute_listener_truth_table(self):
        ''' Compute the listener truth table according to the  the current model's parameters.
        output:
        * listener_truth_table, a 2D array of float, the listener truth table
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        listener = torch.nn.functional.normalize(joint_proba, p=1, dim=1)
        return listener

    def compute_listener_value(self):
        ''' Compute the listener value according to the  the current model's parameters.
        output:
        * listener_value, a float, the listener value
        '''
        eps = 1e-7
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        listener = torch.nn.functional.normalize(joint_proba, p=1, dim=1)
        log_listener = torch.log(listener + eps)
        listener_value = joint_proba.mul(log_listener)
        return torch.sum(listener_value)

    def compute_speaker_entropy(self):
        ''' Compute the speaker entropy according to the  the current model's parameters.
        output:
        * speaker_entropy, a float, the speaker entropy
        '''
        eps = 1e-7
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        log_speaker = torch.log(self.speaker + eps)
        speaker_entropy = - joint_proba.mul(log_speaker)
        return torch.sum(speaker_entropy)
    
    def compute_KL_distance(self):
        ''' Compute the KL distance between the initial lexicon and the speaker's truth table.
        output:
        * KL_distance, a float, the KL distance
        '''
        eps = 1e-7
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        log_distance = torch.log(self.speaker + eps) - torch.log(self.lexicon + eps)
        KL_distance = joint_proba.mul(log_distance)
        return torch.sum(KL_distance)

    def forward(self):
        ''' Compute the loss function.
        output:
        * loss, a float, the loss function
        '''
        listener_value = self.compute_listener_value()
        speaker_entropy = self.compute_speaker_entropy()
        KL_distance = self.compute_KL_distance()
        return - self.alpha * listener_value - speaker_entropy + self.gamma * KL_distance