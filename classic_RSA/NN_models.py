import torch
import torch.nn as nn
from utils import not_converged

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
        self.truth_table = nn.Parameter(torch.randn_like(lexicon), requires_grad=True)
        self.speaker = self.get_speaker()
        self.alpha = alpha
        self.gamma = gamma
        self.priors = priors
        self.lexicon_joint_proba = self.proba_from_lexicon(lexicon)
        self.eps = 1e-7

    def get_speaker(self):
        ''' Compute the speaker's truth table according to the current model's parameters.
        output:
        * speaker, a 2D array of float, the speaker's truth table
        '''
        return torch.nn.functional.normalize(nn.functional.relu(self.truth_table), p=1, dim=0)
    
    def get_listener(self):
        ''' Compute the listener according to the current model's parameters.
        output:
        * listener, a 2D array of float, the listener
        '''
        joint_proba = torch.matmul(nn.functional.relu(self.speaker), torch.diag(self.priors))
        listener = torch.nn.functional.normalize(joint_proba, p=1, dim=1)
        return listener
    
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

    def compute_listener_value(self):
        ''' Compute the listener value according to the  the current model's parameters.
        output:
        * listener_value, a float, the listener value
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        listener = torch.nn.functional.normalize(joint_proba, p=1, dim=1)
        log_listener = torch.log(listener + self.eps)
        listener_value = joint_proba.mul(log_listener)
        return torch.sum(listener_value)

    def compute_speaker_entropy(self):
        ''' Compute the speaker entropy according to the  the current model's parameters.
        output:
        * speaker_entropy, a float, the speaker entropy
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        log_speaker = torch.log(self.speaker + self.eps)
        speaker_entropy = joint_proba.mul(log_speaker)
        return - torch.sum(speaker_entropy)

    def compute_KL_distance(self):
        ''' Compute the KL distance between the initial lexicon and the speaker's truth table.
        output:
        * KL_distance, a float, the KL distance
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        log_distance = torch.log(self.speaker + self.eps) - torch.log(self.lexicon_joint_proba + self.eps)
        KL_distance = joint_proba.mul(log_distance)
        return torch.sum(KL_distance)

    def compute_TV_distance(self):
        ''' Compute the TV distance between the initial lexicon and the speaker's truth table.
        output:
        * TV_distance, a float, the TV distance
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        TV_distance = torch.abs(joint_proba - self.lexicon_joint_proba)
        return torch.sum(TV_distance)
    
    def compute_Jensen_Shannon_divergence(self):
        ''' Compute the Jensen-Shannon divergence between the initial lexicon and the speaker's truth table.
        output:
        * Jensen_Shannon_divergence, a float, the Jensen-Shannon divergence
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        mean_proba = (joint_proba + self.lexicon_joint_proba) / 2
        log_distance = torch.log(self.speaker + self.eps) - torch.log(mean_proba + self.eps)
        JS_divergence = joint_proba.mul(log_distance)
        return torch.sum(JS_divergence)
    
    def compute_Wasserstein_distance(self):
        ''' Compute the Wasserstein distance between the initial lexicon and the speaker's truth table.
        output:
        * wasserstein_distance, a float, the Wasserstein distance
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        wasserstein_distance = torch.abs(joint_proba - self.lexicon_joint_proba)
        return torch.sum(wasserstein_distance)

    def compute_cramer_von_Mises_distance(self):
        ''' Compute the Cramer von Mises distance between the initial lexicon and the speaker's truth table.
        output:
        * cramer_von_Mises_distance, a float, the Cramer von Mises distance
        '''
        joint_proba = torch.matmul(self.speaker, torch.diag(self.priors))
        cramer_von_Mises_distance = (joint_proba - self.lexicon_joint_proba)**2
        return torch.sum(cramer_von_Mises_distance)

    def get_RSA_gain(self):
        ''' Compute the RSA gain function.
        output:
        * gain, a float, the RSA gain function
        '''
        listener_value = self.compute_listener_value()
        speaker_entropy = self.compute_speaker_entropy()
        return self.alpha * listener_value + speaker_entropy

    def forward(self):
        ''' Compute the loss function.
        output:
        * loss, a float, the loss function
        '''
        self.speaker = self.get_speaker()
        listener_value = self.compute_listener_value()
        speaker_entropy = self.compute_speaker_entropy()
        proba_distance = self.compute_cramer_von_Mises_distance()
        return - self.alpha * listener_value - speaker_entropy + self.gamma * proba_distance
    
    
class RSA_Optimal_Transport_model(nn.Module):
    ''' A speaker model whose loss function is the RSA's one to which a KL divergence term is added.
        The KL divergence term is the distance between the initial lexicon and the speaker's truth table.
        The Listener is computed as the bayesian inference of the speaker's truth table according to the priors.
    '''

    def __init__(self, beta, gamma, lamb, priors, lexicon):
        ''' Initialize the model.
        input:
        * alpha, a float, the speaker pragmatism parameter (default=1, the higher the more pragmatic)
        * gamma, a float, the KL divergence weight
        * priors, a 1D array of float, the priors on the meaning list
        * lexicon, a 2D array of float, the initial lexicon
        '''
        super(RSA_Optimal_Transport_model, self).__init__()
        self.truth_table = nn.Parameter(torch.randn_like(lexicon), requires_grad=True)
        self.positive_truth_table = torch.nn.functional.relu(self.truth_table)
        self.beta = beta
        self.gamma = gamma
        self.lamb = lamb
        self.priors = priors
        self.joint_lexicon = self.proba_from_lexicon(lexicon)
        self.eps = 1e-7

    def get_speaker(self):
        ''' Compute the speaker's truth table according to the current model's parameters.
        output:
        * speaker, a 2D array of float, the speaker's truth table
        '''
        return torch.nn.functional.normalize(self.positive_truth_table, p=1, dim=0)
    
    def get_listener(self):
        ''' Compute the listener according to the current model's parameters.
        output:
        * listener, a 2D array of float, the listener
        '''
        speaker = self.get_speaker()
        joint_proba = torch.matmul(speaker, torch.diag(self.priors))
        listener = torch.nn.functional.normalize(joint_proba, p=1, dim=1)
        return listener
    
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

    def compute_KL_distance(self):
        ''' Compute the KL distance between the initial lexicon and the speaker's truth table.
        output:
        * KL_distance, a float, the KL distance
        '''
        joint_proba = torch.matmul(self.positive_truth_table, torch.diag(self.priors))
        log_distance = torch.log(joint_proba + self.eps) - torch.log(self.joint_lexicon + self.eps)
        KL_distance = joint_proba.mul(log_distance)
        return torch.sum(KL_distance)

    def compute_norm_u_constraint(self):
        ''' Compute the Lagrangian constraint that the columns (for a given meaning) of the speaker's truth table must sum to 1.
        output:
        * norm_u_constraint, a float, the Lagrangian constraint
        '''
        norm_u_constraint = self.positive_truth_table @ torch.ones(self.positive_truth_table.shape[1]) - (self.positive_truth_table.shape[1]/self.positive_truth_table.shape[0]) * torch.ones(self.positive_truth_table.shape[0])
        return torch.sum(norm_u_constraint)

    def compute_norm_m_constraint(self):
        ''' Compute the Lagrangian constraint that the rows (for a given utterance u) of the speaker's truth table must sum to M*p(u) (M and p(u) being the number of meanings and the prior on u).
        output:
        * norm_m_constraint, a float, the Lagrangian constraint
        '''
        speaker_T = torch.transpose(self.positive_truth_table, 0, 1)
        norm_m_constraint = speaker_T @ torch.ones(self.positive_truth_table.shape[0]) - torch.ones(self.positive_truth_table.shape[1])
        return torch.sum(norm_m_constraint)
    
    def compute_positivy_constraint(self):
        ''' Compute the Lagrangian constraint that the speaker's truth table must be positive.
        output:
        * positivity_constraint, a float, the Lagrangian constraint
        '''
        return - torch.sum(self.truth_table)
    
    def compute_listener_value(self):
        ''' Compute the listener value according to the  the current model's parameters.
        output:
        * listener_value, a float, the listener value
        '''
        speaker = self.get_speaker()
        joint_proba = torch.matmul(speaker, torch.diag(self.priors))
        listener = torch.nn.functional.normalize(joint_proba, p=1, dim=1)
        log_listener = torch.log(listener + self.eps)
        listener_value = joint_proba.mul(log_listener)
        return torch.sum(listener_value)
    
    def compute_speaker_entropy(self):
        ''' Compute the speaker entropy according to the  the current model's parameters.
        output:
        * speaker_entropy, a float, the speaker entropy
        '''
        speaker = self.get_speaker()
        joint_proba = torch.matmul(speaker, torch.diag(self.priors))
        log_speaker = torch.log(speaker + self.eps)
        speaker_entropy = joint_proba.mul(log_speaker)
        return - torch.sum(speaker_entropy)

    def get_RSA_gain(self):
        ''' Compute the RSA gain function.
        output:
        * gain, a float, the RSA gain function
        '''
        listener_value = self.compute_listener_value()
        speaker_entropy = self.compute_speaker_entropy()
        return listener_value + speaker_entropy

    def forward(self):
        ''' Compute the loss function.
        output:
        * loss, a float, the loss function
        '''
        self.positive_truth_table = torch.nn.functional.relu(self.truth_table)
        KL_distance = self.compute_KL_distance()
        norm_u_constraint = self.compute_norm_u_constraint()
        norm_m_constraint = self.compute_norm_m_constraint()
        positivity_constraint = self.compute_positivy_constraint()
        return KL_distance + self.beta * norm_u_constraint + self.gamma * norm_m_constraint + self.lamb * positivity_constraint