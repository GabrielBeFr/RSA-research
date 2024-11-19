import torch

def literal_listener_B(initial_lexicon: torch.FloatTensor, prior: torch.FloatTensor, verbose: bool):
    '''Compute the literal listener B in a single round.
    input:
    * initial_lexicon, a 2D torch tensor of float, the lexicon of the agent B
    * prior, a 3D torch tensor of float, the prior on the meaning of the agent A (dimension 0), the meaning of the agent B (dimension 1), the world (dimension 2)
    * verbose, a boolean, whether to print the probabilities at each step
    
    output:
    * literal_listener_B, a 2D torch tensor of float, the literal listener B with the meanings as the first dimension and the utterances as the second dimension
    '''
    # Compute the literal listener: L_B(y|mB, u) ~ SUM_{mA} Lex_A(mA, u)*P(mA, mB, y)
    literal_listener_B = torch.einsum('A u, A B y -> B y u', initial_lexicon, prior)

    # Masked division to avoid division by zero errors
    mask = torch.linalg.norm(literal_listener_B, ord=1, dim=1, keepdim=True) 
    #mask = literal_listener_B.sum(dim=1, keepdim=True)
    mask[mask == 0] = 1  # Replace zeros with ones to avoid division by zero
    literal_listener_B = literal_listener_B / mask

    return literal_listener_B

def literal_listener_A(initial_lexicon: torch.FloatTensor, prior: torch.FloatTensor, verbose: bool):
    '''Compute the literal listener A in a single round.
    input:
    * initial_lexicon, a 2D torch tensor of float, the lexicon of the agent A
    * prior, a 3D torch tensor of float, the prior on the meaning of the agent A (dimension 0), the meaning of the agent B (dimension 1), the world (dimension 2)
    * verbose, a boolean, whether to print the probabilities at each step

    output:
    * literal_listener_A, a 2D torch tensor of float, the literal listener A with the meanings as the first dimension and the utterances as the second dimension
    '''
    # Compute the literal listener: L_A(y|mA, v) ~ SUM_{mB} Lex_B(mB, v)*P(mA, mB, y)
    literal_listener_A = torch.einsum('B v, A B y -> A y v', initial_lexicon, prior)

    # Masked division to avoid division by zero errors
    mask = torch.linalg.norm(literal_listener_A, ord=1, dim=1, keepdim=True) 
    #mask = literal_listener_A.sum(dim=1, keepdim=True)
    mask[mask == 0] = 1  # Replace zeros with ones to avoid division by zero
    literal_listener_A = literal_listener_A / mask

    return literal_listener_A

def pragmatic_speaker_A(literal_listener_B: torch.FloatTensor, prior: torch.FloatTensor, alpha: int, RSA_depth: int, verbose: bool):
    '''Compute the pragmatic speaker A in a single round.
    input:
    * literal_listener_B, a 2D torch tensor of float, the literal listener B with the meanings as the first dimension and the utterances as the second dimension
    * prior, a 3D torch tensor of float, the prior on the meaning of the agent A (dimension 0), the meaning of the agent B (dimension 1), the world (dimension 2)
    * alpha, a float, the alpha pragmatism parameter of RSA
    * RSA_depth, an int, the depth of the RSA model
    * verbose, a boolean, whether to print the probabilities at each step

    output:
    * pragmatic_speaker, a 2D torch tensor of float, the pragmatic speaker A with the meanings as the first dimension and the utterances as the second dimension
    '''
    # Compute conditioned prior: P(mA, y|mB) = P(mA, mB, y) / SUM_{mA, y} P(mA, mB, y)
    prior = prior/(prior.sum(dim=2, keepdim=True).sum(dim=0, keepdim=True) + 1e-10)

    # Compute the pragmatic speaker: S_A(u|mA) ~ exp( alpha * SUM_{mB, y} P(mA, y|mB)*L_B(y|mB, u)
    pragmatic_speaker_A = torch.einsum('B y u, A B y -> A u', torch.log(literal_listener_B + 1e-10), prior)
    pragmatic_speaker_A = torch.exp(alpha * pragmatic_speaker_A)

    # Masked division to avoid division by zero errors
    mask = torch.linalg.norm(pragmatic_speaker_A, ord=1, dim=1, keepdim=True) 
    #mask = pragmatic_speaker_A.sum(dim=1, keepdim=True)
    mask[mask == 0] = 1  # Replace zeros with ones to avoid division by zero
    pragmatic_speaker_A = pragmatic_speaker_A / mask

    return pragmatic_speaker_A


def pragmatic_speaker_B(literal_listener_A: torch.FloatTensor, prior: torch.FloatTensor, alpha: int, RSA_depth: int, verbose: bool):
    '''Compute the pragmatic speaker B in a single round.
    input:
    * literal_listener_A, a 2D torch tensor of float, the literal listener A with the meanings as the first dimension and the utterances as the second dimension
    * prior, a 3D torch tensor of float, the prior on the meaning of the agent A (dimension 0), the meaning of the agent B (dimension 1), the world (dimension 2)
    * alpha, a float, the alpha pragmatism parameter of RSA
    * RSA_depth, an int, the depth of the RSA model
    * verbose, a boolean, whether to print the probabilities at each step

    output:
    * pragmatic_speaker, a 2D torch tensor of float, the pragmatic speaker B with the meanings as the first dimension and the utterances as the second dimension
    '''
    # Compute conditioned prior: P(mB, y|mA) = P(mA, mB, y) / SUM_{mB, y} P(mA, mB, y)
    prior = prior/(prior.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) + 1e-10)

    # Compute the pragmatic speaker: S_B(v|mB) ~ exp( alpha * SUM_{mA, y} P(mB, y|mA)*L_A(y|mA, v)
    pragmatic_speaker_B = torch.einsum('A y v, A B y -> B v', torch.log(literal_listener_A + 1e-10), prior)
    pragmatic_speaker_B = torch.exp(alpha * pragmatic_speaker_B)

    # Masked division to avoid division by zero errors
    mask = torch.linalg.norm(pragmatic_speaker_B, ord=1, dim=1, keepdim=True) 
    #mask = pragmatic_speaker_B.sum(dim=1, keepdim=True)
    mask[mask == 0] = 1  # Replace zeros with ones to avoid division by zero
    pragmatic_speaker_B = pragmatic_speaker_B / mask

    return pragmatic_speaker_B


def pragmatic_listener_A(pragmatic_speaker_B: torch.FloatTensor, prior: torch.FloatTensor, alpha: int, RSA_depth: int, verbose: bool):
    '''Compute the pragmatic listener A in a single round.
    input:
    * pragmatic_speaker_B, a 2D torch tensor of float, the pragmatic speaker B with the meanings as the first dimension and the utterances as the second dimension
    * prior, a 3D torch tensor of float, the prior on the meaning of the agent A (dimension 0), the meaning of the agent B (dimension 1), the world (dimension 2)
    * alpha, a float, the alpha pragmatism parameter of RSA
    * RSA_depth, an int, the depth of the RSA model
    * verbose, a boolean, whether to print the probabilities at each step

    output:
    * pragmatic_listenerA, a 2D torch tensor of float, the pragmatic listener A with the meanings as the first dimension and the utterances as the second dimension
    '''

    # Compute the pragmatic listener: L_A(y|mA, v) ~ SUM_{mB} S_B(v|mB)*P(mA, mB, y)
    pragmatic_listener_A = torch.einsum('B v, A B y -> A y v', pragmatic_speaker_B , prior)

    # Masked division to avoid division by zero errors
    mask = torch.linalg.norm(pragmatic_listener_A, ord=1, dim=1, keepdim=True)
    #mask = pragmatic_listener_A.sum(dim=1, keepdim=True)
    mask[mask == 0] = 1  # Replace zeros with ones to avoid division by zero
    pragmatic_listener_A = pragmatic_listener_A / mask

    return pragmatic_listener_A


def pragmatic_listener_B(pragmatic_speaker_A: torch.FloatTensor, prior: torch.FloatTensor, alpha: int, RSA_depth: int, verbose: bool):
    '''Compute the pragmatic listener A in a single round.
    input:
    * pragmatic_speaker_A, a 2D torch tensor of float, the pragmatic speaker A with the meanings as the first dimension and the utterances as the second dimension
    * prior, a 3D torch tensor of float, the prior on the meaning of the agent A (dimension 0), the meaning of the agent B (dimension 1), the world (dimension 2)
    * alpha, a float, the alpha pragmatism parameter of RSA
    * RSA_depth, an int, the depth of the RSA model
    * verbose, a boolean, whether to print the probabilities at each step

    output:
    * pragmatic_listenerB, a 2D torch tensor of float, the pragmatic listener B with the meanings as the first dimension and the utterances as the second dimension
    '''

    # Compute the pragmatic listener: L_B(y|mB, u) ~ SUM_{mA} S_A(u|mA)*P(mA, mB, y)
    pragmatic_listener_B = torch.einsum('A u, A B y -> B y u', pragmatic_speaker_A, prior)

    # Masked division to avoid division by zero errors
    mask = torch.linalg.norm(pragmatic_listener_B, ord=1, dim=1, keepdim=True)
    #mask = pragmatic_listener_B.sum(dim=1, keepdim=True)
    mask[mask == 0] = 1  # Replace zeros with ones to avoid division by zero
    pragmatic_listener_B = pragmatic_listener_B / mask

    return pragmatic_listener_B