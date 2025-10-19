import numpy as np

def create_optimizer_state(weights, beta1=0.9, beta2=0.999, 
                          learning_rate=3e-4, epsilon=1e-8, weight_decay=0.01):
    """
    Initialize optimizer state
    
    Args:
        weights: Dictionary of weight arrays
        beta1, beta2: Moment decay rates
        learning_rate: Step size
        epsilon: Numerical stability constant
        weight_decay: L2 penalty coefficient
    
    Returns:
        optimizer_state: Dictionary containing all optimizer parameters and state
    """
    optimizer_state = {
        'lr': learning_rate,
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon,
        'weight_decay': weight_decay,
        't': 0,  # Time step
        'm': {},  # First moments
        'v': {}   # Second moments
    }
    
    # Initialize moment estimates
    for key in weights:
        optimizer_state['m'][key] = np.zeros_like(weights[key])
        optimizer_state['v'][key] = np.zeros_like(weights[key])
    
    return optimizer_state


def adamw_step(weights, grads, optimizer_state):
    """
    Perform one AdamW optimization step
    
    Args:
        weights: Dictionary of current weights
        grads: Dictionary of gradients
        optimizer_state: Dictionary containing optimizer parameters and state
    
    Returns:
        updated_weights: Dictionary of updated weights
        updated_optimizer_state: Dictionary with updated moments and time step
    """
    # Create copies to avoid modifying inputs
    updated_weights = {}
    updated_state = optimizer_state.copy()
    updated_state['m'] = optimizer_state['m'].copy()
    updated_state['v'] = optimizer_state['v'].copy()
    
    # Increment time step
    updated_state['t'] += 1
    t = updated_state['t']
    
    # Extract hyperparameters
    lr = updated_state['lr']
    beta1 = updated_state['beta1']
    beta2 = updated_state['beta2']
    epsilon = updated_state['epsilon']
    weight_decay = updated_state['weight_decay']
    
    # Update each parameter
    for key in weights:
        # Update biased first moment estimate
        updated_state['m'][key] = beta1 * optimizer_state['m'][key] + (1 - beta1) * grads[key]
        
        # Update biased second moment estimate
        updated_state['v'][key] = beta2 * optimizer_state['v'][key] + (1 - beta2) * (grads[key] ** 2)
        
        # Compute bias-corrected moment estimates
        m_hat = updated_state['m'][key] / (1 - beta1 ** t)
        v_hat = updated_state['v'][key] / (1 - beta2 ** t)
        
        # AdamW update with decoupled weight decay
        updated_weights[key] = weights[key] - lr * (
            m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * weights[key]
        )
    
    return updated_weights, updated_state


def clip_gradients(grads, max_norm=1.0):
    """
    Clip gradients by global norm (optional, helps with stability)
    
    Args:
        grads: Dictionary of gradients
        max_norm: Maximum gradient norm
    
    Returns:
        clipped_grads: Dictionary of clipped gradients
    """
    # Compute global norm
    total_norm = 0
    for grad in grads.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # Clip if necessary
    if clip_coef < 1:
        clipped_grads = {key: grad * clip_coef for key, grad in grads.items()}
    else:
        clipped_grads = grads
    
    return clipped_grads