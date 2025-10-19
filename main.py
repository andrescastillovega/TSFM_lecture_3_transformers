import numpy as np
import pickle
import yaml
from typing import Dict
from io import BufferedReader
import functools
import multiprocessing
from tqdm import tqdm

from tokenizer.EncoderDecoder import encoder, decoder
from tokenizer.tokenizer import tokenize_text
import transformer as tf

import tiktoken

from optimizer import create_optimizer_state, adamw_step, clip_gradients

def generate_text(prompt: str, weights: Dict, cfg: Dict, max_new_tokens: int = 50, temperature: float = 0.8):
    """
    Generate text by sampling from the model
    
    Args:
        prompt: Starting text prompt
        weights: Model weights
        cfg: Model configuration
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
    """
    # Tokenize the prompt
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = list(tokens)  # Convert to list for easier manipulation
    
    print(f"\nGenerating from prompt: '{prompt}'")
    print(f"Initial tokens: {tokens}")
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Prepare input (take last seq_length tokens if longer)
        if len(tokens) > cfg['seq_length']:
            input_tokens = tokens[-cfg['seq_length']:]
        else:
            input_tokens = tokens
        
        # Pad if necessary
        input_tokens = np.array(input_tokens)
        if len(input_tokens) < cfg['seq_length']:
            input_tokens = np.pad(input_tokens, (0, cfg['seq_length'] - len(input_tokens)), constant_values=0)
        
        # Get embeddings
        input_emb = weights['W_TE'][input_tokens] + weights['W_PE'][np.arange(cfg['seq_length'])]
        input_emb = input_emb[np.newaxis, :, :]  # Add batch dimension (1, S, D)
        
        # Forward pass
        logits, _ = forward(input_emb, cfg, weights)  # (1, S, V)
        
        # Get logits for the last position
        next_token_logits = logits[0, len(tokens) - 1 if len(tokens) <= cfg['seq_length'] else -1, :]  # (V,)
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Sample from the distribution
        probs = tf.softmax(next_token_logits.reshape(1, -1))[0]  # (V,)
        next_token = np.random.choice(len(probs), p=probs)
        
        # Add to sequence
        tokens.append(next_token)
        
        # Stop if we generate end of text token (if applicable)
        # endoftext_token = encoder('<|endoftext|>', model='tinystories-2560')[0]
        # if next_token == endoftext_token:
        #     break
    
    # Decode the generated tokens
    generated_text = enc.decode(tokens)
    
    return generated_text

def dataloader(tokens: list, B: int, S: int):
    inputs = []
    targets = []
    print(type(tokens))
    # print(tokens)
    for idx in range(0, len(tokens), B * S):
        if (idx + B * S) < len(tokens):
            inputs.append(np.array(tokens[idx : idx + B * S]).reshape(B, S))
            targets.append(np.array(tokens[idx + 1 : idx + B * S + 1]).reshape(B, S))
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

def initialize_weights(cfg: Dict):
    weights = {
        'W_TE': np.random.rand(cfg['vocab_size'], cfg['model_dim']),
        'W_PE': np.random.rand(cfg['seq_length'], cfg['model_dim']),
        'W_Q': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'W_K': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'W_V': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'W_O': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'GAMMA_1': np.random.rand(cfg['n_layers'], cfg['model_dim'], ),
        'BETA_1': np.random.rand(cfg['n_layers'], cfg['model_dim'], ),
        'W_FFN_E': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['ffn_exp_factor'] * cfg['model_dim']),
        'W_FFN_C': np.random.rand(cfg['n_layers'], cfg['ffn_exp_factor'] * cfg['model_dim'], cfg['model_dim']),
        'GAMMA_2': np.random.rand(cfg['n_layers'], cfg['model_dim'], ),
        'BETA_2': np.random.rand(cfg['n_layers'], cfg['model_dim'], ),
        'W_LOGITS': np.random.rand(cfg['model_dim'], cfg['vocab_size'])
    }

    return weights

def forward(x: np.ndarray, cfg: Dict, weights: Dict):
    cache = {'layer_inputs': []}

    for layer in range(cfg['n_layers']):
        cache['layer_inputs'].append(x.copy())
        x = tf.block_forward(x,
            W_Q= weights['W_Q'][layer],
            W_K= weights['W_K'][layer],
            W_V= weights['W_V'][layer],
            W_O= weights['W_O'][layer],
            W_FF_expand= weights['W_FFN_E'][layer],
            W_FF_contract= weights['W_FFN_C'][layer],
            gamma1= weights['GAMMA_1'][layer],
            beta1= weights['BETA_1'][layer],
            gamma2= weights['GAMMA_2'][layer],
            beta2= weights['BETA_2'][layer])
    
    cache['pre_logits'] = x
    return x @ weights['W_LOGITS'], cache

def backward(dlogits, cache, cfg, weights, batch_input):
    B, S = batch_input.shape
    
    # Initialize gradients dictionary
    grads = {
        'W_Q': [],
        'W_K': [],
        'W_V': [],
        'W_O': [],
        'W_FFN_E': [],
        'W_FFN_C': [],
        'GAMMA_1': [],
        'BETA_1': [],
        'GAMMA_2': [],
        'BETA_2': [],
    }
    
    # ========================================================================
    # Step 1: Backprop through final linear layer (logits projection)
    # ========================================================================
    # dlogits has shape (B, S, V)
    # W_LOGITS has shape (D, V)
    # pre_logits has shape (B, S, D)
    
    pre_logits = cache['pre_logits']  # (B, S, D)
    
    # Gradient w.r.t. W_LOGITS: pre_logits^T @ dlogits
    # Need to reshape: (B*S, D) @ (B*S, V) -> (D, V)
    pre_logits_2d = pre_logits.reshape(-1, cfg['model_dim'])  # (B*S, D)
    dlogits_2d = dlogits.reshape(-1, cfg['vocab_size'])       # (B*S, V)
    grads['W_LOGITS'] = pre_logits_2d.T @ dlogits_2d          # (D, V)
    
    # Gradient w.r.t. pre_logits: dlogits @ W_LOGITS^T
    dx = dlogits_2d @ weights['W_LOGITS'].T                   # (B*S, D)
    dx = dx.reshape(B, S, cfg['model_dim'])                   # (B, S, D)
    
    # ========================================================================
    # Step 2: Backprop through transformer blocks (in reverse order)
    # ========================================================================
    for layer in reversed(range(cfg['n_layers'])):
        # Get input to this block from cache
        block_input = cache['layer_inputs'][layer]  # (B, S, D)
        
        # Backprop through this block
        (dX, dW_Q, dW_K, dW_V, dW_O, 
         dW_FFN_E, dW_FFN_C, 
         dG1, dB1, dG2, dB2) = tf.block_backward(
            x=block_input,
            W_Q=weights['W_Q'][layer],
            W_K=weights['W_K'][layer],
            W_V=weights['W_V'][layer],
            W_O=weights['W_O'][layer],
            W_FF_expand=weights['W_FFN_E'][layer],
            W_FF_contract=weights['W_FFN_C'][layer],
            gamma1=weights['GAMMA_1'][layer],
            beta1=weights['BETA_1'][layer],
            gamma2=weights['GAMMA_2'][layer],
            beta2=weights['BETA_2'][layer],
            dOut=dx
        )
        
        # Store gradients (in reverse order, will reverse again later)
        grads['W_Q'].append(dW_Q)
        grads['W_K'].append(dW_K)
        grads['W_V'].append(dW_V)
        grads['W_O'].append(dW_O)
        grads['W_FFN_E'].append(dW_FFN_E)
        grads['W_FFN_C'].append(dW_FFN_C)
        grads['GAMMA_1'].append(dG1)
        grads['BETA_1'].append(dB1)
        grads['GAMMA_2'].append(dG2)
        grads['BETA_2'].append(dB2)
        
        # Update dx for next layer
        dx = dX
    
    # ========================================================================
    # Step 3: Reverse gradient lists to match forward order
    # ========================================================================
    for key in ['W_Q', 'W_K', 'W_V', 'W_O', 'W_FFN_E', 'W_FFN_C',
                'GAMMA_1', 'BETA_1', 'GAMMA_2', 'BETA_2']:
        grads[key] = np.array(list(reversed(grads[key])))
    
    # ========================================================================
    # Step 4: Backprop through embeddings
    # ========================================================================
    # dx now contains gradients w.r.t. the initial embeddings (after W_TE + W_PE)
    # Shape: (B, S, D)
    
    # Gradient w.r.t. W_TE (token embeddings)
    # W_TE has shape (V, D)
    # We need to accumulate gradients for the tokens that were actually used
    grads['W_TE'] = np.zeros_like(weights['W_TE'])  # (V, D)
    
    # For each position, add gradient to the corresponding token's embedding
    for b in range(B):
        for s in range(S):
            token_idx = batch_input[b, s]
            grads['W_TE'][token_idx] += dx[b, s]
    
    # Alternative (faster) using np.add.at:
    # batch_input_flat = batch_input.reshape(-1)  # (B*S,)
    # dx_flat = dx.reshape(-1, cfg['model_dim'])  # (B*S, D)
    # np.add.at(grads['W_TE'], batch_input_flat, dx_flat)
    
    # Gradient w.r.t. W_PE (positional embeddings)
    # W_PE has shape (S, D)
    # Sum gradients across batch dimension
    grads['W_PE'] = np.sum(dx, axis=0)  # (S, D)
    
    return grads
        

def cross_entropy(logits, targets):
    probas = tf.softmax(logits)
    probas_flatten = probas.reshape(probas.shape[0] * probas.shape[1], probas.shape[2])
    targets_flatten = targets.reshape(targets.shape[0] * targets.shape[1])
    target_probas = probas_flatten[np.arange(probas_flatten.shape[0]), targets_flatten]
    log_probas = np.log(target_probas + 1e-10)  # Add small epsilon for stability
    neg_avg_log_probas = -np.mean(log_probas, axis=-1)

    return neg_avg_log_probas

def cross_entropy_backward(logits, targets):
    B, S, V = logits.shape  # Batch, Sequence, Vocab
    
    # Step 1: Compute softmax probabilities
    probas = tf.softmax(logits)  # (B, S, V)
    # probas[i,j,k] = probability that token at position j in batch i is word k
    
    # Step 2: Flatten for easier indexing
    probas_flat = probas.reshape(B * S, V)      # (B*S, V)
    targets_flat = targets.reshape(B * S)        # (B*S,)
    
    # Step 3: Subtract 1 from the correct class probability
    # This converts softmax probabilities into gradients
    probas_flat[np.arange(B * S), targets_flat] -= 1
    
    # Step 4: Reshape back and normalize
    dlogits = probas_flat.reshape(B, S, V) / (B * S)
    # Divide by (B*S) because we're averaging loss over all tokens
    
    return dlogits

def main():
    print("Hello from lecture-3-code!")

    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        assert cfg['model_dim'] % cfg['num_heads'] == 0, 'model_dim must be divisible by num_heads'

    weights = initialize_weights(cfg)

    # Initialize AdamW optimizer
    optimizer_state = create_optimizer_state(
        weights,
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.1
    )

    # tokens = tokenize_text('tokenizer/data/TinyStoriesV2-GPT4-train.txt', tokenizer='tiny-20k')
    enc = tiktoken.get_encoding("gpt2")
    file = open('tokenizer/data/toy_data3.txt', 'r')
    content = file.read()
    tokens = enc.encode(content, allowed_special={"<|endoftext|>"})
    


    print(len(tokens))
    # print(encoder('<|endoftext|>', model='tiny-20k'), decoder([256], model='tiny-20k'))
    # print(decoder([2560], model='tiny-20k'))

    inputs, targets = dataloader(tokens, cfg['batch_size'], cfg['seq_length'])

    # Test prompts to track progress
    test_prompts = [
        "Hello, I'm ",
        "Once upon a time",
        "The cat"
    ]

    for epoch in range(30):
        epoch_loss = 0
        for batch_idx, (batch_input, batch_target) in enumerate(list(zip(inputs, targets))):
            print(batch_input.shape)
            try:
                input_emb = weights['W_TE'][batch_input] + weights['W_PE'][np.arange(cfg['seq_length'])] # S x D
                logits, forward_cache = forward(input_emb, cfg, weights)
                loss =  cross_entropy(logits, batch_target)

                dlogits = cross_entropy_backward(logits, batch_target)

                # ============================================================
                # Backward pass
                # ============================================================
                dlogits = cross_entropy_backward(logits, batch_target)
                grads = backward(dlogits, forward_cache, cfg, weights, batch_input)
                
                # ============================================================
                # Update weights with AdamW
                # ============================================================
                weights, optimizer_state = adamw_step(weights, grads, optimizer_state)

                epoch_loss += loss

                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(inputs)}, Loss: {loss:.4f}")
            except:
                print(batch_input.shape)
                print(batch_input.max(axis=-1))
                raise
            
        # Print epoch summary
        avg_loss = epoch_loss / len(inputs)
        print(f"{'='*60}")
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        print(f"{'='*60}")

         # ============================================================
        # Generate text samples to see progress
        # ============================================================
        print(f"\n{'='*60}")
        print(f"Text Generation Samples - Epoch {epoch}")
        print(f"{'='*60}")
        
        for prompt in test_prompts:
            generated = generate_text(
                prompt=prompt,
                weights=weights,
                cfg=cfg,
                max_new_tokens=30,
                temperature=0.8
            )
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: '{generated}'")
            print("-" * 40)


    # tokens = encoder("Hello, I'm ", model='tiny-20k')
    # tokens = np.pad(tokens, (0, cfg['seq_length']-len(tokens)))
    # print(tokens)
    # print(tokens.shape)
    # input_emb = weights['W_TE'][tokens] + weights['W_PE'][np.arange(cfg['seq_length'])]
    # logits, forward_cache = forward(input_emb, cfg, weights)

    # print(logits.shape)


if __name__ == "__main__":
    main()
