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

def dataloader(chunks: list, B: int, S: int, stride: int):
    inputs = []
    targets = []
    for tokens, lenght in chunks:
        for idx in range(0, lenght - S - 1, stride):
            inputs.append(tokens[idx:idx + S])
            targets.append(tokens[idx+1:idx + S + 1])

    for i in range(0, len(inputs), B):
        batch_inputs = inputs[i:i + B]
        batch_targets = targets[i:i + B]
        
        if len(batch_inputs) == B:
            yield batch_inputs, batch_targets

    return batch_inputs, batch_targets

def initialize_weights(cfg: Dict):
    weights = {
        'W_TE': np.random.rand(cfg['vocab_size'], cfg['model_dim']),
        'W_PE': np.random.rand(cfg['seq_length'], cfg['model_dim']),
        'W_Q': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'W_K': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'W_V': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'W_O': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['model_dim']),
        'GAMMA_1': np.random.rand(cfg['n_layers'], cfg['seq_length'], 1),
        'BETA_1': np.random.rand(cfg['n_layers'], cfg['seq_length'], 1),
        'W_FFN_E': np.random.rand(cfg['n_layers'], cfg['model_dim'], cfg['ffn_exp_factor'] * cfg['model_dim']),
        'W_FFN_C': np.random.rand(cfg['n_layers'], cfg['ffn_exp_factor'] * cfg['model_dim'], cfg['model_dim']),
        'GAMMA_2': np.random.rand(cfg['n_layers'], cfg['seq_length'], 1),
        'BETA_2': np.random.rand(cfg['n_layers'], cfg['seq_length'], 1),
        'W_LOGITS': np.random.rand(cfg['model_dim'], cfg['vocab_size'])
    }

    return weights

def forward(x: np.ndarray, cfg: Dict, weights: Dict):
    for layer in range(cfg['n_layers']):
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
        print(f"\rLayer: {layer} Done!")
    return x @ weights['W_LOGITS']

def loss(logits, targets):
    assert logits.shape == targets.shape

    # probas_inputs = tf.softmax(logits)
    # probas_targets = tf.softmax(targets)

    pass




def main():
    print("Hello from lecture-3-code!")
    # enc_text = encoder("Hello World!", model='tinystories-2560')
    # dec_text = decoder(enc_text, model='tinystories-2560')

    # enc_text = np.pad(np.array(enc_text), (0, 256 - len(enc_text)), constant_values=0)

    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        assert cfg['model_dim'] % cfg['num_heads'] == 0, 'model_dim must be divisible by num_heads'

    tokens = tokenize_text('tokenizer/data/toy_data.txt', tokenizer='tinystories-2560')
    print(len(tokens))

    # data = dataloader(tokens, cfg['batch_size'], cfg['seq_length'], 100)

    # data = [ batch for batch in data ]

    # print(tokens[0])
    # print("\n\n")
    # print(decoder(tokens[0][0], model='tinystories-2560'))
    # print(decoder([116, 124], model='tinystories-2560'))
    # print(decoder(data[0][0][0], model='tinystories-2560'))
    # print("\n\n")
    # print(decoder(data[0][0][1], model='tinystories-2560'))


    # # print(enc_text, dec_text)

    

    # data = np.random.randint(1, cfg['vocab_size'], (2,cfg['batch_size'],cfg['seq_length']))

    # weights = initialize_weights(cfg)

    # input_emb = weights['W_TE'][enc_text] + weights['W_PE'][np.arange(cfg['seq_length'])] # S x D
    # print(f"data: {data.shape}")
    # print(f"input emb: {input_emb.shape}")

    # blocks_output = forward(input_emb, cfg, weights)

    # logits = tf.softmax(blocks_output)
    # print(logits.shape)
    # print(np.argmax(logits[0,:]))
    # print(np.argmax(logits, axis=-1, keepdims=True)[0])
    # print(decoder(list(np.argmax(logits, axis=-1, keepdims=True)[0]), model='tinystories-2560'))

    # # print(output.shape)

if __name__ == "__main__":
    main()
