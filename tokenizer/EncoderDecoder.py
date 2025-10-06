# Implement the encoder and decoder for the BPE tokenizer here
import os
import sys
from typing import Tuple, Dict
import regex as re

# sys.path.append(os.path.abspath('./utils.py'))

from .utils import load_bpe, render_token

def encoder(text: str, model: str, special_tokens: list = ['<|endoftext|>']):
    merges, vocab = load_bpe(model_name=model)

    pattern = '(' + '|'.join(re.escape(token) for token in special_tokens) + ')'
    chunks = re.split(pattern, text)

    tokens = []
    for chunk in chunks:
        if chunk in special_tokens:
            tokens.extend([vocab[tuple(chunk.encode('utf-8'))]])
        else:
            tokens.extend(list(chunk.encode('utf-8')))

    for merge in merges:
        updated_tokens = []
        i = 0
        while i < len(tokens):
            if i == len(tokens) - 1:
                updated_tokens.append(tokens[i])
                i += 1
            elif merge == (tokens[i], tokens[i + 1]):
                updated_tokens.append(merges[merge])
                i += 2
            else:
                updated_tokens.append(tokens[i])
                i += 1         
        tokens = updated_tokens
    return tokens

def decoder(tokens: list, model: str, decode=True):
    _, vocab = load_bpe(model_name=model)

    text = [ render_token(token, vocab) for token in tokens ]
    if decode:
        return b''.join(text).decode()
    else:
        return b''.join(text)
