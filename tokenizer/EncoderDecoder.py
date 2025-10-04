# Implement the encoder and decoder for the BPE tokenizer here
import os
import sys
from typing import Tuple, Dict

# sys.path.append(os.path.abspath('./utils.py'))

from .utils import load_bpe, render_token

def encoder(text: str, model: str):
    merges, _ = load_bpe(model_name=model)
    text_bytes = list(text.encode('utf-8'))

    tokens = text_bytes
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
