import os
from typing import Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_bpe(merges: Dict, vocab: Dict, model_name: str):
    os.makedirs(f'{SCRIPT_DIR}/models/', exist_ok=True)
    
    with open(f'{SCRIPT_DIR}/models/{model_name}.merges', 'w+') as f:
        for merge_rule, merge_rule_idx in sorted(merges.items(), key=lambda x: x[1]):
            f.write(f'{merge_rule_idx} {merge_rule[0]} {merge_rule[1]}\n')

    with open(f'{SCRIPT_DIR}/models/{model_name}.vocab', 'w+') as f:
        for token, id_vocab in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{id_vocab} {' '.join([str(byte) for byte in token])}\n")


def load_bpe(model_name: str):
    merges = {}
    vocab = {}

    with open(f'{SCRIPT_DIR}/models/{model_name}.merges', 'r') as f:
        for line in f.readlines():
            merge_id, token_a, token_b = [int(x) for x in line.replace("\n","").split(" ")]
            merges[(token_a, token_b)] = merge_id

    with open(f'{SCRIPT_DIR}/models/{model_name}.vocab', 'r') as f:
        for line in f.readlines():
            line_split = [ int(ele) for ele in line.replace("\n","").split(" ") ]
            vocab_id = line_split[0]
            token = tuple(line_split[1:])
            vocab[token] = vocab_id

    return merges, vocab

def render_token(token: int, vocab: Dict):
    inv_vocab = { symbol_id: symbol for symbol, symbol_id in vocab.items() }

    results = b''
    for symbol in inv_vocab[token]:
        if len(inv_vocab[symbol]) == 1:
            results = b''.join([results, bytes([symbol])])
        else:
            results = b''.join([results, render_token(inv_vocab[symbol][0], vocab), render_token(inv_vocab[symbol][1], vocab)])

    return results