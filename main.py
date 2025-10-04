import numpy as np
import pickle

from tokenizer.EncoderDecoder import encoder, decoder
from transformer import (
    input_embedding,
    matmul_backward, 
    input_embedding_backward, 
    qkv_projection_backward, 
    softmax_backward,
    multi_head_attention_backward,
    qkv_projection
)

class Transformer():
    def __init__(self, b, s, d):
        np.random.seed(42)

        self.b = b # b: Batch size
        self.s = s # s: Sequence lenght
        self.d = d # d: Embedding dimension

        self.W_e = np.random.randn(self.b, self.s, self.d)


def forward(x: np.ndarray):
    batch_size, seq_len, model_dim = (1, 512, 384)

    input_embedding 

def main():
    print("Hello from lecture-3-code!")
    enc_text = encoder("Hello World!", model='tinystories')
    dec_text = decoder(enc_text, model='tinystories')

    print(enc_text, dec_text)






if __name__ == "__main__":
    main()
