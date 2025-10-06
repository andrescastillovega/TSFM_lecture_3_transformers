from io import BufferedReader
import functools
import multiprocessing
from tqdm import tqdm

from tokenizer.EncoderDecoder import encoder, decoder

NPROC = 16

def get_chunk_boundaries(file_path: str, special_token: str = '<|endoftext|>'):
    f: BufferedReader = open(file_path, 'rb')
    content = f.read()
    f.close()    

    chunk_starts = [0]
    start_index = 0
    while True:
        match_index = content.find(special_token.encode('utf-8'), start_index)
        if match_index == -1:
            break
        chunk_starts.append(match_index + len(special_token))
        start_index = match_index + len(special_token)
    chunk_ends = chunk_starts[1:]
    chunk_ends.append(len(content))
    chunk_boundaries = list(zip(chunk_starts, chunk_ends))

    return chunk_boundaries

def encode_chunk(chunk_boundaries: tuple, file_path: str, model: str):
    f = open(file_path, 'rb')
    chunk_start, chunk_end = chunk_boundaries
    
    f.seek(chunk_start)

    tokens = encoder(f.read(chunk_end-chunk_start).decode(), model=model)
    f.close()
    return tokens

def tokenize_text(file_path: str, tokenizer:str):
    encode_f = functools.partial(encode_chunk, file_path=file_path, model=tokenizer)

    chunks_boundaries = get_chunk_boundaries(file_path=file_path)

    chunks_tokens = []
    with multiprocessing.Pool(NPROC) as p:
        for chunk in tqdm(p.imap_unordered(encode_f, chunks_boundaries), colour='green', total=len(chunks_boundaries), desc="Tokenizing"):
            chunks_tokens.extend(chunk)
        p.close()
        p.join()
    
    return chunks_tokens