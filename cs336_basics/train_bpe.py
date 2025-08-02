# train a BPE(byte-pair encoding) tokenizer
import regex as re
import time
import multiprocessing
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries, pre_tokenize

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = [] 
    # vocabulary initialization
    idx = 0
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1
    # pre-tokenization
    pre_tokens = {}
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        args = [] # args pass to pre_tokenize
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            args.append((chunk, special_tokens))

        with Pool(num_processes) as pool:
            results = pool.starmap(pre_tokenize, args)
        for sub_pre_tokens in results:
            for k, v in sub_pre_tokens.items():
                if k in pre_tokens:
                    pre_tokens[k] += v
                else:
                    pre_tokens[k] = v
    # compute bpe merges
    while idx < vocab_size:
        pair_to_cnt = {}
        for pre_token in pre_tokens.keys():
            for i, _ in enumerate(pre_token):
                if i + 1 > len(pre_token) - 1:
                    break
                pair_of_bytes = (pre_token[i], pre_token[i + 1])
                if pair_of_bytes in pair_to_cnt:
                    pair_to_cnt[pair_of_bytes] += pre_tokens[pre_token]
                else:
                    pair_to_cnt[pair_of_bytes] = pre_tokens[pre_token]
        pair_to_cnt = dict(sorted(
            pair_to_cnt.items(),
            key = lambda item:  (item[1], item[0]),
            reverse=True
        ))
        if not pair_to_cnt:
            break
        merged_pair = next(iter(pair_to_cnt))
        merges.append(merged_pair)
        changed_pre_tokens = {}
        # merge the pair in the pre_tokens
        for pre_token in pre_tokens.keys():
            new_pre_token = ()
            i = 0
            while i < len(pre_token):
                if i + 1 <= len(pre_token) - 1 and  pre_token[i] == merged_pair[0] and pre_token[i + 1] == merged_pair[1]:
                    new_pre_token = new_pre_token + (pre_token[i] + pre_token[i + 1], )
                    i += 2
                else:
                    new_pre_token = new_pre_token + (pre_token[i], )
                    i += 1
            changed_pre_tokens[new_pre_token] = pre_tokens[pre_token]
        pre_tokens = changed_pre_tokens
        vocab[idx] = merged_pair[0] + merged_pair[1]
        idx += 1
    return vocab, merges

if __name__=="__main__":
    vocab, merges = train_bpe(
        input_path="/home/jingqi/CS336_Assignments/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"]
    )
    # print(len(vocab))
    # print(len(merges))