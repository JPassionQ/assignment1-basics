# train a BPE(byte-pair encoding) tokenizer
import regex as re
import time
import multiprocessing
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries, pre_tokenize
import heapq

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

    # 将特殊token以 bytes 形式放入集合，训练中避免与其相关的 pair
    special_bytes = {t.encode("utf-8") for t in special_tokens}

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
    pair_to_cnt = {} # 统计词频
    for pre_token in pre_tokens.keys():
        for b1, b2 in zip(pre_token[:-1], pre_token[1:]):
            # 跳过包含特殊 token 的pair
            if b1 in special_bytes or b2 in special_bytes:
                continue
            pair_of_bytes = (b1, b2)
            if pair_of_bytes in pair_to_cnt:
                pair_to_cnt[pair_of_bytes] += pre_tokens[pre_token]
            else:
                pair_to_cnt[pair_of_bytes] = pre_tokens[pre_token]
    while idx < vocab_size:
        # 选择最大频 pair， 仍用排序但加入早停
        pair_to_cnt = dict(sorted(
            pair_to_cnt.items(),
            key = lambda item:  (item[1], item[0]),
            reverse=True
        ))
        if not pair_to_cnt:
            break

        merged_pair = next(iter(pair_to_cnt))
        top_cnt = pair_to_cnt[merged_pair]
        # 若最大频次不大于0， 停止
        if top_cnt <= 0:
            break
        
        # 不保留旧键，避免0频键污染/内存膨胀
        del pair_to_cnt[merged_pair]
        merges.append(merged_pair)

        changed_pre_tokens = {}
        # merge the pair in the pre_tokens
        for pre_token in pre_tokens.keys():
            new_pre_token = ()
            i = 0
            while i < len(pre_token):
                if i + 1 < len(pre_token) and pre_token[i] == merged_pair[0] and pre_token[i + 1] == merged_pair[1]:
                    # merge, add new pair, rm old pair
                    merged_tok = pre_token[i] + pre_token[i + 1]
                    new_pre_token = new_pre_token + (merged_tok, )
                    
                    if i - 1 >= 0:
                        old_pair = (pre_token[i - 1], pre_token[i])
                        if old_pair in pair_to_cnt:
                            pair_to_cnt[old_pair] = max(0, pair_to_cnt.get(old_pair, 0) - pre_tokens[pre_token])

                        before_pair = (pre_token[i - 1], merged_tok)
                        if before_pair[0] not in special_bytes and before_pair[1] not in special_bytes:
                            pair_to_cnt[before_pair] = pair_to_cnt.get(before_pair, 0) + pre_tokens[pre_token]
                    if i + 2 < len(pre_token):
                        old_pair = (pre_token[i + 1], pre_token[i + 2])
                        if old_pair in pair_to_cnt:
                            pair_to_cnt[old_pair] = max(0, pair_to_cnt.get(old_pair, 0) - pre_tokens[pre_token])

                        after_pair = (merged_tok, pre_token[i+2])
                        if after_pair[0] not in special_bytes and after_pair[1] not in special_bytes:
                            pair_to_cnt[after_pair] = pair_to_cnt.get(after_pair, 0) + pre_tokens[pre_token]
                    i += 2
                else:
                    new_pre_token = new_pre_token + (pre_token[i], )
                    i += 1
            changed_pre_tokens[new_pre_token] = changed_pre_tokens.get(new_pre_token, 0) + pre_tokens[pre_token]

        pre_tokens = changed_pre_tokens
        vocab[idx] = merged_pair[0] + merged_pair[1]
        idx += 1
    return vocab, merges

if __name__=="__main__":
    vocab, merges = train_bpe(
        input_path="/home/jq/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"]
    )
    # print(len(vocab))
    # print(len(merges))