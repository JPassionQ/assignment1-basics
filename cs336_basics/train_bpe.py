# train a BPE(byte-pair encoding) tokenizer
import regex as re
import time
import multiprocessing
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries, pre_tokenize
import heapq
import os
import json

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
        num_processes = min(os.cpu_count() or 4, 8) # 自适应并发数
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
                pre_tokens[k] = pre_tokens.get(k, 0) + v
    # compute bpe merges
    pair_to_cnt = {}
    # 建立 pair -> 包含该 pair 的 pre_tokens 的索引
    pair_to_tokens = {}

    for pre_token, count in pre_tokens.items():
        for b1, b2 in zip(pre_token[:-1], pre_token[1:]):
            # 跳过包含特殊 token 的pair
            if b1 in special_bytes or b2 in special_bytes:
                continue
            pair_of_bytes = (b1, b2)
            pair_to_cnt[pair_of_bytes] = pair_to_cnt.get(pair_of_bytes, 0) + count
            # 记录哪些 token 包含这个 pair
            if pair_of_bytes not in pair_to_tokens:
                pair_to_tokens[pair_of_bytes] = set()
            pair_to_tokens[pair_of_bytes].add(pre_token)

    # NOTE: 可以使用数据结构 堆 来尽心优化，但是当字节对的频次相同的时候，需要按降序排列，在堆中不好处理
    while idx < vocab_size:
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

        # 优化：只处理包含该 pair 的tokens，而非所有的tokens
        affected_tokens = pair_to_tokens.get(merged_pair, set()).copy()
        new_pre_tokens = {}
        updated_pairs = {} # 记录只需要更新的 pairs

        for pre_token in affected_tokens:
            if pre_token not in pre_tokens:
                continue
            
            count = pre_tokens[pre_token]
            new_pre_token = []
            i = 0
            merged = False
            
            while i < len(pre_token):
                if (i + 1 < len(pre_token) and
                    pre_token[i] == merged_pair[0] and 
                    pre_token[i + 1] == merged_pair[1]):
                    # merge, add new pair, rm old pair
                    merged = True
                    merged_tok = pre_token[i] + pre_token[i + 1]
                    new_pre_token.append(merged_tok)
                    
                    # 更新相邻的 pairs
                    if i > 0:
                        old_pair = (pre_token[i - 1], pre_token[i])
                        updated_pairs[old_pair] = updated_pairs.get(old_pair, 0) - count

                        new_pair = (pre_token[i-1], merged_tok)
                        if (new_pair[0] not in special_bytes and 
                            new_pair[1] not in special_bytes):
                            updated_pairs[new_pair] = updated_pairs.get(new_pair, 0) + count

                    if i + 2 < len(pre_token):
                        old_pair = (pre_token[i + 1], pre_token[i + 2])
                        updated_pairs[old_pair] = updated_pairs.get(old_pair, 0) - count

                        new_pair = (merged_tok, pre_token[i + 2])
                        if (new_pair[0] not in special_bytes and 
                            new_pair[1] not in special_bytes):
                            updated_pairs[new_pair] = updated_pairs.get(new_pair, 0) + count
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    i += 1
            # 使用 list 构建后转 tuple，减少内存分配
            new_pre_token_tuple = tuple(new_pre_token)
            if merged:
                new_pre_tokens[new_pre_token_tuple] = new_pre_tokens.get(new_pre_token_tuple, 0) + count
                del pre_tokens[pre_token]

        # 更新 pre_tokens
        for token, count in new_pre_tokens.items():
            pre_tokens[token] = pre_tokens.get(token, 0) + count
        
        # 优化：批量更新 pair_to_cnt 
        for pair, delta in updated_pairs.items():
            old_cnt = pair_to_cnt.get(pair, 0)
            new_cnt = max(0, old_cnt + delta)

            if new_cnt == 0:
                pair_to_cnt.pop(pair, None)
            else:
                pair_to_cnt[pair] = new_cnt
        
        # 更新 pair_to_tokens 索引:
        if merged_pair in pair_to_tokens:
            del pair_to_tokens[merged_pair]
        
        for token in new_pre_tokens.keys():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                if pair[0] not in special_bytes and pair[1] not in special_bytes:
                    if pair not in pair_to_tokens:
                        pair_to_tokens[pair] = set()
                    pair_to_tokens[pair].add(token)
        vocab[idx] = merged_pair[0] + merged_pair[1]
        idx += 1
    return vocab, merges

if __name__=="__main__":
    vocab, merges = train_bpe(
        input_path="/home/ubuntu/cs336_assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    
    # 【修复 1】使用 repr() 完美保留 bytes 的原始形态，避免乱码或替换
    vocab_json = {str(k): repr(v) for k, v in vocab.items()}
    with open("/home/ubuntu/cs336_assignments/assignment1-basics/utils/tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    with open("/home/ubuntu/cs336_assignments/assignment1-basics/utils/tinystories_merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            # 【修复 2】使用 tab (\t) 作为分隔符，防止和 token 自身的空格冲突
            f.write(f"{repr(pair[0])}\t{repr(pair[1])}\n")
    
    print(f"Vocab saved to vocab.json ({len(vocab)} entries)")
    print(f"Merges saved to merges.txt ({len(merges)} merges)")