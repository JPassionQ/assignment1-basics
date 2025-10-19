# train a BPE(byte-pair encoding) tokenizer
# import regex as re
# import time
# import multiprocessing
# from multiprocessing import Pool
# from .pretokenization_example import find_chunk_boundaries, pre_tokenize
# import heapq
# import os
# import json

# def train_bpe(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     vocab = {}
#     merges = [] 
#     # vocabulary initialization
#     idx = 0
#     for special_token in special_tokens:
#         vocab[idx] = special_token.encode("utf-8")
#         idx += 1
#     for i in range(256):
#         vocab[idx] = bytes([i])
#         idx += 1

#     # 将特殊token以 bytes 形式放入集合，训练中避免与其相关的 pair
#     special_bytes = {t.encode("utf-8") for t in special_tokens}

#     # pre-tokenization
#     pre_tokens = {}
#     with open(input_path, "rb") as f:
#         num_processes = min(os.cpu_count() or 4, 8) # 自适应并发数
#         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#         args = [] # args pass to pre_tokenize
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk = f.read(end - start).decode("utf-8", errors="ignore")
#             args.append((chunk, special_tokens))

#         with Pool(num_processes) as pool:
#             results = pool.starmap(pre_tokenize, args)
#         for sub_pre_tokens in results:
#             for k, v in sub_pre_tokens.items():
#                 pre_tokens[k] = pre_tokens.get(k, 0) + v
#     # compute bpe merges
#     pair_to_cnt = {}
#     # 建立 pair -> 包含该 pair 的 pre_tokens 的索引
#     pair_to_tokens = {}

#     for pre_token, count in pre_tokens.items():
#         for b1, b2 in zip(pre_token[:-1], pre_token[1:]):
#             # 跳过包含特殊 token 的pair
#             if b1 in special_bytes or b2 in special_bytes:
#                 continue
#             pair_of_bytes = (b1, b2)
#             pair_to_cnt[pair_of_bytes] = pair_to_cnt.get(pair_of_bytes, 0) + count
#             # 记录哪些 token 包含这个 pair
#             if pair_of_bytes not in pair_to_tokens:
#                 pair_to_tokens[pair_of_bytes] = set()
#             pair_to_tokens[pair_of_bytes].add(pre_token)

#     # NOTE: 可以使用数据结构 堆 来尽心优化，但是当字节对的频次相同的时候，需要按降序排列，在堆中不好处理
#     while idx < vocab_size:
#         pair_to_cnt = dict(sorted(
#             pair_to_cnt.items(),
#             key = lambda item:  (item[1], item[0]),
#             reverse=True
#         ))
#         if not pair_to_cnt:
#             break

#         merged_pair = next(iter(pair_to_cnt))
#         top_cnt = pair_to_cnt[merged_pair]
#         # 若最大频次不大于0， 停止
#         if top_cnt <= 0:
#             break
        
#         # 不保留旧键，避免0频键污染/内存膨胀
#         del pair_to_cnt[merged_pair]
#         merges.append(merged_pair)

#         # 优化：只处理包含该 pair 的tokens，而非所有的tokens
#         affected_tokens = pair_to_tokens.get(merged_pair, set()).copy()
#         new_pre_tokens = {}
#         updated_pairs = {} # 记录只需要更新的 pairs

#         for pre_token in affected_tokens:
#             if pre_token not in pre_tokens:
#                 continue
            
#             count = pre_tokens[pre_token]
#             new_pre_token = []
#             i = 0
#             merged = False
            
#             while i < len(pre_token):
#                 if (i + 1 < len(pre_token) and
#                     pre_token[i] == merged_pair[0] and 
#                     pre_token[i + 1] == merged_pair[1]):
#                     # merge, add new pair, rm old pair
#                     merged = True
#                     merged_tok = pre_token[i] + pre_token[i + 1]
#                     new_pre_token.append(merged_tok)
                    
#                     # 更新相邻的 pairs
#                     if i > 0:
#                         old_pair = (pre_token[i - 1], pre_token[i])
#                         updated_pairs[old_pair] = updated_pairs.get(old_pair, 0) - count

#                         new_pair = (pre_token[i-1], merged_tok)
#                         if (new_pair[0] not in special_bytes and 
#                             new_pair[1] not in special_bytes):
#                             updated_pairs[new_pair] = updated_pairs.get(new_pair, 0) + count

#                     if i + 2 < len(pre_token):
#                         old_pair = (pre_token[i + 1], pre_token[i + 2])
#                         updated_pairs[old_pair] = updated_pairs.get(old_pair, 0) - count

#                         new_pair = (merged_tok, pre_token[i + 2])
#                         if (new_pair[0] not in special_bytes and 
#                             new_pair[1] not in special_bytes):
#                             updated_pairs[new_pair] = updated_pairs.get(new_pair, 0) + count
#                     i += 2
#                 else:
#                     new_pre_token.append(pre_token[i])
#                     i += 1
#             # 使用 list 构建后转 tuple，减少内存分配
#             new_pre_token_tuple = tuple(new_pre_token)
#             if merged:
#                 new_pre_tokens[new_pre_token_tuple] = new_pre_tokens.get(new_pre_token_tuple, 0) + count
#                 del pre_tokens[pre_token]

#         # 更新 pre_tokens
#         for token, count in new_pre_tokens.items():
#             pre_tokens[token] = pre_tokens.get(token, 0) + count
        
#         # 优化：批量更新 pair_to_cnt 
#         for pair, delta in updated_pairs.items():
#             old_cnt = pair_to_cnt.get(pair, 0)
#             new_cnt = max(0, old_cnt + delta)

#             if new_cnt == 0:
#                 pair_to_cnt.pop(pair, None)
#             else:
#                 pair_to_cnt[pair] = new_cnt
        
#         # 更新 pair_to_tokens 索引:
#         if merged_pair in pair_to_tokens:
#             del pair_to_tokens[merged_pair]
        
#         for token in new_pre_tokens.keys():
#             for i in range(len(token) - 1):
#                 pair = (token[i], token[i + 1])
#                 if pair[0] not in special_bytes and pair[1] not in special_bytes:
#                     if pair not in pair_to_tokens:
#                         pair_to_tokens[pair] = set()
#                     pair_to_tokens[pair].add(token)
#         vocab[idx] = merged_pair[0] + merged_pair[1]
#         idx += 1
#     return vocab, merges

import regex as re
import time
import multiprocessing
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries, pre_tokenize
import heapq
import os
import json
import psutil  # 需要安装: pip install psutil
import sys

def get_memory_usage():
    """获取当前进程的内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = time.time()
    vocab = {}
    merges = [] 
    
    print(f"开始训练 BPE tokenizer")
    print(f"目标词表大小: {vocab_size}")
    print(f"特殊token: {special_tokens}")
    print("-" * 60)
    
    # vocabulary initialization
    idx = 0
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1

    special_bytes = {t.encode("utf-8") for t in special_tokens}

    # pre-tokenization
    print("步骤 1/3: 预分词处理...")
    pretok_start = time.time()
    
    pre_tokens = {}
    with open(input_path, "rb") as f:
        num_processes = min(os.cpu_count() or 4, 8)
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        args = []
        
        print(f"使用 {num_processes} 个进程处理 {len(boundaries)-1} 个数据块")
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            args.append((chunk, special_tokens))

        with Pool(num_processes) as pool:
            results = pool.starmap(pre_tokenize, args)
        
        for sub_pre_tokens in results:
            for k, v in sub_pre_tokens.items():
                pre_tokens[k] = pre_tokens.get(k, 0) + v
    
    pretok_time = time.time() - pretok_start
    print(f"预分词完成: {len(pre_tokens)} 个唯一token, 耗时 {pretok_time:.2f}s")
    print(f"当前内存占用: {get_memory_usage():.2f} MB")
    print("-" * 60)
    
    # compute bpe merges
    print("步骤 2/3: 计算初始字节对...")
    pair_start = time.time()
    
    pair_to_cnt = {}
    pair_to_tokens = {}

    for pre_token, count in pre_tokens.items():
        for b1, b2 in zip(pre_token[:-1], pre_token[1:]):
            if b1 in special_bytes or b2 in special_bytes:
                continue
            pair_of_bytes = (b1, b2)
            pair_to_cnt[pair_of_bytes] = pair_to_cnt.get(pair_of_bytes, 0) + count
            if pair_of_bytes not in pair_to_tokens:
                pair_to_tokens[pair_of_bytes] = set()
            pair_to_tokens[pair_of_bytes].add(pre_token)
    
    pair_time = time.time() - pair_start
    print(f"初始字节对计算完成: {len(pair_to_cnt)} 个唯一pair, 耗时 {pair_time:.2f}s")
    print(f"当前内存占用: {get_memory_usage():.2f} MB")
    print("-" * 60)
    
    # BPE merging
    print("步骤 3/3: 执行 BPE 合并...")
    merge_start = time.time()
    
    initial_vocab_size = idx
    target_merges = vocab_size - initial_vocab_size
    
    merge_count = 0
    last_print_time = time.time()
    print_interval = 1.0  # 每秒打印一次进度
    
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
        if top_cnt <= 0:
            break
        
        del pair_to_cnt[merged_pair]
        merges.append(merged_pair)

        affected_tokens = pair_to_tokens.get(merged_pair, set()).copy()
        new_pre_tokens = {}
        updated_pairs = {}

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
                    merged = True
                    merged_tok = pre_token[i] + pre_token[i + 1]
                    new_pre_token.append(merged_tok)
                    
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
            
            new_pre_token_tuple = tuple(new_pre_token)
            if merged:
                new_pre_tokens[new_pre_token_tuple] = new_pre_tokens.get(new_pre_token_tuple, 0) + count
                del pre_tokens[pre_token]

        for token, count in new_pre_tokens.items():
            pre_tokens[token] = pre_tokens.get(token, 0) + count
        
        for pair, delta in updated_pairs.items():
            old_cnt = pair_to_cnt.get(pair, 0)
            new_cnt = max(0, old_cnt + delta)

            if new_cnt == 0:
                pair_to_cnt.pop(pair, None)
            else:
                pair_to_cnt[pair] = new_cnt
        
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
        merge_count += 1
        
        # 显示进度
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            progress = merge_count / target_merges * 100
            elapsed = current_time - merge_start
            eta = (elapsed / merge_count * (target_merges - merge_count)) if merge_count > 0 else 0
            
            # 使用 \r 实现同行更新
            sys.stdout.write(
                f"\r合并进度: {merge_count}/{target_merges} "
                f"({progress:.1f}%) | "
                f"已用时间: {elapsed:.1f}s | "
                f"预计剩余: {eta:.1f}s | "
                f"内存: {get_memory_usage():.1f}MB"
            )
            sys.stdout.flush()
            last_print_time = current_time
    
    # 打印最终进度
    sys.stdout.write("\n")
    merge_time = time.time() - merge_start
    print(f"BPE 合并完成: {merge_count} 次合并, 耗时 {merge_time:.2f}s")
    print("-" * 60)
    
    # 总结
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总耗时: {total_time:.2f}s")
    print(f"  - 预分词: {pretok_time:.2f}s ({pretok_time/total_time*100:.1f}%)")
    print(f"  - 计算初始pair: {pair_time:.2f}s ({pair_time/total_time*100:.1f}%)")
    print(f"  - BPE合并: {merge_time:.2f}s ({merge_time/total_time*100:.1f}%)")
    print(f"最终内存占用: {final_memory:.2f} MB")
    print(f"词表大小: {len(vocab)}")
    print(f"合并次数: {len(merges)}")
    print("=" * 60)
    
    return vocab, merges

if __name__=="__main__":
    vocab, merges = train_bpe(
        input_path="/home/jq/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    # 将 vocab 写入 JSON 文件
    # 注意：JSON 的键必须是字符串，值需要转换为可序列化格式
    vocab_json = {str(k): v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    with open("/home/jq/cs336/assignment1-basics/utils/tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    # 将 merges 写入 TXT 文件
    # 每行一个 merge，格式：byte1 byte2
    with open("/home/jq/cs336/assignment1-basics/utils/tinystories_merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            # 将 bytes 转为十六进制字符串或可读格式
            b1_str = pair[0].decode('utf-8', errors='replace')
            b2_str = pair[1].decode('utf-8', errors='replace')
            f.write(f"{b1_str} {b2_str}\n")
    
    print(f"Vocab saved to vocab.json ({len(vocab)} entries)")
    print(f"Merges saved to merges.txt ({len(merges)} merges)")