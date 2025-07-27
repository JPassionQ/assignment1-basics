# train a BPE(byte-pair encoding) tokenizer
import regex as re
import time

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
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "r") as f:
        text = f.read()
            # line = line.rstrip()
            # handle the special_token
            # st_time = time.time()
            # for i in range(len(special_tokens)):
            #     special_tokens[i] = re.escape(special_tokens[i])
            # print(time.time() - st_time)
            # 使用上面这段处理过程会造成处理时间指数上升
        delimiter = "|".join(map(re.escape, special_tokens))
        docs = re.split(delimiter, text)
        for doc in docs:
            matches = re.finditer(pattern, doc)
            for match in matches:
                pre_token = ()
                token = match.group()
                token = token.encode("utf-8")
                # 这里不要用chr来做遍历的单位，因为一个Unicode字符可能对应多个字节
                for i in range(len(token)):
                    pre_token = pre_token + (token[i:i+1], )
                if pre_token in pre_tokens:
                    pre_tokens[pre_token] += 1
                else:
                    pre_tokens[pre_token] = 1
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
    print(vocab)
    print(merges)
    with open("/home/jingqi/CS336_Assignments/assignment1-basics/utils/vocab_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge}\n")
    with open("/home/jingqi/CS336_Assignments/assignment1-basics/utils/vocab1.txt", "w") as f:
        for k, v in vocab.items():
            f.write(f"{v}\n")