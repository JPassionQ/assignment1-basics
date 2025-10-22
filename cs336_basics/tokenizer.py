import json
import regex as re
from typing import Iterator, Iterable

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
        self.byte_to_id = {v:k for k, v in vocab.items()}
        if special_tokens is not None:
            self.special_bytes = {t.encode("utf-8") for t in special_tokens}
        else:
            self.special_bytes = None

        # 将特殊字符添加到词汇表的末尾
        idx = len(self.vocab)
        if self.special_bytes is not None:
            for special_byte in self.special_bytes:
                self.vocab[idx] = special_byte
                idx += 1
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [line.strip() for line in f]
        
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer
    
    def _apply_merges(self, pre_token: list[bytes]) -> list[bytes]:
        "对预分词后的token 应用 BPE merges"
        while len(pre_token) > 1:
            # 找到优先级最高（priority 最小） 的相邻字节 pair
            best_pair = None
            best_idx = -1
            best_priority = float('inf')

            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i+1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_idx = i
                        best_pair = pair
            if best_pair is None:
                break

            # 进行 merge
            pre_token[best_idx] = pre_token[best_idx] + pre_token[best_idx + 1]
            pre_token.pop(best_idx + 1)
        return pre_token
    def encode(self, text: str) -> list[int]:
        pre_tokens = []
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 用捕获组保留分隔符
        if self.special_tokens:
            # 按照长度降序排序，让长的 token 优先匹配，可以解决special_token有单个连续出现的情况
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            delimiter = "(" + "|".join(map(re.escape, sorted_special_tokens))+")"
            docs = re.split(delimiter, text)
        else:
            docs = [text]
        for doc in docs:
            # 检查是否是special token
            if self.special_tokens and doc in self.special_tokens:
                # 直接将 special_token 添加到 pre_tokens中
                pre_tokens.append((doc.encode("utf-8"), ))
            else:
                matches = re.finditer(pattern, doc)
                for match in matches:
                    pre_token = ()
                    token = match.group().encode("utf-8")
                    pre_token = tuple(token[i:i+1] for i in range(len(token)))
                    pre_tokens.append(pre_token)
        # 按照 merge 的顺序进行编码
        encoded_text = []
        for pre_token in pre_tokens:
            # 处理 special token
            if self.special_tokens and len(pre_token) == 1 and pre_token[0] in self.special_bytes:
                encoded_text.append(self.byte_to_id[pre_token[0]])
                continue

            # 应用 merges
            tokens = self._apply_merges(list(pre_token))

            # 转换为 vocab id
            encoded_text.extend(self.byte_to_id[token] for token in tokens)
            
        return encoded_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        # 先拼接所有的字节序列，再统一解码
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_sequence.decode(encoding="utf-8", errors="replace")