# train a BPE(byte-pair encoding) tokenizer

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = [] 
    # vocabulary initialization
    for i in range(256):
        vocab[i] = bytes(i)
    idx = 256
    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1
    # pre-tokenization