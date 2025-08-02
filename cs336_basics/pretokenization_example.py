import os
from typing import BinaryIO
import multiprocessing
from multiprocessing import Pool
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(
    chunk: str,
    special_tokens: list[str],
) -> dict[tuple, int]:
    pre_tokens = {}
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    delimiter = "|".join(map(re.escape, special_tokens))
    docs = re.split(delimiter, chunk)
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
    return pre_tokens

if __name__=="__main__":
    ## Usage
    with open("/home/jingqi/CS336_Assignments/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", "rb") as f:
        num_processes = 10
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        args = []
        special_tokens = ["<|endoftext|>"]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            args.append((chunk, special_tokens))
            # Run pre-tokenization on your chunk and store the counts for each pre-token
        with Pool(num_processes) as pool:
            results = pool.starmap(pre_tokenize, args)
        pre_tokens = {}
        for res in results:
            for k, v in res.items():
                if k in pre_tokens:
                    pre_tokens[k] += v
                else:
                    pre_tokens[k] = v
        with open("/home/jingqi/CS336_Assignments/assignment1-basics/utils/pre_tokens_parallel.txt", "w") as f:
            for k, v in pre_tokens.items():
                f.write(f"{k}, {v}\n")
