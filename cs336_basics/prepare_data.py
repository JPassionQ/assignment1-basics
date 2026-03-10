import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from .tokenizer import Tokenizer # 你的 tokenizer 路径

def process_chunk(chunk):
    # 初始化 tokenizer，因为多进程中无法序列化某些对象
    vocab_file = "/home/ubuntu/cs336_assignments/assignment1-basics/utils/tinystories_vocab.json"
    merges_file = "/home/ubuntu/cs336_assignments/assignment1-basics/utils/tinystories_merges.txt"
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=['<|endoftext|>'])
    return tokenizer.encode(chunk)

if __name__ == '__main__':
    corpus_file = "/home/ubuntu/cs336_assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    
    print("读取文件内容...")
    # 按行读取，或者按文章分割（TinyStories 通常用 <|endoftext|> 分割）
    with open(corpus_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 将小行合并成稍大一点的块，提高多进程效率
    chunk_size = 1000
    chunks = ["".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
    
    print(f"分为 {len(chunks)} 个 chunks，开始多进程 Tokenize...")
    # 使用所有可用的 CPU 核心
    with mp.Pool(mp.cpu_count()) as pool:
        # 使用 tqdm 显示进度条
        encoded_chunks = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))
    
    print("合并并保存...")
    # 展平列表
    encoded_text = [token for chunk in encoded_chunks for token in chunk]
    
    # 训练的时候加载的数据类型要和这里保持一致
    encoded_text_np = np.array(encoded_text, dtype=np.uint16)
    
    # 保存为 numpy 文件
    np.save('/home/ubuntu/cs336_assignments/assignment1-basics/data/tinystories_valid_encoded.npy', encoded_text_np)
    print("预处理完成！保存在 tinystories_train_encoded.npy")