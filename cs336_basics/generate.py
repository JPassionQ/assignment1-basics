import torch
import math
from .transformer_lm import Transformer_LM
from .tokenizer import Tokenizer
from .utils import load_checkpoint, softmax
from .adamw import AdamW

@torch.no_grad()
def generate_text(
    model: Transformer_LM, 
    tokenizer: Tokenizer, 
    prompt: str, 
    max_new_tokens: int, 
    context_length: int, 
    temperature: float = 1.0, 
    top_p: float = 1.0, 
    device: str = "cpu"
) -> str:
    """
    根据给定的 prompt 生成文本 [cite: 1082-1085]。
    """
    model.eval()
    
    # 编码 prompt [cite: 312]
    # inputs 是一个二维数组，第一维是batch size
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # 获取停止符的 token ID (假设 special token 直接被 encode)
    stop_token_id = tokenizer.encode("<|endoftext|>")[0] 
    
    for _ in range(max_new_tokens):
        # 截断 context_length，防止超出模型最大上下文长度 [cite: 1067]
        cond_ids = input_ids if input_ids.size(1) <= context_length else input_ids[:, -context_length:]
        
        # 前向传播，获取 logits [cite: 1066-1067]
        logits = model(cond_ids)
        # 我们只需要最后一个时间步的预测分布 [cite: 1067]
        next_token_logits = logits[:, -1, :] # 维度变化 [bs, vocab_size]
        
        # 1. 温度缩放 (Temperature Scaling) [cite: 1070-1071]
        if temperature == 0.0:
            # 贪婪解码: 当 tau 趋近于 0 时，相当于取 argmax [cite: 1073]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # 应用温度参数 [cite: 1071]
            next_token_logits = next_token_logits / temperature
            # 使用你 utils.py 中实现的 softmax 将 logits 转换为概率 [cite: 1061, 1071]
            probs = softmax(next_token_logits, dim=-1)
            
            # 2. 核采样 (Nucleus / Top-p Sampling) 
            if top_p < 1.0:
                # 将概率降序排序 [cite: 1078]
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                print(sorted_probs[:5])
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 截断累积概率超过 top_p 的 token [cite: 1077-1078]
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将索引向右平移一位，确保第一个超过阈值的 token 也被保留到 V(p) 集合中 [cite: 1078]
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 恢复到原始词表的索引顺序，并将被截断的 token 概率置为 0 [cite: 1077]
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                
                # 重新归一化概率分布以进行采样 [cite: 1075]
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 根据最终的概率分布采样下一个 token [cite: 1063, 1077]
            next_token = torch.multinomial(probs, num_samples=1)

        
        # 将生成的 token 拼接到当前序列中 [cite: 1068]
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # 如果生成了 <|endoftext|>，则提前停止生成 [cite: 1068]
        if next_token.item() == stop_token_id:
            break
            
    # 解码为文本字符串 [cite: 315]
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text

if __name__ == "__main__":
    # 配置你的硬件设备 [cite: 991]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. 还原模型结构参数 (务必与训练时保持完全一致) [cite: 1114-1122]
    config = {
        "context_length": 256,
        "d_model": 512,
        "n_heads": 16,
        "layers": 4,
        "vocab_size": 10000 
    }
    
    # 2. 加载 Tokenizer 
    # 假设你的 BPE Tokenizer 训练后保存为了这两个文件 [cite: 307]
    tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/ubuntu/cs336_assignments/assignment1-basics/utils/tinystories_vocab.json", 
        merges_filepath="/home/ubuntu/cs336_assignments/assignment1-basics/utils/tinystories_merges.txt", 
        special_tokens=["<|endoftext|>"]
    )
    
    # 3. 初始化模型
    model = Transformer_LM(
        d_model=config["d_model"], 
        num_heads=config["n_heads"], 
        d_ff=4 * config["d_model"], 
        vocab_size=config["vocab_size"], 
        context_length=config["context_length"], 
        num_layers=config["layers"], 
        apply_rope=True, 
        theta=10000.0, 
        device=device
    )
    
    # 4. 加载 Checkpoint
    # 因为你实现的 load_checkpoint 需要传入 optimizer，我们构造一个 dummy optimizer [cite: 1033-1040]
    dummy_optimizer = AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = "/home/ubuntu/cs336_assignments/assignment1-basics/checkpoints/ckpt_iter_2000.pt" # 替换为你的真实权重路径
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    iteration = load_checkpoint(src=checkpoint_path, model=model, optimizer=dummy_optimizer)
    print(f"Model loaded at training iteration {iteration}.")
    
    # 5. 执行生成测试
    prompt_text = "Once upon a time, there was a little boy named Ben."
    
    
    print(f"\n--- Prompt ---\n{prompt_text}\n")
    print("--- Generating... ---")
    
    generated_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_new_tokens=256,
        context_length=config["context_length"],
        temperature=1.0,   # 尝试调整温度: 0.8 或 1.0 [cite: 1084]
        top_p=0.9,         # 尝试调整 top-p: 0.9 [cite: 1085]
        device=device
    )
    
    print(f"\n--- Final Output ---\n{generated_output}")