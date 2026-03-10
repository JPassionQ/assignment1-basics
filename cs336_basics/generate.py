import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_p=1.0, stop_token_id=None):
    """
    根据给定的 prompt (input_ids) 生成文本。
    
    参数:
        model: 训练好的 Transformer_LM 模型
        input_ids: 形状为 (batch_size, sequence_length) 的 prompt token IDs
        max_new_tokens: 最大生成的 token 数量 [cite: 1083]
        temperature: softmax 温度调节 [cite: 1084]
        top_p: Nucleus 采样阈值 [cite: 1085]
        stop_token_id: 遇到该 token 时停止生成 (如 <|endoftext|>)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # 截断 context_length，防止超出模型的最大位置编码限制
        context_length = model.config.get("context_length", 256)
        idx_cond = input_ids if input_ids.size(1) <= context_length else input_ids[:, -context_length:]
        
        # 前向传播，获取最后一个时间步的 logits
        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :] 
        
        # 1. 温度缩放 [cite: 1070]
        if temperature == 0.0:
            # 贪婪解码
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 2. Top-p (Nucleus) 采样 [cite: 1074]
            if top_p < 1.0:
                # 降序排序概率
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 找到累积概率超过 top_p 的索引
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将索引向右平移一位，确保第一个超过阈值的 token 也被保留
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 恢复到原始索引并将需要移除的 token 概率置为 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                
                # 重新归一化
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)
        
        # 将生成的 token 拼接到输入中
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # 检查是否生成了停止符 [cite: 1082]
        if stop_token_id is not None and (next_token == stop_token_id).all():
            break
            
    return input_ids