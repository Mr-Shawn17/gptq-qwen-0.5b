import os, torch
from safetensors import safe_open

p = os.path.expanduser("~/quant_gptq/models/Qwen2-0.5B-Instruct-GPTQ-Int4-official/model.safetensors")
f = safe_open(p, framework="pt")
emb = f.get_tensor("model.embed_tokens.weight")
lm  = f.get_tensor("lm_head.weight")
print("embed_tokens shape:", tuple(emb.shape), emb.dtype)
print("lm_head     shape:", tuple(lm.shape),  lm.dtype)
same = torch.equal(emb, lm)
if same:
    print("结论: 两者逐元素完全相同 —— lm_head 只是 embed 的一份冗余拷贝")
else:
    diff = (emb != lm).float().mean().item()
    print(f"结论: 数值不同（差异元素占比 {diff*100:.2f}%）—— 官方流水线存了一份独立处理过的 lm_head")
