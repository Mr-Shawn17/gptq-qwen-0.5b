# Qwen2-0.5B GPTQ 量化实战（GTX 1650 · WSL2）

在一张 4GB 显存的图灵卡（GTX 1650）上，用 [GPTQModel](https://github.com/modelcloud/gptqmodel) 把 Qwen2-0.5B-Instruct 量化成 W4A16（INT4 权重 + FP16 计算），并与官方 GPTQ-Int4 产物、BF16 原模型做同口径性能与质量对照。

**一句话结论：GPTQ W4A16 走 vLLM 是 4GB 老卡上的最优推理方案——长输出解码速度 2.56 倍于基线，模型体积省 56%，质量五题对照无新增错误。**

## 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GTX 1650（图灵 SM75，4GB，无 BF16 计算单元） |
| 系统 | Windows + WSL2 Ubuntu 22.04 |
| 驱动 | 610.74 |
| 量化环境 | Python 3.10 · gptqmodel 7.6.0 · torch 2.14.1+cu130（conda 环境约 6.2GB） |
| 推理环境 | vLLM 0.8.5.post1（`VLLM_USE_V1=0`，`--dtype half`） |
| 源模型 | Qwen2-0.5B-Instruct（BF16，988MB，来自 ModelScope） |

> 本机为图灵架构无 BF16 计算单元，vLLM 加载 BF16 权重时自动转 FP16 计算（日志可见 `Casting torch.bfloat16 to torch.float16`）。Ampere 及以上显卡会以 BF16 运行。

## 结果速览（max_tokens=1024 自然收尾，预热 3 + 5 次中位数）

| 指标 | BF16 基线（服务 FP16） | 官方 GPTQ（W4A16） | 自产 GPTQ（W4A16） |
|---|---|---|---|
| 解码速度 (tok/s) | 23.34 | 59.54 | **59.87** |
| TTFT (ms) | 220.5 | 65.4 | **60.2** |
| 自然输出长度 (tok) | 308 | 339 | 129（最贴近"两百字"题目要求） |
| 显存峰值 (MiB) | 2031 | 2132 | 2132 |
| 模型体积 (MB) | 988 | 699 | **439** |

完整逐次明细与质量五题对照见 [docs/执行报告.md](docs/执行报告.md)。

## 仓库结构

```
├── scripts/                  # 全部可复用脚本
│   ├── quant_gptq.py         # GPTQ 量化（--model/--out/--calib 参数化）
│   ├── bench_openai.py       # OpenAI 兼容接口性能测试（v1 单题 / v2 五题库双模式）
│   ├── run_bench_1024.sh     # max_tokens=1024 单题×5 次 bench 封装
│   ├── eval5.sh              # 五题质量对照（temp=0）
│   ├── wait_ready.sh         # vLLM 服务就绪轮询
│   ├── inspect_json.py       # bench JSON 口径检查
│   └── compare_lm_head.py    # 官方 vs 自产产物 lm_head 冗余分析
├── bench_results/            # 6 份 bench JSON（*_1024.json 为报告口径，*_http.json 为 128 上限存档）
├── logs/                     # 实测日志原样存档（量化/TRT 构建/vLLM 启动/五题答案）
└── docs/                     # 量化方案 + 执行报告（中文）
```

> 注：logs/ 保留了原始日志（含运行机器的本地路径），仅作过程存档参考；scripts/ 与 docs/ 已做去本机化处理。

## 快速开始

### 1. 环境准备

```bash
# 量化环境（gptqmodel）
conda create -n gptq python=3.10 -y
conda activate gptq
pip install gptqmodel torch --index-url https://download.pytorch.org/whl/cu121

# 推理环境（vllm）
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm==0.8.5.post1
```

### 2. 下载源模型

```bash
# ModelScope（国内快）
pip install modelscope
modelscope download --model Qwen/Qwen2-0.5B-Instruct --local_dir ~/models/Qwen2-0.5B-Instruct
```

### 3. 量化（约 5 分钟，显存 < 2.5GB）

```bash
conda activate gptq
python scripts/quant_gptq.py \
    --model ~/models/Qwen2-0.5B-Instruct \
    --out ./Qwen2-0.5B-Instruct-GPTQ-Int4-local

# 校准集三选一：zh_wiki（默认，需联网）/ c4_en（需联网）/ builtin（离线，需 --calib-dir 指向本地 .md 目录）
```

### 4. 起服务 + 测试

```bash
conda activate vllm
export VLLM_USE_V1=0
vllm serve ./Qwen2-0.5B-Instruct-GPTQ-Int4-local \
    --served-model-name qwen2-gptq --dtype half --quantization gptq \
    --gpu-memory-utilization 0.6 --max-model-len 2048 --port 8000
# 注意：--quantization gptq 必须显式指定，否则 vLLM 0.8.5 自动选 bitblas 后端会报 No module named bitblas

# 另开终端：性能测试 / 质量对照
python3 scripts/bench_openai.py --model qwen2-gptq --max-tokens 1024 \
    --label my_test --out-dir ./bench_results
bash scripts/eval5.sh qwen2-gptq my_eval .
```

## 踩坑记录（都在日志里可查证）

| 坑 | 现象 | 解法 |
|---|---|---|
| vLLM 自动选错 GPTQ 后端 | `No module named bitblas` | 显式 `--quantization gptq` |
| 校准集下载失败 | hf-mirror 偶发拉不动 zh_wiki | `--calib builtin --calib-dir <本地md目录>` 兜底，实测质量合格 |
| gptqmodel 保存的 tokenizer 版本太新 | 老 vLLM 加载报错 | 从源模型目录拷原版 tokenizer 四件套覆盖 |
| TRT-LLM 引擎构建必崩 | `Can't free tmp workspace for GEMM tactics profiling` | 图灵 4GB 卡 tactic 搜索撑爆显存，环境级不可行（官方产物同样崩）；详见 docs/执行报告.md §2 阶段 5 |

## 官方 GPTQ vs 自产 GPTQ 深度对比

两者量化参数完全一致（bits=4 / group_size=128 / desc_act=False / sym=True，量化矩阵数同为 168 个），实测运行时等价，但文件组织与实测表现有几处值得玩味的差异。

### 总对比表

| 维度 | 官方 GPTQ-Int4 | 自产 GPTQ | 说明 |
|---|---|---|---|
| **量化参数** | 4bit / gs=128 / desc_act=False / sym=True | 完全一致 | quantize_config.json 逐项对齐 |
| **量化矩阵数** | 168 个（24 层 × 7 Linear） | 168 个 | safetensors qweight 键数相同 |
| **模型体积** | 699MB（safetensors 732MB） | **439MB（-37%）** | 差异全部来自 lm_head 冗余（见下） |
| **safetensors 键数** | 891（含 `lm_head.weight`） | 794（不含） | 官方多存一份 lm_head |
| **lm_head 处理** | 冗余存一份与 embed_tokens 逐元素相同的拷贝 | 只存 embed_tokens 一份 | 实测 `torch.equal` 逐元素相同 |
| **权重绑定** | config 标 tie=true，但文件里存两份 | config 标 tie=true，文件里存一份 | vLLM 加载时都按 tie 重新绑定 |
| **运行时计算图** | 相同 | 相同 | 多余拷贝被丢弃，速度/显存理论无差 |
| **解码速度（1024 自然收尾）** | 59.54 tok/s | **59.87 tok/s**（+0.6%） | 长输出下收敛到同一 kernel 稳态 |
| **解码速度（128 上限轮）** | 50.83 tok/s | **60.18 tok/s**（+18%） | 短轮对 launch 开销敏感，差异被放大 |
| **TTFT（1024）** | 65.4 ms | **60.2 ms** | |
| **质量五题对照** | 无新增错误 | 无新增错误 | 两版 GPTQ 均与 BF16 基线对照合格 |
| **输出风格差异** | "两百字"题写 339 tok（超纲） | 129 tok（≈190 字，最贴题） | 单题样本，仅作观察 |
| **复现方式** | ModelScope 下载 | `quant_gptq.py` 一条命令 ~5 分钟 | 自产可换校准集再迭代 |

### 体积差 273MB 的来龙去脉

Qwen2-0.5B 的 `tie_word_embeddings=true`——embedding 矩阵（151936 词表 × 896 维）与 lm_head 共享同一份权重，这样 0.5B 的小模型能省 272MB 参数（占总参数 34%）。

- **自产产物（gptqmodel 导出）**：忠实于绑定关系，safetensors 只存 `model.embed_tokens.weight` 一份（459MB）
- **官方产物（AutoGPTQ 一脉工具链）**：量化流水线不特判 tie 关系，导出时把 lm_head 也序列化了一份——实测该拷贝与 embed_tokens **逐元素完全相同**（`torch.equal` 为 True），纯粹是磁盘冗余

vLLM 加载时读到 `tie_word_embeddings=true` 都会重新绑定，官方那份多余的 lm_head 直接被丢弃——所以两者**运行时计算图完全一致**，速度与显存理论无差（1024 轮实测 59.54 vs 59.87 tok/s，0.6% 属噪声级）。

用 `python3 scripts/compare_lm_head.py` 可复现"逐元素相同"这一结论（需本地有官方产物）。

### 那 128 轮的 18% 速度差是哪来的？

不是 lm_head 冗余（它根本不参与计算），而是**短轮次对 launch 开销的敏感度**：128 tok 截断轮的有效解码窗口只有 ~2 秒，权重加载布局、g_idx 排布、CPU 缓存命中的微小差异都会被放大；1024 长轮次下两边收敛到同一稳态（≈60 tok/s），恰好证明运行时等价。

### 选型建议

- **纯推理部署**：两种产物等价，自产版省 37% 磁盘与分发带宽——没有理由选冗余版
- **要在 lm_head 上做 LoRA/微调**：需要解开绑定（`tie_word_embeddings=False` 后显式复制），官方那种"显式双份"存储才是你要的形态
- **往小众/老旧 runtime 移植**：官方的显式双份更保险（不依赖框架支持 tie）

一句话：**那 273MB 是"保险费"，不是"性能费"——买到的是兼容性，不是速度。**

## License

仅供学习研究。模型权重遵循 Qwen 原始许可（Tongyi Qianwen LICENSE）。
