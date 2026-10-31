#!/usr/bin/env bash
# max_tokens=1024 单题×5次 bench（v1 口径，与 128 那轮完全同口径，仅输出上限不同）
# 用法: run_bench_1024.sh <model_name> <label>
set -e
MODEL="$1"; LABEL="$2"
python3 ~/quant_gptq/scripts/bench_openai.py \
    --base-url http://127.0.0.1:8000 \
    --model "$MODEL" \
    --prompt "请用两百字介绍一下人工智能的发展历史，要求逻辑清晰、分阶段说明。" \
    --max-tokens 1024 --warmup 3 --runs 5 \
    --label "$LABEL" \
    --out-dir ~/quant_gptq/bench_results
