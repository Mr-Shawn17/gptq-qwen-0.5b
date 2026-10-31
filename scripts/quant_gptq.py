#!/usr/bin/env python3
"""
Qwen2-0.5B-Instruct GPTQ 4bit 量化脚本
参数与 Qwen 官方 GPTQ-Int4 产物对齐：bits=4 / group_size=128 / desc_act=False / sym=True

校准集三选一：
  zh_wiki  中文维基（默认，走 HF 镜像，需联网）
  c4_en    英文 C4（需联网）
  builtin  零下载兜底——扫描 --calib-dir 目录下的 .md 文件切段生成

用法：
  python quant_gptq.py --model ~/models/Qwen2-0.5B-Instruct --out ./out-gptq
  python quant_gptq.py --calib builtin --calib-dir ~/my-docs
"""
import argparse
import glob
import os
import re
import time

from gptqmodel import GPTQModel, GPTQConfig


def load_calib(kind: str, calib_dir: str, n: int = 128):
    """加载校准样本 n 条（每条截断到 1024 字符）"""
    if kind in ("zh_wiki", "c4_en"):
        from datasets import load_dataset
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        if kind == "zh_wiki":
            ds = load_dataset("wikimedia/wikipedia", "20231101.zh",
                              split="train", streaming=True)
        else:
            ds = load_dataset("allenai/c4",
                              data_files="en/c4-train.00001-of-01024.json.gz",
                              split="train", streaming=True)
        return [next(iter(ds))["text"][:1024] for _ in range(n)]

    # builtin：扫描本地 .md 文档，按空行切段，取合适长度的段落
    if not calib_dir:
        raise SystemExit("[error] builtin 校准集需要 --calib-dir 指定本地文档目录")
    texts = []
    for f in glob.glob(os.path.join(calib_dir, "*.md")):
        with open(f, encoding="utf-8") as fh:
            content = fh.read()
        for para in re.split(r"\n{2,}", content):
            if 100 < len(para) < 2000:
                texts.append(para)
    if not texts:
        raise SystemExit(f"[error] {calib_dir} 下未找到可用的 .md 段落（要求段长 100~2000 字符）")
    return (texts * 8)[:n]


def main():
    ap = argparse.ArgumentParser(description="Qwen2 GPTQ 4bit 量化")
    ap.add_argument("--model", required=True, help="源模型目录（BF16/FP16 HF 格式）")
    ap.add_argument("--out", required=True, help="量化产物输出目录")
    ap.add_argument("--calib", choices=["zh_wiki", "c4_en", "builtin"],
                    default="zh_wiki", help="校准集（默认 zh_wiki）")
    ap.add_argument("--calib-dir", default="",
                    help="builtin 模式下本地 .md 文档目录")
    ap.add_argument("--samples", type=int, default=128, help="校准样本数（默认 128）")
    ap.add_argument("--batch-size", type=int, default=1, help="校准批大小（默认 1）")
    args = ap.parse_args()

    calib_data = load_calib(args.calib, args.calib_dir, args.samples)
    print(f"[calib] 校准样本 {len(calib_data)} 条，首条预览: {calib_data[0][:60]}...", flush=True)

    print("[load] BF16 原生加载到 GPU，逐层 Hessian 校准 ...", flush=True)
    t0 = time.time()
    quant_config = GPTQConfig(bits=4, group_size=128, desc_act=False)
    model = GPTQModel.load(args.model, quant_config)
    model.quantize(calib_data, batch_size=args.batch_size)
    print(f"[quant] 量化完成，耗时 {time.time() - t0:.0f}s", flush=True)

    model.save(args.out)
    print(f"[save] 产物目录: {args.out}", flush=True)

    out = model.generate("中国的首都是", max_new_tokens=16)
    print("[smoke] 冒烟输出:", model.tokenizer.decode(out[0]), flush=True)


if __name__ == "__main__":
    main()
