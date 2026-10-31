#!/usr/bin/env python3
"""
OpenAI 兼容接口的对称性能测试（vLLM / TRT-LLM 通用同一套口径）

v2 口径（默认）：内置五题库，每题各测 1 次
  - 五道长输出题（都要求展开说明，自然写满 max_tokens），输入长度刻意拉开梯度
  - 逐次记录并展示 input / output token 数
  - 汇总取中位数 + 极差；引擎稳定性结论引用历史"同题×5"轮次（v1 口径）

v1 口径（兼容）：显式传 --prompt 时回到单题 × --runs 次的旧模式

四指标：
  1. TTFT        首 token 延迟 (ms)          —— 流式请求发出到收到第一个内容块
  2. 总耗时      端到端墙钟 (s)
  3. 解码速度    tok/s = (completion_tokens - 1) / (总耗时 - TTFT)
  4. 显存峰值    推理期间 nvidia-smi 轮询峰值 (MiB)

用法:
  python3 bench_openai.py --base-url http://127.0.0.1:8000 --model Qwen2-0.5B-Instruct \
      --label vllm_v2 --max-tokens 128 --warmup 3
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

# 五题库：均为长输出题；输入长度约 25~45 tok 梯度分布（两边引擎同题同序）
QUESTION_BANK = [
    "请用两百字介绍人工智能的发展历史，要求逻辑清晰、分阶段说明。",
    "请用两百字解释深度学习为什么在最近十年才取得突破，分点说明主要原因。",
    "请用两百字介绍 Transformer 架构的核心思想，要求条理清晰、通俗准确。",
    "请用两百字比较 GPU 和 CPU 在并行计算能力上的差异，并分点阐述各自适用场景。",
    "请用两百字说明训练一个大语言模型需要经历哪些关键步骤、消耗哪些核心资源，按流程顺序展开。",
]


class GpuSampler(threading.Thread):
    """后台轮询显存占用，取峰值"""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                v = int(out.stdout.strip().splitlines()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=3)


def one_request(base_url, model, prompt, max_tokens, timeout=600):
    """发一次流式请求，返回 (ttft_ms, total_s, completion_tokens, prompt_tokens, finish_reason, text)"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})

    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    finish_reason = ""
    text_parts = []

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
                prompt_tokens = obj["usage"].get("prompt_tokens", prompt_tokens)
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    text_parts.append(piece)
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr
    total = time.perf_counter() - t0

    if completion_tokens == 0:
        completion_tokens = len(text_parts)  # 兜底
    if ttft is None:
        ttft = total * 1000
    if not finish_reason:
        finish_reason = "length" if completion_tokens >= max_tokens else "stop"
    return ttft, total, completion_tokens, prompt_tokens, finish_reason, "".join(text_parts)


def run_and_record(base_url, model, prompt, max_tokens, tag):
    """单次请求 + 指标计算 + 逐次打印（含 input/output token 与 finish_reason）"""
    ttft, total, ctok, ptok, reason, text = one_request(base_url, model, prompt, max_tokens)
    decode_tps = (ctok - 1) / (total - ttft / 1000) if total > ttft / 1000 and ctok > 1 else 0
    row = {"tag": tag, "question": prompt,
           "prompt_tokens": ptok, "completion_tokens": ctok,
           "finish_reason": reason,
           "ttft_ms": round(ttft, 1), "total_s": round(total, 3),
           "decode_tps": round(decode_tps, 2),
           "output_excerpt": text[:160]}
    print(f"  {tag:<4} 输入 {ptok:3d} tok | 输出 {ctok:4d} tok [{reason:<5}] | "
          f"TTFT {ttft:7.1f} ms | 总 {total:6.2f} s | 解码 {decode_tps:6.2f} tok/s")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen2-0.5B-Instruct")
    ap.add_argument("--prompt", default=None,
                    help="传入则回到 v1 单题模式（--runs 生效）；默认用内置五题库（v2）")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5, help="仅 v1 单题模式生效")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/bench_results"))
    args = ap.parse_args()

    bank_mode = args.prompt is None
    protocol = "v2-bank(5题各1次)" if bank_mode else "v1-single(单题xN次)"

    print(f"=== 目标 {args.base_url}  模型 {args.model}  "
          f"max_tokens={args.max_tokens}  口径 {protocol} ===")
    try:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/v1/models", timeout=15) as r:
            print("模型列表:", json.loads(r.read())["data"][0]["id"])
    except Exception as e:
        print(f"[ERROR] 无法访问服务: {e}")
        return 1

    warm_prompt = QUESTION_BANK[0] if bank_mode else args.prompt
    print(f"预热 {args.warmup} 次（不计时）...")
    for _ in range(args.warmup):
        one_request(args.base_url, args.model, warm_prompt, 32)

    sampler = GpuSampler()
    sampler.start()
    rows = []

    if bank_mode:
        print(f"正式测试：五题各 1 次 ...")
        for i, q in enumerate(QUESTION_BANK, 1):
            rows.append(run_and_record(args.base_url, args.model, q, args.max_tokens, f"Q{i}"))
    else:
        print(f"正式测试：同题 {args.runs} 次 ...")
        for i in range(args.runs):
            rows.append(run_and_record(args.base_url, args.model, args.prompt,
                                       args.max_tokens, f"#{i+1}"))

    sampler.stop()

    med = lambda k: statistics.median([r[k] for r in rows])
    rng = lambda k: (min(r[k] for r in rows), max(r[k] for r in rows))
    result = {
        "label": args.label,
        "protocol": protocol,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "questions": QUESTION_BANK if bank_mode else [args.prompt],
        "ttft_ms_median": round(med("ttft_ms"), 1),
        "ttft_ms_range": rng("ttft_ms"),
        "total_s_median": round(med("total_s"), 3),
        "decode_tps_median": round(med("decode_tps"), 2),
        "decode_tps_range": rng("decode_tps"),
        "prompt_tokens_per_run": [r["prompt_tokens"] for r in rows],
        "completion_tokens_per_run": [r["completion_tokens"] for r in rows],
        "finish_reasons": [r["finish_reason"] for r in rows],
        "gpu_peak_mib": sampler.peak,
        "samples": rows,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    tt_lo, tt_hi = result["ttft_ms_range"]
    dc_lo, dc_hi = result["decode_tps_range"]
    print(f"\n================ 汇总（中位数 / 极差）================")
    print(f"  标签        : {args.label}  [{protocol}]")
    print(f"  输入 token  : {' / '.join(map(str, result['prompt_tokens_per_run']))}")
    print(f"  输出 token  : {' / '.join(map(str, result['completion_tokens_per_run']))}")
    print(f"  TTFT        : {result['ttft_ms_median']} ms  (极差 {tt_lo}~{tt_hi})")
    print(f"  总耗时      : {result['total_s_median']} s")
    print(f"  解码速度    : {result['decode_tps_median']} tok/s  (极差 {dc_lo}~{dc_hi})")
    print(f"  结束原因    : {' / '.join(result['finish_reasons'])}")
    print(f"  显存峰值    : {result['gpu_peak_mib']} MiB")
    print(f"  结果文件    : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
