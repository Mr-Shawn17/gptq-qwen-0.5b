#!/usr/bin/env python3
"""检查 bench JSON 的口径信息：标签 / 协议 / 题数 / 输入输出 token
用法: python3 inspect_json.py [glob_pattern]
默认扫描 ./bench_results/*.json
"""
import json
import glob
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "bench_results/*.json"
files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit(f"[error] 未匹配到文件: {pattern}")

for f in files:
    d = json.load(open(f))
    qs = d.get("questions", ["?"])
    pt = d.get("prompt_tokens_per_run") or [s.get("prompt_tokens") for s in d.get("samples", [])]
    ct = d.get("completion_tokens_per_run") or [s.get("completion_tokens") for s in d.get("samples", [])]
    proto = d.get("protocol", "v1(旧版无此字段)")
    print(d["label"], "|", proto, "| 题数", len(qs), "| 输入tok", pt, "| 输出tok", ct)
    if len(qs) == 1:
        print("    题目:", qs[0][:40])
