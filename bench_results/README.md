# bench 数据说明

| 文件 | 口径 | 说明 |
|---|---|---|
| *_http_1024.json（3 份） | **报告主口径** | max_tokens=1024 自然收尾，同题×5 次取中位数 |
| *_http.json（3 份） | 历史存档 | max_tokens=128 上限截断轮（输出恒 128 = length 截断，非自然长度） |

三组对照：fp16_ref（BF16 基线，服务端 FP16）/ gptq_official（官方 GPTQ-Int4）/ gptq_local（自产 GPTQ）。

用 `python3 ../scripts/inspect_json.py` 可查看每份 JSON 的口径摘要。
