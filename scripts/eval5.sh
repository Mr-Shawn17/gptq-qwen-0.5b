#!/bin/bash
# 五题质量对照脚本：向 vLLM 服务发 5 道固定题，逐题记录回答
# 用法: eval5.sh <model_name> <tag> [out_dir]
#   model_name  vLLM --served-model-name 指定的名字
#   tag         输出文件标签（生成 eval_<tag>.txt）
#   out_dir     输出目录（默认当前目录）
set -e
TAG="${1:?用法: eval5.sh <model_name> <tag> [out_dir]}"
MODEL="${2:?缺少 model_name}"
OUT_DIR="${3:-.}"
OUT="${OUT_DIR}/eval_${TAG}.txt"
: > "$OUT"
ask() {
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":128,\"temperature\":0}" \
  | python3 -c "import json,sys
try:
    r=json.load(sys.stdin)
    print(r['choices'][0]['message']['content'][:260])
except Exception as e:
    print('ERROR:', e)"
}
Q1="用一句话介绍你自己"
Q2="中国的首都是哪里"
Q3="1+2+3+...+100 等于多少？请说明你的思路。"
Q4="把『今天天气真好』翻译成英文。"
Q5="列出三个以字母 W 开头的英文单词。"
i=1
for q in "$Q1" "$Q2" "$Q3" "$Q4" "$Q5"; do
  echo "===== Q$i: $q" | tee -a "$OUT"
  ask "$q" | tee -a "$OUT"
  echo | tee -a "$OUT"
  i=$((i+1))
done
echo "[done] $OUT"
