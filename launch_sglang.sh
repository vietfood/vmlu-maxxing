#!/bin/bash
# Note: For single GPU (A6000/5090) setup, we use --tp-size 1.
# Ensure SGLang is installed: pip install sglang[all]

echo "Launching SGLang Server for Qwen3.5-27B..."
echo "Host: http://localhost:8000"

python3 -m sglang.launch_server \
        --port 8000 \
        --model-path Qwen/Qwen3.5-27B \
        --tp-size 1 \
        --attention-backend triton \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder
