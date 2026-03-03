#!/bin/bash
# VLM Jenga Baseline 启动脚本
# API 配置 (也可通过环境变量 OPENAI_API_KEY / OPENAI_BASE_URL 设置)
API_KEY="sk-A6WNIdgvHjFkR1vaIyqgEyoP6KnXQgld6Zy75b2Vypz1DCE0"
BASE_URL="https://www.chataiapi.com/v1"

python vlm_jenga_baseline.py \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model gemini-2.5-pro \
    --camera multi \
    --save_images
