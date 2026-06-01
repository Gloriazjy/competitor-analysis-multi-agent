# -*- coding: utf-8 -*-
"""
config.py — 智能竞品分析多Agent系统 全局配置

配置加载优先级：
  1. 环境变量
  2. 本文件硬编码值（仅作兜底，建议通过环境变量设置）

LLM后端选择：
  - LLM_PROVIDER = "doubao"   → 字节跳动豆包API（默认）
  - LLM_PROVIDER = "qianfan"  → 百度千帆API（云端）
  - LLM_PROVIDER = "aliyun"   → 阿里云通义千问API
  - LLM_PROVIDER = "openai"   → OpenAI API
  - LLM_PROVIDER = "ollama"   → 本机Ollama（本地）
"""

import os

# ========================
# LLM 后端选择
# ========================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "doubao")

# ========================
# 豆包API配置（LLM_PROVIDER = "doubao" 时生效）
# 模型: Doubao-Seed-2.0-lite
# EP: ep-20260514111325-xjmj7
# ========================
DOUBAO_API_KEY = os.environ.get(
    "DOUBAO_API_KEY",
    "your_api_key",
)
DOUBAO_MODEL = os.environ.get(
    "DOUBAO_MODEL",
    "ep-20260514111325-xjmj7",
)
DOUBAO_BASE_URL = os.environ.get(
    "DOUBAO_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3",
)

# ========================
# 阿里云API配置（LLM_PROVIDER = "aliyun" 时生效）
# ========================
ALI_API_KEY = os.environ.get(
    "ALI_API_KEY",
    "your_api_key",
)
ALI_MODEL = os.environ.get("ALI_MODEL", "qwen-plus")
ALI_BASE_URL = os.environ.get(
    "ALI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========================
# 千帆API配置（LLM_PROVIDER = "qianfan" 时生效）
# ========================
BAIDU_API_KEY = os.environ.get(
    "BAIDU_API_KEY",
    "your_api_key",
)
BAIDU_SECRET_KEY = os.environ.get(
    "BAIDU_SECRET_KEY",
    "your_api_key",
)
QIANFAN_MODEL = os.environ.get("QIANFAN_MODEL", "glm-5.1")

# 保留旧命名兼容
QIANFAN_API_KEY = BAIDU_API_KEY
QIANFAN_SECRET_KEY = BAIDU_SECRET_KEY

# ========================
# OpenAI API配置（LLM_PROVIDER = "openai" 时生效）
# ========================
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "your_api_key",
)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ========================
# Ollama配置（LLM_PROVIDER = "ollama" 时生效）
# ========================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:0.8b")

# ========================
# 百度AI搜索配置
# ========================
BAIDU_SEARCH_API_KEY = os.environ.get(
    "BAIDU_SEARCH_API_KEY",
    "your_api_key",
)
BAIDU_SEARCH_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
BAIDU_SEARCH_SOURCE = "baidu_search_v2"
BAIDU_SEARCH_RECENCY = os.environ.get("BAIDU_SEARCH_RECENCY", "month")

# 搜索间隔（秒），避免限流
SEARCH_DELAY_SECONDS = float(os.environ.get("SEARCH_DELAY_SECONDS", "2.0"))

# ========================
# 系统模式配置
# ========================
ENABLE_LLM = True

# ========================
# 竞品分析参数
# ========================
# 默认竞品数量范围
MIN_COMPETITORS = 3
MAX_COMPETITORS = 8
DEFAULT_COMPETITOR_COUNT = 5

# ========================
# LLM调用参数
# ========================
LLM_TEMPERATURE = 0.3       # 适中温度，保证分析既准确又有洞察
LLM_MAX_TOKENS = 4096       # 竞品数据较多，增大输出上限

