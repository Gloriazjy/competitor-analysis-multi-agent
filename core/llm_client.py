# -*- coding: utf-8 -*-
"""
core/llm_client.py — LLM调用封装（支持5种后端）

架构设计：
  - 竞品分析Agent的LLM调用统一入口
  - 根据 config.LLM_PROVIDER 自动路由到对应后端：
    - "doubao"  → 字节跳动豆包API（OpenAI兼容）
    - "aliyun"  → 阿里云通义千问API（OpenAI兼容）
    - "qianfan" → 百度千帆API（云端）
    - "openai"  → OpenAI API
    - "ollama"  → 本机Ollama（本地）
  - 带重试和降级机制（LLM失败 → 规则引擎兜底）
  - 带调用统计与详细日志
"""

import json
import re
import time
import config


# ============================
# Access Token 缓存（千帆OAuth2方式）
# ============================
_token_cache = {"token": "", "expires_at": 0}


# ============================
# OpenAI兼容API通用调用函数
# ============================
def _call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    agent_id: str,
    provider_name: str,
) -> str:
    """
    调用OpenAI兼容API（通用实现）

    适用于：豆包、阿里云、OpenAI、Ollama（OpenAI兼容接口）
    """
    import requests

    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(2):
        try:
            print(f"  [{provider_name}] [{agent_id}] 🔄 调用API (attempt {attempt+1})...")
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            result = resp.json()

            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                print(f"  [{provider_name}] [{agent_id}] ❌ API错误 (attempt {attempt+1}): {error_msg}")
                if attempt == 0:
                    time.sleep(2)
                    continue
                return ""

            content = (result.get("choices", [{}])
                      [0].get("message", {}).get("content", ""))

            # 兼容思考型模型（reasoning字段）
            if not content:
                reasoning = (result.get("choices", [{}])
                            [0].get("message", {}).get("reasoning", ""))
                if reasoning:
                    content = reasoning
                    print(f"  [{provider_name}] [{agent_id}] ℹ️  思考型模型：content为空，"
                          f"使用reasoning字段 ({len(reasoning)}字)")

            if content:
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "?")
                completion_tokens = usage.get("completion_tokens", "?")
                print(f"  [{provider_name}] [{agent_id}] ✅ 调用成功 "
                      f"(tokens: {prompt_tokens}+{completion_tokens}, "
                      f"输出长度: {len(content)}字)")
                return content
            else:
                print(f"  [{provider_name}] [{agent_id}] ❌ 返回内容为空")
                return ""

        except requests.exceptions.Timeout:
            print(f"  [{provider_name}] [{agent_id}] ⏱️ 请求超时 (attempt {attempt+1})")
            if attempt == 0:
                continue
            return ""
        except requests.exceptions.ConnectionError:
            print(f"  [{provider_name}] [{agent_id}] ❌ 连接失败")
            return ""
        except Exception as e:
            print(f"  [{provider_name}] [{agent_id}] ❌ 异常: {e}")
            if attempt == 0:
                time.sleep(2)
                continue
            return ""

    return ""


# ============================
# 豆包后端
# ============================
def _call_doubao(system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int,
                 agent_id: str) -> str:
    """调用豆包API（OpenAI兼容）"""
    return _call_openai_compat(
        base_url=config.DOUBAO_BASE_URL,
        api_key=config.DOUBAO_API_KEY,
        model=config.DOUBAO_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_id=agent_id,
        provider_name="豆包",
    )


# ============================
# 阿里云后端
# ============================
def _call_aliyun(system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int,
                 agent_id: str) -> str:
    """调用阿里云通义千问API（OpenAI兼容）"""
    return _call_openai_compat(
        base_url=config.ALI_BASE_URL,
        api_key=config.ALI_API_KEY,
        model=config.ALI_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_id=agent_id,
        provider_name="阿里云",
    )


# ============================
# OpenAI后端
# ============================
def _call_openai(system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int,
                 agent_id: str) -> str:
    """调用OpenAI API"""
    return _call_openai_compat(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL,
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_id=agent_id,
        provider_name="OpenAI",
    )


# ============================
# 千帆后端
# ============================
def _is_bce_v3_key(api_key: str) -> bool:
    """判断是否为 bce-v3 格式的直接认证密钥"""
    return api_key.startswith("bce-v3/")


def _get_bearer_token() -> str:
    """
    获取千帆Bearer Token

    优先使用 bce-v3 直接认证，否则走 OAuth2 获取 access_token
    """
    # 方式一：bce-v3 直接认证
    if _is_bce_v3_key(config.QIANFAN_API_KEY):
        return config.QIANFAN_API_KEY

    # 方式二：OAuth2 获取 access_token
    if not config.QIANFAN_SECRET_KEY:
        return ""

    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    try:
        import requests
        url = (
            f"https://aip.baidubce.com/oauth/2.0/token"
            f"?grant_type=client_credentials"
            f"&client_id={config.QIANFAN_API_KEY}"
            f"&client_secret={config.QIANFAN_SECRET_KEY}"
        )
        resp = requests.post(url, timeout=10)
        data = resp.json()
        token = data.get("access_token", "")
        expires_in = data.get("expires_in", 2592000)

        if token:
            _token_cache["token"] = token
            _token_cache["expires_at"] = now + expires_in - 86400
            return token
        else:
            print(f"  [千帆] ❌ Token获取失败: {data}")
            return ""
    except ImportError:
        print("  [千帆] ❌ requests库未安装 (pip install requests)")
        return ""
    except Exception as e:
        print(f"  [千帆] ❌ Token请求异常: {e}")
        return ""


def _call_qianfan(system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int,
                 agent_id: str) -> str:
    """调用千帆API"""
    import requests

    token = _get_bearer_token()
    if not token:
        print(f"  [千帆] [{agent_id}] ⚠️ Token获取失败，降级到规则引擎")
        return ""

    api_url = "https://qianfan.baidubce.com/v2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "model": config.QIANFAN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    for attempt in range(2):
        try:
            print(f"  [千帆] [{agent_id}] 🔄 调用千帆API (attempt {attempt+1})...")
            resp = requests.post(api_url, headers=headers,
                                json=payload, timeout=120)
            result = resp.json()

            if "error_code" in result:
                error_msg = f"{result.get('error_code')} {result.get('error_msg', '')}"
                print(f"  [千帆] [{agent_id}] ❌ API错误 (attempt {attempt+1}): {error_msg}")
                if attempt == 0:
                    time.sleep(2)
                    continue
                return ""

            content = (result.get("choices", [{}])
                      [0].get("message", {}).get("content", ""))

            # 兼容思考型模型（reasoning字段）
            if not content:
                reasoning = (result.get("choices", [{}])
                            [0].get("message", {}).get("reasoning", ""))
                if reasoning:
                    content = reasoning
                    print(f"  [千帆] [{agent_id}] ℹ️  思考型模型：content为空，"
                          f"使用reasoning字段 ({len(reasoning)}字)")

            if content:
                usage = result.get("usage", {})
                total_tokens = usage.get("total_tokens", "?")
                print(f"  [千帆] [{agent_id}] ✅ 调用成功 "
                      f"(tokens: {total_tokens}, 输出长度: {len(content)}字)")
                return content
            else:
                print(f"  [千帆] [{agent_id}] ❌ 返回内容为空")
                return ""

        except requests.exceptions.Timeout:
            print(f"  [千帆] [{agent_id}] ⏱️ 请求超时 (attempt {attempt+1})")
            if attempt == 0:
                continue
            return ""

    return ""


# ============================
# Ollama后端
# ============================
def _check_ollama_available() -> tuple:
    """检查Ollama服务是否可用"""
    try:
        import requests
        base = config.OLLAMA_BASE_URL.rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return True, model_names
        return False, []
    except ImportError:
        return False, ["(requests库未安装)"]
    except Exception as e:
        return False, [str(e)]


def _call_ollama(system_prompt: str, user_message: str,
                temperature: float, max_tokens: int,
                agent_id: str) -> str:
    """调用Ollama API（使用OpenAI兼容接口）"""
    import requests

    base = config.OLLAMA_BASE_URL.rstrip("/")
    api_url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    for attempt in range(2):
        try:
            print(f"  [Ollama] [{agent_id}] 🔄 调用Ollama (attempt {attempt+1})...")
            resp = requests.post(api_url, headers=headers,
                                json=payload, timeout=300)
            result = resp.json()

            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                print(f"  [Ollama] [{agent_id}] ❌ API错误 (attempt {attempt+1}): {error_msg}")
                if attempt == 0:
                    time.sleep(1)
                    continue
                return ""

            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            if not content:
                reasoning = message.get("reasoning", "")
                if reasoning:
                    content = reasoning
                    print(f"  [Ollama] [{agent_id}] ℹ️  思考型模型：content为空，"
                          f"使用reasoning字段 ({len(reasoning)}字)")

            if content:
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "?")
                completion_tokens = usage.get("completion_tokens", "?")
                print(f"  [Ollama] [{agent_id}] ✅ 调用成功 "
                      f"(tokens: {prompt_tokens}+{completion_tokens}, "
                      f"输出长度: {len(content)}字)")
                return content
            else:
                print(f"  [Ollama] [{agent_id}] ❌ 返回内容为空")
                return ""

        except requests.exceptions.ConnectionError:
            print(f"  [Ollama] [{agent_id}] ❌ 连接失败：Ollama服务未启动？"
                  f" (请运行 ollama serve)")
            return ""
        except requests.exceptions.Timeout:
            print(f"  [Ollama] [{agent_id}] ⏱️ 请求超时 (attempt {attempt+1})")
            if attempt == 0:
                continue
            return ""

    return ""


# ============================
# 统一调度入口
# ============================
_call_stats = {"total": 0, "success": 0, "fallback": 0, "errors": []}


def llm_call(system_prompt: str, user_message: str,
             temperature: float = 0.3, max_tokens: int = 4096,
             agent_id: str = "") -> str:
    """
    调用LLM获取决策（带后端路由、重试和降级）

    根据 config.LLM_PROVIDER 自动选择后端：
      - "doubao"  → 字节跳动豆包API
      - "aliyun"  → 阿里云通义千问API
      - "qianfan" → 百度千帆API
      - "openai"  → OpenAI API
      - "ollama"  → 本机Ollama
    """
    _call_stats["total"] += 1
    call_label = f"[{agent_id}]" if agent_id else ""

    if not config.ENABLE_LLM:
        _call_stats["fallback"] += 1
        print(f"  [LLM] {call_label} ⏭️ LLM未启用，使用规则引擎")
        return ""

    provider = config.LLM_PROVIDER.lower()

    # ----------------------------
    # 豆包
    # ----------------------------
    if provider == "doubao":
        if not config.DOUBAO_API_KEY:
            _call_stats["fallback"] += 1
            print(f"  [豆包] {call_label} ⚠️ API密钥未配置，降级到规则引擎")
            return ""
        try:
            result = _call_doubao(system_prompt, user_message,
                                temperature, max_tokens, agent_id)
            if result:
                _call_stats["success"] += 1
                return result
            else:
                _call_stats["fallback"] += 1
                return ""
        except Exception as e:
            print(f"  [豆包] {call_label} ❌ 异常: {e}")
            _call_stats["errors"].append(str(e))
            _call_stats["fallback"] += 1
            return ""

    # ----------------------------
    # 阿里云
    # ----------------------------
    elif provider == "aliyun":
        if not config.ALI_API_KEY:
            _call_stats["fallback"] += 1
            print(f"  [阿里云] {call_label} ⚠️ API密钥未配置，降级到规则引擎")
            return ""
        try:
            result = _call_aliyun(system_prompt, user_message,
                                temperature, max_tokens, agent_id)
            if result:
                _call_stats["success"] += 1
                return result
            else:
                _call_stats["fallback"] += 1
                return ""
        except Exception as e:
            print(f"  [阿里云] {call_label} ❌ 异常: {e}")
            _call_stats["errors"].append(str(e))
            _call_stats["fallback"] += 1
            return ""

    # ----------------------------
    # 千帆
    # ----------------------------
    elif provider == "qianfan":
        if not config.QIANFAN_API_KEY:
            _call_stats["fallback"] += 1
            print(f"  [千帆] {call_label} ⚠️ API密钥未配置，降级到规则引擎")
            return ""
        try:
            result = _call_qianfan(system_prompt, user_message,
                                temperature, max_tokens, agent_id)
            if result:
                _call_stats["success"] += 1
                return result
            else:
                _call_stats["fallback"] += 1
                return ""
        except ImportError:
            print(f"  [千帆] {call_label} ❌ requests未安装 (pip install requests)")
            _call_stats["fallback"] += 1
            return ""
        except Exception as e:
            print(f"  [千帆] {call_label} ❌ 异常: {e}")
            _call_stats["errors"].append(str(e))
            _call_stats["fallback"] += 1
            return ""

    # ----------------------------
    # OpenAI
    # ----------------------------
    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            _call_stats["fallback"] += 1
            print(f"  [OpenAI] {call_label} ⚠️ API密钥未配置，降级到规则引擎")
            return ""
        try:
            result = _call_openai(system_prompt, user_message,
                                temperature, max_tokens, agent_id)
            if result:
                _call_stats["success"] += 1
                return result
            else:
                _call_stats["fallback"] += 1
                return ""
        except Exception as e:
            print(f"  [OpenAI] {call_label} ❌ 异常: {e}")
            _call_stats["errors"].append(str(e))
            _call_stats["fallback"] += 1
            return ""

    # ----------------------------
    # Ollama
    # ----------------------------
    elif provider == "ollama":
        try:
            result = _call_ollama(system_prompt, user_message,
                                temperature, max_tokens, agent_id)
            if result:
                _call_stats["success"] += 1
                return result
            else:
                _call_stats["fallback"] += 1
                return ""
        except ImportError:
            print(f"  [Ollama] {call_label} ❌ requests未安装 (pip install requests)")
            _call_stats["fallback"] += 1
            return ""
        except Exception as e:
            print(f"  [Ollama] {call_label} ❌ 异常: {e}")
            _call_stats["errors"].append(str(e))
            _call_stats["fallback"] += 1
            return ""

    else:
        _call_stats["fallback"] += 1
        print(f"  [LLM] {call_label} ❌ 未知的LLM_PROVIDER: {provider}")
        return ""


def check_llm_backend() -> dict:
    """检查当前LLM后端的可用性"""
    provider = config.LLM_PROVIDER.lower()

    if provider == "doubao":
        if not config.DOUBAO_API_KEY:
            return {"provider": "doubao", "available": False,
                    "model": config.DOUBAO_MODEL, "detail": "API密钥未配置"}
        return {"provider": "doubao", "available": True,
                "model": config.DOUBAO_MODEL, "detail": "API密钥已配置"}

    elif provider == "aliyun":
        if not config.ALI_API_KEY:
            return {"provider": "aliyun", "available": False,
                    "model": config.ALI_MODEL, "detail": "API密钥未配置"}
        return {"provider": "aliyun", "available": True,
                "model": config.ALI_MODEL, "detail": "API密钥已配置"}

    elif provider == "qianfan":
        if not config.QIANFAN_API_KEY:
            return {"provider": "qianfan", "available": False,
                    "model": config.QIANFAN_MODEL, "detail": "API密钥未配置"}
        return {"provider": "qianfan", "available": True,
                "model": config.QIANFAN_MODEL, "detail": "API密钥已配置"}

    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            return {"provider": "openai", "available": False,
                    "model": config.OPENAI_MODEL, "detail": "API密钥未配置"}
        return {"provider": "openai", "available": True,
                "model": config.OPENAI_MODEL, "detail": "API密钥已配置"}

    elif provider == "ollama":
        available, model_names = _check_ollama_available()
        target = config.OLLAMA_MODEL
        if available:
            model_found = any(target in m or m.startswith(target.split(":")[0])
                            for m in model_names)
            if model_found:
                return {"provider": "ollama", "available": True,
                        "model": target, "detail": f"Ollama服务可用，模型 {target} 已就绪"}
            else:
                return {"provider": "ollama", "available": True,
                        "model": target,
                        "detail": f"Ollama服务可用，但模型 {target} 未找到 (请运行 ollama pull {target})"}
        else:
            return {"provider": "ollama", "available": False,
                    "model": target, "detail": "Ollama服务未启动 (请运行 ollama serve)"}

    else:
        return {"provider": provider, "available": False,
                "model": "", "detail": f"未知的LLM_PROVIDER: {provider}"}


# ============================
# JSON解析工具
# ============================

def parse_llm_json(text: str) -> dict:
    """解析LLM返回的JSON（支持多种格式）"""
    if not text:
        return {}

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  [LLM] ⚠️ JSON解析失败，原始文本前200字: {text[:200]}...")
    return {}


# ============================
# 调用统计
# ============================

def get_llm_stats() -> dict:
    """获取LLM调用统计"""
    return _call_stats.copy()


def reset_llm_stats():
    """重置LLM调用统计"""
    global _call_stats
    _call_stats = {"total": 0, "success": 0, "fallback": 0, "errors": []}

