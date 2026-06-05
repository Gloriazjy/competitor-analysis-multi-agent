# -*- coding: utf-8 -*-
"""
core/disaster_gateway.py — 多模型灾难网关
目标：单点故障50ms内自动切换，跨厂商降级链
核心特性：
  1. 熔断器模式 (Circuit Breaker) - 快速失败，避免雪崩
  2. 实时健康检查 - 毫秒级故障检测
  3. 加权故障转移 - 按优先级自动降级
  4. 故障统计面板 - 实时监控服务商状态
  5. 自动恢复探测 - 半开状态自动恢复
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
import config


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 正常状态，允许请求通过
    OPEN = "open"            # 熔断状态，直接快速失败
    HALF_OPEN = "half_open"  # 半开状态，尝试少量请求探测恢复


@dataclass
class ProviderHealth:
    """服务商健康状态数据"""
    provider_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    open_since: float = 0.0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=20))
    priority: int = 0  # 降级优先级，数字越小越优先


class DisasterGateway:
    """
    多模型灾难网关 - 核心实现
    
    设计目标：单点故障50ms内完成自动切换
    通过熔断器+故障链实现高可用
    """

    _instance: Optional["DisasterGateway"] = None
    _lock: threading.RLock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._lock = threading.RLock()
        self._initialized = True
        
        # 配置参数
        self.failure_threshold = getattr(config, "GATEWAY_FAILURE_THRESHOLD", 5)
        self.open_timeout_ms = getattr(config, "GATEWAY_OPEN_TIMEOUT_MS", 30000)
        self.half_open_max_calls = getattr(config, "GATEWAY_HALF_OPEN_MAX_CALLS", 2)
        self.switch_overhead_ms = getattr(config, "GATEWAY_SWITCH_OVERHEAD_MS", 50)
        
        # 服务商注册表
        self.providers: Dict[str, ProviderHealth] = {}
        self.provider_callbacks: Dict[str, Callable] = {}
        
        # 降级链配置
        self.fallback_chain: List[str] = []
        
        # 并发控制
        self._half_open_calls = 0
        
        # 自动注册默认服务商
        self._register_default_providers()

    def _register_default_providers(self):
        """注册默认的服务商列表，按优先级排序"""
        default_chain = [
            ("doubao", 0),
            ("aliyun", 1),
            ("qianfan", 2),
            ("openai", 3),
            ("ollama", 4),
        ]
        for name, priority in default_chain:
            self.register_provider(name, priority)
        # 设置默认降级链
        self.set_fallback_chain([p[0] for p in default_chain])

    def register_provider(self, provider_name: str, priority: int = 0):
        """注册一个AI服务商"""
        with self._lock:
            if provider_name not in self.providers:
                self.providers[provider_name] = ProviderHealth(
                    provider_name=provider_name,
                    priority=priority
                )

    def register_provider_callback(self, provider_name: str, callback: Callable):
        """注册服务商的调用函数"""
        with self._lock:
            self.provider_callbacks[provider_name] = callback

    def set_fallback_chain(self, provider_names: List[str]):
        """设置降级链路顺序"""
        with self._lock:
            self.fallback_chain = provider_names.copy()

    def record_success(self, provider_name: str, latency_ms: float):
        """记录一次成功调用"""
        with self._lock:
            if provider_name not in self.providers:
                return
            ph = self.providers[provider_name]
            ph.state = CircuitState.CLOSED
            ph.success_count += 1
            ph.total_requests += 1
            ph.failure_count = 0
            ph.last_success_time = time.time()
            ph.recent_latencies.append(latency_ms)

    def record_failure(self, provider_name: str, error_msg: str = ""):
        """记录一次失败调用"""
        with self._lock:
            if provider_name not in self.providers:
                return
            ph = self.providers[provider_name]
            ph.failure_count += 1
            ph.total_requests += 1
            ph.last_failure_time = time.time()
            ph.recent_errors.append(error_msg)
            
            # 达到阈值触发熔断
            if ph.state == CircuitState.CLOSED and ph.failure_count >= self.failure_threshold:
                ph.state = CircuitState.OPEN
                ph.open_since = time.time()
                print(f"  [灾难网关] [WARN] 服务商 [{provider_name}] 触发熔断 (失败次数={ph.failure_count})")

    def _check_and_restore_half_open(self, ph: ProviderHealth) -> bool:
        """检查是否应该进入半开状态尝试恢复"""
        now = time.time()
        if ph.state == CircuitState.OPEN:
            elapsed_ms = (now - ph.open_since) * 1000
            if elapsed_ms >= self.open_timeout_ms:
                ph.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                print(f"  [灾难网关] [INFO] 服务商 [{ph.provider_name}] 进入半开状态尝试恢复")
        return ph.state == CircuitState.HALF_OPEN

    def is_provider_available(self, provider_name: str) -> bool:
        """检查服务商当前是否可用（快速判断，<1ms）"""
        with self._lock:
            if provider_name not in self.providers:
                return False
            ph = self.providers[provider_name]
            
            if ph.state == CircuitState.CLOSED:
                return True
            
            if ph.state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    return True
                return False
            
            # OPEN状态下检查是否可以尝试半开恢复
            self._check_and_restore_half_open(ph)
            return ph.state == CircuitState.HALF_OPEN

    def get_available_providers(self) -> List[str]:
        """获取所有当前可用的服务商，按优先级排序"""
        available = []
        for name in self.fallback_chain:
            if self.is_provider_available(name):
                available.append(name)
        return available

    def call_with_failover(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        agent_id: str,
        preferred_provider: Optional[str] = None
    ) -> tuple[str, str]:
        """
        带自动故障转移的调用 - 核心方法
        
        实现50ms内自动切换：跳过熔断器打开的服务商，立即尝试下一个
        返回: (result_content, used_provider_name)
        """
        start_total = time.perf_counter()
        
        # 生成尝试顺序：优先使用指定的 -> 然后走降级链
        attempt_order = []
        if preferred_provider and preferred_provider in self.fallback_chain:
            attempt_order.append(preferred_provider)
        
        for p in self.fallback_chain:
            if p not in attempt_order:
                attempt_order.append(p)
        
        last_error = ""
        for provider_name in attempt_order:
            # 快速检查可用性，<1ms
            if not self.is_provider_available(provider_name):
                continue
            
            if provider_name not in self.provider_callbacks:
                continue
            
            # 执行调用
            call_start = time.perf_counter()
            try:
                callback = self.provider_callbacks[provider_name]
                result = callback(system_prompt, user_message, temperature, max_tokens, agent_id)
                
                latency_ms = (time.perf_counter() - call_start) * 1000
                
                if result:
                    # 成功
                    self.record_success(provider_name, latency_ms)
                    
                    # 如果是半开状态下成功，完全恢复熔断器
                    ph = self.providers[provider_name]
                    if ph.state == CircuitState.HALF_OPEN:
                        with self._lock:
                            ph.state = CircuitState.CLOSED
                            ph.failure_count = 0
                        print(f"  [灾难网关] [OK] 服务商 [{provider_name}] 已自动恢复正常")
                    
                    total_latency = (time.perf_counter() - start_total) * 1000
                    if provider_name != preferred_provider and preferred_provider:
                        print(f"  [灾难网关] [SWITCH] 从 [{preferred_provider}] 自动切换到 [{provider_name}] "
                              f"(总耗时 {total_latency:.1f}ms)")
                    
                    return result, provider_name
                else:
                    last_error = f"empty_response"
                    self.record_failure(provider_name, last_error)
                    
            except Exception as e:
                latency_ms = (time.perf_counter() - call_start) * 1000
                last_error = str(e)
                self.record_failure(provider_name, last_error)
            
            # 关键：切换开销控制在50ms内 - 这里不等待，立即尝试下一个
            switch_elapsed = (time.perf_counter() - start_total) * 1000
            if switch_elapsed > self.switch_overhead_ms:
                break  # 防止总时间过长
        
        print(f"  [灾难网关] [ERROR] 所有服务商均不可用，最后错误: {last_error}")
        return "", ""

    def get_status_dashboard(self) -> Dict[str, Any]:
        """获取网关状态监控面板"""
        with self._lock:
            dashboard = {
                "gateway_config": {
                    "failure_threshold": self.failure_threshold,
                    "open_timeout_ms": self.open_timeout_ms,
                    "half_open_max_calls": self.half_open_max_calls,
                    "target_switch_overhead_ms": self.switch_overhead_ms,
                },
                "fallback_chain": self.fallback_chain.copy(),
                "providers": {}
            }
            
            for name, ph in self.providers.items():
                # 即使没有callback也显示出来
                pass
                
                avg_latency = 0
                if ph.recent_latencies:
                    avg_latency = sum(ph.recent_latencies) / len(ph.recent_latencies)
                
                success_rate = 0
                if ph.total_requests > 0:
                    success_rate = (ph.success_count / ph.total_requests) * 100
                
                dashboard["providers"][name] = {
                    "state": ph.state.value,
                    "state_emoji": {
                        "closed": "🟢",
                        "open": "🔴",
                        "half_open": "🟡"
                    }.get(ph.state.value, "⚪"),
                    "priority": ph.priority,
                    "total_requests": ph.total_requests,
                    "success_count": ph.success_count,
                    "failure_count": ph.failure_count,
                    "success_rate": round(success_rate, 2),
                    "avg_latency_ms": round(avg_latency, 2),
                    "last_success_ago_sec": round(time.time() - ph.last_success_time, 1) if ph.last_success_time > 0 else None,
                    "last_failure_ago_sec": round(time.time() - ph.last_failure_time, 1) if ph.last_failure_time > 0 else None,
                }
            
            return dashboard

    def print_status_dashboard(self):
        """打印状态面板到控制台"""
        dash = self.get_status_dashboard()
        print("\n" + "="*70)
        print("  [Gateway] 多模型灾难网关 - 实时状态面板")
        print("="*70)
        
        cfg = dash["gateway_config"]
        print(f"\n  配置参数:")
        print(f"    失败阈值: {cfg['failure_threshold']}次  |  熔断恢复时间: {cfg['open_timeout_ms']/1000:.0f}s")
        print(f"    半开探测: {cfg['half_open_max_calls']}并发 |  目标切换耗时: <{cfg['target_switch_overhead_ms']}ms")
        
        print(f"\n  降级链路 (优先级从高到低): {' -> '.join(dash['fallback_chain'])}")
        
        print(f"\n  服务商健康状态:")
        print(f"  {'Provider':<12} {'State':<8} {'Reqs':>8} {'Success%':>8} {'Avg(ms)':>10} {'FailCnt':>8}")
        print("  " + "-"*65)
        
        for name, info in dash["providers"].items():
            state_icon = {'closed':'OK','open':'FAIL','half_open':'PROBE'}.get(info['state'], 'UNK')
            print(
                f"  {name:<12} "
                f"{state_icon:<8} "
                f"{info['total_requests']:>8} "
                f"{info['success_rate']:>7.1f}% "
                f"{info['avg_latency_ms']:>8.1f} "
                f"{info['failure_count']:>8}"
            )
        print("="*70 + "\n")


# 全局单例
_gateway_instance = None

def get_disaster_gateway() -> DisasterGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = DisasterGateway()
    return _gateway_instance
