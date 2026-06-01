# -*- coding: utf-8 -*-
"""
core/graph_state.py — LangGraph 全局状态定义

统一管理竞品分析全流程的所有中间数据和状态标记
"""

from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import add_messages
from models.domain import (
    CompetitorList, CompetitorData,
    ProductAnalysis, PricingAnalysis, MarketAnalysis,
    StrategyReport
)
import operator


class GraphState(TypedDict):
    """
    LangGraph 全局状态
    
    所有Agent节点都从这里读取数据，并写入新的状态
    """
    # ── 输入参数 ──
    product_description: str
    max_competitors: int
    
    # ── Phase 1: 竞品发现结果 ──
    competitor_list: Optional[CompetitorList]
    
    # ── Phase 2: 数据采集结果 ──
    competitors_data: Sequence[CompetitorData]
    
    # ── Phase 3: 三维分析结果 ──
    product_analysis: Optional[ProductAnalysis]
    pricing_analysis: Optional[PricingAnalysis]
    market_analysis: Optional[MarketAnalysis]
    
    # ── Phase 4: 策略报告 ──
    strategy_report: Optional[StrategyReport]
    
    # ── 质检相关: 问题队列和质量打分 ──
    quality_check_passed: bool
    quality_score: float
    issues_found: Sequence[dict]
    retry_count: int
    max_retries: int
    
    # ── 时间统计和全链路日志 ──
    timings: dict[str, float]
    execution_logs: Sequence[dict]
