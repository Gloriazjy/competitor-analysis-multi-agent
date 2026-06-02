# -*- coding: utf-8 -*-
"""
core/orchestrator_graph.py — LangGraph 主控编排器

架构:
  - 使用 LangGraph StateGraph 构建带动态循环的DAG
  - 质检Agent发现问题后可自动回退重做
  - LangSmith 全链路追踪可视化
  - 完全兼容 LangGraph 自动序列化 dataclass → dict 反向转换
"""

import asyncio
import time
import os

from langgraph.graph import StateGraph, END
from core.graph_state import GraphState
from core.type_utils import (
    to_competitor_list, to_competitor_data,
    to_product_analysis, to_pricing_analysis, to_market_analysis
)
from agents.research.discovery_agent import DiscoveryAgent
from agents.research.collection_agent import CollectionAgent
from agents.analysis.product_agent import ProductAgent
from agents.analysis.pricing_agent import PricingAgent
from agents.analysis.market_agent import MarketAgent
from agents.analysis.quality_check_agent import QualityCheckAgent
from agents.reporting.strategy_agent import StrategyAgent
from agents.reporting.report_formatter import ReportFormatter
import config


class LangGraphOrchestrator:
    """
    基于LangGraph的竞品分析编排器
    
    DAG流程图:
        START → 竞品发现 → 数据采集 → 并行三维分析 → 质检
              ↑                                          ↓
              ────────── 不通过(有问题) ←─────────────────  通过 → 策略报告 → END
    """

    def __init__(self):
        self.discovery_agent = DiscoveryAgent()
        self.collection_agent = CollectionAgent()
        self.product_agent = ProductAgent()
        self.pricing_agent = PricingAgent()
        self.market_agent = MarketAgent()
        self.strategy_agent = StrategyAgent()
        self.report_formatter = ReportFormatter()
        self.quality_check_agent = QualityCheckAgent()
        
        self._app = None

    # ── 节点定义 ──
    async def node_discovery(self, state: GraphState) -> dict:
        """节点：竞品发现"""
        start = time.time()
        print("\n" + "═"*60)
        print("  🔍 [LangGraph Node] 竞品发现")
        print("═"*60)
        product_desc = state["product_description"]
        max_comp = state["max_competitors"]
        result = await self.discovery_agent.run(product_desc, max_comp)
        state["timings"]["discovery"] = time.time() - start
        return {"competitor_list": result}

    async def node_collection(self, state: GraphState) -> dict:
        """节点：数据采集"""
        start = time.time()
        print("\n" + "═"*60)
        print("  📊 [LangGraph Node] 数据采集")
        print("═"*60)
        
        raw_cl = state.get("competitor_list")
        comp_list = to_competitor_list(raw_cl)
        
        product_desc = state["product_description"]
        result = await self.collection_agent.run(
            product_desc,
            comp_list,
            quality_feedback=state.get("quality_feedback", []),
        )
        state["timings"]["collection"] = time.time() - start
        return {"competitors_data": result}

    async def node_parallel_analysis(self, state: GraphState) -> dict:
        """节点：并行三维分析"""
        start = time.time()
        print("\n" + "═"*60)
        print("  ⚡ [LangGraph Node] 三维并行分析")
        print("═"*60)
        
        raw_cl = state.get("competitor_list")
        comp_list = to_competitor_list(raw_cl)
        product_name = comp_list.product_name
        
        raw_cd_list = state.get("competitors_data") or []
        comp_data_list = [to_competitor_data(cd) for cd in raw_cd_list]
        
        # 把 list 转成 dict[str, CompetitorData]，兼容各Agent的接口
        comp_data_dict = {cd.name: cd for cd in comp_data_list}
        
        pa, pra, ma = await asyncio.gather(
            self.product_agent.run(
                product_name,
                comp_data_dict,
                quality_feedback=state.get("quality_feedback", []),
            ),
            self.pricing_agent.run(
                product_name,
                comp_data_dict,
                quality_feedback=state.get("quality_feedback", []),
            ),
            self.market_agent.run(
                product_name,
                comp_data_dict,
                quality_feedback=state.get("quality_feedback", []),
            )
        )
        state["timings"]["parallel_analysis"] = time.time() - start
        return {
            "product_analysis": pa,
            "pricing_analysis": pra,
            "market_analysis": ma
        }

    async def node_quality_check(self, state: GraphState) -> dict:
        """节点：质检"""
        start = time.time()
        print("\n" + "═"*60)
        print("  🧐 [LangGraph Node] 质量检测Agent")
        print("═"*60)
        raw_state = dict(state)
        result = await self.quality_check_agent.run(raw_state)
        state["timings"]["quality_check"] = time.time() - start
        return result

    async def node_strategy(self, state: GraphState) -> dict:
        """节点：策略生成"""
        start = time.time()
        print("\n" + "═"*60)
        print("  🎯 [LangGraph Node] 生成策略报告")
        print("═"*60)
        
        raw_cl = state.get("competitor_list")
        comp_list = to_competitor_list(raw_cl)
        comp_count = len(comp_list.competitors) if comp_list else 0
        
        raw_pa = state.get("product_analysis")
        raw_pra = state.get("pricing_analysis")
        raw_ma = state.get("market_analysis")
        
        pa = to_product_analysis(raw_pa)
        pra = to_pricing_analysis(raw_pra)
        ma = to_market_analysis(raw_ma)
        
        result = await self.strategy_agent.run(
            comp_list.product_name,
            comp_count,
            pa,
            pra,
            ma
        )
        state["timings"]["strategy"] = time.time() - start
        return {"strategy_report": result}

    # ── 条件边路由 ──
    def router_after_quality_check(self, state: GraphState) -> str:
        """质检后的条件路由：通过就去策略报告，不通过就按问题目标回退"""
        if state.get("quality_check_passed"):
            print(f"  ✅ 质检通过 (得分 {state.get('quality_score', 0):.2f}) → 进入策略生成")
            return "strategy"

        target = state.get("rollback_target") or "collection"
        allowed_targets = {"discovery", "collection", "parallel_analysis"}
        if target not in allowed_targets:
            target = "collection"
        print(
            f"  🔄 质检未通过 (得分 {state.get('quality_score', 0):.2f}) "
            f"→ 回退到 {target}"
        )
        return target

    def build_graph(self):
        """构建完整LangGraph StateGraph"""
        workflow = StateGraph(GraphState)
        
        # 注册所有节点
        workflow.add_node("discovery", self.node_discovery)
        workflow.add_node("collection", self.node_collection)
        workflow.add_node("parallel_analysis", self.node_parallel_analysis)
        workflow.add_node("quality_check", self.node_quality_check)
        workflow.add_node("strategy", self.node_strategy)

        # 设置边
        workflow.set_entry_point("discovery")
        workflow.add_edge("discovery", "collection")
        workflow.add_edge("collection", "parallel_analysis")
        workflow.add_edge("parallel_analysis", "quality_check")
        
        # 条件边（质检分叉）
        workflow.add_conditional_edges(
            "quality_check",
            self.router_after_quality_check,
            {
                "discovery": "discovery",
                "collection": "collection",
                "parallel_analysis": "parallel_analysis",
                "strategy": "strategy"
            }
        )
        workflow.add_edge("strategy", END)
        
        self._app = workflow.compile()
        return self._app

    def save_graph_visualization(self, output_path: str = "output/langgraph_dag.png"):
        """生成DAG可视化图片"""
        if not self._app:
            self.build_graph()
        try:
            png_data = self._app.get_graph().draw_mermaid_png()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(png_data)
            print(f"🖼️  LangGraph DAG图已保存: {output_path}")
        except Exception as e:
            print(f"⚠️  可视化生成跳过: {e}")

    async def run(self, product_description: str, max_competitors: int = 5):
        """执行完整LangGraph工作流"""
        if not self._app:
            self.build_graph()
            
        total_start = time.time()
        initial_state: GraphState = {
            "product_description": product_description,
            "max_competitors": max_competitors,
            "competitor_list": None,
            "competitors_data": [],
            "product_analysis": None,
            "pricing_analysis": None,
            "market_analysis": None,
            "strategy_report": None,
            "quality_check_passed": False,
            "quality_score": 0.0,
            "issues_found": [],
            "rollback_target": "",
            "quality_feedback": [],
            "retry_count": 0,
            "max_retries": 1,
            "timings": {},
            "execution_logs": []
        }
        
        final_state = await self._app.ainvoke(initial_state)
        final_state["timings"]["total"] = time.time() - total_start
        return final_state
