# -*- coding: utf-8 -*-
"""
agents/strategy_agent.py — 策略建议Agent

职责：综合三维分析，输出差异化定位建议和行动方案
LLM调用：1次
外部工具：无
提示词来源：prompts/strategy_agent.md
"""

from agents.base_agent import BaseAgent
from models.domain import (
    ProductAnalysis, PricingAnalysis, MarketAnalysis,
    StrategyReport, ActionItem
)
from core.prompt_loader import load as load_prompts
import config
import json


class StrategyAgent(BaseAgent):
    """策略建议Agent — 综合三维分析输出策略"""

    def __init__(self):
        prompts = load_prompts("strategy_agent")
        super().__init__(
            agent_id="StrategyAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_strategy = prompts["prompt_strategy"]

    async def run(self, product_name: str,
                  competitor_count: int,
                  product_analysis: ProductAnalysis,
                  pricing_analysis: PricingAnalysis,
                  market_analysis: MarketAnalysis) -> StrategyReport:
        """
        主运行逻辑：综合三维分析输出策略

        Args:
            product_name: 产品名称
            competitor_count: 竞品数量
            product_analysis: 产品分析结果
            pricing_analysis: 定价分析结果
            market_analysis: 市场分析结果

        Returns:
            StrategyReport: 策略建议报告
        """
        self._log("🎯 开始策略建议...")

        # 构建三维分析汇总文本
        analysis_text = self._build_analysis_text(
            product_name, product_analysis, pricing_analysis, market_analysis
        )

        if config.ENABLE_LLM:
            prompt = self._prompt_strategy.format(
                product_name=product_name,
                analysis_text=analysis_text,
            )
            result = await self.ask_llm_json_async(prompt, max_tokens=4096)
            if result:
                report = self._parse_strategy_report(product_name, competitor_count, result)
                self._log(f"✅ 策略建议完成: {len(report.action_plan)}项行动方案")
                return report
            else:
                self._log("⚠️ LLM策略建议失败，降级到规则引擎")

        return self._rule_strategy(product_name, competitor_count,
                                    product_analysis, pricing_analysis, market_analysis)

    def _build_analysis_text(self, product_name: str,
                              product_analysis: ProductAnalysis,
                              pricing_analysis: PricingAnalysis,
                              market_analysis: MarketAnalysis) -> str:
        """构建三维分析汇总文本"""
        lines = []

        # 产品分析
        lines.append("## 一、产品分析")
        if product_analysis.feature_matrix:
            features = [fm.feature for fm in product_analysis.feature_matrix]
            lines.append(f"对比功能维度: {', '.join(features[:10])}")
        if product_analysis.competitive_advantages:
            for adv in product_analysis.competitive_advantages[:5]:
                lines.append(f"- vs {adv.competitor}: 我方优势={adv.our_advantage}, 对方优势={adv.their_advantage}")
        if product_analysis.differentiation_points:
            lines.append(f"差异化点: {', '.join(product_analysis.differentiation_points[:5])}")
        lines.append(f"摘要: {product_analysis.summary}")

        # 定价分析
        lines.append("\n## 二、定价分析")
        if pricing_analysis.pricing_comparison:
            for pc in pricing_analysis.pricing_comparison[:5]:
                lines.append(f"- {pc.competitor}: 免费={pc.free_tier}, 付费={pc.paid_tier}, 模式={pc.pricing_model}")
        lines.append(f"策略分析: {pricing_analysis.pricing_strategy_analysis}")
        if pricing_analysis.value_ranking:
            lines.append(f"性价比排名: {' > '.join(pricing_analysis.value_ranking)}")
        lines.append(f"摘要: {pricing_analysis.summary}")

        # 市场分析
        lines.append("\n## 三、市场分析")
        if market_analysis.market_share_data:
            for ms in market_analysis.market_share_data[:5]:
                lines.append(f"- {ms.competitor}: 份额={ms.share_estimate}, 趋势={ms.trend}")
        lines.append(f"增长趋势: {market_analysis.growth_trends}")
        lines.append(f"渠道分析: {market_analysis.channel_analysis}")
        lines.append(f"摘要: {market_analysis.summary}")

        return "\n".join(lines)

    def _parse_strategy_report(self, product_name: str, competitor_count: int,
                                result: dict) -> StrategyReport:
        """解析LLM返回的策略报告"""
        action_plan = []
        for ap in result.get("action_plan", []):
            action_plan.append(ActionItem(
                priority=ap.get("priority", "P2"),
                action=ap.get("action", ""),
                timeline=ap.get("timeline", ""),
                expected_impact=ap.get("expected_impact", ""),
            ))

        return StrategyReport(
            product_name=product_name,
            competitor_count=competitor_count,
            overall_positioning=result.get("overall_positioning", ""),
            differentiation_strategy=result.get("differentiation_strategy", {}),
            action_plan=action_plan,
            risk_assessment=result.get("risk_assessment", ""),
            product_analysis_summary=result.get("product_analysis_summary", ""),
            pricing_analysis_summary=result.get("pricing_analysis_summary", ""),
            market_analysis_summary=result.get("market_analysis_summary", ""),
            summary=result.get("summary", ""),
        )

    def _rule_strategy(self, product_name: str, competitor_count: int,
                        product_analysis: ProductAnalysis,
                        pricing_analysis: PricingAnalysis,
                        market_analysis: MarketAnalysis) -> StrategyReport:
        """规则引擎策略建议（SWOT模板）"""
        # 从三维分析中提取关键词
        diff_points = product_analysis.differentiation_points[:3] if product_analysis.differentiation_points else []
        diff_text = "、".join(diff_points) if diff_points else "需进一步分析"

        return StrategyReport(
            product_name=product_name,
            competitor_count=competitor_count,
            overall_positioning=f"{product_name}应基于{diff_text}等差异化优势进行市场定位",
            differentiation_strategy={
                "core_differentiator": diff_text,
                "supporting_points": diff_points,
            },
            action_plan=[
                ActionItem(priority="P0", action="深入调研竞品最新动态", timeline="1-2周",
                           expected_impact="建立竞品情报基线"),
                ActionItem(priority="P1", action="强化差异化功能投入", timeline="1-3月",
                           expected_impact="巩固竞争优势"),
                ActionItem(priority="P2", action="制定针对性市场策略", timeline="3-6月",
                           expected_impact="提升市场份额"),
            ],
            risk_assessment="(规则引擎分析，详情请启用LLM)",
            product_analysis_summary=product_analysis.summary[:100],
            pricing_analysis_summary=pricing_analysis.summary[:100],
            market_analysis_summary=market_analysis.summary[:100],
            summary="基于SWOT模板的简单策略建议（建议启用LLM获得深度分析）",
        )

