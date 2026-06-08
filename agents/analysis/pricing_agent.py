# -*- coding: utf-8 -*-
"""
agents/pricing_agent.py — 定价分析Agent

职责：对比各竞品定价策略、促销模式、性价比
LLM调用：1次
外部工具：无
提示词来源：prompts/pricing_agent.md
"""

from agents.base_agent import BaseAgent
from models.domain import CompetitorData, PricingAnalysis, PricingItem
from core.prompt_loader import load as load_prompts
from core.scenario_profile import detect_scenario
import config
import json


class PricingAgent(BaseAgent):
    """定价分析Agent — 价格策略对比"""

    def __init__(self):
        prompts = load_prompts("pricing_agent")
        super().__init__(
            agent_id="PricingAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_analyze = prompts["prompt_analyze"]

    async def run(self, product_name: str,
                  competitors_data: dict[str, CompetitorData],
                  quality_feedback: list[dict] = None) -> PricingAnalysis:
        """
        主运行逻辑：全量数据分析定价对比

        Args:
            product_name: 用户产品名称
            competitors_data: 竞品采集数据
            quality_feedback: 质检打回的定价分析问题

        Returns:
            PricingAnalysis: 定价分析结果
        """
        self._log("💰 开始定价分析...")
        if quality_feedback:
            self._log(f"   收到质检反馈: {len(quality_feedback)}条")

        competitors_text = self._build_competitors_text(product_name, competitors_data)

        if config.ENABLE_LLM:
            prompt = self._prompt_analyze.format(
                product_name=product_name,
                competitors_text=competitors_text,
            )
            feedback_text = self._format_feedback(quality_feedback or [])
            if feedback_text:
                prompt += f"\n\n## 质检返工要求\n{feedback_text}\n请补齐定价对比项，并避免无依据的定价判断。"
            result = await self.ask_llm_json_async(prompt, max_tokens=4096)
            if result:
                analysis = self._parse_pricing_analysis(result)
                self._log(f"✅ 定价分析完成: {len(analysis.pricing_comparison)}个竞品定价对比")
                return analysis
            else:
                self._log("⚠️ LLM定价分析失败，降级到规则引擎")

        return self._rule_analyze(product_name, competitors_data)

    def _build_competitors_text(self, product_name: str,
                                 competitors_data: dict[str, CompetitorData]) -> str:
        """构建竞品定价数据文本"""
        lines = []
        for name, data in competitors_data.items():
            label = name if name != product_name else f"{name}(我方产品)"
            lines.append(f"\n### {label}")
            lines.append(f"- 定价信息: {data.pricing_info[:300]}")
            lines.append(f"- 定价采集状态: {(data.field_status or {}).get('pricing_info', 'unknown')}")
            if data.offers:
                lines.append(f"- 报价方案: {json.dumps(data.offers[:3], ensure_ascii=False)[:800]}")
            lines.append(f"- 优势: {data.strengths[:200]}")
            lines.append(f"- 劣势: {data.weaknesses[:200]}")
        return "\n".join(lines)

    def _parse_pricing_analysis(self, result: dict) -> PricingAnalysis:
        """解析LLM返回的定价分析结果"""
        pricing_comparison = []
        for pc in result.get("pricing_comparison", []):
            pricing_comparison.append(PricingItem(
                competitor=pc.get("competitor", ""),
                free_tier=pc.get("free_tier", ""),
                paid_tier=pc.get("paid_tier", ""),
                pricing_model=pc.get("pricing_model", ""),
            ))

        return PricingAnalysis(
            pricing_comparison=pricing_comparison,
            pricing_strategy_analysis=result.get("pricing_strategy_analysis", ""),
            value_ranking=result.get("value_ranking", []),
            summary=result.get("summary", ""),
        )

    def _rule_analyze(self, product_name: str,
                       competitors_data: dict[str, CompetitorData]) -> PricingAnalysis:
        """规则引擎定价分析"""
        import re
        scenario_text = product_name + " " + " ".join(
            f"{data.product_features} {data.pricing_info} {data.market_share} {data.user_reviews}"
            for data in competitors_data.values()
        )
        profile = detect_scenario(scenario_text)
        pricing_comparison = []
        for name, data in competitors_data.items():
            pricing_text = data.pricing_info or ""
            if data.offers:
                for offer in data.offers[:3]:
                    price = str(offer.get("price", "") or "未知")
                    included = offer.get("included", [])
                    excluded = offer.get("excluded", [])
                    pricing_comparison.append(PricingItem(
                        competitor=name,
                        free_tier="包含: " + "、".join(included[:4]) if included else "包含项需核实",
                        paid_tier=f"{price} {offer.get('unit', '')}".strip(),
                        pricing_model=self._infer_pricing_model(
                            pricing_text + " " + json.dumps(offer, ensure_ascii=False),
                            profile.pricing_dimensions,
                        ),
                    ))
            else:
                paid_tier = pricing_text[:120] if pricing_text else "未公开价格/需咨询"
                pricing_comparison.append(PricingItem(
                    competitor=name,
                    free_tier=self._extract_pricing_hint(pricing_text, ["免费", "free", "试用"]),
                    paid_tier=paid_tier,
                    pricing_model=self._infer_pricing_model(pricing_text, profile.pricing_dimensions),
                ))

        return PricingAnalysis(
            pricing_comparison=pricing_comparison,
            pricing_strategy_analysis=f"按{profile.category}场景关注: {', '.join(profile.pricing_dimensions)}",
            value_ranking=[],
            summary=f"基于{profile.category}场景维度的规则定价信息提取（建议启用LLM获得深度分析）",
        )

    @staticmethod
    def _extract_pricing_hint(text: str, keywords: list[str]) -> str:
        if not text:
            return "未知"
        lowered = text.lower()
        return "有免费/试用线索" if any(keyword.lower() in lowered for keyword in keywords) else "未发现免费线索"

    @staticmethod
    def _infer_pricing_model(text: str, dimensions: tuple[str, ...]) -> str:
        lowered = (text or "").lower()
        if "订阅" in lowered or "subscription" in lowered:
            return "订阅制"
        if "按量" in lowered or "用量" in lowered:
            return "按量计费"
        if "买断" in lowered or "硬件" in lowered:
            return "一次性购买/硬件售价"
        return "需核实：" + "、".join(dimensions[:2])

    @staticmethod
    def _format_feedback(quality_feedback: list[dict]) -> str:
        lines = [
            f"- {item.get('message', '')}"
            for item in quality_feedback
            if item.get("type") == "pricing_analysis_insufficient"
        ]
        return "\n".join(line for line in lines if line.strip())
