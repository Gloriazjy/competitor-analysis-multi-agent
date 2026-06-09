# -*- coding: utf-8 -*-
"""
agents/product_agent.py — 产品分析Agent

职责：逐竞品对比功能矩阵，标注优势/劣势/差异点
LLM调用：1次
外部工具：无
提示词来源：prompts/product_agent.md
"""

from agents.base_agent import BaseAgent
from models.domain import CompetitorData, ProductAnalysis, FeatureComparison, CompetitiveAdvantage
from core.prompt_loader import load as load_prompts
from core.scenario_profile import detect_scenario
import config
import json


class ProductAgent(BaseAgent):
    """产品分析Agent — 功能对比矩阵"""

    def __init__(self):
        prompts = load_prompts("product_agent")
        super().__init__(
            agent_id="ProductAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_analyze = prompts["prompt_analyze"]

    async def run(self, product_name: str,
                  competitors_data: dict[str, CompetitorData],
                  quality_feedback: list[dict] = None) -> ProductAnalysis:
        """
        主运行逻辑：全量数据分析产品对比

        Args:
            product_name: 用户产品名称
            competitors_data: 竞品采集数据
            quality_feedback: 质检打回的产品分析问题

        Returns:
            ProductAnalysis: 产品分析结果
        """
        self._log("🔧 开始产品分析...")
        if quality_feedback:
            self._log(f"   收到质检反馈: {len(quality_feedback)}条")

        # 构建竞品数据摘要
        competitors_text = self._build_competitors_text(product_name, competitors_data)

        if config.ENABLE_LLM:
            prompt = self._prompt_analyze.format(
                product_name=product_name,
                competitors_text=competitors_text,
            )
            feedback_text = self._format_feedback(quality_feedback or [])
            if feedback_text:
                prompt += f"\n\n## 质检返工要求\n{feedback_text}\n请只输出采集材料能支撑的产品分析结论。"
            result = self.ask_llm_json(prompt, max_tokens=4096)
            if result:
                analysis = self._parse_product_analysis(result)
                self._log(f"✅ 产品分析完成: {len(analysis.feature_matrix)}个功能维度, "
                          f"{len(analysis.differentiation_points)}个差异点")
                return analysis
            else:
                self._log("⚠️ LLM产品分析失败，降级到规则引擎")

        # Fallback: 规则引擎分析
        return self._rule_analyze(product_name, competitors_data)

    def _build_competitors_text(self, product_name: str,
                                 competitors_data: dict[str, CompetitorData]) -> str:
        """构建竞品数据文本"""
        lines = []
        for name, data in competitors_data.items():
            label = name if name != product_name else f"{name}(我方产品)"
            lines.append(f"\n### {label}")
            lines.append(f"- 产品功能: {data.product_features[:300]}")
            lines.append(f"- 优势: {data.strengths[:200]}")
            lines.append(f"- 劣势: {data.weaknesses[:200]}")
        return "\n".join(lines)

    def _parse_product_analysis(self, result: dict) -> ProductAnalysis:
        """解析LLM返回的产品分析结果"""
        feature_matrix = []
        for fm in result.get("feature_matrix", []):
            feature_matrix.append(FeatureComparison(
                feature=fm.get("feature", ""),
                values=fm.get("values", {}),
            ))

        advantages = []
        for adv in result.get("competitive_advantages", []):
            advantages.append(CompetitiveAdvantage(
                competitor=adv.get("competitor", ""),
                our_advantage=adv.get("our_advantage", ""),
                their_advantage=adv.get("their_advantage", ""),
            ))

        return ProductAnalysis(
            feature_matrix=feature_matrix,
            competitive_advantages=advantages,
            differentiation_points=result.get("differentiation_points", []),
            summary=result.get("summary", ""),
        )

    def _rule_analyze(self, product_name: str,
                       competitors_data: dict[str, CompetitorData]) -> ProductAnalysis:
        """规则引擎产品分析"""
        scenario_text = product_name + " " + " ".join(
            f"{data.product_features} {data.pricing_info} {data.market_share} {data.user_reviews}"
            for data in competitors_data.values()
        )
        profile = detect_scenario(scenario_text)
        feature_keywords = self._build_feature_keywords(profile.product_dimensions)

        feature_matrix = []
        for feature, keywords in feature_keywords.items():
            values = {}
            # 先检查我方产品（使用产品描述+竞品数据中标注为"我方"的信息）
            product_text = product_name.lower()
            for name, data in competitors_data.items():
                product_text += f" {data.product_features} {data.strengths}".lower()
            if any(kw.lower() in product_text for kw in keywords):
                values[product_name] = "✅"
            else:
                values[product_name] = "❌"
            # 再检查每个竞品
            for name, data in competitors_data.items():
                text = f"{data.product_features} {data.strengths}".lower()
                if any(kw.lower() in text for kw in keywords):
                    values[name] = "✅"
                else:
                    values[name] = "❌"
            feature_matrix.append(FeatureComparison(feature=feature, values=values))

        return ProductAnalysis(
            feature_matrix=feature_matrix,
            competitive_advantages=[],
            differentiation_points=[f"{profile.category}场景下需重点验证: {profile.product_dimensions[0]}"],
            summary=f"基于{profile.category}场景维度的规则产品对比（建议启用LLM获得深度分析）",
        )

    @staticmethod
    def _build_feature_keywords(dimensions: tuple[str, ...]) -> dict[str, list[str]]:
        feature_keywords = {}
        for dimension in dimensions:
            words = [dimension]
            if "AI" in dimension or "智能" in dimension:
                words.extend(["AI", "智能", "助手", "算法"])
            if "价格" in dimension or "定价" in dimension:
                words.extend(["价格", "收费", "订阅"])
            if "用户" in dimension or "体验" in dimension:
                words.extend(["用户", "体验", "评价"])
            if "安全" in dimension or "合规" in dimension:
                words.extend(["安全", "合规", "权限"])
            if "生态" in dimension or "集成" in dimension:
                words.extend(["生态", "集成", "接口", "插件"])
            feature_keywords[dimension] = list(dict.fromkeys(words))
        return feature_keywords

    @staticmethod
    def _format_feedback(quality_feedback: list[dict]) -> str:
        relevant = {
            "product_analysis_insufficient",
            "product_claim_unsupported",
        }
        lines = [
            f"- {item.get('message', '')}"
            for item in quality_feedback
            if item.get("type") in relevant
        ]
        return "\n".join(line for line in lines if line.strip())
