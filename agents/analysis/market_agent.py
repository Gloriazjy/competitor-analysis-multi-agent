# -*- coding: utf-8 -*-
"""
agents/market_agent.py — 市场分析Agent

职责：分析市场份额、增长趋势、用户口碑、渠道策略
LLM调用：1次
外部工具：无
提示词来源：prompts/market_agent.md
"""

from agents.base_agent import BaseAgent
from models.domain import CompetitorData, MarketAnalysis, MarketShareItem, UserReputation
from core.prompt_loader import load as load_prompts
from core.scenario_profile import detect_scenario
import config
import json


class MarketAgent(BaseAgent):
    """市场分析Agent — 市场格局与趋势"""

    def __init__(self):
        prompts = load_prompts("market_agent")
        super().__init__(
            agent_id="MarketAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_analyze = prompts["prompt_analyze"]

    async def run(self, product_name: str,
                  competitors_data: dict[str, CompetitorData],
                  quality_feedback: list[dict] = None) -> MarketAnalysis:
        """
        主运行逻辑：全量数据分析市场格局

        Args:
            product_name: 用户产品名称
            competitors_data: 竞品采集数据
            quality_feedback: 质检打回的市场分析问题

        Returns:
            MarketAnalysis: 市场分析结果
        """
        self._log("📈 开始市场分析...")
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
                prompt += f"\n\n## 质检返工要求\n{feedback_text}\n请补齐市场份额、趋势和用户口碑维度，并避免无来源判断。"
            result = self.ask_llm_json(prompt, max_tokens=4096)
            if result:
                analysis = self._parse_market_analysis(result)
                self._log(f"✅ 市场分析完成: {len(analysis.market_share_data)}个竞品市场数据")
                return analysis
            else:
                self._log("⚠️ LLM市场分析失败，降级到规则引擎")

        return self._rule_analyze(product_name, competitors_data)

    def _build_competitors_text(self, product_name: str,
                                 competitors_data: dict[str, CompetitorData]) -> str:
        """构建竞品市场数据文本"""
        lines = []
        for name, data in competitors_data.items():
            label = name if name != product_name else f"{name}(我方产品)"
            lines.append(f"\n### {label}")
            lines.append(f"- 市场份额: {data.market_share[:300]}")
            lines.append(f"- 市场份额采集状态: {(data.field_status or {}).get('market_share', 'unknown')}")
            lines.append(f"- 用户评价: {data.user_reviews[:300]}")
            lines.append(f"- 用户评价采集状态: {(data.field_status or {}).get('user_reviews', 'unknown')}")
            lines.append(f"- 渠道策略: {data.channels[:200]}")
            if data.evidence_notes:
                lines.append(f"- 证据摘要: {'; '.join(data.evidence_notes[:2])[:300]}")
        return "\n".join(lines)

    def _parse_market_analysis(self, result: dict) -> MarketAnalysis:
        """解析LLM返回的市场分析结果"""
        market_share_data = []
        for ms in result.get("market_share_data", []):
            market_share_data.append(MarketShareItem(
                competitor=ms.get("competitor", ""),
                share_estimate=ms.get("share_estimate", ""),
                trend=ms.get("trend", ""),
            ))

        user_reputation = {}
        for name, rep in result.get("user_reputation", {}).items():
            user_reputation[name] = UserReputation(
                score=rep.get("score", ""),
                keywords=rep.get("keywords", []),
            )

        return MarketAnalysis(
            market_share_data=market_share_data,
            growth_trends=result.get("growth_trends", ""),
            user_reputation=user_reputation,
            channel_analysis=result.get("channel_analysis", ""),
            summary=result.get("summary", ""),
        )

    def _rule_analyze(self, product_name: str,
                       competitors_data: dict[str, CompetitorData]) -> MarketAnalysis:
        """规则引擎市场分析"""
        scenario_text = product_name + " " + " ".join(
            f"{data.product_features} {data.pricing_info} {data.market_share} {data.user_reviews}"
            for data in competitors_data.values()
        )
        profile = detect_scenario(scenario_text)
        market_share_data = []
        user_reputation = {}
        for name, data in competitors_data.items():
            market_share_data.append(MarketShareItem(
                competitor=name,
                share_estimate=data.market_share[:100] if data.market_share else "无公开份额，需用替代指标评估",
                trend=self._infer_trend(data.market_share + " " + data.user_reviews + " " + " ".join(data.risk_flags)),
            ))
            if data.user_reviews or data.risk_flags or (data.field_status or {}).get("user_reviews") in {"not_found", "not_public"}:
                user_reputation[name] = UserReputation(
                    score="需核实" if data.user_reviews or data.risk_flags else "未发现可验证公开评论",
                    keywords=self._extract_reputation_keywords(data.user_reviews + " " + " ".join(data.risk_flags)),
                )

        return MarketAnalysis(
            market_share_data=market_share_data,
            growth_trends=f"按{profile.category}场景关注: {', '.join(profile.market_dimensions)}",
            user_reputation=user_reputation,
            channel_analysis=self._build_channel_analysis(competitors_data),
            summary=f"基于{profile.category}场景维度的规则市场信息提取（建议启用LLM获得深度分析）",
        )

    @staticmethod
    def _infer_trend(text: str) -> str:
        if any(word in text for word in ("增长", "上升", "热门", "领先")):
            return "增长/领先线索"
        if any(word in text for word in ("下降", "下滑", "投诉", "流失")):
            return "下滑/风险线索"
        return "未知"

    @staticmethod
    def _extract_reputation_keywords(text: str) -> list[str]:
        candidates = ("好评", "差评", "易用", "贵", "稳定", "卡顿", "服务", "体验", "安全", "续航")
        return [word for word in candidates if word in text][:5]

    @staticmethod
    def _build_channel_analysis(competitors_data: dict[str, CompetitorData]) -> str:
        parts = []
        for name, data in competitors_data.items():
            channels = []
            if data.channels:
                channels.append(data.channels[:60])
            if data.contact_methods:
                channels.append("有联系方式")
            if data.source_urls:
                channels.append("有可跳转来源")
            if channels:
                parts.append(f"{name}: {'; '.join(channels)}")
        return "\n".join(parts) if parts else "需结合官网、电商/应用商店、行业榜单继续核实"

    @staticmethod
    def _format_feedback(quality_feedback: list[dict]) -> str:
        lines = [
            f"- {item.get('message', '')}"
            for item in quality_feedback
            if item.get("type") == "market_analysis_insufficient"
        ]
        return "\n".join(line for line in lines if line.strip())
