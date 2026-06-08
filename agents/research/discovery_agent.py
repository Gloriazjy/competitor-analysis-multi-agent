# -*- coding: utf-8 -*-
"""
agents/discovery_agent.py -- 竞品发现Agent

职责：根据用户产品描述，搜索并筛选出3~8个核心竞品
LLM调用：2次（关键词生成 + 结果筛选）
外部工具：百度AI搜索
提示词来源：prompts/discovery_agent.md
"""

from agents.base_agent import BaseAgent
from models.domain import CompetitorInfo, CompetitorList
from core.prompt_loader import load as load_prompts
from core.search_client import SearchClient
from core.scenario_profile import detect_scenario, infer_product_name
import config
import json


class DiscoveryAgent(BaseAgent):
    """竞品发现Agent -- 搜索并筛选核心竞品"""

    def __init__(self):
        prompts = load_prompts("discovery_agent")
        super().__init__(
            agent_id="DiscoveryAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_keywords = prompts["prompt_keywords"]
        self._prompt_filter = prompts["prompt_filter"]
        self.search_client = SearchClient()

    async def run(self, product_description: str,
                  max_competitors: int = config.DEFAULT_COMPETITOR_COUNT,
                  quality_feedback: list[dict] = None) -> CompetitorList:
        """
        主运行逻辑：生成搜索关键词 -> 搜索 -> 筛选竞品

        Args:
            product_description: 用户产品描述
            max_competitors: 最大竞品数量
            quality_feedback: 质检打回的竞品发现问题

        Returns:
            CompetitorList: 发现的竞品列表
        """
        self._log(f"🔍 开始发现竞品: {product_description[:50]}...")

        needs_more_competitors = False
        if quality_feedback:
            self._log(f"   收到质检反馈: {len(quality_feedback)}条")
            for feedback in quality_feedback:
                issue_type = feedback.get("type", "")
                if issue_type in ["competitor_list_empty", "competitor_count_low"]:
                    needs_more_competitors = True
                    self._log(f"   🔄 检测到竞品数量问题，将增加搜索力度")
                    break

        profile = detect_scenario(product_description)
        self._log(f"   识别场景: {profile.category}")

        # --- 步骤1: 生成搜索关键词 ---
        keywords = await self._generate_keywords(product_description, needs_more_competitors)
        self._log(f"   生成搜索关键词: {keywords}")

        # --- 步骤2: 执行搜索 ---
        search_results = await self._search(keywords, needs_more_competitors)
        self._log(f"   搜索完成，获得{len(search_results)}组结果")

        # --- 步骤3: 筛选竞品 ---
        target_count = max_competitors + 2 if needs_more_competitors else max_competitors
        competitor_list = await self._filter_competitors(
            product_description, search_results, target_count
        )

        self._log(f"✅ 发现{len(competitor_list.competitors)}个核心竞品")
        for c in competitor_list.competitors:
            self._log(f"   • {c.name} ({c.relevance}): {c.brief[:40]}...")

        return competitor_list

    async def _generate_keywords(self, product_description: str, needs_more: bool = False) -> list[str]:
        """生成搜索关键词（LLM + 规则引擎降级）"""
        if config.ENABLE_LLM:
            prompt = self._prompt_keywords.format(
                product_description=product_description,
                count=8 if needs_more else 5,
            )
            result = await self.ask_llm_json_async(prompt)
            if result and "keywords" in result:
                keywords = result["keywords"]
                self._log(f"   LLM生成关键词: {keywords}")
                return keywords[:10] if needs_more else keywords[:8]
            else:
                self._log("   LLM关键词生成失败，降级到规则引擎")

        return self._rule_keywords(product_description, needs_more)

    def _rule_keywords(self, product_description: str, needs_more: bool = False) -> list[str]:
        """规则引擎生成搜索关键词"""
        name = infer_product_name(product_description)
        profile = detect_scenario(product_description)
        keywords = [
            f"{name}竞品分析",
            f"{name}替代产品",
            f"{name}同类产品对比",
            f"类似{name}的产品",
            f"{name}竞争对手",
            f"{name}市场份额",
        ]
        for modifier in profile.search_modifiers[:6] if needs_more else profile.search_modifiers[:4]:
            keywords.append(f"{name} {modifier}")
        keywords.append(f"{profile.category} 主流产品 排名")
        keywords.append(f"{profile.category} 品牌排行榜")
        return list(dict.fromkeys(keywords))[:10] if needs_more else list(dict.fromkeys(keywords))[:8]

    async def _search(self, keywords: list[str], needs_more: bool = False) -> list[dict]:
        """执行搜索"""
        results = await self.search_client.batch_search_async(keywords)
        if needs_more and len(results) < len(keywords):
            self._log("   搜索结果不足，补充搜索...")
            additional_keywords = [f"{k} 官网" for k in keywords[:3]]
            additional_results = await self.search_client.batch_search_async(additional_keywords)
            results.extend(additional_results)
        return results

    async def _filter_competitors(self, product_description: str,
                                  search_results: list[dict],
                                  max_competitors: int) -> CompetitorList:
        """筛选核心竞品（LLM + 规则引擎降级）"""
        all_text = ""
        for sr in search_results:
            query = sr.get("query", "")
            result = sr.get("result")
            text = SearchClient.extract_text(result) if result else ""
            if text:
                all_text += f"\n--- 搜索: {query} ---\n{text[:1000]}\n"

        if config.ENABLE_LLM and all_text:
            prompt = self._prompt_filter.format(
                product_description=product_description,
                search_results=all_text[:8000] if len(all_text) > 6000 else all_text[:6000],
                max_competitors=max_competitors,
            )
            result = await self.ask_llm_json_async(prompt, max_tokens=4096)
            if result and "competitors" in result:
                competitors = []
                for c in result["competitors"]:
                    competitors.append(CompetitorInfo(
                        name=c.get("name", ""),
                        brief=c.get("brief", ""),
                        relevance=c.get("relevance", "MEDIUM"),
                    ))
                return CompetitorList(
                    product_name=result.get("product_name", product_description),
                    product_category=result.get("product_category", "") or detect_scenario(product_description).category,
                    competitors=competitors[:max_competitors],
                    search_keywords_used=[sr.get("query", "") for sr in search_results],
                )
            else:
                self._log("   LLM筛选失败，降级到规则引擎")

        return self._rule_filter(product_description, search_results, max_competitors)

    def _rule_filter(self, product_description: str,
                     search_results: list[dict],
                     max_competitors: int) -> CompetitorList:
        """规则引擎筛选竞品（从搜索文本中提取产品名 + 预设领域库）"""
        import re
        competitors = []
        seen_names = set()
        product_name = infer_product_name(product_description)
        seen_names.add(product_name)
        profile = detect_scenario(product_description)

        if profile.competitor_candidates:
            self._log(f"   📚 匹配场景库: {profile.category}")
            candidate_count = max_competitors + 2 if len(profile.competitor_candidates) > max_competitors else max_competitors
            for name in profile.competitor_candidates[:candidate_count]:
                if name not in seen_names:
                    seen_names.add(name)
                    competitors.append(CompetitorInfo(
                        name=name,
                        brief=f"{profile.category}主流竞品",
                        relevance="HIGH",
                    ))

        if len(competitors) < max_competitors:
            self._log("   从搜索结果补充提取...")
            for sr in search_results:
                result = sr.get("result")
                if not result:
                    continue
                text = SearchClient.extract_text(result)
                if not text:
                    continue

                name_patterns = re.findall(r'[《「]([^」》]+)[」》]', text)
                for name in name_patterns:
                    name = name.strip()
                    if name and name not in seen_names and len(name) < 30:
                        seen_names.add(name)
                        competitors.append(CompetitorInfo(
                            name=name,
                            brief=f"从搜索结果中发现的相关产品",
                            relevance="MEDIUM",
                        ))
                        if len(competitors) >= max_competitors:
                            break
                if len(competitors) >= max_competitors:
                    break

        if not competitors:
            self._log("   ⚠️ 未发现竞品，使用通用模式生成")
            for i in range(1, max_competitors + 1):
                name = f"竞品{i}"
                competitors.append(CompetitorInfo(
                    name=name,
                    brief="同类产品中的主要竞争对手",
                    relevance="MEDIUM",
                ))

        return CompetitorList(
            product_name=product_name,
            product_category=profile.category,
            competitors=competitors[:max_competitors],
            search_keywords_used=[sr.get("query", "") for sr in search_results],
        )
