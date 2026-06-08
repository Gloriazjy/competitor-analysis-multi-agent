# -*- coding: utf-8 -*-
"""
agents/collection_agent.py — 数据采集Agent

职责：对每个竞品，采集产品功能、定价、用户评价、市场份额等信息
LLM调用：1+N次（维度拆解 + 逐竞品汇总）
外部工具：百度AI搜索
提示词来源：prompts/collection_agent.md
"""

from agents.base_agent import BaseAgent
from models.domain import CompetitorList, CompetitorData
from core.prompt_loader import load as load_prompts
from core.search_client import SearchClient
from core.scenario_profile import detect_scenario
import config
import asyncio
import json
import re


class CollectionAgent(BaseAgent):
    """数据采集Agent — 逐竞品深度采集"""

    def __init__(self):
        prompts = load_prompts("collection_agent")
        super().__init__(
            agent_id="CollectionAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_collect = prompts["prompt_collect"]
        self.search_client = SearchClient()

    async def run(self, product_description: str,
                  competitor_list: CompetitorList,
                  quality_feedback: list[dict] = None) -> dict[str, CompetitorData]:
        """
        主运行逻辑：逐竞品搜索+汇总

        Args:
            product_description: 用户产品描述
            competitor_list: 竞品列表
            quality_feedback: 质检打回的采集问题

        Returns:
            dict[str, CompetitorData]: 竞品名称 → 采集数据
        """
        self._log(f"📊 开始采集数据，共{len(competitor_list.competitors)}个竞品")
        if quality_feedback:
            self._log(f"   收到质检反馈: {len(quality_feedback)}条")

        product_name = competitor_list.product_name
        competitors = list(competitor_list.competitors)
        semaphore = asyncio.Semaphore(max(1, config.COLLECTION_COMPETITOR_CONCURRENCY))

        async def collect_one(index, comp):
            async with semaphore:
                self._log(f"   采集 {index+1}/{len(competitors)}: {comp.name}")
                data = await self._collect_competitor(
                    product_name,
                    product_description,
                    comp.name,
                    quality_feedback=quality_feedback or [],
                )
                return comp.name, data

        pairs = await asyncio.gather(
            *(collect_one(i, comp) for i, comp in enumerate(competitors))
        )
        result_data = dict(pairs)

        self._log(f"✅ 数据采集完成: {len(result_data)}个竞品")
        return result_data

    async def _collect_competitor(self, product_name: str,
                                  product_description: str,
                                  competitor_name: str,
                                  quality_feedback: list[dict] = None) -> CompetitorData:
        """采集单个竞品数据"""
        # 生成搜索查询
        profile = detect_scenario(product_description)
        queries = [
            f"{competitor_name} 产品功能介绍",
            f"{competitor_name} 定价 价格 收费标准",
            f"{competitor_name} 市场份额 用户量 评测",
            f"{competitor_name} vs {product_name} 对比",
        ]
        for source_query in profile.source_queries[:4]:
            queries.append(f"{competitor_name} {source_query}")
        for modifier in profile.search_modifiers[:3]:
            queries.append(f"{competitor_name} {modifier}")
        for feedback in quality_feedback or []:
            evidence = feedback.get("evidence", {})
            if evidence.get("competitor") and evidence["competitor"] != competitor_name:
                continue
            if feedback.get("type") == "source_insufficient":
                queries.extend([
                    f"{competitor_name} 官网 产品 定价",
                    f"{competitor_name} 用户评价 真实评测",
                ])
            elif feedback.get("type") in {"competitor_data_incomplete", "competitor_data_missing"}:
                queries.extend([
                    f"{competitor_name} 功能 定价 市场 用户评价",
                    f"{competitor_name} 产品文档 收费 用户口碑",
                ])

        # 执行搜索
        queries = list(dict.fromkeys(queries))
        search_results = await self.search_client.batch_search_async(queries)

        # 提取搜索文本
        all_text = ""
        sources = []
        source_urls = []
        for sr in search_results:
            query = sr.get("query", "")
            result = sr.get("result")
            text = SearchClient.extract_text(result) if result else ""
            urls = SearchClient.extract_urls(result) if result else []
            if text:
                all_text += f"\n--- 搜索: {query} ---\n{text[:1500]}\n"
                sources.append(text[:500])
            source_urls.extend(urls)

        # LLM汇总提取
        if config.ENABLE_LLM and all_text:
            feedback_text = self._format_feedback(quality_feedback or [], competitor_name)
            prompt = self._prompt_collect.format(
                product_name=product_name,
                product_description=product_description,
                competitor_name=competitor_name,
                search_results=all_text[:8000],
            )
            if feedback_text:
                prompt += f"\n\n## 质检返工要求\n{feedback_text}\n请优先补齐上述缺口，避免再次输出无来源或空字段。"
            prompt += self._build_universal_extract_instruction(profile)
            result = await self.ask_llm_json_async(prompt, max_tokens=4096)
            if result:
                return CompetitorData(
                    name=competitor_name,
                    category=profile.category,
                    product_features=result.get("product_features", ""),
                    pricing_info=result.get("pricing_info", ""),
                    market_share=result.get("market_share", ""),
                    user_reviews=result.get("user_reviews", ""),
                    strengths=result.get("strengths", ""),
                    weaknesses=result.get("weaknesses", ""),
                    channels=result.get("channels", ""),
                    offers=result.get("offers", []),
                    contact_methods=result.get("contact_methods", []),
                    source_urls=result.get("source_urls", []) or source_urls,
                    evidence_notes=result.get("evidence_notes", []),
                    risk_flags=result.get("risk_flags", []),
                    search_sources=sources,
                    field_status=self._normalize_field_status(result.get("field_status", {}), result, sources),
                )
            else:
                self._log(f"   ⚠️ {competitor_name} LLM汇总失败，降级到规则引擎")

        # Fallback: 规则引擎提取
        offers = self._extract_offers(all_text, profile)
        field_values = {
            "product_features": all_text[:500] if all_text else "",
            "pricing_info": "; ".join(item.get("price", "") for item in offers if item.get("price")),
            "market_share": self._extract_market_proxy(all_text),
            "user_reviews": self._extract_review_summary(all_text),
        }
        return CompetitorData(
            name=competitor_name,
            category=profile.category,
            product_features=field_values["product_features"] or "数据采集失败",
            pricing_info=field_values["pricing_info"] or ("咨询报价/未公开价格" if offers else ""),
            market_share=field_values["market_share"],
            user_reviews=field_values["user_reviews"],
            strengths=self._extract_strengths(all_text, profile),
            weaknesses="; ".join(self._extract_risk_flags(all_text, profile)),
            channels="; ".join(self._extract_contacts(all_text)),
            offers=offers,
            contact_methods=self._extract_contacts(all_text),
            source_urls=list(dict.fromkeys(source_urls)),
            evidence_notes=sources[:3],
            risk_flags=self._extract_risk_flags(all_text, profile),
            search_sources=sources,
            field_status=self._infer_field_status(field_values, sources, offers),
        )

    @staticmethod
    def _format_feedback(quality_feedback: list[dict], competitor_name: str) -> str:
        lines = []
        for item in quality_feedback:
            evidence = item.get("evidence", {})
            if evidence.get("competitor") and evidence["competitor"] != competitor_name:
                continue
            lines.append(f"- {item.get('message', '')}")
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _build_universal_extract_instruction(profile) -> str:
        return f"""

## 通用可比较对象抽取要求
请在原有JSON字段外，尽量补充以下字段，字段缺失时返回空数组或空字符串，不要编造：
- offers: 数组，每项包含 name、price、currency、unit、included、excluded、constraints、booking_url、contact、evidence
- contact_methods: 数组，报名/购买/销售/客服联系方式或可联系入口
- source_urls: 数组，原始来源URL
- evidence_notes: 数组，支撑价格、服务、口碑、风险判断的证据摘要
- risk_flags: 数组，隐形消费、口碑争议、退款限制、价格口径不清等风险
- field_status: 对象，必须包含 product_features、pricing_info、market_share、user_reviews 四个键；
  值只能是 found、not_public、not_found、proxy_available。公开资料未披露时请诚实标记，不要编造。
本场景为：{profile.category}
重点比较维度：{", ".join(profile.offer_dimensions)}
重点质检风险：{", ".join(profile.quality_risks)}
"""

    @classmethod
    def _normalize_field_status(cls, status: dict, result: dict, sources: list[str]) -> dict[str, str]:
        values = {
            "product_features": result.get("product_features", ""),
            "pricing_info": result.get("pricing_info", ""),
            "market_share": result.get("market_share", ""),
            "user_reviews": result.get("user_reviews", ""),
        }
        offers = result.get("offers", [])
        normalized = cls._infer_field_status(values, sources, offers)
        allowed = {"found", "not_public", "not_found", "proxy_available"}
        for key in normalized:
            value = str((status or {}).get(key, "")).strip()
            if value in allowed:
                normalized[key] = value
        return normalized

    @staticmethod
    def _infer_field_status(values: dict[str, str], sources: list[str], offers: list[dict]) -> dict[str, str]:
        has_sources = bool(sources)
        pricing_text = values.get("pricing_info", "")
        market_text = values.get("market_share", "")
        review_text = values.get("user_reviews", "")
        return {
            "product_features": "found" if values.get("product_features") else ("not_found" if has_sources else "not_public"),
            "pricing_info": "found" if pricing_text or offers else ("not_public" if has_sources else "not_found"),
            "market_share": "found" if market_text and "替代指标" not in market_text else ("proxy_available" if market_text else ("not_public" if has_sources else "not_found")),
            "user_reviews": "found" if review_text else ("not_found" if has_sources else "not_public"),
        }

    @staticmethod
    def _extract_market_proxy(text: str) -> str:
        if not text:
            return ""
        keywords = ("用户量", "下载量", "销量", "门店", "覆盖", "排名", "融资", "营收", "GMV", "订单")
        snippets = []
        for sentence in re.split(r"[。！？\n]", text):
            clean = sentence.strip()
            if clean and any(keyword in clean for keyword in keywords):
                snippets.append(clean[:120])
            if len(snippets) >= 3:
                break
        return "替代指标: " + "；".join(snippets) if snippets else ""

    @staticmethod
    def _extract_contacts(text: str) -> list[str]:
        if not text:
            return []
        patterns = [
            r"1[3-9]\d{9}",
            r"(?:电话|热线|客服|联系|微信|WhatsApp|邮箱|Email)[:：]?\s*[A-Za-z0-9_@.+\-]+",
        ]
        contacts = []
        for pattern in patterns:
            contacts.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        return list(dict.fromkeys(str(item).strip() for item in contacts if str(item).strip()))[:8]

    @staticmethod
    def _extract_offers(text: str, profile) -> list[dict]:
        if not text:
            return []
        price_patterns = re.findall(
            r"(?:¥|￥|CNY|RMB)?\s?(\d{2,6}(?:\.\d+)?)\s?(?:元|块|/人|每人|起|RMB|CNY)?",
            text,
            flags=re.IGNORECASE,
        )
        offers = []
        for price in list(dict.fromkeys(price_patterns))[:5]:
            offers.append({
                "name": "公开报价线索",
                "price": price,
                "currency": "CNY",
                "unit": "需核实",
                "included": [],
                "excluded": [],
                "constraints": [],
                "booking_url": "",
                "contact": "",
                "evidence": f"搜索文本出现价格线索: {price}",
            })
        if not offers and any(word in text for word in ("电询", "咨询", "联系客服", "报价")):
            offers.append({
                "name": "咨询报价",
                "price": "电询/咨询",
                "currency": "",
                "unit": "",
                "included": [],
                "excluded": [],
                "constraints": [],
                "booking_url": "",
                "contact": "",
                "evidence": "搜索文本仅出现咨询报价线索",
            })
        return offers

    @staticmethod
    def _extract_risk_flags(text: str, profile) -> list[str]:
        risks = []
        for risk in profile.quality_risks:
            key = risk[:2]
            if key and key in text:
                risks.append(risk)
        generic = {
            "隐形消费": ("隐形消费", "强制购物", "自费", "另付"),
            "价格口径不清": ("起", "电询", "咨询", "价格以"),
            "退款规则需核实": ("退款", "退改", "取消"),
            "口碑需核实": ("差评", "投诉", "避雷"),
        }
        for label, words in generic.items():
            if any(word in text for word in words):
                risks.append(label)
        return list(dict.fromkeys(risks))[:8]

    @staticmethod
    def _extract_review_summary(text: str) -> str:
        if not text:
            return ""
        review_words = ("好评", "差评", "评价", "口碑", "投诉", "推荐", "避雷")
        snippets = [line.strip() for line in text.splitlines() if any(word in line for word in review_words)]
        return "\n".join(snippets[:5])

    @staticmethod
    def _extract_strengths(text: str, profile) -> str:
        if not text:
            return ""
        hits = [dimension for dimension in profile.offer_dimensions if dimension and dimension in text]
        return "覆盖维度: " + "、".join(hits) if hits else ""
