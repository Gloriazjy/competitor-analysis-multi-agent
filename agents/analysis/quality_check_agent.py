# -*- coding: utf-8 -*-
"""
agents/quality_check_agent.py — 质检Agent

职责：对采集和分析结果进行质量检测，识别缺失数据/低质量内容
返回质检结果：是否通过、质量分数、问题列表、目标回退节点
"""

from agents.base_agent import BaseAgent
from core.graph_state import GraphState
from core.prompt_loader import load as load_prompts
from core.type_utils import (
    to_product_analysis,
    to_pricing_analysis,
    to_market_analysis,
)
from core.scenario_profile import detect_scenario
import config
import time


class QualityCheckAgent(BaseAgent):
    """数据与分析质量审核Agent

    双层质检：
      第1层 规则引擎：结构完整性（字段/来源/数量），零成本快速过滤。
      第2层 LLM事实核查：把采集原文与分析结论一起交给LLM，识别幻觉/编造/矛盾，
                          产出"无信源支撑"类问题并据此打回。LLM不可用时自动跳过，
                          只用第1层，不影响原流程。
    """

    MIN_SOURCE_COUNT = 2
    MIN_COMPETITOR_COUNT = 3
    MIN_FEATURE_COUNT = 3
    MIN_PRICING_ITEM_COUNT = 2
    MIN_MARKET_ITEM_COUNT = 2
    # 事实核查得分低于此阈值视为"分析脱离材料"，强制打回
    FACTUAL_FAIL_THRESHOLD = 0.6

    def __init__(self):
        prompts = load_prompts("quality_check_agent")
        super().__init__(
            agent_id="QualityCheckAgent",
            system_prompt=prompts["system_prompt"],
        )
        self._prompt_fact_check = prompts["prompt_fact_check"]

    def _safe_convert_competitor_data(self, cd):
        """安全转换：如果LangGraph把dataclass变成dict或字符串，就转回来"""
        from models.domain import CompetitorData

        # 情况1: 已经是 CompetitorData 对象
        if isinstance(cd, CompetitorData):
            return cd

        # 情况2: 字符串形式（LangGraph序列化可能把dict变成字符串）
        if isinstance(cd, str):
            cd = cd.strip()
            # 尝试解析JSON字符串
            if cd.startswith("{"):
                try:
                    import json
                    cd = json.loads(cd)
                except:
                    print(f"⚠️  无法解析字符串: {cd[:50]}...")
                    return CompetitorData(name="未知竞品")
            else:
                # 纯字符串可能是竞品名
                return CompetitorData(name=cd)

        # 情况3: 字典形式
        if isinstance(cd, dict):
            return CompetitorData(
                name=cd.get("name", ""),
                category=cd.get("category", ""),
                product_features=cd.get("product_features", ""),
                pricing_info=cd.get("pricing_info", ""),
                market_share=cd.get("market_share", ""),
                user_reviews=cd.get("user_reviews", ""),
                strengths=cd.get("strengths", ""),
                weaknesses=cd.get("weaknesses", ""),
                channels=cd.get("channels", ""),
                offers=cd.get("offers", []),
                contact_methods=cd.get("contact_methods", []),
                source_urls=cd.get("source_urls", []),
                evidence_notes=cd.get("evidence_notes", []),
                risk_flags=cd.get("risk_flags", []),
                search_sources=cd.get("search_sources", []),
                field_status=cd.get("field_status", {})
            )

        # 默认情况
        return CompetitorData(name="未知竞品")

    def _normalize_competitor_data_list(self, raw_data) -> list:
        """兼容采集接口的 dict 输出和 LangGraph 序列化后的 list 输出。"""
        if isinstance(raw_data, dict):
            raw_items = raw_data.values()
        else:
            raw_items = raw_data or []
        return [self._safe_convert_competitor_data(cd) for cd in raw_items]

    def _safe_convert_competitor_list(self, cl):
        if isinstance(cl, dict):
            from models.domain import CompetitorList, CompetitorInfo
            comps = []
            for c in cl.get("competitors", []):
                if isinstance(c, dict):
                    comps.append(CompetitorInfo(
                        name=c.get("name", ""),
                        brief=c.get("brief", ""),
                        relevance=c.get("relevance", "MEDIUM")
                    ))
                else:
                    comps.append(c)
            return CompetitorList(
                product_name=cl.get("product_name", ""),
                product_category=cl.get("product_category", ""),
                competitors=comps,
                search_keywords_used=cl.get("search_keywords_used", [])
            )
        return cl

    @staticmethod
    def _is_blank(value: str) -> bool:
        return not value or not str(value).strip()

    @staticmethod
    def _contains_any(text: str, words: list[str]) -> bool:
        haystack = (text or "").lower()
        return any(word.lower() in haystack for word in words if word)

    @staticmethod
    def _field_status(cd, field: str) -> str:
        return str((getattr(cd, "field_status", {}) or {}).get(field, "")).strip()

    @classmethod
    def _is_unavailable(cls, cd, field: str) -> bool:
        return cls._field_status(cd, field) in {"not_public", "not_found", "proxy_available"}

    @staticmethod
    def _issue(issue_type: str, severity: str, message: str,
               rollback_to: str, evidence: dict = None) -> dict:
        return {
            "type": issue_type,
            "severity": severity,
            "message": message,
            "rollback_to": rollback_to,
            "evidence": evidence or {},
        }

    @staticmethod
    def _select_rollback_target(issues: list[dict]) -> str:
        """按最高影响面选择一个回退目标：发现 > 采集 > 分析。"""
        if not issues:
            return ""
        for issue in issues:
            if issue.get("severity") == "critical" and issue.get("rollback_to") == "discovery":
                return "discovery"
        factual_types = {"unsupported_claim", "contradiction", "fabricated_data", "factual_score_low"}
        for issue in issues:
            if issue.get("severity") == "high" and issue.get("type") in factual_types:
                return "parallel_analysis"
        priority = {
            "discovery": 0,
            "collection": 1,
            "parallel_analysis": 2,
            "product_analysis": 2,
            "pricing_analysis": 2,
            "market_analysis": 2,
        }
        target = min(
            (issue.get("rollback_to", "collection") for issue in issues),
            key=lambda item: priority.get(item, 9),
        )
        if target in {"product_analysis", "pricing_analysis", "market_analysis"}:
            return "parallel_analysis"
        return target

    @staticmethod
    def _build_feedback(issues: list[dict], rollback_target: str) -> list[dict]:
        feedback = []
        for issue in issues:
            target = issue.get("rollback_to", "")
            if target in {"product_analysis", "pricing_analysis", "market_analysis"}:
                target = "parallel_analysis"
            if target == rollback_target:
                feedback.append({
                    "type": issue.get("type", ""),
                    "severity": issue.get("severity", ""),
                    "message": issue.get("message", ""),
                    "evidence": issue.get("evidence", {}),
                })
        return feedback

    async def run(self, state: GraphState) -> dict:
        """
        执行质量质检

        返回质量报告，决定是否继续往下走还是回退重做
        """
        self._log("🔍 开始执行全流程质量检测...")
        issues = []
        score = 1.0

        # ── 兼容LangGraph序列化后变成dict的问题 ──
        competitor_list_raw = state.get("competitor_list")
        competitor_list = self._safe_convert_competitor_list(competitor_list_raw)

        # ── 检查1: 竞品列表有效性 ──
        if not competitor_list or not competitor_list.competitors:
            issues.append(self._issue(
                "competitor_list_empty",
                "critical",
                "竞品列表为空，需重新执行发现流程",
                "discovery",
            ))
            score -= 0.4
            self._log("   ❌ 竞品列表为空")
        else:
            competitor_count = len(competitor_list.competitors)
            self._log(f"   ✅ 竞品列表有效: {competitor_count}个竞品")
            if competitor_count < self.MIN_COMPETITOR_COUNT:
                issues.append(self._issue(
                    "competitor_count_low",
                    "high",
                    f"竞品数量不足，仅发现{competitor_count}个，建议至少{self.MIN_COMPETITOR_COUNT}个",
                    "discovery",
                    {"competitor_count": competitor_count},
                ))
                score -= 0.2

        # ── 检查2: 数据采集完整性 ──
        competitors_data_raw = state.get("competitors_data") or []
        competitors_data = self._normalize_competitor_data_list(competitors_data_raw)
        expected_count = len(competitor_list.competitors) if competitor_list else 0
        if expected_count and len(competitors_data) < expected_count:
            issues.append(self._issue(
                "competitor_data_missing",
                "high",
                f"采集结果数量不足，应有{expected_count}个，实际{len(competitors_data)}个",
                "collection",
                {"expected_count": expected_count, "actual_count": len(competitors_data)},
            ))
            score -= 0.2
        current_retry_count = state.get("retry_count", 0)
        for cd in competitors_data:
            missing_retry_fields = []
            if self._is_blank(cd.product_features) and not self._is_unavailable(cd, "product_features"):
                missing_retry_fields.append("product_features")
            if self._is_blank(cd.pricing_info) and not cd.offers and not self._is_unavailable(cd, "pricing_info"):
                missing_retry_fields.append("pricing_info")
            if self._is_blank(cd.user_reviews) and not self._is_unavailable(cd, "user_reviews"):
                missing_retry_fields.append("user_reviews")

            if missing_retry_fields:
                issues.append(self._issue(
                    "competitor_data_incomplete",
                    "high",
                    f"竞品[{cd.name}]可补采字段缺失: {', '.join(missing_retry_fields)}",
                    "collection",
                    {"competitor": cd.name, "missing_fields": missing_retry_fields},
                ))
                score -= 0.15
                self._log(f"   ⚠️  竞品[{cd.name}]可补采字段缺失: {missing_retry_fields}")

            if self._is_blank(cd.pricing_info) and not cd.offers and self._is_unavailable(cd, "pricing_info"):
                issues.append(self._issue(
                    "pricing_public_unavailable",
                    "medium",
                    f"竞品[{cd.name}]未发现公开价格，需在定价分析中按咨询报价/价格透明度处理",
                    "pricing_analysis",
                    {"competitor": cd.name, "field_status": self._field_status(cd, "pricing_info")},
                ))
                score -= 0.02

            if self._is_blank(cd.market_share) and not self._is_unavailable(cd, "market_share"):
                issues.append(self._issue(
                    "market_proxy_needed",
                    "medium",
                    f"竞品[{cd.name}]未发现市场份额，需市场分析使用用户量、渠道、排名等替代指标",
                    "market_analysis" if current_retry_count else "collection",
                    {"competitor": cd.name},
                ))
                score -= 0.06

            if self._is_blank(cd.user_reviews) and self._is_unavailable(cd, "user_reviews"):
                issues.append(self._issue(
                    "reviews_public_unavailable",
                    "low",
                    f"竞品[{cd.name}]未发现可验证公开评论，报告中需明确不编造口碑",
                    "market_analysis",
                    {"competitor": cd.name, "field_status": self._field_status(cd, "user_reviews")},
                ))
            source_count = len(cd.search_sources) + len(cd.source_urls)
            if source_count < self.MIN_SOURCE_COUNT:
                issues.append(self._issue(
                    "source_insufficient",
                    "high",
                    f"竞品[{cd.name}]来源数量不足，仅{source_count}条",
                    "collection",
                    {"competitor": cd.name, "source_count": source_count},
                ))
                score -= 0.1
                self._log(f"   ⚠️  竞品[{cd.name}]来源不足")
            profile = detect_scenario(f"{cd.category} {cd.product_features} {cd.pricing_info} {cd.channels}")
            if profile.offer_dimensions and not cd.offers and self._is_blank(cd.pricing_info) and not self._is_unavailable(cd, "pricing_info"):
                issues.append(self._issue(
                    "offer_missing",
                    "high",
                    f"竞品[{cd.name}]缺少可比较报价/套餐信息",
                    "collection" if current_retry_count == 0 else "pricing_analysis",
                    {"competitor": cd.name, "expected_offer_dimensions": list(profile.offer_dimensions)},
                ))
                score -= 0.12
            if profile.scenario_id in {"service_package", "general"} and not cd.contact_methods and self._is_blank(cd.channels):
                contact_retryable = current_retry_count == 0
                issues.append(self._issue(
                    "contact_missing" if contact_retryable else "contact_public_unavailable",
                    "medium" if contact_retryable else "low",
                    f"竞品[{cd.name}]缺少报名/购买/咨询联系方式" if contact_retryable else f"竞品[{cd.name}]未发现公开联系方式，报告中需提示人工核实",
                    "collection" if contact_retryable else "market_analysis",
                    {"competitor": cd.name},
                ))
                score -= 0.08 if contact_retryable else 0.0
            if cd.offers:
                for offer in cd.offers[:3]:
                    if not offer.get("price"):
                        issues.append(self._issue(
                            "offer_price_missing",
                            "medium",
                            f"竞品[{cd.name}]存在报价方案但价格字段缺失",
                            "collection",
                            {"competitor": cd.name, "offer": offer.get("name", "")},
                        ))
                        score -= 0.05
                        break

        # ── 检查3: 产品分析完整性 ──
        product_analysis = to_product_analysis(state.get("product_analysis"))
        if not product_analysis or len(product_analysis.feature_matrix) < self.MIN_FEATURE_COUNT:
            issues.append(self._issue(
                "product_analysis_insufficient",
                "medium",
                f"产品分析特征矩阵维度太少，需至少{self.MIN_FEATURE_COUNT}个",
                "product_analysis",
                {"feature_count": len(product_analysis.feature_matrix) if product_analysis else 0},
            ))
            score -= 0.2
            self._log("   ⚠️  产品分析特征维度不足")
        else:
            self._log(f"   ✅ 产品分析OK: {len(product_analysis.feature_matrix)}个特征")

        # ── 检查4: 定价分析完整性 ──
        pricing_analysis = to_pricing_analysis(state.get("pricing_analysis"))
        if not pricing_analysis or len(pricing_analysis.pricing_comparison) < self.MIN_PRICING_ITEM_COUNT:
            issues.append(self._issue(
                "pricing_analysis_insufficient",
                "medium",
                f"定价对比项不足，需至少{self.MIN_PRICING_ITEM_COUNT}项",
                "pricing_analysis",
                {"pricing_item_count": len(pricing_analysis.pricing_comparison) if pricing_analysis else 0},
            ))
            score -= 0.15
            self._log("   ⚠️  定价分析数据不足")
        else:
            self._log(f"   ✅ 定价分析OK: {len(pricing_analysis.pricing_comparison)}个定价项")

        # ── 检查5: 市场分析完整性 ──
        market_analysis = to_market_analysis(state.get("market_analysis"))
        if not market_analysis or len(market_analysis.market_share_data) < self.MIN_MARKET_ITEM_COUNT:
            issues.append(self._issue(
                "market_analysis_insufficient",
                "medium",
                f"市场份额数据不足，需至少{self.MIN_MARKET_ITEM_COUNT}项",
                "market_analysis",
                {"market_item_count": len(market_analysis.market_share_data) if market_analysis else 0},
            ))
            score -= 0.15
            self._log("   ⚠️  市场分析数据不足")
        else:
            self._log(f"   ✅ 市场分析OK: {len(market_analysis.market_share_data)}个市场数据")

        # ── 第2层: LLM事实核查（识别幻觉/编造/矛盾）──
        # 把采集原文与分析结论一起交给LLM判断"结论是否被原文支撑"。
        # 这是真正的事实依据打回，区别于第1层的"字段是否为空"。
        source_text = "\n".join(
            f"### {cd.name}\n"
            f"产品功能: {cd.product_features}\n定价: {cd.pricing_info}\n"
            f"市场: {cd.market_share}\n用户评价: {cd.user_reviews}\n"
            f"优势: {cd.strengths}\n劣势: {cd.weaknesses}\n渠道: {cd.channels}"
            for cd in competitors_data
        )
        factual_score = 1.0
        if config.ENABLE_LLM and competitors_data:
            factual_score, llm_issues = self._llm_fact_check(
                source_text, product_analysis, pricing_analysis, market_analysis
            )
            if llm_issues:
                issues.extend(llm_issues)
            # 事实分按权重并入总分（材料脱节比字段缺失更严重）
            score = min(score, 0.4 + 0.6 * factual_score)
            if factual_score < self.FACTUAL_FAIL_THRESHOLD:
                issues.append(self._issue(
                    "factual_score_low",
                    "high",
                    f"事实核查得分过低({factual_score:.2f})，分析结论整体缺少采集材料支撑",
                    "parallel_analysis",
                    {"factual_score": factual_score, "threshold": self.FACTUAL_FAIL_THRESHOLD},
                ))
            self._log(f"   🔬 LLM事实核查得分: {factual_score:.2f}（已并入总分）")
        else:
            # LLM不可用时降级到原有的简易子串覆盖检查
            if product_analysis and product_analysis.differentiation_points:
                unsupported = [
                    point for point in product_analysis.differentiation_points[:5]
                    if not self._contains_any(source_text, [point])
                ]
                if len(unsupported) >= 3:
                    issues.append(self._issue(
                        "product_claim_unsupported",
                        "medium",
                        "产品差异化结论缺少采集材料支撑",
                        "product_analysis",
                        {"unsupported_points": unsupported},
                    ))
                    score -= 0.1

        # ── 最终判定 ──
        score = max(0.0, min(1.0, score))
        retry_count_before = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        has_high_factual_issue = any(
            issue.get("severity") == "high"
            and issue.get("type") in {
                "unsupported_claim",
                "contradiction",
                "fabricated_data",
                "factual_score_low",
            }
            for issue in issues
        )
        # 通过条件：得分达标且无高危事实问题；或已用尽重试次数（强制降级通过）
        retries_exhausted = retry_count_before >= max_retries
        quality_ok = score >= 0.7 and not has_high_factual_issue
        passed = quality_ok or retries_exhausted
        forced = passed and not quality_ok and retries_exhausted
        retry_count = retry_count_before + 1
        rollback_target = "" if passed else self._select_rollback_target(issues)
        quality_feedback = [] if passed else self._build_feedback(issues, rollback_target)

        if forced:
            self._log(
                f"📊 质检完成: 得分 {score:.2f} / 1.0, 已达最大重试{max_retries}次 → "
                f"强制通过并标注置信度，问题数={len(issues)}"
            )
        else:
            self._log(
                f"📊 质检完成: 得分 {score:.2f} / 1.0, 通过={passed}, "
                f"回退目标={rollback_target or '无'}, 重试次数={retry_count}"
            )

        decision_log = {
            "step": "quality_check",
            "decision": "pass" if passed else "rollback",
            "rollback_target": rollback_target,
            "reason": self._build_decision_reason(score, factual_score, issues, forced),
            "issues": [issue.get("type", "") for issue in issues[:8]],
            "retry_count": retry_count,
            "forced_pass": forced,
        }

        return {
            "quality_check_passed": passed,
            "quality_score": score,
            "factual_score": factual_score,
            "quality_forced_pass": forced,
            "issues_found": issues,
            "rollback_target": rollback_target,
            "quality_feedback": quality_feedback,
            "retry_count": retry_count,
            "decision_logs": [decision_log],
            "execution_logs": [{
                "agent": "QualityCheckAgent",
                "timestamp": time.time(),
                "score": score,
                "factual_score": factual_score,
                "passed": passed,
                "forced_pass": forced,
                "issue_count": len(issues)
            }]
        }

    @staticmethod
    def _build_decision_reason(score: float, factual_score: float,
                               issues: list[dict], forced: bool) -> str:
        if forced:
            return "达到最大重试次数，保留问题并降级通过"
        if score >= 0.7 and factual_score >= 0.7:
            unavailable = [
                issue.get("type", "")
                for issue in issues
                if "unavailable" in issue.get("type", "")
            ]
            if unavailable:
                return "公开不可得字段已标注，不再重复回采"
            return "质量分和事实核查均达标"
        high = [issue.get("type", "") for issue in issues if issue.get("severity") in {"critical", "high"}]
        return "存在需回退处理的问题: " + "、".join(high[:3])

    def _llm_fact_check(self, source_text, product_analysis,
                        pricing_analysis, market_analysis) -> tuple[float, list[dict]]:
        """第2层：调用LLM核查分析结论是否被采集原文支撑。

        返回 (事实得分, 问题清单)。LLM失败时返回(1.0, [])不影响主流程。
        """
        claims_text = self._build_claims_text(
            product_analysis, pricing_analysis, market_analysis
        )
        if not claims_text.strip():
            return 1.0, []

        prompt = self._prompt_fact_check.format(
            evidence_text=source_text[:8000],
            claims_text=claims_text[:4000],
        )
        result = self.ask_llm_json(prompt, temperature=0.0, max_tokens=2048)
        if not result:
            self._log("   ⚠️ LLM事实核查无返回，跳过第2层（仅用规则层）")
            return 1.0, []

        factual_score = result.get("overall_factual_score", 1.0)
        try:
            factual_score = max(0.0, min(1.0, float(factual_score)))
        except (TypeError, ValueError):
            factual_score = 1.0

        target_map = {
            "product_analysis": "product_analysis",
            "pricing_analysis": "pricing_analysis",
            "market_analysis": "market_analysis",
            "collection": "collection",
        }
        issues = []
        for item in result.get("issues", []):
            if not isinstance(item, dict):
                continue
            issue_type = item.get("type", "unsupported_claim")
            severity = item.get("severity", "medium")
            target = target_map.get(item.get("target", ""), "parallel_analysis")
            claim = str(item.get("claim", ""))[:120]
            reason = str(item.get("reason", ""))[:200]
            issues.append(self._issue(
                issue_type,
                severity,
                f"事实核查: {reason}（涉及结论: {claim}）",
                target,
                {"claim": claim, "reason": reason, "source": "llm_fact_check"},
            ))
            self._log(f"   🚩 [{severity}] {issue_type}: {claim[:40]}")
        return factual_score, issues

    @staticmethod
    def _build_claims_text(product_analysis, pricing_analysis, market_analysis) -> str:
        """把三维分析的关键结论汇总成待核查文本。"""
        lines = []
        if product_analysis:
            if product_analysis.differentiation_points:
                lines.append("【产品-差异化结论】")
                lines.extend(f"- {p}" for p in product_analysis.differentiation_points[:6])
            for adv in (product_analysis.competitive_advantages or [])[:5]:
                lines.append(f"- 对比{adv.competitor}: 我方优势={adv.our_advantage}; 对方优势={adv.their_advantage}")
            if product_analysis.summary:
                lines.append(f"【产品分析摘要】{product_analysis.summary}")
        if pricing_analysis:
            for pc in (pricing_analysis.pricing_comparison or [])[:6]:
                lines.append(f"【定价】{pc.competitor}: 免费={pc.free_tier}; 付费={pc.paid_tier}; 模式={pc.pricing_model}")
            if pricing_analysis.pricing_strategy_analysis:
                lines.append(f"【定价策略】{pricing_analysis.pricing_strategy_analysis}")
        if market_analysis:
            for ms in (market_analysis.market_share_data or [])[:6]:
                lines.append(f"【市场份额】{ms.competitor}: 估算={ms.share_estimate}; 趋势={ms.trend}")
            if market_analysis.growth_trends:
                lines.append(f"【市场趋势】{market_analysis.growth_trends}")
        return "\n".join(lines)
