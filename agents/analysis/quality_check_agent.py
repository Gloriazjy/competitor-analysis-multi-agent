# -*- coding: utf-8 -*-
"""
agents/quality_check_agent.py — 质检Agent

职责：对采集和分析结果进行质量检测，识别缺失数据/低质量内容
返回质检结果：是否通过、质量分数、问题列表、目标回退节点
"""

from agents.base_agent import BaseAgent
from core.graph_state import GraphState
from core.type_utils import (
    to_product_analysis,
    to_pricing_analysis,
    to_market_analysis,
)
import time


class QualityCheckAgent(BaseAgent):
    """数据与分析质量审核Agent"""

    MIN_SOURCE_COUNT = 2
    MIN_COMPETITOR_COUNT = 3
    MIN_FEATURE_COUNT = 3
    MIN_PRICING_ITEM_COUNT = 2
    MIN_MARKET_ITEM_COUNT = 2

    def __init__(self):
        super().__init__(
            agent_id="QualityCheckAgent",
            system_prompt="",
        )

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
                product_features=cd.get("product_features", ""),
                pricing_info=cd.get("pricing_info", ""),
                market_share=cd.get("market_share", ""),
                user_reviews=cd.get("user_reviews", ""),
                strengths=cd.get("strengths", ""),
                weaknesses=cd.get("weaknesses", ""),
                channels=cd.get("channels", ""),
                search_sources=cd.get("search_sources", [])
            )
        
        # 默认情况
        return CompetitorData(name="未知竞品")

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
        competitors_data = [self._safe_convert_competitor_data(cd) for cd in competitors_data_raw]
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
        for cd in competitors_data:
            missing_fields = [
                field for field in ("product_features", "pricing_info", "market_share", "user_reviews")
                if self._is_blank(getattr(cd, field, ""))
            ]
            if len(missing_fields) >= 3:
                issues.append(self._issue(
                    "competitor_data_incomplete",
                    "high",
                    f"竞品[{cd.name}]核心采集字段缺失: {', '.join(missing_fields)}",
                    "collection",
                    {"competitor": cd.name, "missing_fields": missing_fields},
                ))
                score -= 0.15
                self._log(f"   ⚠️  竞品[{cd.name}]采集字段缺失: {missing_fields}")
            if len(cd.search_sources) < self.MIN_SOURCE_COUNT:
                issues.append(self._issue(
                    "source_insufficient",
                    "high",
                    f"竞品[{cd.name}]来源数量不足，仅{len(cd.search_sources)}条",
                    "collection",
                    {"competitor": cd.name, "source_count": len(cd.search_sources)},
                ))
                score -= 0.1
                self._log(f"   ⚠️  竞品[{cd.name}]来源不足")

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

        # ── 检查6: 简单证据覆盖，避免分析结论脱离采集材料 ──
        source_text = "\n".join(
            f"{cd.name} {cd.product_features} {cd.pricing_info} {cd.market_share} "
            f"{cd.user_reviews} {cd.strengths} {cd.weaknesses} {cd.channels}"
            for cd in competitors_data
        )
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
        passed = score >= 0.7 or (state.get("retry_count", 0) >= state.get("max_retries", 2))
        retry_count = state.get("retry_count", 0) + 1
        rollback_target = "" if passed else self._select_rollback_target(issues)
        quality_feedback = [] if passed else self._build_feedback(issues, rollback_target)

        self._log(
            f"📊 质检完成: 得分 {score:.2f} / 1.0, 通过={passed}, "
            f"回退目标={rollback_target or '无'}, 重试次数={retry_count}"
        )

        return {
            "quality_check_passed": passed,
            "quality_score": score,
            "issues_found": issues,
            "rollback_target": rollback_target,
            "quality_feedback": quality_feedback,
            "retry_count": retry_count,
            "execution_logs": [{
                "agent": "QualityCheckAgent",
                "timestamp": time.time(),
                "score": score,
                "passed": passed,
                "issue_count": len(issues)
            }]
        }
