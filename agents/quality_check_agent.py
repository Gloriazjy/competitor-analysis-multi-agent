# -*- coding: utf-8 -*-
"""
agents/quality_check_agent.py — 质检Agent

职责：对采集和分析结果进行质量检测，识别缺失数据/低质量内容
返回质检结果：是否通过、质量分数、问题列表、目标回退节点
"""

from agents.base_agent import BaseAgent
from core.graph_state import GraphState
import time


class QualityCheckAgent(BaseAgent):
    """数据与分析质量审核Agent"""

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
            issues.append({
                "type": "competitor_list_empty",
                "severity": "critical",
                "message": "竞品列表为空，需重新执行发现流程",
                "rollback_to": "discovery"
            })
            score -= 0.4
            self._log("   ❌ 竞品列表为空")
        else:
            competitor_count = len(competitor_list.competitors)
            self._log(f"   ✅ 竞品列表有效: {competitor_count}个竞品")

        # ── 检查2: 数据采集完整性 ──
        competitors_data_raw = state.get("competitors_data") or []
        competitors_data = [self._safe_convert_competitor_data(cd) for cd in competitors_data_raw]
        missing_data_count = 0
        for cd in competitors_data:
            if not cd.product_features and not cd.pricing_info and not cd.market_share:
                missing_data_count += 1
                issues.append({
                    "type": "competitor_data_empty",
                    "severity": "high",
                    "message": f"竞品[{cd.name}]数据完全缺失",
                    "rollback_to": "collection"
                })
                score -= 0.15
                self._log(f"   ⚠️  竞品[{cd.name}]数据为空")

        # ── 检查3: 产品分析完整性 ──
        product_analysis = state.get("product_analysis")
        if not product_analysis or len(product_analysis.feature_matrix) < 2:
            issues.append({
                "type": "product_analysis_insufficient",
                "severity": "medium",
                "message": "产品分析特征矩阵维度太少",
                "rollback_to": "product_analysis"
            })
            score -= 0.2
            self._log("   ⚠️  产品分析特征维度不足")
        else:
            self._log(f"   ✅ 产品分析OK: {len(product_analysis.feature_matrix)}个特征")

        # ── 检查4: 定价分析完整性 ──
        pricing_analysis = state.get("pricing_analysis")
        if not pricing_analysis or len(pricing_analysis.pricing_comparison) < 2:
            issues.append({
                "type": "pricing_analysis_insufficient",
                "severity": "medium",
                "message": "定价对比项不足",
                "rollback_to": "pricing_analysis"
            })
            score -= 0.15
            self._log("   ⚠️  定价分析数据不足")
        else:
            self._log(f"   ✅ 定价分析OK: {len(pricing_analysis.pricing_comparison)}个定价项")

        # ── 检查5: 市场分析完整性 ──
        market_analysis = state.get("market_analysis")
        if not market_analysis or len(market_analysis.market_share_data) < 2:
            issues.append({
                "type": "market_analysis_insufficient",
                "severity": "medium",
                "message": "市场份额数据不足",
                "rollback_to": "market_analysis"
            })
            score -= 0.15
            self._log("   ⚠️  市场分析数据不足")
        else:
            self._log(f"   ✅ 市场分析OK: {len(market_analysis.market_share_data)}个市场数据")

        # ── 最终判定 ──
        score = max(0.0, min(1.0, score))
        passed = score >= 0.7 or (state.get("retry_count", 0) >= state.get("max_retries", 2))
        retry_count = state.get("retry_count", 0) + 1

        self._log(f"📊 质检完成: 得分 {score:.2f} / 1.0, 通过={passed}, 重试次数={retry_count}")

        return {
            "quality_check_passed": passed,
            "quality_score": score,
            "issues_found": issues,
            "retry_count": retry_count,
            "execution_logs": [{
                "agent": "QualityCheckAgent",
                "timestamp": time.time(),
                "score": score,
                "passed": passed,
                "issue_count": len(issues)
            }]
        }
