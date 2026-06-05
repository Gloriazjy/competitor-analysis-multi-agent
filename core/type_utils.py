# -*- coding: utf-8 -*-
"""
core/type_utils.py — LangGraph 序列化兼容工具

解决LangGraph自动把Python dataclass转成普通dict的问题
"""

from models.domain import (
    CompetitorInfo, CompetitorList, CompetitorData,
    FeatureComparison, CompetitiveAdvantage, ProductAnalysis,
    PricingItem, PricingAnalysis, MarketShareItem, UserReputation, MarketAnalysis,
    ActionItem, StrategyReport
)


def to_competitor_data(cd):
    """安全转成 CompetitorData"""
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


def to_competitor_list(cl):
    """安全转成 CompetitorList"""
    if isinstance(cl, dict):
        comps = []
        for c in cl.get("competitors", []):
            comps.append(to_competitor_info(c))
        return CompetitorList(
            product_name=cl.get("product_name", ""),
            product_category=cl.get("product_category", ""),
            competitors=comps,
            search_keywords_used=cl.get("search_keywords_used", [])
        )
    return cl


def to_competitor_info(ci):
    if isinstance(ci, dict):
        return CompetitorInfo(
            name=ci.get("name", ""),
            brief=ci.get("brief", ""),
            relevance=ci.get("relevance", "MEDIUM")
        )
    return ci


def to_product_analysis(pa):
    if isinstance(pa, dict):
        fms = []
        for fm_dict in pa.get("feature_matrix", []):
            if isinstance(fm_dict, dict):
                fms.append(FeatureComparison(
                    feature=fm_dict.get("feature", ""),
                    values=fm_dict.get("values", {})
                ))
        cas = []
        for ca_dict in pa.get("competitive_advantages", []):
            if isinstance(ca_dict, dict):
                cas.append(CompetitiveAdvantage(
                    competitor=ca_dict.get("competitor", ""),
                    our_advantage=ca_dict.get("our_advantage", ""),
                    their_advantage=ca_dict.get("their_advantage", "")
                ))
        return ProductAnalysis(
            feature_matrix=fms,
            competitive_advantages=cas,
            differentiation_points=pa.get("differentiation_points", []),
            summary=pa.get("summary", "")
        )
    return pa


def to_pricing_analysis(pra):
    if isinstance(pra, dict):
        pis = []
        for pi_dict in pra.get("pricing_comparison", []):
            if isinstance(pi_dict, dict):
                pis.append(PricingItem(
                    competitor=pi_dict.get("competitor", ""),
                    free_tier=pi_dict.get("free_tier", ""),
                    paid_tier=pi_dict.get("paid_tier", ""),
                    pricing_model=pi_dict.get("pricing_model", "")
                ))
        return PricingAnalysis(
            pricing_comparison=pis,
            pricing_strategy_analysis=pra.get("pricing_strategy_analysis", ""),
            value_ranking=pra.get("value_ranking", []),
            summary=pra.get("summary", "")
        )
    return pra


def to_market_analysis(ma):
    if isinstance(ma, dict):
        msis = []
        for msi_dict in ma.get("market_share_data", []):
            if isinstance(msi_dict, dict):
                msis.append(MarketShareItem(
                    competitor=msi_dict.get("competitor", ""),
                    share_estimate=msi_dict.get("share_estimate", ""),
                    trend=msi_dict.get("trend", "")
                ))
        urs = {}
        for k, v in ma.get("user_reputation", {}).items():
            if isinstance(v, dict):
                urs[k] = UserReputation(
                    score=v.get("score", ""),
                    keywords=v.get("keywords", [])
                )
        return MarketAnalysis(
            market_share_data=msis,
            growth_trends=ma.get("growth_trends", ""),
            user_reputation=urs,
            channel_analysis=ma.get("channel_analysis", ""),
            summary=ma.get("summary", "")
        )
    return ma
