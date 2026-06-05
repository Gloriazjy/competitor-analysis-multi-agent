# -*- coding: utf-8 -*-
"""
eval/run_eval.py — 竞品分析系统评测脚本

对评测集中的每个用例跑完整 LangGraph 流程，采集可量化指标，输出实验结果。

用法:
  python eval/run_eval.py                # 跑全部用例（LLM模式）
  python eval/run_eval.py --rule         # 规则引擎模式（快速、零API消耗）
  python eval/run_eval.py --case saas_notion   # 只跑指定用例
  python eval/run_eval.py --limit 2      # 只跑前2个用例

指标:
  - scenario_hit        场景识别是否命中预期
  - competitor_count    发现竞品数
  - avg_source_count    平均每竞品来源数（溯源能力）
  - field_completeness  关键字段(功能/定价/市场/口碑)非空比例
  - quality_score       质检综合得分
  - factual_score       LLM事实核查得分（幻觉抑制）
  - retry_count         质检触发的重做次数
  - analysis_rollback   分析Agent主动回退采集次数
  - forced_pass         是否因达重试上限被迫通过
  - duration_sec        端到端耗时
  - llm_success_rate    LLM调用成功率
"""

import sys
import os
import json
import time
import asyncio
import argparse

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.orchestrator_graph import LangGraphOrchestrator
from core.type_utils import to_competitor_list, to_competitor_data
from core.scenario_profile import detect_scenario
from core.llm_client import get_llm_stats, reset_llm_stats
from core.observability import latest_decision_summary, setup_langsmith

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _field_completeness(comp_data_list) -> float:
    """关键字段非空比例（功能/定价/市场/口碑）"""
    if not comp_data_list:
        return 0.0
    total = 0
    filled = 0
    for cd in comp_data_list:
        for field in ("product_features", "pricing_info", "market_share", "user_reviews"):
            total += 1
            if (getattr(cd, field, "") or "").strip():
                filled += 1
    return round(filled / total, 3) if total else 0.0


def _data_availability_score(comp_data_list) -> float:
    """字段可信可用率：有值、公开不可得说明、替代指标都算可解释。"""
    if not comp_data_list:
        return 0.0
    total = 0
    explained = 0
    acceptable = {"found", "not_public", "not_found", "proxy_available"}
    for cd in comp_data_list:
        status = getattr(cd, "field_status", {}) or {}
        for field in ("product_features", "pricing_info", "market_share", "user_reviews"):
            total += 1
            if (getattr(cd, field, "") or "").strip() or status.get(field) in acceptable:
                explained += 1
    return round(explained / total, 3) if total else 0.0


def _decision_summary(final_state: dict) -> dict:
    """评测只保留决策依据摘要，不把路径轨迹作为核心指标。"""
    return latest_decision_summary(final_state)


def _avg_source_count(comp_data_list) -> float:
    if not comp_data_list:
        return 0.0
    counts = [len(cd.search_sources) + len(cd.source_urls) for cd in comp_data_list]
    return round(sum(counts) / len(counts), 2)


async def run_case(case: dict) -> dict:
    reset_llm_stats()
    product = case["product"]
    max_comp = case.get("max_competitors", 4)
    print("\n" + "#" * 70)
    print(f"  评测用例: {case['id']} | 产品: {product}")
    print("#" * 70)

    start = time.time()
    orchestrator = LangGraphOrchestrator()
    try:
        final_state = await orchestrator.run(product, max_comp)
    except Exception as e:
        return {"id": case["id"], "product": product, "error": str(e)}
    duration = round(time.time() - start, 2)

    comp_list = to_competitor_list(final_state.get("competitor_list"))
    comp_data_list = [to_competitor_data(cd) for cd in (final_state.get("competitors_data") or [])]

    profile = detect_scenario(product)
    llm_stats = get_llm_stats()
    llm_success_rate = (
        round(llm_stats["success"] / llm_stats["total"], 3)
        if llm_stats["total"] else 0.0
    )

    return {
        "id": case["id"],
        "product": product,
        "scenario_expected": case.get("scenario_expected", ""),
        "scenario_detected": profile.scenario_id,
        "scenario_hit": profile.scenario_id == case.get("scenario_expected", ""),
        "competitor_count": len(comp_list.competitors) if comp_list else 0,
        "avg_source_count": _avg_source_count(comp_data_list),
        "field_completeness": _field_completeness(comp_data_list),
        "data_availability_score": _data_availability_score(comp_data_list),
        "quality_score": round(final_state.get("quality_score", 0.0), 3),
        "factual_score": round(final_state.get("factual_score", 1.0), 3),
        "retry_count": final_state.get("retry_count", 0),
        "analysis_rollback": final_state.get("analysis_rollback_count", 0),
        "forced_pass": final_state.get("quality_forced_pass", False),
        "issue_count": len(final_state.get("issues_found", [])),
        "report_generated": final_state.get("strategy_report") is not None,
        "duration_sec": duration,
        "llm_total": llm_stats["total"],
        "llm_success_rate": llm_success_rate,
        "decision_summary": _decision_summary(final_state),
    }


def summarize(results: list) -> dict:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"cases": len(results), "all_failed": True}
    n = len(ok)
    def avg(key):
        return round(sum(r[key] for r in ok) / n, 3)
    return {
        "cases": len(results),
        "succeeded": n,
        "scenario_hit_rate": round(sum(1 for r in ok if r["scenario_hit"]) / n, 3),
        "avg_competitor_count": avg("competitor_count"),
        "avg_source_count": avg("avg_source_count"),
        "avg_field_completeness": avg("field_completeness"),
        "avg_data_availability_score": avg("data_availability_score"),
        "avg_quality_score": avg("quality_score"),
        "avg_factual_score": avg("factual_score"),
        "total_retries": sum(r["retry_count"] for r in ok),
        "total_analysis_rollback": sum(r["analysis_rollback"] for r in ok),
        "forced_pass_count": sum(1 for r in ok if r["forced_pass"]),
        "report_success_rate": round(sum(1 for r in ok if r["report_generated"]) / n, 3),
        "avg_duration_sec": avg("duration_sec"),
        "avg_llm_success_rate": avg("llm_success_rate"),
    }


async def main():
    parser = argparse.ArgumentParser(description="竞品分析系统评测")
    parser.add_argument("--rule", action="store_true", help="规则引擎模式（不调用LLM）")
    parser.add_argument("--case", type=str, default="", help="只跑指定用例id")
    parser.add_argument("--limit", type=int, default=0, help="只跑前N个用例")
    args = parser.parse_args()

    config.ENABLE_LLM = not args.rule

    with open(os.path.join(EVAL_DIR, "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)
    cases = dataset["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.limit:
        cases = cases[:args.limit]

    print(f"\n评测模式: {'规则引擎' if args.rule else 'LLM(豆包)'} | 用例数: {len(cases)}")
    trace_status = setup_langsmith(run_name=f"eval-{time.strftime('%Y%m%d-%H%M%S')}")
    print(
        "Trace模式: "
        + ("LangSmith" if trace_status["enabled"] and trace_status["has_api_key"] else "local-only")
    )

    results = []
    for case in cases:
        results.append(await run_case(case))

    summary = summarize(results)
    report = {
        "mode": "rule" if args.rule else "llm",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "observability": {
            "langsmith_enabled": trace_status["enabled"],
            "langsmith_project": trace_status["project"],
            "note": "LangSmith records process traces; eval metrics score output quality and decision validity.",
        },
        "summary": summary,
        "details": results,
    }

    out_path = os.path.join(EVAL_DIR, "eval_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  评测汇总")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
