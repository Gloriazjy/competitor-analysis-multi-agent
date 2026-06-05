#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 智能竞品分析 (LangGraph 版本)

运行示例:
  python main.py "小度学习机"
  python main.py --rule "小度学习机"
  python main.py help

启用 LangSmith 全链路追踪:
  环境变量:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=你的-langchain-api-key
"""

import asyncio
import sys
import os

# Windows 控制台默认 GBK，无法打印 emoji，这里统一切到 UTF-8，避免 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.orchestrator_graph import LangGraphOrchestrator
from core.observability import latest_decision_summary, langsmith_hint, setup_langsmith as configure_langsmith


def print_banner():
    banner = """
╔═════════════════════════════════════════════════════════════════════════╗
║                                                                         ║
║   智能竞品分析 — LangGraph 多Agent系统                                   ║
║   Dynamic Quality Loop + LangSmith Tracing                              ║
║                                                                         ║
║   带质检自动回退机制 🔄  全链路可视化可追溯 📊                           ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def setup_langsmith():
    """自动检查并提示启用LangSmith追踪"""
    status = setup_langsmith_trace("main")
    print("\n🔭 " + langsmith_hint(status))
    if not status["enabled"]:
        print("   LangSmith 只用于过程可观测；核心评测仍由 eval/run_eval.py 计算。")
        print()


def setup_langsmith_trace(run_name: str) -> dict:
    return configure_langsmith(run_name=run_name)


async def run_graph_analysis(product_description: str, use_llm: bool = True, max_competitors: int = 5):
    """执行 LangGraph 工作流"""
    config.ENABLE_LLM = use_llm

    print_banner()
    setup_langsmith()
    decision_mode = "🧠 LLM智能分析 (带动态质检回退)" if use_llm else "📋 规则引擎分析"
    print(f"  模式: {decision_mode}")
    print(f"  分析目标: {product_description}")
    print(f"  最大竞品数: {max_competitors}")
    print()

    orchestrator = LangGraphOrchestrator()

    # 生成并保存DAG图
    orchestrator.save_graph_visualization()

    final_state = await orchestrator.run(product_description, max_competitors)

    report = final_state.get("strategy_report")
    if not report:
        print("❌ 分析未完成，未生成报告")
        return

    # 保存输出
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(report_dir, exist_ok=True)

    html_content = orchestrator.report_formatter.format_html_report(
        report,
        product_analysis=final_state.get("product_analysis"),
        pricing_analysis=final_state.get("pricing_analysis"),
        market_analysis=final_state.get("market_analysis"),
        competitor_list=final_state.get("competitor_list"),
        competitors_data=final_state.get("competitors_data"),
        timings=final_state.get("timings", {}),
    )
    html_path = os.path.join(report_dir, report.product_name + "_analysis_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n💾 HTML报告已保存: {html_path}")

    import json
    json_path = os.path.join(report_dir, report.product_name + "_analysis_report.json")
    print(f"💾 JSON报告: {json_path}")
    report_data = {
        "product_name": report.product_name,
        "competitor_count": report.competitor_count,
        "overall_positioning": report.overall_positioning,
        "differentiation_strategy": report.differentiation_strategy,
        "action_plan": [
            {"priority": ap.priority, "action": ap.action, "timeline": ap.timeline, "expected_impact": ap.expected_impact}
            for ap in report.action_plan
        ],
        "risk_assessment": report.risk_assessment,
        "timings": final_state.get("timings", {}),
        "quality_score": final_state.get("quality_score", 0),
        "factual_score": final_state.get("factual_score", 1.0),
        "retry_count": final_state.get("retry_count", 0),
        "analysis_rollback_count": final_state.get("analysis_rollback_count", 0),
        "quality_forced_pass": final_state.get("quality_forced_pass", False),
        "issues_found": final_state.get("issues_found", []),
        "decision_summary": latest_decision_summary(final_state),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"💾 [LangGraph版本] JSON报告: {json_path}")

    print(f"\n📊 质量统计: 最终得分 {final_state.get('quality_score', 0):.2f}, "
          f"事实核查 {final_state.get('factual_score', 1.0):.2f}, "
          f"质检重做 {final_state.get('retry_count', 0)}次, "
          f"分析主动补采 {final_state.get('analysis_rollback_count', 0)}次")
    if final_state.get("quality_forced_pass"):
        print("⚠️  注意: 已达最大重试次数，本次为降级强制通过，结论置信度偏低")
    decision = latest_decision_summary(final_state)
    if decision:
        print(f"🧭 关键决策: {decision.get('decision')} | {decision.get('reason')}")
    print(f"⏱️  总耗时: {final_state.get('timings', {}).get('total', 0):.2f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    use_rule = "--rule" in args
    if use_rule:
        args.remove("--rule")
    use_llm = not use_rule

    mode = args[0] if args else ""
    if mode in ("help", "-h", "--help"):
        print("""
╔═══════════════════════════════════════════════════════════════╗
║     智能竞品分析 - LangGraph 版本 运行指南                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  python main_graph.py "产品名"     默认模式                      ║
║  python main_graph.py --rule "产品名"  规则引擎模式             ║
║  python main_graph.py help         显示帮助                     ║
║                                                               ║
║  启用 LangSmith 追踪:                                          ║
║    set LANGCHAIN_TRACING_V2=true                               ║
║    set LANGCHAIN_API_KEY=<你的API Key>                          ║
║                                                               ║
║  DAG 动态流程:                                                 ║
║    START → 发现 → 采集 → 三维并行分析 → 质检                    ║
║               ↑                                        ↓        ║
║               ──── 不通过 ←──────────  通过 → 策略 → END        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
        sys.exit(0)
    if not mode:
        print("❌ 请提供产品描述，例如: python main_graph.py \"小度学习机\"")
        sys.exit(1)

    product_description = mode
    asyncio.run(run_graph_analysis(product_description, use_llm=use_llm, max_competitors=5))
