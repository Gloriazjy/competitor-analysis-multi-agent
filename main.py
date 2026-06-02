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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.orchestrator_graph import LangGraphOrchestrator


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
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() != "true":
        print("\n💡 提示: 如需启用 LangSmith 全链路追踪，请设置以下环境变量:")
        print("   set LANGCHAIN_TRACING_V2=true")
        print("   set LANGCHAIN_API_KEY=<your-langchain-api-key>")
        print("   (前往 https://smith.langchain.com/ 免费获取 API Key)")
        print()
    else:
        api_key = os.environ.get("LANGCHAIN_API_KEY", "")
        if api_key:
            print("✅ LangSmith 追踪已启用! 所有运行将自动上传到 LangSmith 面板")
        else:
            print("⚠️  LANGCHAIN_TRACING_V2 已打开但未找到 LANGCHAIN_API_KEY")


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
        "retry_count": final_state.get("retry_count", 0),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"💾 [LangGraph版本] JSON报告: {json_path}")
    
    print(f"\n📊 质量统计: 最终得分 {final_state.get('quality_score', 0):.2f}, 重试次数 {final_state.get('retry_count', 0)}")
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
