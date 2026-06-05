# -*- coding: utf-8 -*-
"""Observability helpers: LangSmith trace setup and decision-log summaries."""

import os

import config


def setup_langsmith(run_name: str = "") -> dict:
    """Configure LangSmith tracing from env/config without making it a metric."""
    enabled = config.LANGSMITH_TRACING
    if enabled:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    if config.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", config.LANGSMITH_API_KEY)
        os.environ.setdefault("LANGSMITH_API_KEY", config.LANGSMITH_API_KEY)
    if config.LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", config.LANGSMITH_PROJECT)
        os.environ.setdefault("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT)
    if config.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", config.LANGSMITH_ENDPOINT)
        os.environ.setdefault("LANGSMITH_ENDPOINT", config.LANGSMITH_ENDPOINT)
    if run_name:
        os.environ.setdefault("LANGCHAIN_RUN_NAME", run_name)

    return {
        "enabled": enabled,
        "has_api_key": bool(config.LANGSMITH_API_KEY or os.environ.get("LANGCHAIN_API_KEY")),
        "project": config.LANGSMITH_PROJECT,
        "run_name": run_name,
    }


def langsmith_hint(status: dict) -> str:
    if not status.get("enabled"):
        return (
            "LangSmith trace 未启用；如需可观测追踪，设置 "
            "LANGSMITH_TRACING=true 和 LANGSMITH_API_KEY。"
        )
    if not status.get("has_api_key"):
        return "LangSmith trace 已请求启用，但缺少 LANGSMITH_API_KEY。"
    return f"LangSmith trace 已启用，项目: {status.get('project') or 'default'}。"


def latest_decision_summary(state: dict) -> dict:
    logs = state.get("decision_logs", []) or []
    if not logs:
        return {}
    last = logs[-1]
    return {
        "decision": last.get("decision", ""),
        "rollback_target": last.get("rollback_target", ""),
        "reason": last.get("reason", ""),
        "issue_types": last.get("issues", []),
        "retry_count": last.get("retry_count", 0),
        "forced_pass": last.get("forced_pass", False),
    }
