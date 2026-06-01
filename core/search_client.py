# -*- coding: utf-8 -*-
"""
core/search_client.py — 百度AI搜索客户端

调用千帆AI搜索接口（baidu_search_v2数据源），返回搜索结果摘要。
"""

import json
import time
import requests

import config


class SearchClient:
    """百度AI搜索原生HTTP客户端"""

    def __init__(
        self,
        api_key: str = config.BAIDU_SEARCH_API_KEY,
        base_url: str = config.BAIDU_SEARCH_URL,
        search_source: str = config.BAIDU_SEARCH_SOURCE,
        recency: str = config.BAIDU_SEARCH_RECENCY,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.search_source = search_source
        self.recency = recency

    def search(self, query: str, recency: str | None = None) -> dict:
        """
        执行一次搜索查询。

        Args:
            query: 搜索关键词
            recency: 时间范围过滤 (week/month/year)，默认使用配置值

        Returns:
            搜索结果字典
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = json.dumps(
            {
                "messages": [{"role": "user", "content": query}],
                "edition": "standard",
                "search_source": self.search_source,
                "search_recency_filter": recency or self.recency,
            },
            ensure_ascii=False,
        )

        resp = requests.post(
            self.base_url,
            headers=headers,
            data=payload.encode("utf-8"),
            timeout=60,
        )
        resp.encoding = "utf-8"
        resp.raise_for_status()
        return resp.json()

    def batch_search(self, queries: list[str],
                     delay: float = config.SEARCH_DELAY_SECONDS) -> list[dict]:
        """
        批量搜索，逐条调用并附带间隔，避免限流。

        Args:
            queries: 搜索关键词列表
            delay: 每次搜索间隔秒数

        Returns:
            搜索结果列表，每项包含 query 和 result 两个键
        """
        results = []
        total = len(queries)
        for i, q in enumerate(queries):
            print(f"  [SearchClient] 搜索 {i+1}/{total}: {q[:50]}...")
            try:
                result = self.search(q)
                results.append({"query": q, "result": result})
            except Exception as e:
                print(f"  [SearchClient] 搜索失败: {q[:50]}... | 错误: {e}")
                results.append({"query": q, "result": None, "error": str(e)})
            if i < total - 1:
                time.sleep(delay)
        return results

    @staticmethod
    def extract_text(search_result: dict) -> str:
        """从AI搜索返回结构中提取纯文本内容"""
        if not search_result:
            return ""

        texts = []

        # 提取AI摘要
        choices = search_result.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, str) and content:
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            texts.append(text)
                    elif isinstance(item, str):
                        texts.append(item)

        # 提取搜索结果片段
        search_results = search_result.get("references", [])
        for sr in search_results:
            title = sr.get("title", "")
            snippet = sr.get("content", "") or sr.get("snippet", "")
            if title or snippet:
                texts.append(f"【{title}】{snippet}")

        return "\n".join(texts)
