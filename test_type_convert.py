# -*- coding: utf-8 -*-
"""测试LangGraph状态序列化后的类型转换"""

import sys
sys.path.insert(0, '.')

from core.type_utils import to_competitor_data, to_competitor_list
from models.domain import CompetitorData, CompetitorList, CompetitorInfo

print("=== 测试类型转换 ===")

# 测试1: 原始对象
cd1 = CompetitorData(name="测试竞品1", product_features="功能1")
result1 = to_competitor_data(cd1)
print(f"1. 原始对象: {type(result1).__name__}, name={result1.name}")

# 测试2: 字典形式
cd2 = {"name": "测试竞品2", "product_features": "功能2", "pricing_info": "价格2"}
result2 = to_competitor_data(cd2)
print(f"2. 字典转换: {type(result2).__name__}, name={result2.name}, features={result2.product_features}")

# 测试3: JSON字符串形式
import json
cd3 = json.dumps({"name": "测试竞品3", "product_features": "功能3"})
result3 = to_competitor_data(cd3)
print(f"3. JSON字符串转换: {type(result3).__name__}, name={result3.name}")

# 测试4: 纯字符串形式
cd4 = "测试竞品4"
result4 = to_competitor_data(cd4)
print(f"4. 纯字符串转换: {type(result4).__name__}, name={result4.name}")

print("\n=== 测试通过 ===")
