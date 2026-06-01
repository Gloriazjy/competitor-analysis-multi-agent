# 智能竞品分析多Agent系统

通过多个AI智能体协同工作，将复杂的竞品分析任务拆解为竞品发现、数据采集、多维分析和策略建议等专业化子任务，实现从竞品搜索到战略输出的全流程自动化。**现已升级为 LangGraph 架构**，支持动态质检自动回退机制和 LangSmith 全链路可视化追踪。

---

## 🚀 新特性 (v2.0)

- ✅ **LangGraph 编排引擎**：支持带动态循环的 DAG 工作流
- 🔄 **质检自动回退**：质检Agent检测数据质量，不合格自动打回重做
- 🎯 **5种LLM后端支持**：豆包、阿里云、百度千帆、OpenAI、Ollama
- 📊 **LangSmith 全链路追踪**：一键启用可视化追踪
- 🎨 **DAG流程图自动生成**：自动保存Mermaid图到output目录

---

## 项目背景

企业产品团队在进行竞品分析时，通常需要经历"信息搜集 → 功能对比 → 用户评价整理 → SWOT 分析 → 结构化报告输出"等多个环节，流程重复性高、信息源分散，且对分析人员的行业认知要求较高。本课题实现一个多 Agent 协作的竞品分析系统，模拟一个"数字调研小组"，由 3-4 个专职 Agent 自动完成从公开信息采集到结构化竞品报告的全链路产出，并通过 Agent 间的交叉审查与反馈机制实现自我校验。

## 一、系统总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LangGraph 动态DAG工作流                                 │
│                                                                             │
│  ┌──────────────────┐                                                        │
│  │  竞品发现Agent   │ →  生成初始列表                                       │
│  └───────┬──────────┘                                                        │
│          ↓                                                                 │
│  ┌──────────────────┐                                                        │
│  │  数据采集Agent   │ →  逐竞品深度采集                                       │
│  └───────┬──────────┘                                                        │
│          ↓                                                                 │
│  ┌──────────────────────────────────────┐                                   │
│  │  三维并行分析 (asyncio.gather)        │                                   │
│  │  ├ 产品分析  ┊ 定价分析  ┊ 市场分析 │                                   │
│  └───────┬──────────────────────────────┘                                   │
│          ↓                                                                 │
│  ┌─────────────────────────────────────────────┐                          │
│  │  🧐  质检Agent (QualityCheckAgent)          │←───┐                     │
│  │  → 质量打分 0.0~1.0                          │     │                     │
│  └───────┬─────────────────────────────────────┘     │                     │
│          │ (通过?  得分 >=0.7 或 达到最大重试)        │                     │
│          ↓ No →────────────────────────────────────┘  回退重做            │
│    打回采集 & 重分析                                                     │
│          ↓ Yes                                                            │
│  ┌──────────────────┐                                                        │
│  │  策略生成Agent   │ → 最终报告输出                                          │
│  └──────────────────┘                                                        │
│                                                                             │
│  LLM后端支持: 豆包  阿里云  百度千帆  OpenAI  Ollama                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**协作模式**：LangGraph 状态机 + 动态循环 混合模式

**核心理念**：
- **动态质检循环**：不合格数据自动回退重做，避免低质量输出
- **OpenAI兼容统一接口**：豆包/阿里云/OpenAI 共享一套调用逻辑
- **三维并行分析**：产品/定价/市场三个维度独立，结果JSON格式传递给策略Agent
- **质检打分系统**：0.0-1.0 量化数据质量
- **LangSmith追踪**：一键开启，全流程可追溯

## 二、Agent角色定义

### 1. 竞品发现Agent（DiscoveryAgent）
- **职责**：根据用户产品描述，搜索并筛选出3~8个核心竞品
- **LLM调用**：2次（关键词生成 + 结果筛选）
- **外部工具**：AI搜索
- **输入**：用户产品描述（string）
- **输出**：CompetitorList（竞品名称+简介列表）
- **领域预设库**：学习机、协同办公、手机、电动汽车等常见品类内置竞品列表

### 2. 数据采集Agent（CollectionAgent）
- **职责**：对每个竞品，采集产品功能、定价、用户评价、市场份额等信息
- **LLM调用**：1+N次（拆解采集维度 + 逐竞品汇总）
- **外部工具**：AI搜索
- **输入**：CompetitorList + 用户产品描述
- **输出**：dict[str, CompetitorData]（每竞品一份数据）
- **降级策略**：直接使用固定搜索模板采集

### 3. 产品分析Agent（ProductAgent）
- **职责**：逐竞品对比功能矩阵，标注优势/劣势/差异点
- **LLM调用**：1次
- **外部工具**：无
- **输入**：全部竞品数据
- **输出**：ProductAnalysis（含功能对比矩阵）
- **降级策略**：基于关键词匹配生成简单对比

### 4. 定价分析Agent（PricingAgent）
- **职责**：对比各竞品定价策略、促销模式、性价比
- **LLM调用**：1次
- **外部工具**：无
- **输入**：全部竞品数据
- **输出**：PricingAnalysis（含定价对比表）
- **降级策略**：提取价格数字进行简单排序

### 5. 市场分析Agent（MarketAgent）
- **职责**：分析市场份额、增长趋势、用户口碑、渠道策略
- **LLM调用**：1次
- **外部工具**：无
- **输入**：全部竞品数据
- **输出**：MarketAnalysis
- **降级策略**：基于采集数据中的关键词统计

### 6. 质量检测Agent（QualityCheckAgent）⭐ 新增
- **职责**：全链路数据质量审核，识别问题并打回重做
- **LLM调用**：0次（纯规则引擎，快速）
- **输入**：所有中间数据
- **输出**：质量报告（得分、问题列表、重试计数）
- **判定逻辑**：
  - 数据质量得分 >=0.7 → 通过
  - 得分 <0.7 → 回退重做
  - 达到max_retries → 强制通过，继续生成报告

### 7. 策略建议Agent（StrategyAgent）
- **职责**：综合三维分析，输出差异化定位建议和行动方案
- **LLM调用**：1次
- **外部工具**：无
- **输入**：ProductAnalysis + PricingAnalysis + MarketAnalysis
- **输出**：StrategyReport
- **降级策略**：基于SWOT模板生成简单建议

## 三、5种LLM后端配置说明

系统支持5种LLM后端，完全通过环境变量切换：

| 后端名称 | 环境变量设置 | 说明 |
|---------|------------|------|
| **豆包** (默认) | `set LLM_PROVIDER=doubao` | 字节跳动火山引擎Ark，Doubao-Seed-2.0-lite |
| **阿里云** | `set LLM_PROVIDER=aliyun` | 通义千问，OpenAI兼容接口 |
| **百度千帆** | `set LLM_PROVIDER=qianfan` | 百度文心大模型，支持bce-v3直接认证 |
| **OpenAI** | `set LLM_PROVIDER=openai` | GPT-4o / GPT系列 |
| **Ollama** | `set LLM_PROVIDER=ollama` | 本地大模型，完全离线运行 |

## 四、快速开始

### 环境准备

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
.\venv\Scripts\activate.bat

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行方式（LangGraph版本）

```bash
# 规则引擎模式（零依赖，直接体验动态质检循环）
python main_graph.py --rule "小度学习机"

# 豆包模式（默认，启用LLM）
python main_graph.py "小度学习机"

# 指定阿里云后端
set LLM_PROVIDER=aliyun
python main_graph.py "小度学习机"

# 启用 LangSmith 全链路追踪
set LANGCHAIN_TRACING_V2=true
set LANGCHAIN_API_KEY=你的key
python main_graph.py "小度学习机"
```

## 五、项目结构

```
competitor-analysis-mas-v2/
├── README.md                    # 本文档
├── config.py                    # 全局配置（5种LLM后端）
├── requirements.txt             # 依赖清单
├── main.py                      # 旧asyncio版本主入口
├── main_graph.py                # ✨ LangGraph版本主入口（推荐）
├── core/
│   ├── __init__.py
│   ├── graph_state.py           # LangGraph全局State定义
│   ├── llm_client.py            # LLM统一调用封装（5种后端）
│   ├── type_utils.py            # LangGraph序列化兼容工具
│   ├── orchestrator_graph.py    # LangGraph编排器（带动态循环）
│   ├── search_client.py         # 百度AI搜索客户端
│   └── prompt_loader.py         # 提示词模板加载器
├── agents/
│   ├── __init__.py
│   ├── base_agent.py            # Agent基类
│   ├── discovery_agent.py       # 竞品发现Agent（含领域预设库）
│   ├── collection_agent.py      # 数据采集Agent
│   ├── product_agent.py         # 产品分析Agent
│   ├── pricing_agent.py         # 定价分析Agent
│   ├── market_agent.py          # 市场分析Agent
│   ├── strategy_agent.py        # 策略建议Agent
│   └── quality_check_agent.py   # ✨ 质检Agent
├── models/
│   ├── __init__.py
│   └── domain.py                # 领域模型（TypedDict数据Schema）
├── prompts/                     # 提示词模板（.md格式）
├── output/                      # 报告输出目录
│   ├── langgraph_dag.png        # 自动生成的DAG图
│   └── 产品名_analysis_report.html
└── data/                        # 示例数据
```

## 六、技术栈

- **语言**：Python 3.10+
- **编排引擎**：LangGraph 0.2+
- **LLM调用**：统一OpenAI兼容接口
- **搜索**：百度AI Search
- **并行执行**：asyncio.gather
- **状态管理**：LangGraph StateGraph + 类型安全转换
- **可观测性**：LangSmith 全链路追踪

## 七、输出结果

运行成功后，`output/`目录下生成：

| 文件名 | 说明 |
|--------|------|
| `产品名_analysis_report.html` | 完整HTML可视化竞品分析报告 |
| `产品名_analysis_report.json` | JSON结构化数据 |
| `langgraph_dag.png` | LangGraph自动生成的DAG流程图 |

---

**让竞品分析从"凭经验"升级为"凭数据与智能"** 🚀

