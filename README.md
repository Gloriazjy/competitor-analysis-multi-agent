# 智能竞品分析多Agent系统

通过多个AI智能体协同工作，将复杂的竞品分析任务拆解为竞品发现、数据采集、多维分析和策略建议等专业化子任务，实现从竞品搜索到战略输出的全流程自动化。**现已升级为 LangGraph 架构**，支持动态质检自动回退机制和 LangSmith 全链路可视化追踪。

---

## 🚀 新特性 (v2.1)

- ✅ **LangGraph 编排引擎**：带动态循环的 DAG 工作流
- 🔬 **双层质检 + 真打回闭环**：规则层查"字段完整性" + LLM事实核查层查"结论是否被原文支撑"，识别幻觉/编造/矛盾并据此打回重做
- 🤝 **Agent 自由协作**：分析Agent发现信息不足时，可主动回退到采集Agent补数据，而非只走固定流程
- 🎯 **多LLM后端支持**：豆包（默认，对接大赛模型资源）、阿里云、百度千帆、OpenAI、Ollama 可一键切换
- 📊 **三层可观测与评测组合**：LangSmith 看过程 Trace，Eval 脚本算质量指标，decision_logs 记录关键决策依据
- 🧪 **评测集**：`eval/` 目录提供多行业测试用例与评测脚本，输出可量化实验结果
- 🎨 **DAG流程图自动生成**：自动保存到 output 目录

---

## 项目背景

企业产品团队在进行竞品分析时，通常需要经历"信息搜集 → 功能对比 → 用户评价整理 → SWOT 分析 → 结构化报告输出"等多个环节，流程重复性高、信息源分散，且对分析人员的行业认知要求较高。本课题实现一个多 Agent 协作的竞品分析系统，模拟一个"数字调研小组"，由 3-4 个专职 Agent 自动完成从公开信息采集到结构化竞品报告的全链路产出，并通过 Agent 间的交叉审查与反馈机制实现自我校验。

## 一、系统总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LangGraph 动态DAG工作流                                 │
│                                                                             │
│   START → 竞品发现 → 数据采集 ←──────────┐                                  │
│                         ↓                 │ 分析Agent主动补采请求            │
│              ┌────────────────────────┐   │                                  │
│              │  三维并行分析           │───┘ (信息不足时回退采集)            │
│              │  产品 ┊ 定价 ┊ 市场     │                                     │
│              └───────────┬────────────┘                                     │
│                          ↓ (信息充分)                                       │
│              ┌────────────────────────────────┐                            │
│              │  🔬 双层质检 QualityCheckAgent │←────┐                       │
│              │  L1 规则: 字段/来源完整性       │     │                       │
│              │  L2 LLM : 结论是否被原文支撑     │     │ 质检打回重做          │
│              └───────────┬────────────────────┘     │ (发现>采集>分析)      │
│                          │ 不通过 ──────────────────┘                       │
│                          ↓ 通过                                             │
│                   策略生成 → 报告输出 → END                                 │
│                                                                             │
│  LLM后端: 豆包(默认) ┊ 阿里云 ┊ 百度千帆 ┊ OpenAI ┊ Ollama                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**双重反馈闭环**：
1. **分析主动协作**：分析Agent判断采集材料不足 → 主动回退采集补数据
2. **质检事实打回**：质检发现结论缺原文支撑/编造 → 打回对应Agent重做

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

### 6. 质量检测Agent（QualityCheckAgent）⭐ 双层质检
- **职责**：全链路质量审核，识别问题并打回重做
- **第1层 规则引擎**（0次LLM，快速）：检查字段完整性、来源数量、竞品/特征/定价/市场数据是否达标
- **第2层 LLM事实核查**（1次LLM）：把"采集原文"与"分析结论"一起交给LLM，识别原文中不存在的编造功能/价格/市场份额/评分，输出 `unsupported_claim`/`contradiction`/`fabricated_data` 三类问题
- **输入**：所有中间数据
- **输出**：质量报告（综合得分、事实得分、问题列表、回退目标、重试计数）
- **判定逻辑**：
  - 得分 ≥0.7、事实核查分 ≥0.6 且无 high 级事实问题 → 通过
  - 否则 → 选择影响面最大的目标回退重做（发现 > 采集 > 分析）
  - 达到 max_retries → 强制降级通过，并在报告中标注置信度偏低

### 🤝 Agent 自由协作（动态路由）
除固定流程与质检回退外，**分析Agent可主动发起协作**：当三维分析发现过半竞品缺关键维度（产品功能/定价）时，自动生成补采清单回退到采集Agent，补齐后再继续分析。由独立计数器 `max_analysis_rollback` 控制，避免无限循环。

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

### 运行方式

```bash
# 默认模式（启用豆包LLM + 双层质检 + 动态回退）
python main.py "小度学习机"

# 规则引擎模式（不调用LLM，快速体验流程）
python main.py --rule "小度学习机"

# 指定阿里云后端
set LLM_PROVIDER=aliyun
python main.py "小度学习机"

# 启用 LangSmith 全链路追踪
set LANGCHAIN_TRACING_V2=true
set LANGCHAIN_API_KEY=你的key
python main.py "小度学习机"
```

### 运行评测集（生成实验结果）

```bash
# 跑全部用例（LLM模式）
python eval/run_eval.py

# 规则引擎模式（快速、零API消耗）
python eval/run_eval.py --rule

# 只跑指定用例 / 限制数量
python eval/run_eval.py --case saas_notion
python eval/run_eval.py --limit 2
```

评测结果输出到 `eval/eval_result.json`，包含场景命中率、平均来源数、字段完整率、字段可解释率、质检得分、事实核查得分、重做次数、耗时等指标。

### Agent 评估与可观测分工

本项目不把“固定路径轨迹”作为核心评分项，因为动态回退、自评补采和质检打回本来就会让不同输入走不同路径。更合理的组合是：

- **LangSmith Trace**：记录 Agent/工具调用、输入输出、耗时、Token、报错，用于调试和答辩展示。
- **Eval 指标**：评估输出质量，包括来源覆盖、字段可解释率、事实核查分、质检得分、重试次数、耗时。
- **decision_logs**：记录关键决策依据，例如为什么打回采集、为什么进入分析、为什么标记 `not_public` 后继续。

因此我们监控过程，但不迷信“理想路径”。核心评估对象是证据可信度、决策合理性和闭环是否真正改善。

采集性能方面，Discovery 与 Collection 均使用并发搜索：

- `SEARCH_CONCURRENCY` 控制单个竞品内 query 并发数。
- `COLLECTION_COMPETITOR_CONCURRENCY` 控制同时采集几个竞品。
- `SEARCH_STAGGER_SECONDS` 用于错峰启动，兼顾速度与限流风险。

## 五、项目结构

```
competitor-analysis-multi-agent/
├── README.md                    # 本文档
├── COMPLIANCE.md                # 合规与数据来源说明
├── config.py                    # 全局配置（多LLM后端）
├── requirements.txt             # 依赖清单
├── main.py                      # 主入口（LangGraph版本）
├── core/
│   ├── graph_state.py           # LangGraph全局State定义
│   ├── llm_client.py            # LLM统一调用封装（多后端 + 重试退避）
│   ├── type_utils.py            # LangGraph序列化兼容工具
│   ├── orchestrator_graph.py    # LangGraph编排器（质检回退 + 分析主动协作）
│   ├── orchestrator.py          # 旧asyncio线性编排器（保留参考）
│   ├── scenario_profile.py      # 通用场景画像（多行业维度库）
│   ├── search_client.py         # 百度AI搜索客户端
│   └── prompt_loader.py         # 提示词模板加载器
├── agents/
│   ├── base_agent.py            # Agent基类
│   ├── research/                # 佳怡：发现 + 采集
│   │   ├── discovery_agent.py
│   │   └── collection_agent.py
│   ├── analysis/                # 三维分析 + 质检
│   │   ├── product_agent.py
│   │   ├── pricing_agent.py
│   │   ├── market_agent.py
│   │   └── quality_check_agent.py   # ✨ 双层质检
│   └── reporting/               # 策略 + 报告
│       ├── strategy_agent.py
│       └── report_formatter.py
├── models/
│   └── domain.py                # 领域模型（dataclass数据Schema）
├── prompts/                     # 提示词模板（.md格式）
├── eval/                        # ✨ 评测集
│   ├── dataset.json             # 多行业测试用例
│   ├── run_eval.py              # 评测脚本
│   └── eval_result.json         # 评测输出（运行后生成）
└── output/                      # 报告输出目录
    ├── langgraph_dag.png        # 自动生成的DAG图
    └── 产品名_analysis_report.html
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
