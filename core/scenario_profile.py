# -*- coding: utf-8 -*-
"""
core/scenario_profile.py — 通用竞品分析场景画像

用轻量规则识别产品所属场景，为发现、采集和分析 Agent 提供统一维度。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScenarioProfile:
    scenario_id: str
    category: str
    keywords: tuple[str, ...]
    competitor_candidates: tuple[str, ...]
    search_modifiers: tuple[str, ...]
    product_dimensions: tuple[str, ...]
    pricing_dimensions: tuple[str, ...]
    market_dimensions: tuple[str, ...]
    source_queries: tuple[str, ...] = field(default_factory=tuple)
    offer_dimensions: tuple[str, ...] = field(default_factory=tuple)
    quality_risks: tuple[str, ...] = field(default_factory=tuple)


SCENARIO_PROFILES: tuple[ScenarioProfile, ...] = (
    ScenarioProfile(
        scenario_id="education_hardware",
        category="教育硬件/学习产品",
        keywords=("学习机", "教育硬件", "家教机", "AI学习", "智能学习", "在线教育"),
        competitor_candidates=("步步高学习机", "作业帮学习机", "科大讯飞学习机", "学而思学习机", "普通平板电脑"),
        search_modifiers=("功能评测", "家长评价", "课程资源", "硬件参数", "价格"),
        product_dimensions=("课程资源", "AI辅导", "错题管理", "护眼能力", "硬件性能", "家长管控"),
        pricing_dimensions=("硬件售价", "课程订阅", "增值服务", "售后保修"),
        market_dimensions=("目标年级", "家长口碑", "渠道覆盖", "教育资源合作"),
        source_queries=("官网", "京东", "评测", "家长评价"),
        offer_dimensions=("报价", "包含服务", "售后", "适用人群"),
        quality_risks=("夸大AI能力", "课程资源不透明", "售后限制"),
    ),
    ScenarioProfile(
        scenario_id="enterprise_saas",
        category="企业 SaaS/办公协作",
        keywords=("SaaS", "协作", "办公", "CRM", "ERP", "项目管理", "知识库", "会议", "文档"),
        competitor_candidates=("飞书", "钉钉", "企业微信", "Notion", "腾讯会议", "Zoom"),
        search_modifiers=("功能对比", "收费标准", "企业版", "客户案例", "安全合规"),
        product_dimensions=("协作能力", "权限管理", "集成生态", "AI能力", "数据安全", "部署与运维"),
        pricing_dimensions=("免费版", "团队版", "企业版", "按席位计费", "私有化部署"),
        market_dimensions=("目标客群", "行业案例", "生态伙伴", "用户口碑"),
        source_queries=("官网", "价格页", "帮助中心", "客户案例"),
        offer_dimensions=("套餐价格", "席位数", "功能权限", "服务支持", "合同周期"),
        quality_risks=("隐藏增购", "私有化费用不透明", "安全合规表述无证据"),
    ),
    ScenarioProfile(
        scenario_id="consumer_app",
        category="消费级 App/互联网产品",
        keywords=("App", "应用", "社区", "社交", "内容", "短视频", "工具", "小程序"),
        competitor_candidates=("小红书", "抖音", "快手", "B站", "知乎", "微信小程序"),
        search_modifiers=("用户评价", "下载量", "商业模式", "功能对比", "版本更新"),
        product_dimensions=("核心功能", "用户体验", "内容生态", "推荐算法", "社交关系", "增长机制"),
        pricing_dimensions=("免费模式", "会员订阅", "广告变现", "增值服务"),
        market_dimensions=("用户画像", "下载热度", "社区活跃度", "口碑趋势"),
        source_queries=("应用商店", "七麦数据", "用户评价", "版本记录"),
        offer_dimensions=("会员价格", "免费权益", "增值服务", "广告/内购"),
        quality_risks=("刷量口碑", "隐私风险", "隐藏自动续费"),
    ),
    ScenarioProfile(
        scenario_id="hardware_device",
        category="智能硬件/消费电子",
        keywords=("手机", "平板", "耳机", "手表", "智能硬件", "机器人", "摄像头", "设备"),
        competitor_candidates=("Apple", "华为", "小米", "OPPO", "vivo", "三星"),
        search_modifiers=("参数对比", "评测", "价格", "销量", "用户评价"),
        product_dimensions=("核心参数", "性能", "续航", "工业设计", "生态兼容", "售后"),
        pricing_dimensions=("官方售价", "渠道价", "套餐", "保修服务"),
        market_dimensions=("销量", "渠道覆盖", "品牌口碑", "用户评价"),
        source_queries=("官网", "电商平台", "评测", "参数"),
        offer_dimensions=("官方售价", "套装权益", "保修", "渠道价"),
        quality_risks=("渠道价不稳定", "售后限制", "参数虚标"),
    ),
    ScenarioProfile(
        scenario_id="vehicle_mobility",
        category="汽车/出行产品",
        keywords=("汽车", "电动车", "新能源", "出行", "车机", "自动驾驶", "充电"),
        competitor_candidates=("特斯拉", "比亚迪", "小鹏汽车", "理想汽车", "蔚来汽车"),
        search_modifiers=("车型对比", "售价", "续航", "销量", "车主评价"),
        product_dimensions=("续航", "智能座舱", "辅助驾驶", "空间", "补能", "售后服务"),
        pricing_dimensions=("官方指导价", "选装价格", "金融方案", "补贴政策"),
        market_dimensions=("销量趋势", "目标用户", "渠道网络", "车主口碑"),
        source_queries=("官网", "懂车帝", "汽车之家", "销量"),
        offer_dimensions=("指导价", "选装", "金融方案", "保养/补能权益"),
        quality_risks=("优惠口径不一致", "选装成本", "交付周期"),
    ),
    ScenarioProfile(
        scenario_id="finance_service",
        category="金融/支付/保险服务",
        keywords=("支付", "银行", "理财", "保险", "贷款", "金融", "风控", "证券"),
        competitor_candidates=("支付宝", "微信支付", "招商银行", "平安保险", "京东金融"),
        search_modifiers=("费率", "合规", "产品功能", "用户评价", "风险控制"),
        product_dimensions=("核心服务", "风控能力", "合规能力", "账户体系", "服务体验"),
        pricing_dimensions=("手续费", "服务费率", "会员权益", "优惠政策"),
        market_dimensions=("目标客群", "渠道合作", "品牌信任", "投诉口碑"),
        source_queries=("官网", "费率", "监管", "投诉"),
        offer_dimensions=("费率", "服务费", "权益", "风控/保障"),
        quality_risks=("费率口径不清", "合规风险", "投诉风险"),
    ),
    ScenarioProfile(
        scenario_id="game_content",
        category="游戏/内容产品",
        keywords=("游戏", "手游", "端游", "直播", "动漫", "影视", "内容平台"),
        competitor_candidates=("王者荣耀", "原神", "和平精英", "网易游戏", "腾讯游戏", "B站"),
        search_modifiers=("玩法", "付费模式", "用户评价", "流水", "版本更新"),
        product_dimensions=("核心玩法", "内容更新", "美术表现", "社交系统", "运营活动"),
        pricing_dimensions=("免费游玩", "内购", "月卡", "皮肤道具", "会员订阅"),
        market_dimensions=("活跃用户", "流水表现", "社区口碑", "版本热度"),
        source_queries=("官网", "TapTap", "七麦数据", "玩家评价"),
        offer_dimensions=("内购价格", "月卡", "会员", "活动权益"),
        quality_risks=("氪金压力", "流水口径不明", "玩家口碑两极化"),
    ),
    ScenarioProfile(
        scenario_id="service_package",
        category="服务套餐/团购方案",
        keywords=("旅游团", "跟团游", "报团", "团购", "套餐", "服务商", "报价", "大环线", "小团", "包车", "纯玩"),
        competitor_candidates=("携程", "飞猪", "途牛", "马蜂窝", "Klook", "本地旅行社", "定制服务商"),
        search_modifiers=("价格", "费用包含", "联系方式", "用户评价", "隐形消费", "退款政策"),
        product_dimensions=("服务内容", "交付流程", "附赠服务", "履约保障", "灵活性", "售后/退款"),
        pricing_dimensions=("人均价格", "总价", "费用包含", "费用不含", "隐藏成本", "退款政策"),
        market_dimensions=("平台口碑", "销量/热度", "真实评价", "供应商资质"),
        source_queries=("官网", "平台页面", "联系方式", "用户评价", "费用包含", "退款政策"),
        offer_dimensions=("价格", "天数/周期", "人数限制", "包含服务", "不含费用", "联系方式", "预订链接"),
        quality_risks=("隐形消费", "虚假低价", "评论不可验证", "联系方式缺失", "退款规则不清"),
    ),
)


DEFAULT_PROFILE = ScenarioProfile(
    scenario_id="general",
    category="通用产品/服务",
    keywords=(),
    competitor_candidates=(),
    search_modifiers=("竞品分析", "替代产品", "同类产品对比", "用户评价", "价格"),
    product_dimensions=("核心功能", "用户体验", "差异化能力", "集成能力", "服务支持"),
    pricing_dimensions=("免费能力", "付费方案", "定价模型", "性价比"),
    market_dimensions=("目标用户", "市场热度", "用户口碑", "渠道策略"),
    source_queries=("官网", "价格", "评测", "用户评价"),
    offer_dimensions=("价格", "包含权益", "限制条件", "联系方式/购买入口"),
    quality_risks=("来源不足", "价格口径不清", "口碑无证据", "隐藏成本"),
)


def detect_scenario(product_description: str) -> ScenarioProfile:
    """根据产品描述匹配最相关的场景画像。"""
    text = product_description.lower()
    best_profile = DEFAULT_PROFILE
    best_score = 0
    for profile in SCENARIO_PROFILES:
        score = sum(
            10 + len(keyword)
            for keyword in profile.keywords
            if keyword.lower() in text
        )
        if score > best_score:
            best_score = score
            best_profile = profile
    return best_profile


def infer_product_name(product_description: str) -> str:
    """从用户输入中取一个稳定的产品名兜底值。"""
    raw = product_description.strip()
    for sep in ("，", ",", "：", ":", "\n"):
        raw = raw.split(sep)[0]
    return raw[:30] if raw else "待分析产品"
