"""股票代码 → 名称映射（A 股）。

两层来源：

1. ``BUILTIN_STOCK_NAME_MAP`` —— 内嵌沪深 300 核心池名称，离线兜底，无需网络/文件。
2. ``data/stock_names.json`` —— 全市场缓存（由 akshare 生成，``quant-agent update-names``
   可刷新）。缓存优先于内嵌。

``STOCK_NAME_MAP`` 为两者合并结果，引擎与 CLI 均从此读取。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认池（沪深300核心 ~150 只）的名称，离线兜底。
BUILTIN_STOCK_NAME_MAP: dict[str, str] = {
    "000001": "平安银行",
    "000002": "万科A",
    "000063": "中兴通讯",
    "000333": "美的集团",
    "000338": "潍柴动力",
    "000425": "徐工机械",
    "000538": "云南白药",
    "000568": "泸州老窖",
    "000651": "格力电器",
    "000661": "长春高新",
    "000725": "京东方A",
    "000776": "广发证券",
    "000858": "五粮液",
    "000895": "双汇发展",
    "000938": "紫光股份",
    "001979": "招商蛇口",
    "002007": "华兰生物",
    "002120": "韵达股份",
    "002129": "TCL中环",
    "002142": "宁波银行",
    "002230": "科大讯飞",
    "002236": "大华股份",
    "002241": "歌尔股份",
    "002304": "洋河股份",
    "002352": "顺丰控股",
    "002415": "海康威视",
    "002459": "晶澳科技",
    "002460": "赣锋锂业",
    "002475": "立讯精密",
    "002493": "荣盛石化",
    "002555": "三七互娱",
    "002594": "比亚迪",
    "002601": "龙佰集团",
    "002602": "世纪华通",
    "002709": "天赐材料",
    "002714": "牧原股份",
    "002812": "恩捷股份",
    "002841": "视源股份",
    "002916": "深南电路",
    "003816": "中国广核",
    "300003": "乐普医疗",
    "300014": "亿纬锂能",
    "300015": "爱尔眼科",
    "300033": "同花顺",
    "300059": "东方财富",
    "300124": "汇川技术",
    "300142": "沃森生物",
    "300223": "北京君正",
    "300274": "阳光电源",
    "300347": "泰格医药",
    "300394": "天孚通信",
    "300408": "三环集团",
    "300413": "芒果超媒",
    "300418": "昆仑万维",
    "300433": "蓝思科技",
    "300454": "深信服",
    "300457": "赢时胜",
    "300496": "中科创达",
    "300529": "健帆生物",
    "300601": "康泰生物",
    "300628": "亿联网络",
    "300676": "华大基因",
    "300750": "宁德时代",
    "300760": "迈瑞医疗",
    "300782": "卓胜微",
    "300832": "新产业",
    "300896": "爱美客",
    "600009": "上海机场",
    "600010": "包钢股份",
    "600016": "民生银行",
    "600019": "宝钢股份",
    "600025": "华能水电",
    "600028": "中国石化",
    "600029": "南方航空",
    "600030": "中信证券",
    "600031": "三一重工",
    "600036": "招商银行",
    "600048": "保利发展",
    "600050": "中国联通",
    "600061": "国投资本",
    "600085": "同仁堂",
    "600089": "特变电工",
    "600104": "上汽集团",
    "600111": "北方稀土",
    "600115": "中国东航",
    "600150": "中国船舶",
    "600153": "建发股份",
    "600160": "巨化股份",
    "600176": "中国巨石",
    "600177": "雅戈尔",
    "600196": "复星医药",
    "600208": "新湖中宝",
    "600219": "南山铝业",
    "600230": "沧州大化",
    "600271": "航天信息",
    "600276": "恒瑞医药",
    "600282": "南钢股份",
    "600299": "安迪苏",
    "600309": "万华化学",
    "600332": "白云山",
    "600346": "恒力石化",
    "600362": "江西铜业",
    "600369": "西南证券",
    "600383": "金地集团",
    "600390": "五矿资本",
    "600406": "国电南瑞",
    "600436": "片仔癀",
    "600438": "通威股份",
    "600486": "扬农化工",
    "600489": "中金黄金",
    "600498": "烽火通信",
    "600519": "贵州茅台",
    "600521": "华海药业",
    "600570": "恒生电子",
    "600588": "用友网络",
    "600596": "新安股份",
    "600600": "青岛啤酒",
    "600606": "绿地控股",
    "600690": "海尔智家",
    "600703": "三安光电",
    "600745": "闻泰科技",
    "600809": "山西汾酒",
    "600837": "海通证券",
    "600845": "宝信软件",
    "600859": "王府井",
    "600887": "伊利股份",
    "600893": "航发动力",
    "600900": "长江电力",
    "600905": "三峡能源",
    "600918": "中泰证券",
    "600919": "江苏银行",
    "600941": "中国移动",
    "601006": "大秦铁路",
    "601012": "隆基绿能",
    "601066": "中信建投",
    "601088": "中国神华",
    "601111": "中国国航",
    "601127": "赛力斯",
    "601138": "工业富联",
    "601166": "兴业银行",
    "601211": "国泰君安",
    "601225": "陕西煤业",
    "601236": "红塔证券",
    "601288": "农业银行",
    "601318": "中国平安",
    "601328": "交通银行",
    "601336": "新华保险",
    "601390": "中国中铁",
    "601398": "工商银行",
    "601601": "中国太保",
    "601628": "中国人寿",
    "601633": "长城汽车",
    "601668": "中国建筑",
    "601669": "中国电建",
    "601688": "华泰证券",
    "601728": "中国电信",
    "601766": "中国中车",
    "601788": "光大证券",
    "601799": "星宇股份",
    "601818": "光大银行",
    "601857": "中国石油",
    "601881": "中国银河",
    "601899": "紫金矿业",
    "601901": "方正证券",
    "601919": "中远海控",
    "601939": "建设银行",
    "601985": "中国核电",
    "601989": "中国重工",
    "603160": "汇顶科技",
    "603259": "药明康德",
    "603288": "海天味业",
    "603369": "今世缘",
    "603501": "韦尔股份",
    "603799": "华友钴业",
    "603833": "欧派家居",
    "603986": "兆易创新",
}

# 全市场缓存文件（仓库内），``update_stock_names()`` 生成/刷新。
# stock_names.py 位于 src/quant_agent/screener/，父级第 3 层即仓库根目录。
_DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "stock_names.json"


def _load_cache(path: Path = _DEFAULT_CACHE) -> dict[str, str]:
    """加载 data/stock_names.json 全市场缓存；缺失/损坏则返回空字典。"""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning("加载股票名称缓存失败 %s: %s", path, e)
    return {}


CACHE_STOCK_NAME_MAP: dict[str, str] = _load_cache()

# 合并：缓存优先覆盖内嵌（线上刷新后全市场名称更准）。
STOCK_NAME_MAP: dict[str, str] = {**BUILTIN_STOCK_NAME_MAP, **CACHE_STOCK_NAME_MAP}


def get_stock_name(code: str) -> str:
    """返回股票名称，未知时返回空字符串。"""
    return STOCK_NAME_MAP.get(code, "")


def search_stocks(query: str, limit: int = 10) -> list[dict[str, str]]:
    """按代码或名称模糊搜索股票，返回 ``[{"code", "name"}, ...]``。

    匹配与排序规则（相关度由高到低）：

    1. 代码完全相等
    2. 代码以 query 前缀开头
    3. 名称以 query 前缀开头
    4. 代码包含 query
    5. 名称包含 query

    query 为空时返回空列表。搜索大小写不敏感、忽略首尾空白。
    """
    q = (query or "").strip()
    if not q:
        return []
    q_lower = q.lower()

    scored: list[tuple[int, str, str]] = []
    for code, name in STOCK_NAME_MAP.items():
        name_lower = name.lower()
        rank: int | None = None
        if code == q:
            rank = 0
        elif code.startswith(q):
            rank = 1
        elif name.startswith(q):
            rank = 2
        elif q_lower in code.lower():
            rank = 3
        elif q_lower in name_lower:
            rank = 4
        if rank is not None:
            scored.append((rank, code, name))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [{"code": c, "name": n} for _, c, n in scored[: max(1, limit)]]


def update_stock_names(path: Path = _DEFAULT_CACHE) -> int:
    """通过 akshare 拉取全市场 A 股代码→名称并写回缓存，返回条目数。

    需要联网；成功后会更新本模块的 ``CACHE_STOCK_NAME_MAP`` 与 ``STOCK_NAME_MAP``。
    """
    import akshare as ak

    logger.info("正在从 akshare 拉取全市场 A 股代码→名称…")
    df = ak.stock_info_a_code_name()
    code_col = "code" if "code" in df.columns else df.columns[0]
    name_col = "name" if "name" in df.columns else df.columns[1]

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        c = str(row[code_col]).strip()
        n = str(row[name_col]).strip()
        if c and n and n != "nan":
            mapping[c] = n

    if not mapping:
        raise RuntimeError("akshare 返回为空，未更新名称缓存")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    # 进程内热更新，使后续调用立即生效。
    global CACHE_STOCK_NAME_MAP, STOCK_NAME_MAP
    CACHE_STOCK_NAME_MAP = mapping
    STOCK_NAME_MAP = {**BUILTIN_STOCK_NAME_MAP, **CACHE_STOCK_NAME_MAP}
    logger.info("已更新股票名称缓存：%d 条 -> %s", len(mapping), path)
    return len(mapping)
