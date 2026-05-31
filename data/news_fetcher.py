"""
新闻获取模块。

流程：
  1. 查数据库 24h 内已分析新闻 >= NEWS_CACHE_MIN_ITEMS → 直接复用
  2. 否则 → 调 LLM 获取 → 交给 FinBERT 分析（由 analysis_service 写入 DB）
  3. LLM 失败 / 未配置 Key → 降级为库内历史已分析新闻
"""

import json
import logging
import re

from data.models import NewsItem
from data.database import Database
from indicators.constants import (
    NEWS_CACHE_HOURS,
    NEWS_CACHE_MIN_ITEMS,
    NEWS_FETCH_LIMIT,
)

logger = logging.getLogger(__name__)

_NEWS_PROMPT_EN = """You are a professional financial news editor. Search for the latest real news about {name} ({code}) from reputable financial websites (Reuters, Bloomberg, CNBC, etc.).

Return {limit} news items sorted by date descending (newest first). Each item must include: date (YYYY-MM-DD), title, full content, source name.
Output MUST be in English. Output ONLY the JSON array below, nothing else:

[
  {{"date": "2026-05-27", "title": "News Title", "content": "Full news content here", "source": "Reuters"}},
  {{"date": "2026-05-26", "title": "News Title", "content": "Full news content here", "source": "Bloomberg"}}
]"""

_NEWS_PROMPT_CN = """你是一位专业的财经新闻编辑。请从正规财经网站（东方财富、财联社、证券时报、 Reuters、Bloomberg 等）获取关于 {name}（{code}）近一周的真实新闻。

返回 {limit} 条新闻，按日期从新到旧排列。每条包含：日期(YYYY-MM-DD)、标题、完整内容、来源名称。
请用中文输出。只输出以下 JSON 数组，不要其他内容：

[
  {{"date": "2026-05-27", "title": "新闻标题", "content": "完整新闻内容", "source": "东方财富"}},
  {{"date": "2026-05-26", "title": "新闻标题", "content": "完整新闻内容", "source": "财联社"}}
]"""


def _cache_min_items(limit: int) -> int:
    return min(NEWS_CACHE_MIN_ITEMS, limit)


def _load_cached(code: str, limit: int, hours: int = NEWS_CACHE_HOURS) -> list[NewsItem]:
    db = Database()
    cached = db.get_recent_news_with_sentiment(code, hours=hours, limit=limit)
    return cached[:limit]


def _fallback_news(code: str, limit: int) -> list[NewsItem]:
    """LLM 不可用时的降级：先 24h 缓存，再全库已分析新闻。"""
    recent = _load_cached(code, limit, hours=NEWS_CACHE_HOURS)
    if recent:
        logger.info(f"新闻降级: 使用 24h 内缓存 {len(recent)} 条")
        return recent
    historical = Database().get_news_with_sentiment(code, limit=limit)
    if historical:
        logger.info(f"新闻降级: 使用历史已分析新闻 {len(historical)} 条")
    return historical


def fetch_news(
    name: str, code: str, market: str,
    model: str, base_url: str, api_key: str,
    limit: int | None = None,
) -> list[NewsItem]:
    """
    获取股票新闻（缓存优先，LLM 兜底）。

    Args:
        name: 股票名称
        code: 股票代码
        market: 市场 (A/US)
        model/base_url/api_key: LLM 配置
        limit: 最大条数（默认 NEWS_FETCH_LIMIT）

    Returns:
        NewsItem 列表（缓存命中时含情感标签；LLM 新拉取的不含，需 FinBERT）
    """
    limit = limit or NEWS_FETCH_LIMIT
    min_items = _cache_min_items(limit)

    cached = _load_cached(code, limit)
    if len(cached) >= min_items:
        logger.info(
            f"新闻缓存命中: {len(cached)} 条 "
            f"({NEWS_CACHE_HOURS}h 内已分析, 阈值 {min_items})"
        )
        return cached

    if not (api_key or "").strip():
        logger.warning("LLM API Key 未配置，跳过在线抓取")
        return _fallback_news(code, limit)

    logger.info(
        f"缓存不足 ({len(cached)}/{min_items} 条)，调用 LLM 获取新闻..."
    )

    prompt_template = _NEWS_PROMPT_EN if market == "US" else _NEWS_PROMPT_CN
    prompt = prompt_template.format(name=name, code=code, limit=limit)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        response = completion.choices[0].message.content or ""
        logger.info(f"LLM 返回 {len(response)} 字符")

        items = _parse_llm_json(response, code, limit)
        logger.info(f"LLM 新闻: 解析出 {len(items)} 条")
        if not items:
            logger.warning(
                f"LLM 新闻解析为空，原始响应前 300 字符: {response[:300]}"
            )
            return _fallback_news(code, limit)
        return items

    except Exception as e:
        logger.error(f"LLM 新闻获取失败: {e}", exc_info=True)
        return _fallback_news(code, limit)


def _parse_llm_json(response: str, code: str, limit: int) -> list[NewsItem]:
    """解析 LLM 返回的 JSON 新闻列表。"""
    response = re.sub(r"```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```", "", response)
    response = response.strip()

    match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", response)
    json_str = match.group(0) if match else response

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"JSON 解析失败: {json_str[:300]}")
        return []

    items = []
    for item in data[:limit]:
        try:
            items.append(NewsItem(
                code=code,
                date=str(item.get("date", ""))[:10],
                title=str(item.get("title", "")),
                source=str(item.get("source", "")),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    items.sort(key=lambda n: str(n.date), reverse=True)
    return items
