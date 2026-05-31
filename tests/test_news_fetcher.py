"""
新闻获取与数据库缓存单元测试。
"""

import sys
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.models import NewsItem
from data.database import Database
from data.news_fetcher import fetch_news, _parse_llm_json, _fallback_news


def _reset_db(path: str) -> Database:
    Database._instance = None
    return Database.init(path)


def _temp_db() -> tuple[Database, str]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return _reset_db(path), path


def _sample_news(code: str, title: str, sentiment: str = "positive") -> NewsItem:
    return NewsItem(
        code=code,
        date=datetime.now().strftime("%Y-%m-%d"),
        title=title,
        source="测试",
        sentiment=sentiment,
        confidence=0.85,
    )


def test_insert_news_upsert_no_duplicates():
    db, path = _temp_db()
    try:
        item = _sample_news("600519", "茅台发布年报")
        db.insert_news([item])
        db.insert_news([item])
        rows = db.get_news("600519", limit=20)
        assert len(rows) == 1
        assert rows[0].sentiment == "positive"
    finally:
        os.unlink(path)
        Database._instance = None


def test_get_recent_news_with_sentiment_respects_cached_at():
    db, path = _temp_db()
    try:
        db.insert_news([_sample_news("600519", "旧闻", "neutral")])
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
        db._execute_write(
            "UPDATE news_sentiment SET cached_at = ? WHERE title = ?",
            (old_ts, "旧闻"),
        )
        db.insert_news([_sample_news("600519", "新闻", "positive")])

        recent = db.get_recent_news_with_sentiment("600519", hours=24, limit=10)
        titles = {n.title for n in recent}
        assert "新闻" in titles
        assert "旧闻" not in titles
    finally:
        os.unlink(path)
        Database._instance = None


def test_fetch_news_cache_hit_skips_llm():
    db, path = _temp_db()
    try:
        items = [_sample_news("600519", f"缓存新闻{i}") for i in range(5)]
        db.insert_news(items)

        with patch("openai.OpenAI") as mock_client:
            result = fetch_news(
                name="贵州茅台", code="600519", market="A",
                model="gpt-4o", base_url="http://x", api_key="sk-test",
                limit=5,
            )
            mock_client.assert_not_called()

        assert len(result) == 5
        assert all(n.sentiment for n in result)
    finally:
        os.unlink(path)
        Database._instance = None


def test_fetch_news_no_api_key_uses_fallback():
    db, path = _temp_db()
    try:
        db.insert_news([_sample_news("600519", "历史新闻")])

        with patch("openai.OpenAI") as mock_client:
            result = fetch_news(
                name="贵州茅台", code="600519", market="A",
                model="", base_url="", api_key="",
                limit=5,
            )
            mock_client.assert_not_called()

        assert len(result) == 1
        assert result[0].title == "历史新闻"
    finally:
        os.unlink(path)
        Database._instance = None


def test_fetch_news_llm_failure_fallback():
    db, path = _temp_db()
    try:
        db.insert_news([_sample_news("AAPL", "苹果财报超预期")])

        mock_completion = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network down")

        with patch("openai.OpenAI", return_value=mock_client):
            result = fetch_news(
                name="Apple", code="AAPL", market="US",
                model="gpt-4o", base_url="http://x", api_key="sk-test",
                limit=5,
            )

        assert len(result) == 1
        assert result[0].code == "AAPL"
    finally:
        os.unlink(path)
        Database._instance = None


def test_parse_llm_json_strips_markdown():
    raw = """```json
[{"date": "2026-05-28", "title": "测试标题", "source": "Reuters"}]
```"""
    items = _parse_llm_json(raw, "NVDA", 5)
    assert len(items) == 1
    assert items[0].title == "测试标题"
    assert items[0].code == "NVDA"


def test_fallback_news_prefers_recent_cache():
    db, path = _temp_db()
    try:
        db.insert_news([_sample_news("TSLA", "特斯拉交付创新高")])
        result = _fallback_news("TSLA", limit=5)
        assert len(result) == 1
    finally:
        os.unlink(path)
        Database._instance = None


if __name__ == "__main__":
    tests = [
        test_insert_news_upsert_no_duplicates,
        test_get_recent_news_with_sentiment_respects_cached_at,
        test_fetch_news_cache_hit_skips_llm,
        test_fetch_news_no_api_key_uses_fallback,
        test_fetch_news_llm_failure_fallback,
        test_parse_llm_json_strips_markdown,
        test_fallback_news_prefers_recent_cache,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"OK  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
