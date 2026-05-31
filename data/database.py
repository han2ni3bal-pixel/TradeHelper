"""
SQLite 数据库操作层

使用单例模式管理 SQLite 连接，提供以下功能：
  - 数据库初始化与自动建表（4 张核心表）
  - 股票信息 CRUD
  - 股价历史数据批量读写
  - 分析报告增删改查（含评分更新）
  - 新闻情感分析结果存储

表结构：
  stocks       — 股票基本信息缓存（主键：code）
  price_history — 日 K 线历史数据（联合主键：code + date）
  reports      — 分析报告记录（含用户评分，外键关联 stocks）
  news_sentiment — 新闻情感分析结果缓存

【扩展点】如需新增表或字段：
  1. 在 CREATE_TABLES_SQL 中添加 DDL 语句
  2. 在 data/models.py 中定义对应的 dataclass
  3. 在本模块添加对应的 CRUD 方法
"""

import sqlite3
import logging
import threading
from datetime import datetime
from typing import Optional

from data.models import StockInfo, PriceData, AnalysisReport, NewsItem
from config.settings import Settings

logger = logging.getLogger(__name__)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str, default: str = "''"):
    """SQLite 兼容的"加列如果不存在"——遍历 PRAGMA 列名。"""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {row[1] for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}")
        logger.info(f"DB migrate: added column {table}.{column}")


# ======================== 数据库建表 DDL ========================

CREATE_TABLES_SQL = """
-- 股票基本信息表：缓存从 API 获取的股票元数据
CREATE TABLE IF NOT EXISTS stocks (
    code TEXT PRIMARY KEY,           -- 股票代码（A 股 6 位数字 / 美股字母）
    name TEXT NOT NULL,              -- 股票名称
    market TEXT NOT NULL,            -- 市场："A" / "US"
    industry TEXT DEFAULT '',        -- 所属行业
    description TEXT DEFAULT '',     -- 公司简介
    update_time TEXT DEFAULT ''     -- 信息更新时间
);

-- 日 K 线历史数据表：存储股价 OHLCV 数据
CREATE TABLE IF NOT EXISTS price_history (
    code TEXT NOT NULL,              -- 股票代码
    date TEXT NOT NULL,              -- 交易日期 "YYYY-MM-DD"
    open REAL NOT NULL,              -- 开盘价
    high REAL NOT NULL,              -- 最高价
    low REAL NOT NULL,               -- 最低价
    close REAL NOT NULL,             -- 收盘价
    volume REAL NOT NULL,            -- 成交量
    PRIMARY KEY (code, date)         -- 同一天同一股票只有一条记录
);

-- 索引：加速按股票代码和日期查询
CREATE INDEX IF NOT EXISTS idx_price_code ON price_history(code);
CREATE INDEX IF NOT EXISTS idx_price_date ON price_history(date);

-- 分析报告表：存储每次分析生成的完整报告
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    code TEXT NOT NULL,                     -- 股票代码
    name TEXT NOT NULL,                     -- 股票名称（冗余，便于展示）
    market TEXT NOT NULL,                   -- 市场
    backtest_period TEXT NOT NULL,         -- 回测周期
    create_time TEXT NOT NULL,             -- 创建时间
    content TEXT NOT NULL,                  -- 报告正文（Markdown）
    chart_path TEXT DEFAULT '',            -- K 线图 PNG 路径（生成报告时写入）
    pdf_path TEXT DEFAULT '',              -- PDF 导出路径
    rating INTEGER DEFAULT NULL,           -- 用户评分 1-5
    rated_at TEXT DEFAULT ''              -- 评分时间
);

CREATE INDEX IF NOT EXISTS idx_reports_code ON reports(code);
CREATE INDEX IF NOT EXISTS idx_reports_time ON reports(create_time);

-- 新闻情感分析结果表：缓存分析与情感标签
CREATE TABLE IF NOT EXISTS news_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    code TEXT NOT NULL,                     -- 关联股票代码
    date TEXT NOT NULL,                     -- 新闻日期
    title TEXT NOT NULL,                    -- 新闻标题
    source TEXT DEFAULT '',                 -- 新闻来源
    sentiment TEXT DEFAULT '',              -- 情感标签
    confidence REAL DEFAULT 0.0,            -- 置信度
    cached_at TEXT DEFAULT '',              -- 入库/更新时间（ISO8601，用于 24h 复用）
    UNIQUE(code, date, title)
);

CREATE INDEX IF NOT EXISTS idx_news_code ON news_sentiment(code);
CREATE INDEX IF NOT EXISTS idx_news_date ON news_sentiment(date);
CREATE INDEX IF NOT EXISTS idx_news_cached_at ON news_sentiment(cached_at);
"""


def _migrate_news_sentiment(conn: sqlite3.Connection):
    """news_sentiment：去重、补 cached_at、确保业务唯一索引。"""
    _ensure_column(conn, "news_sentiment", "cached_at", "TEXT", "''")
    conn.execute(
        """UPDATE news_sentiment
           SET cached_at = date || 'T12:00:00'
           WHERE cached_at = '' OR cached_at IS NULL"""
    )
    conn.execute(
        """DELETE FROM news_sentiment
           WHERE id NOT IN (
               SELECT MIN(id) FROM news_sentiment GROUP BY code, date, title
           )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_news_code_date_title
           ON news_sentiment(code, date, title)"""
    )


class Database:
    """
    SQLite 数据库单例类。

    使用方式：
        db = Database.init()              # 初始化（使用默认路径）
        db = Database.init("/custom/path/tradehelper.db")  # 自定义路径
        db = Database()                   # 后续任意位置获取实例

    线程安全：使用 check_same_thread=False 允许多线程共享连接。
    WAL 模式：允许读写并发，提升性能。
    """

    _instance: Optional["Database"] = None  # 单例实例

    def __new__(cls):
        """确保全局只有一个 Database 实例。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conn = None
            # 写操作锁：分析线程与 UI 线程共享同一连接时序列化写入，
            # 防止 "database is locked" 错误（WAL 仅保护读读并发）。
            cls._instance._write_lock = threading.Lock()
        return cls._instance

    @classmethod
    def init(cls, db_path: str | None = None) -> "Database":
        """
        初始化数据库连接并建表。

        Args:
            db_path: SQLite 文件路径，默认从 Settings 中读取

        Returns:
            Database 单例实例
        """
        instance = cls()
        if db_path is None:
            db_path = Settings().db_path
        instance._connect(db_path)
        return instance

    def _open(self, db_path: str) -> sqlite3.Connection:
        """
        打开一个新的 SQLite 连接并应用全部 PRAGMA / 建表 / schema 迁移。

        所有连接入口（init / 懒加载重连）都走这里，避免重连时丢失 PRAGMA。
        """
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")    # Write-Ahead Logging
        conn.execute("PRAGMA foreign_keys=ON")      # 启用外键约束
        conn.executescript(CREATE_TABLES_SQL)       # 自动建表
        # 老版本数据库 schema 迁移（chart_path 是后期新增字段）
        _ensure_column(conn, "reports", "chart_path", "TEXT", "''")
        _migrate_news_sentiment(conn)
        conn.commit()
        return conn

    def _connect(self, db_path: str):
        """对外建立连接入口（首次 init 时调用）。"""
        self._conn = self._open(db_path)
        self._db_path = db_path
        logger.info(f"Database connected: {db_path}")

    @property
    def conn(self) -> sqlite3.Connection:
        """获取数据库连接（懒加载：如已关闭则自动重连，复用 _open 应用 PRAGMA）。"""
        if self._conn is None:
            db_path = getattr(self, "_db_path", None) or Settings().db_path
            self._conn = self._open(db_path)
            self._db_path = db_path
            logger.info(f"Database reconnected: {db_path}")
        return self._conn

    def execute(self, sql: str, params=None):
        """执行 SQL 语句的快捷方法（仅适用于读；写请走 _execute_write）。"""
        return self.conn.execute(sql, params or ())

    def _execute_write(self, sql: str, params=None):
        """串行化执行单条写 SQL，并在同一锁内提交。"""
        with self._write_lock:
            cursor = self.conn.execute(sql, params or ())
            self.conn.commit()
            return cursor

    def _executemany_write(self, sql: str, seq_of_params):
        """串行化执行批量写 SQL。"""
        with self._write_lock:
            self.conn.executemany(sql, seq_of_params)
            self.conn.commit()


    # ======================== 股票信息 ========================

    def upsert_stock(self, stock: StockInfo):
        """
        插入或更新股票信息（使用 INSERT OR REPLACE）。

        Args:
            stock: StockInfo 实例
        """
        sql = """INSERT OR REPLACE INTO stocks (code, name, market, industry, description, update_time)
                 VALUES (?, ?, ?, ?, ?, ?)"""
        self._execute_write(sql, (stock.code, stock.name, stock.market,
                                  stock.industry, stock.description,
                                  stock.update_time))


    # ======================== 股价历史 ========================

    def insert_prices(self, prices: list[PriceData]):
        """
        批量插入股价数据（使用 INSERT OR REPLACE 避免重复）。

        Args:
            prices: PriceData 列表
        """
        if not prices:
            return
        sql = """INSERT OR REPLACE INTO price_history (code, date, open, high, low, close, volume)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        data = [(p.code, p.date, p.open, p.high, p.low, p.close, p.volume)
                for p in prices]
        self._executemany_write(sql, data)

    def get_prices(self, code: str, start_date: str = "", end_date: str = "") -> list[PriceData]:
        """
        按代码和日期范围查询股价数据。

        Args:
            code: 股票代码
            start_date: 起始日期（可选，含）
            end_date: 结束日期（可选，含）

        Returns:
            按日期升序排列的 PriceData 列表
        """
        conditions = ["code = ?"]
        params = [code]
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        sql = f"SELECT * FROM price_history WHERE {' AND '.join(conditions)} ORDER BY date ASC"
        rows = self.execute(sql, params).fetchall()
        return [PriceData.from_dict(dict(r)) for r in rows]


    # ======================== 分析报告 ========================

    def insert_report(self, report: AnalysisReport) -> int:
        """
        插入一条分析报告记录。

        Args:
            report: AnalysisReport 实例

        Returns:
            新记录的 ID（数据库自增主键）
        """
        sql = """INSERT INTO reports
                 (code, name, market, backtest_period, create_time, content,
                  chart_path, pdf_path, rating, rated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor = self._execute_write(sql, (
            report.code, report.name, report.market, report.backtest_period,
            report.create_time, report.content,
            report.chart_path or "", report.pdf_path or "",
            report.rating, report.rated_at or ""
        ))
        return cursor.lastrowid


    def get_all_reports(self, limit: int = 50) -> list[AnalysisReport]:
        """
        获取所有报告（按时间倒序）。

        Args:
            limit: 最大返回数量

        Returns:
            AnalysisReport 列表
        """
        rows = self.execute(
            "SELECT * FROM reports ORDER BY create_time DESC LIMIT ?", (limit,)
        ).fetchall()
        return [AnalysisReport.from_dict(dict(r)) for r in rows]

    def get_reports_by_code(self, code: str) -> list[AnalysisReport]:
        """
        按股票代码筛选报告。

        Args:
            code: 股票代码

        Returns:
            该股票的所有报告（时间倒序）
        """
        rows = self.execute(
            "SELECT * FROM reports WHERE code = ? ORDER BY create_time DESC", (code,)
        ).fetchall()
        return [AnalysisReport.from_dict(dict(r)) for r in rows]

    def update_report_rating(self, report_id: int, rating: int):
        """
        更新报告的用户评分。

        评分数据存储后可用于：
          - 分析用户偏好的策略类型
          - 为后续策略参数优化提供反馈信号

        Args:
            report_id: 报告 ID
            rating: 评分（1-5）
        """
        now = datetime.now().isoformat()
        self._execute_write(
            "UPDATE reports SET rating = ?, rated_at = ? WHERE id = ?",
            (rating, now, report_id)
        )

    def update_report_pdf(self, report_id: int, pdf_path: str):
        """
        更新报告的 PDF 文件路径（PDF 导出后调用）。

        Args:
            report_id: 报告 ID
            pdf_path: PDF 文件的绝对路径
        """
        self._execute_write(
            "UPDATE reports SET pdf_path = ? WHERE id = ?",
            (pdf_path, report_id)
        )


    def delete_report(self, report_id: int):
        """
        删除指定报告（注意：不会删除磁盘上的 PDF 文件）。

        Args:
            report_id: 报告 ID
        """
        self._execute_write("DELETE FROM reports WHERE id = ?", (report_id,))

    # ======================== 新闻情感 ========================

    def insert_news(self, news_list: list[NewsItem]):
        """
        批量插入新闻情感分析结果。

        Args:
            news_list: NewsItem 列表
        """
        if not news_list:
            return
        now = datetime.now().isoformat(timespec="seconds")
        sql = """INSERT OR REPLACE INTO news_sentiment
                 (code, date, title, source, sentiment, confidence, cached_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        data = [
            (n.code, n.date, n.title, n.source, n.sentiment, n.confidence, now)
            for n in news_list
        ]
        self._executemany_write(sql, data)

    def get_news(self, code: str, limit: int = 20) -> list[NewsItem]:
        """
        获取某股票最近的新闻情感记录。

        Args:
            code: 股票代码
            limit: 最大返回条数

        Returns:
            NewsItem 列表（按日期倒序）
        """
        rows = self.execute(
            "SELECT * FROM news_sentiment WHERE code = ? ORDER BY date DESC LIMIT ?",
            (code, limit)
        ).fetchall()
        return [NewsItem.from_dict(dict(r)) for r in rows]

    def get_recent_news_with_sentiment(
        self, code: str, hours: int = 24, limit: int = 50
    ) -> list[NewsItem]:
        """
        获取指定时间窗口内、且已经完成情感分析（sentiment 非空）的新闻。

        用于新闻 24 小时复用：避免每次分析都重新拉取并跑 FinBERT。

        Args:
            code: 股票代码
            hours: 时间窗口（小时），默认 24h
            limit: 最大返回条数

        Returns:
            NewsItem 列表（按日期倒序）；窗口内无缓存时返回空列表
        """
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")
        rows = self.execute(
            """SELECT * FROM news_sentiment
               WHERE code = ? AND cached_at >= ? AND sentiment != ''
               ORDER BY date DESC LIMIT ?""",
            (code, cutoff, limit),
        ).fetchall()
        return [NewsItem.from_dict(dict(r)) for r in rows]

    def get_news_with_sentiment(self, code: str, limit: int = 20) -> list[NewsItem]:
        """获取已完成情感分析的新闻（不限时间窗口，按新闻日期倒序）。"""
        rows = self.execute(
            """SELECT * FROM news_sentiment
               WHERE code = ? AND sentiment != ''
               ORDER BY date DESC LIMIT ?""",
            (code, limit),
        ).fetchall()
        return [NewsItem.from_dict(dict(r)) for r in rows]
