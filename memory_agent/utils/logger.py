"""
日志配置模块。

提供 setup_logging 函数，支持可读文本和结构化 JSON 两种输出格式，
并自动抑制 chromadb 的 DEBUG 日志以减少输出噪音。
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """JSON 结构化日志格式化器。

    每行输出一条 JSON，包含 timestamp、level、module、message 字段。
    """

    def format(self, record: logging.LogRecord) -> str:
        """将日志记录格式化为单行 JSON 字符串。

        Args:
            record: Python logging 日志记录对象。

        Returns:
            单行 JSON 字符串。
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        # 若存在异常信息，附加到日志条目中
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(config: "MemoryConfig") -> None:  # type: ignore  # noqa: F821
    """配置 Python 标准库 logging。

    从 config.log_level 读取日志级别，支持 text 和 json 两种输出格式。
    格式通过 LOGGING_FORMAT 环境变量切换（默认 text）。
    同时将 chromadb 的日志级别提升至 WARNING，避免调试日志泛滥。

    Args:
        config: MemoryConfig 实例，提供 log_level 配置。
    """
    log_level = config.log_level_int

    # 确定日志格式
    logging_format = os.getenv("LOGGING_FORMAT", "text").strip().lower()

    # 创建 stderr handler
    handler = logging.StreamHandler(sys.stderr)

    if logging_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        # 默认 text 格式
        text_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(text_format))

    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # 移除已有 handler 避免重复添加（例如在测试中多次调用）
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # 抑制 chromadb 的 DEBUG 日志
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    # 抑制其他频繁输出 DEBUG 日志的第三方库
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
