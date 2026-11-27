"""
coding_agent.llm - 模型调用适配层

统一重导出 `coding_agent.llm.client` 中的核心接口，保持与原 `_legacy_ref/common/__init__.py`
一致的导入习惯。

Why:
    此处刻意不在包级 `__init__.py` 重导出同名变量 `client`，避免覆盖同级子模块
    `coding_agent.llm.client` 的命名空间（导致 `from coding_agent.llm import client`
    或 `import coding_agent.llm.client` 取到对象实例而非模块）。需要底层实例时请调用 `get_client()`。
"""

from coding_agent.llm.client import (
    API_KEY,
    BASE_URL,
    DEFAULT_MODEL,
    assistant_message_to_dict,
    call_llm,
    call_llm_response,
    get_client,
    parse_tool_arguments,
)

__all__ = [
    "API_KEY",
    "BASE_URL",
    "DEFAULT_MODEL",
    "get_client",
    "call_llm",
    "call_llm_response",
    "assistant_message_to_dict",
    "parse_tool_arguments",
]
