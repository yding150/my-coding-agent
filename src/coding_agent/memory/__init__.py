"""
coding_agent.memory - 跨会话持久记忆管理模块

在里程碑 M3.2 中迁入，暴露：
    - `MemoryManager`: 记忆写入、索引维护、按需召回、持久性提取与阈值归并管理器
    - 常量与工具：`MEMORY_TYPES`, `TEMPORARY_MEMORY_MARKERS`, `extract_json_array`,
      `memory_slug`, `parse_frontmatter`
"""

from coding_agent.memory.store import (
    MEMORY_TYPES,
    TEMPORARY_MEMORY_MARKERS,
    MemoryManager,
    extract_json_array,
    memory_slug,
    parse_frontmatter,
)

__all__ = [
    "MEMORY_TYPES",
    "TEMPORARY_MEMORY_MARKERS",
    "MemoryManager",
    "extract_json_array",
    "memory_slug",
    "parse_frontmatter",
]
