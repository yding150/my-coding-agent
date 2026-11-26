"""
coding_agent.tools - 统一工具管理与核心基础工具库

当前阶段（M1.2）暴露：
    - 工具管理器：`ToolManager`, `ToolDefinition`
    - 5 大核心工具与装配器：`CORE_SCHEMAS`, `register_core_tools`,
      `run_shell`, `read_file`, `write_file`, `edit_file`, `glob_files`, `_safe_path`

注：`todo_tools` 将于里程碑 M2.3 按增量规划迁入。
"""

from coding_agent.tools.core_tools import (
    CORE_SCHEMAS,
    _safe_path,
    edit_file,
    glob_files,
    read_file,
    register_core_tools,
    run_shell,
    write_file,
)
from coding_agent.tools.manager import ToolDefinition, ToolManager

__all__ = [
    "ToolManager",
    "ToolDefinition",
    "CORE_SCHEMAS",
    "register_core_tools",
    "run_shell",
    "read_file",
    "write_file",
    "edit_file",
    "glob_files",
    "_safe_path",
]
