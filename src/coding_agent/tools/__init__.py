"""
coding_agent.tools - 统一工具管理与核心基础工具库

当前阶段（M2.3）暴露：
    - 工具管理器：`ToolManager`, `ToolDefinition`
    - 5 大核心工具与装配器：`CORE_SCHEMAS`, `register_core_tools`,
      `run_shell`, `read_file`, `write_file`, `edit_file`, `glob_files`, `_safe_path`
    - 会话级 Todo 工具：`TodoManager`, `TODO_TOOL_SCHEMA`, `register_todo_tools`
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
from coding_agent.tools.todo_tools import (
    TODO_TOOL_SCHEMA,
    TodoManager,
    register_todo_tools,
)

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
    "TodoManager",
    "TODO_TOOL_SCHEMA",
    "register_todo_tools",
]
