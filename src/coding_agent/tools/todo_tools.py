"""
todo_tools.py - 会话级 Todo 工具兼容重导出模块

Why:
    原教学代码同时存在 `common/tasks/todo.py` 与 `common/tools/todo_tools.py` 两份实现。
    新仓库将唯一事实实现收拢于 `coding_agent.tasks.todo`，本模块仅作兼容重导出，
    避免重复维护两份逻辑与签名漂移。
"""

from coding_agent.tasks.todo import (
    TODO_TOOL_SCHEMA,
    TodoManager,
    register_todo_tools,
)

__all__ = [
    "TODO_TOOL_SCHEMA",
    "TodoManager",
    "register_todo_tools",
]
