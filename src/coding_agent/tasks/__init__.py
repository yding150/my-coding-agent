"""
coding_agent.tasks - 双轨计划系统

当前阶段（M2.3）已迁入第一轨：
    - `TodoManager`, `TODO_TOOL_SCHEMA`, `register_todo_tools`（会话级轻量内存待办清单）

注：第二轨跨会话持久 DAG 任务图（`TaskStore`、`register_task_tools`）将于 M4.2 迁入。
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
