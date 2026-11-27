"""
coding_agent.tasks - 双轨计划系统

已完成双轨迁入：
    - 第一轨（M2.3）：`TodoManager`, `TODO_TOOL_SCHEMA`, `register_todo_tools`（会话级轻量内存待办清单）
    - 第二轨（M4.2）：`TASK_ID_PATTERN`, `TASK_STATUSES`, `Task`, `TaskStore`, `register_task_tools`（`.agent/tasks/` 跨会话持久 DAG 任务依赖图）
"""

from coding_agent.tasks.store import (
    TASK_ID_PATTERN,
    TASK_STATUSES,
    Task,
    TaskStore,
    register_task_tools,
)
from coding_agent.tasks.todo import (
    TODO_TOOL_SCHEMA,
    TodoManager,
    register_todo_tools,
)

__all__ = [
    "TASK_ID_PATTERN",
    "TASK_STATUSES",
    "Task",
    "TaskStore",
    "register_task_tools",
    "TODO_TOOL_SCHEMA",
    "TodoManager",
    "register_todo_tools",
]
