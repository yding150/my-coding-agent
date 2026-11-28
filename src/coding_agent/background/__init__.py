"""
coding_agent.background - 异步后台任务管理（M5.1 已迁入）

暴露的核心接口：
    - `BackgroundTaskManager`: 非阻塞子进程执行、超时强杀与 `<task_notification>` 通知注入
    - `patch_run_shell_with_background`: 为 `run_shell` 增加 `run_in_background` 后台派发参数
    - `register_background_tools`: 一键升级 `run_shell` 并挂载 `check_bg_task`/`kill_bg_task`/`list_bg_tasks`
"""

from coding_agent.background.manager import BackgroundTaskManager
from coding_agent.background.tools import (
    patch_run_shell_with_background,
    register_background_tools,
)

__all__ = [
    "BackgroundTaskManager",
    "patch_run_shell_with_background",
    "register_background_tools",
]
