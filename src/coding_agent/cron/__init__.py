"""
coding_agent.cron - Cron 定时调度引擎（M5.2 已迁入）

暴露的核心接口：
    - `CronJob`, `validate_cron`, `cron_matches`: 定时任务模型与五段式 Cron 表达式匹配器
    - `CronSchedulerManager`, `DEFAULT_SCHEDULED_TASKS_REL`: 持久化调度管理与交付队列状态机
    - `register_cron_tools`: 一键挂载 `schedule_cron`、`list_crons`、`cancel_cron` 工具
"""

from coding_agent.cron.job import CronJob, cron_matches, validate_cron
from coding_agent.cron.manager import (
    DEFAULT_SCHEDULED_TASKS_REL,
    CronSchedulerManager,
)
from coding_agent.cron.tools import register_cron_tools

__all__ = [
    "CronJob",
    "cron_matches",
    "validate_cron",
    "DEFAULT_SCHEDULED_TASKS_REL",
    "CronSchedulerManager",
    "register_cron_tools",
]
