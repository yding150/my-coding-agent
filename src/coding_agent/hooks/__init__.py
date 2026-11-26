"""
coding_agent.hooks - 统一生命周期钩子管理与切面库

在里程碑 M2.1 中迁入，暴露：
    - 生命周期管理器：`HookEvent`, `HookEntry`, `HookManager`
    - 内置切面与装配器：`DEFAULT_DANGEROUS_PATTERNS`, `tool_logger_hook`,
      `create_permission_hook`, `large_output_warning_hook`,
      `session_summary_hook`, `register_default_hooks`
"""

from coding_agent.hooks.builtins import (
    DEFAULT_DANGEROUS_PATTERNS,
    create_permission_hook,
    large_output_warning_hook,
    register_default_hooks,
    session_summary_hook,
    tool_logger_hook,
)
from coding_agent.hooks.manager import HookEntry, HookEvent, HookManager

__all__ = [
    "HookManager",
    "HookEntry",
    "HookEvent",
    "tool_logger_hook",
    "create_permission_hook",
    "large_output_warning_hook",
    "session_summary_hook",
    "register_default_hooks",
    "DEFAULT_DANGEROUS_PATTERNS",
]
