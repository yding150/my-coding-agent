"""
coding_agent.subagent - 一次性隔离子代理模块（Ephemeral Subagent）

在里程碑 M4.1 中迁入，暴露：
    - `FORBIDDEN_SUBAGENT_TOOLS`: 防递归剥离工具名单
    - `SubagentRunner`: 独立干净上下文的同步子代理运行器
    - `register_subagent_tools`: 向父级 `ToolManager` 挂载 `task` 工具
"""

from coding_agent.subagent.runner import FORBIDDEN_SUBAGENT_TOOLS, SubagentRunner
from coding_agent.subagent.tools import register_subagent_tools

__all__ = [
    "FORBIDDEN_SUBAGENT_TOOLS",
    "SubagentRunner",
    "register_subagent_tools",
]
