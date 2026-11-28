"""
coding_agent.teams - 多 Agent 团队协同运行时（M6.2 全量就绪）

暴露的核心接口：
    - 通信总线：`DEFAULT_MAILBOX_DIR`, `MessageBus`
    - 控制协议：`PLAN_GATED_TOOLS`, `ProtocolState`, `TeamProtocols`
    - 队友运行时：`TeammateRuntime`, `TeamManager`
    - 团队工具装配：`register_team_tools`
"""

from coding_agent.teams.bus import DEFAULT_MAILBOX_DIR, MessageBus
from coding_agent.teams.protocol import (
    PLAN_GATED_TOOLS,
    ProtocolState,
    TeamProtocols,
)
from coding_agent.teams.runtime import TeamManager, TeammateRuntime
from coding_agent.teams.tools import register_team_tools

__all__ = [
    "DEFAULT_MAILBOX_DIR",
    "MessageBus",
    "PLAN_GATED_TOOLS",
    "ProtocolState",
    "TeamProtocols",
    "TeammateRuntime",
    "TeamManager",
    "register_team_tools",
]
