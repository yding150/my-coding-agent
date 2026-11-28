"""
coding_agent.teams - 多 Agent 团队协同运行时

当前阶段（M6.1）已迁入基础通信与控制协议层：
    - `DEFAULT_MAILBOX_DIR`, `MessageBus`: 收拢于 `.agent/mailboxes/` 的 JSONL 异步消息总线
    - `PLAN_GATED_TOOLS`, `ProtocolState`, `TeamProtocols`: 计划审批闸门与平滑关机协议状态机

注：常驻队友线程 (`TeammateRuntime`, `TeamManager`) 与团队工具 (`register_team_tools`) 将于 M6.2 迁入并接入 `runtime.py`。
"""

from coding_agent.teams.bus import DEFAULT_MAILBOX_DIR, MessageBus
from coding_agent.teams.protocol import (
    PLAN_GATED_TOOLS,
    ProtocolState,
    TeamProtocols,
)

__all__ = [
    "DEFAULT_MAILBOX_DIR",
    "MessageBus",
    "PLAN_GATED_TOOLS",
    "ProtocolState",
    "TeamProtocols",
]
