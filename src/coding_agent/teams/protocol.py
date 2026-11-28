"""
protocol.py - 团队控制协议与计划审批闸门（TeamProtocols）

架构角色：
    定义多 Agent 团队协作中的强类型控制协议状态机与安全权限闸门：
    1. `ProtocolState`：携带唯一 `request_id` 的请求/审批状态记录，杜绝意图混淆与重放；
    2. 计划审批闸门（Plan Approval Gate）：在 `require_plan=True` 约束下锁定修改型工具
       （`write_file`, `edit_file`, `run_shell`），放行只读工具以便先调研再提交计划；
    3. 平滑关机协议（Graceful Shutdown）：Lead 发起 `shutdown` 请求，队友在安全边界响应确认。

暴露的核心接口：
    - `PLAN_GATED_TOOLS`: 受计划审批闸门拦截的写/执行工具名集合
    - `ProtocolState`: 团队控制握手状态实体
    - `TeamProtocols`: 线程安全的计划审批与平滑关机协议管理器
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

PLAN_GATED_TOOLS: tuple[str, ...] = ("write_file", "edit_file", "run_shell")


@dataclass
class ProtocolState:
    """团队控制协议握手状态实体。"""

    request_id: str  # 全局唯一请求 ID（形如 req_a1b2c3d4）
    type: str  # "shutdown" | "plan_approval"
    sender: str  # 发起方 Agent 名称
    target: str  # 接收方/被控制方 Agent 名称
    status: str  # "pending" | "approved" | "rejected"
    payload: str  # 请求正文（如方案描述或关机原因）
    task_id: Optional[str] = None
    feedback: str = ""  # Lead 审批通过或驳回时附加的说明


class TeamProtocols:
    """
    团队控制协议管理器与修改型工具权限闸门。

    生命周期：
        在 `TeamManager` 或运行期单会话中常驻，负责跨线程维护每名队友的计划审批态
        与待决关机握手记录。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.pending_requests: Dict[str, ProtocolState] = {}
        # agent_name -> "not_required" | "required" | "pending" | "approved" | "rejected"
        self.plan_gates: Dict[str, str] = {}
        self.plan_request_ids: Dict[str, str] = {}

    def new_request_id(self) -> str:
        """生成全局唯一请求 ID。"""
        return f"req_{secrets.token_hex(4)}"

    # ----------------------------------------------------
    # 1. 计划审批闸门（Plan Approval Gate）
    # ----------------------------------------------------
    def set_plan_requirement(self, agent: str, required: bool) -> None:
        """设定指定 Agent 是否在修改文件/执行命令前受计划审批约束。"""
        with self._lock:
            self.plan_gates[agent] = "required" if required else "not_required"
            if not required:
                self.plan_request_ids.pop(agent, None)

    def get_plan_gate(self, agent: str) -> str:
        """查询指定 Agent 当前所处的计划审批状态。"""
        with self._lock:
            return self.plan_gates.get(agent, "not_required")

    def submit_plan(self, agent: str, plan_text: str, task_id: Optional[str] = None) -> ProtocolState:
        """队友提交执行计划方案，状态切换为 `pending` 并生成关联 `request_id`。"""
        with self._lock:
            req_id = self.new_request_id()
            req = ProtocolState(
                request_id=req_id,
                type="plan_approval",
                sender=agent,
                target="lead",
                status="pending",
                payload=plan_text,
                task_id=task_id,
            )
            self.pending_requests[req_id] = req
            self.plan_gates[agent] = "pending"
            self.plan_request_ids[agent] = req_id
            return req

    def approve_plan(
        self,
        request_id: str,
        approve: bool,
        feedback: str = "",
    ) -> Optional[ProtocolState]:
        """Lead 对计划审批请求执行放行 (`approved`) 或驳回 (`rejected`)。"""
        with self._lock:
            req = self.pending_requests.get(request_id)
            if not req or req.type != "plan_approval":
                return None

            req.status = "approved" if approve else "rejected"
            req.feedback = feedback
            self.plan_gates[req.sender] = req.status
            return req

    def is_tool_allowed(self, agent: str, tool_name: str) -> Tuple[bool, str]:
        """
        检查指定工具是否被团队计划闸门放行。

        Why:
            只读探查工具（如 `read_file`、`glob_files`）始终放行，保证队友在提出靠谱方案前能先阅读代码；
            修改型工具（`write_file`、`edit_file`、`run_shell`）在计划未 `approved` 前一律拦截，
            防止队友未征得 Lead 同意即直接破坏工作区。
        """
        with self._lock:
            gate = self.plan_gates.get(agent, "not_required")
            if tool_name in PLAN_GATED_TOOLS:
                if gate in ("required", "pending", "rejected"):
                    return (
                        False,
                        f"操作被团队计划闸门拦截: 当前计划状态为 '{gate}'。"
                        "在计划获批前禁止执行写文件、编辑文件或 Shell 操作。请先调用 submit_plan 提交计划方案等待批准。",
                    )
            return True, ""

    # ----------------------------------------------------
    # 2. 平滑关机协议（Graceful Shutdown）
    # ----------------------------------------------------
    def create_shutdown_request(self, target_agent: str) -> ProtocolState:
        """Lead 为指定队友创建带唯一 `request_id` 的关机握手请求。"""
        with self._lock:
            req_id = self.new_request_id()
            req = ProtocolState(
                request_id=req_id,
                type="shutdown",
                sender="lead",
                target=target_agent,
                status="pending",
                payload="请在当前步骤完成后安全停止",
            )
            self.pending_requests[req_id] = req
            return req

    def acknowledge_shutdown(self, request_id: str, agent: str) -> bool:
        """队友校验目标身份与 `request_id` 匹配后确认关机请求。"""
        with self._lock:
            req = self.pending_requests.get(request_id)
            if req and req.type == "shutdown" and req.target == agent:
                req.status = "approved"
                return True
            return False
