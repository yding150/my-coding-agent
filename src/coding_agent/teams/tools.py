"""
tools.py - 团队管理工具集封装（register_team_tools）

架构角色：
    向 Lead（主 Agent）的 `ToolManager` 挂载多 Agent 团队协同与审批控制 6 大工具：
    - `spawn_teammate`: 派生并启动持久队友线程；
    - `send_message`: 向指定队友 JSONL 信箱发送指令或补充说明；
    - `request_plan`: 对队友开启计划审批闸门锁住修改权限并通知提交方案；
    - `approve_plan`: 审批队友提交的计划并向其回信解锁或要求修改；
    - `request_shutdown`: 向指定队友发送平滑关机握手请求；
    - `list_teammates`: 查看全队当前状态与计划闸门状态。

暴露的核心函数：
    - `register_team_tools`: 一键挂载上述 6 个 Lead 团队管理工具
"""

from __future__ import annotations

from typing import Optional

from coding_agent.teams.bus import MessageBus
from coding_agent.teams.protocol import TeamProtocols
from coding_agent.teams.runtime import TeamManager
from coding_agent.tools.manager import ToolManager


def register_team_tools(
    tool_mgr: ToolManager,
    team_mgr: TeamManager,
    bus: MessageBus,
    protocols: TeamProtocols,
) -> ToolManager:
    """
    向 Lead 的 `ToolManager` 挂载团队协作与控制工具集（共 6 个工具）。

    :param tool_mgr: 目标父级 `ToolManager`（严禁挂载至 `shared_tools` 或子代理）。
    :param team_mgr: 团队管理器实例。
    :param bus: 团队消息总线实例。
    :param protocols: 团队控制协议实例。
    :return: 注册完成后的原 `ToolManager` 实例。
    """

    @tool_mgr.register(
        name="spawn_teammate",
        description=(
            "启动一个专职持久队友（Teammate）并行推进开发。\n"
            "注意：在调用本工具前，必须已经向用户提出团队分工方案并获得了用户的明确确认（如'开始吧'）。\n"
            "可传入已创建的 task_id 作为初始任务，若 require_plan=true 则强制队友必须先提交方案获批后才能修改代码。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "队友名称代号，如 'alice', 'bob'"},
                "role": {"type": "string", "description": "队友职责描述，如 '后端数据库开发'"},
                "task_id": {"type": "string", "description": "初始认领的任务 ID（可选）"},
                "require_plan": {
                    "type": "boolean",
                    "description": "是否开启计划审批闸门。默认为 false。",
                },
            },
            "required": ["name", "role"],
        },
    )
    def run_spawn_teammate(
        name: str,
        role: str,
        task_id: Optional[str] = None,
        require_plan: bool = False,
    ) -> str:
        try:
            team_mgr.spawn(name, role, task_id=task_id, require_plan=require_plan)
            plan_str = " (已开启计划审批闸门)" if require_plan else ""
            task_str = f"，初始认领: {task_id}" if task_id else ""
            return f"成功启动持久队友 '{name}' ({role}){task_str}{plan_str}。队友已进入工作循环。"
        except Exception as exc:
            return f"启动队友失败: {exc}"

    def _check_online_teammate(name: str) -> Optional[str]:
        """校验目标队友存在且处于在线可接收指令状态（未 SHUTDOWN），不满足时返回错误说明。"""
        mate = team_mgr.get(name)
        if not mate:
            return f"未找到名为 '{name}' 的队友。"
        if mate.status == "SHUTDOWN":
            return (
                f"队友 '{name}' 已关机离线 (SHUTDOWN)，无法再接收消息。"
                f"如需再次启用，请调用 spawn_teammate(name='{name}', ...) 重新上线。"
            )
        return None

    @tool_mgr.register(
        name="send_message",
        description="向指定在线队友发送直接业务指示或补充说明。",
        parameters={
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "接收消息的队友代号，如 'alice'"},
                "content": {"type": "string", "description": "要发送的消息内容"},
            },
            "required": ["to_agent", "content"],
        },
    )
    def run_send_message(to_agent: str, content: str) -> str:
        err = _check_online_teammate(to_agent)
        if err:
            return f"发送失败：{err}"
        bus.send(from_agent="lead", to_agent=to_agent, content=content, msg_type="message")
        return f"已向队友 '{to_agent}' 发送消息。"

    @tool_mgr.register(
        name="request_plan",
        description="要求指定在线队友制定并提交当前任务的详细计划方案，锁定其修改权限直到获批。",
        parameters={
            "type": "object",
            "properties": {
                "teammate": {"type": "string", "description": "队友名称"},
            },
            "required": ["teammate"],
        },
    )
    def run_request_plan(teammate: str) -> str:
        err = _check_online_teammate(teammate)
        if err:
            return err
        protocols.set_plan_requirement(teammate, required=True)
        bus.send(
            from_agent="lead",
            to_agent=teammate,
            content="组长要求你在修改文件前必须先提交计划方案（submit_plan）并等待审批。",
            msg_type="plan_request",
        )
        return f"已向队友 '{teammate}' 下达计划审批约束，修改权限已被锁定。"

    @tool_mgr.register(
        name="approve_plan",
        description="审批队友提交的计划方案。approve=true 将开启其代码修改权限；false 将驳回并要求修改。",
        parameters={
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "计划审批请求 ID，形如 req_xxxx"},
                "approve": {"type": "boolean", "description": "是否批准该方案"},
                "feedback": {"type": "string", "description": "审批意见或修改反馈（可选）"},
            },
            "required": ["request_id", "approve"],
        },
    )
    def run_approve_plan(request_id: str, approve: bool, feedback: str = "") -> str:
        req = protocols.approve_plan(request_id, approve=approve, feedback=feedback)
        if not req:
            return f"未找到请求 ID 为 '{request_id}' 的计划审批项。"
        bus.send(
            from_agent="lead",
            to_agent=req.sender,
            content=feedback or ("计划已获批准，请着手推进！" if approve else "计划已被驳回，请调整后重新提交。"),
            msg_type="plan_approval_response",
            metadata={"request_id": request_id, "approved": approve},
        )
        status_str = "批准放行" if approve else "驳回修改"
        return f"已成功对队友 '{req.sender}' 的方案予以 {status_str}。"

    @tool_mgr.register(
        name="request_shutdown",
        description="向指定队友发出平滑关机请求。队友将在完成当前手头工作后安全下线。",
        parameters={
            "type": "object",
            "properties": {
                "teammate": {"type": "string", "description": "待关机的队友名称"}
            },
            "required": ["teammate"],
        },
    )
    def run_request_shutdown(teammate: str) -> str:
        mate = team_mgr.get(teammate)
        if not mate:
            return f"未找到队友 '{teammate}'"
        if mate.status == "SHUTDOWN":
            return f"队友 '{teammate}' 当前已处于关机状态 (SHUTDOWN)，无需重复关机。"
        req = protocols.create_shutdown_request(teammate)
        bus.send(
            from_agent="lead",
            to_agent=teammate,
            content="团队请求平滑关机，请在当前步骤完成后退出。",
            msg_type="shutdown_request",
            metadata={"request_id": req.request_id},
        )
        return f"已向队友 '{teammate}' 发出平滑关机请求 (Request ID: {req.request_id})。"

    @tool_mgr.register(
        name="list_teammates",
        description="列出当前团队全体队友的在线状态（WORK / IDLE / SHUTDOWN）与当前认领任务。",
        parameters={"type": "object", "properties": {}},
    )
    def run_list_teammates() -> str:
        mates = team_mgr.list_all()
        if not mates:
            return "当前尚未启动任何团队队友。"
        lines = ["【团队在线成员看板】:"]
        for m in mates:
            if m.status == "SHUTDOWN":
                task_str = "已关机离线（可重新 spawn_teammate 复用）"
            elif m.current_task_id:
                task_str = f"认领: {m.current_task_id}"
            else:
                task_str = "空闲等待任务"
            gate = protocols.get_plan_gate(m.name)
            lines.append(f"  - [{m.status}] {m.name} ({m.role}) -> {task_str} [闸门: {gate}]")
        return "\n".join(lines)

    @tool_mgr.register(
        name="read_inbox",
        description=(
            "原子读取并消费 Lead 收件箱中的全部未读队友消息（读取后立即出队清空）。"
            "禁止用 read_file 直接读取 .agent/mailboxes/lead.jsonl，否则不会出队导致后台重复唤醒。"
        ),
        parameters={"type": "object", "properties": {}},
    )
    def run_read_inbox() -> str:
        msgs = bus.read_inbox("lead")
        if not msgs:
            return "当前 Lead 收件箱暂无未读消息。"
        lines = [f"已读取并消费 {len(msgs)} 条 Lead 收件箱消息:"]
        for m in msgs:
            meta = m.get("metadata") or {}
            meta_str = f" (metadata={meta})" if meta else ""
            lines.append(
                f"- [{m.get('type', 'message')}] 来自 '{m.get('from', 'unknown')}': {m.get('content', '')}{meta_str}"
            )
        return "\n".join(lines)

    return tool_mgr
