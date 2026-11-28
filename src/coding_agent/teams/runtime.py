"""
runtime.py - 常驻队友运行时与团队主管协调器（TeammateRuntime & TeamManager）

架构角色：
    为 Coding Agent 提供可跨任务存活的多 Agent 团队并发运行引擎（代理第二轨）：
    1. `TeammateRuntime`：运行在独立守护线程上的持久执行单元，围绕 `WORK <-> IDLE <-> SHUTDOWN`
       状态机循环推进，与一次性 `SubagentRunner` 不同，队友会保留上下文并在空闲时自动接单；
    2. `TeamManager`：统筹派生队友 (`spawn`)、绑定制约初始任务 (`task_id`/`require_plan`)、
       查询成员看板及安全关闭清理所有常驻线程 (`shutdown_all`)。

解决的核心痛点：
    1. 共享任务板原子自认领：队友进入 `IDLE` 且收件箱无信件时，自动扫描 `.agent/tasks/` 中
       前置依赖全部满足且无人认领的 `pending` 任务并调用 `claim()` 抢占执行；
    2. 计划审批闸门拦截：在 `require_plan=True` 模式下，修改型工具在 `_execute_tool_with_gate()`
       层面直接被 `TeamProtocols` 拦下，必须调用 `submit_plan()` 投递至 Lead 并进入 `IDLE` 等审批；
    3. 协议安全历史裁剪：超过 20 条消息时按窗口滑动裁剪并回退保护 `assistant(tool_calls)` -> `tool` 配对；
    4. 双事件异步上报：任务阶段完成后按 `result` -> `idle_notification` 顺序投递至 Lead 邮箱。

暴露的核心类：
    - `TeammateRuntime`: 单个常驻队友后台线程状态机
    - `TeamManager`: 多队友生命周期协调器
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from coding_agent.llm import assistant_message_to_dict, call_llm, parse_tool_arguments
from coding_agent.tasks.store import TaskStore
from coding_agent.teams.bus import MessageBus
from coding_agent.teams.protocol import TeamProtocols
from coding_agent.tools.core_tools import register_core_tools
from coding_agent.tools.manager import ToolManager


class TeammateRuntime:
    """
    单个常驻队友执行单元（WORK <-> IDLE <-> SHUTDOWN）。

    生命周期：
        由 `TeamManager.spawn()` 启动独立后台守护线程；工作时在内部局部 `messages` 上运行
        最多 15 轮工具调用，空闲时调用 `bus.wait_for_messages()` 低功耗等待信件或扫描任务板，
        收到 `shutdown_request` 或调用 `stop_event.set()` 后安全下线。
    """

    def __init__(
        self,
        name: str,
        role: str,
        bus: MessageBus,
        task_store: TaskStore,
        protocols: TeamProtocols,
        workdir: Optional[Path] = None,
        initial_task_id: Optional[str] = None,
        require_plan: bool = False,
        llm_caller: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.name: str = name
        self.role: str = role
        self.bus: MessageBus = bus
        self.task_store: TaskStore = task_store
        self.protocols: TeamProtocols = protocols
        self.workdir: Path = (workdir or Path.cwd()).resolve()
        self._llm_caller: Optional[Callable[..., Any]] = llm_caller

        self.current_task_id: Optional[str] = initial_task_id
        self.status: str = "IDLE" if initial_task_id is None else "WORK"
        self.stop_event: threading.Event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.consecutive_work_turns: int = 0

        # 初始化该队友在协议栈中的计划审批约束
        self.protocols.set_plan_requirement(self.name, require_plan)

        # 每个队友维护完全独立隔离的对话历史上下文
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

        # 仅挂载基础核心工具 + 团队协作 3 工具（不挂载 spawn_teammate/task 防递归）
        self.tools: ToolManager = ToolManager()
        register_core_tools(self.tools, workdir=self.workdir)
        self._register_teammate_tools()

    def _build_system_prompt(self) -> str:
        """构建绑定队友身份、工作区与审批协作纪律的系统提示词。"""
        return (
            f"你是团队中的专职技术队友 '{self.name}'，职责角色: {self.role}。\n"
            f"工作根目录: {self.workdir}\n"
            f"【协作规则】:\n"
            f"1. 专注当前认领的任务。任务完成后务必调用 `complete_task` 工具标记完成并解锁下游。\n"
            f"2. 若当前任务需要计划审批，必须先调用 `submit_plan` 提交方案并等待批准，获批前写操作将被系统拦截。\n"
            f"3. 遇到困难或需要向组长汇报，可调用 `send_to_lead` 发送消息。\n"
            f"任务完成后直接给出清晰的工作总结。"
        )

    def _register_teammate_tools(self) -> None:
        """挂载队友专属协作工具 (`complete_task`, `submit_plan`, `send_to_lead`)。"""

        @self.tools.register(
            name="complete_task",
            description="完成当前认领的任务，推进状态机至 completed 并自动解锁下游依赖任务。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID，如 task_1a2b3c4d"}
                },
                "required": ["task_id"],
            },
        )
        def run_complete_task(task_id: str) -> str:
            res = self.task_store.complete(task_id, owner=self.name)
            if "已完成任务" in res or "已标记完成" in res or "成功" in res:
                self.current_task_id = None
            return res

        @self.tools.register(
            name="submit_plan",
            description="向组长 (Lead) 提交当前任务的技术实施计划方案，请求审批以解锁写权限。",
            parameters={
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "计划方案详细说明"}
                },
                "required": ["plan"],
            },
        )
        def run_submit_plan(plan: str) -> str:
            req = self.protocols.submit_plan(self.name, plan, task_id=self.current_task_id)
            self.bus.send(
                from_agent=self.name,
                to_agent="lead",
                content=f"队友 '{self.name}' 提交了执行计划 (请求ID: {req.request_id}):\n{plan}",
                msg_type="plan_approval_request",
                metadata={"request_id": req.request_id, "task_id": self.current_task_id},
            )
            # Why: 提交计划后主动挂起转入 IDLE 并唤醒收件箱等待，避免模型在等待审批期间空转烧 Token
            self.status = "IDLE"
            return (
                f"计划已成功提交至 Lead 审批中 (Request ID: {req.request_id})。"
                "当前已挂起等待审批通知，切勿重复提交。"
            )

        @self.tools.register(
            name="send_to_lead",
            description="向团队组长 (Lead) 发送说明信息或汇报进度。",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "消息内容"}
                },
                "required": ["content"],
            },
        )
        def run_send_to_lead(content: str) -> str:
            self.bus.send(from_agent=self.name, to_agent="lead", content=content, msg_type="message")
            return "消息已成功发送至 Lead 收件箱。"

    def start(self) -> None:
        """启动队友后台守护线程。"""
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run_loop,
            name=f"Teammate-{self.name}",
            daemon=True,
        )
        self.thread.start()
        print(f"  \033[35m[Teammate 启动]\033[0m 队友 '{self.name}' ({self.role}) 已上线")

    def _execute_tool_with_gate(self, name: str, args: dict) -> str:
        """执行工具，并在修改型工具前强制经过团队计划审批闸门校验。"""
        allowed, reason = self.protocols.is_tool_allowed(self.name, name)
        if not allowed:
            return reason
        return str(self.tools.execute(name, args))

    def _run_loop(self) -> None:
        """核心 WORK <-> IDLE 状态机循环（带连续 15 轮熔断保护）。"""
        while not self.stop_event.is_set():
            if self.status == "WORK":
                self.consecutive_work_turns += 1
                if self.consecutive_work_turns > 15:
                    print(
                        f"  \033[33m[Teammate 熔断]\033[0m 队友 '{self.name}' 连续执行达到 15 轮安全上限，挂起进入 IDLE 待命"
                    )
                    self.status = "IDLE"
                    self.consecutive_work_turns = 0
                    self.bus.send(
                        from_agent=self.name,
                        to_agent="lead",
                        content=f"队友 '{self.name}' 连续工作 15 轮已触发安全熔断挂起。请检查任务进展或重新分派。",
                        msg_type="idle_notification",
                    )
                    continue
                self._execute_work_turn()
            elif self.status == "IDLE":
                self.consecutive_work_turns = 0
                self._idle_wait_and_scan()

    @staticmethod
    def _get_role(msg: Any) -> str:
        """安全提取消息角色（兼容 `dict` 与 `ChatCompletionMessage` 对象）。"""
        if isinstance(msg, dict):
            return str(msg.get("role", ""))
        return str(getattr(msg, "role", "") or "")

    def _execute_work_turn(self) -> None:
        """执行单轮 `WORK` 状态思考与工具调用（含 OpenAI 协议安全裁剪）。"""
        if len(self.messages) > 20:
            system_msg = self.messages[0]
            start_idx = len(self.messages) - 10
            # Why: tool 消息必须紧跟在携带对应 tool_calls 的 assistant 消息之后；
            # 若窗口切片起始处恰好切到 tool，必须向前回退对齐到所属 assistant 节点，防止 API 报 400。
            while start_idx > 1 and self._get_role(self.messages[start_idx]) == "tool":
                start_idx -= 1

            trimmed = self.messages[start_idx:]
            if trimmed and self._get_role(trimmed[0]) != "user":
                self.messages = [
                    system_msg,
                    {
                        "role": "user",
                        "content": "（前序长程操作记录已自动折叠，请基于以下最新上下文继续推进当前任务）",
                    },
                ] + trimmed
            else:
                self.messages = [system_msg] + trimmed

        caller = self._llm_caller or call_llm
        try:
            response = caller(messages=self.messages, tools=self.tools.get_schemas())
        except Exception as exc:
            print(f"  \033[31m[Teammate 异常]\033[0m {self.name} 模型调用错误: {exc}")
            # Wait cleanly on stop_event instead of uninterruptible sleep so shutdown stays responsive
            self.stop_event.wait(2.0)
            return

        assistant_dict = assistant_message_to_dict(response)
        self.messages.append(assistant_dict)
        tool_calls = assistant_dict.get("tool_calls") or []

        if not tool_calls:
            result_summary = str(assistant_dict.get("content") or "") or "(无总结文本)"
            print(f"  \033[32m[Teammate 产出]\033[0m {self.name} 完成阶段工作: {result_summary[:60]}...")

            # 先汇报成果再发就绪通知，让 Lead 收件箱按因果顺序接收两类事件
            self.bus.send(
                from_agent=self.name,
                to_agent="lead",
                content=result_summary,
                msg_type="result",
                metadata={"task_id": self.current_task_id},
            )
            self.status = "IDLE"
            self.bus.send(
                from_agent=self.name,
                to_agent="lead",
                content=f"队友 '{self.name}' 已完成当前工作，进入空闲状态 (IDLE)，准备接手后续任务。",
                msg_type="idle_notification",
            )
            return

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            parsed_args, parse_err = parse_tool_arguments(raw_args)
            if parse_err is not None:
                out = parse_err
            else:
                out = self._execute_tool_with_gate(fn_name, parsed_args or {})

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": out,
                }
            )

    def _idle_wait_and_scan(self) -> None:
        """`IDLE` 状态：先在收件箱等待新信件，若无信件则自动扫描任务板认领就绪任务。"""
        inbox = self.bus.wait_for_messages(self.name, timeout=1.0)
        if inbox:
            should_work = False
            for msg in inbox:
                m_type = msg.get("type")
                if m_type == "shutdown_request":
                    req_id = msg.get("metadata", {}).get("request_id")
                    if req_id:
                        self.protocols.acknowledge_shutdown(req_id, self.name)
                    self.bus.send(
                        from_agent=self.name,
                        to_agent="lead",
                        content=f"队友 '{self.name}' 已确认关机指令，安全离线。",
                        msg_type="shutdown_response",
                        metadata={"request_id": req_id},
                    )
                    self.status = "SHUTDOWN"
                    self.stop_event.set()
                    print(f"  \033[33m[Teammate 关机]\033[0m 队友 '{self.name}' 已平滑下线。")
                    return

                if m_type == "plan_approval_response":
                    approved = msg.get("metadata", {}).get("approved", False)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[组长计划审批结果]: {'已批准 (Approved)' if approved else '已驳回 (Rejected)'}\n"
                                f"反馈: {msg.get('content')}"
                            ),
                        }
                    )
                    should_work = True
                    continue

                self.messages.append(
                    {
                        "role": "user",
                        "content": f"[来自 {msg.get('from')} 的消息]: {msg.get('content')}",
                    }
                )
                should_work = True

            # Why: 先完整消化完本批次已取出的所有 inbox 消息后再切换至 WORK，防止早返回丢失同批后续信件
            if should_work and not self.stop_event.is_set():
                self.status = "WORK"
            return

        if self.stop_event.is_set():
            return

        tasks = self.task_store.list()
        candidates = [
            t
            for t in tasks
            if t.status == "pending" and t.owner is None and self.task_store.can_start(t.id)
        ]

        if candidates:
            target = candidates[0]
            claim_res = self.task_store.claim(target.id, owner=self.name)
            if "已认领" in claim_res:
                self.current_task_id = target.id
                self.status = "WORK"
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[自动认领新任务 {target.id}]\n"
                            f"标题: {target.subject}\n"
                            f"要求: {target.description}\n"
                            f"请专注完成本项任务，完成后及时调用 complete_task。"
                        ),
                    }
                )
                print(
                    f"  \033[36m[Teammate 自动接单]\033[0m 队友 '{self.name}' 自主认领任务: {target.subject}"
                )


class TeamManager:
    """多 Agent 团队主管协调器。"""

    def __init__(
        self,
        bus: MessageBus,
        task_store: TaskStore,
        protocols: TeamProtocols,
        workdir: Path,
        llm_caller: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.bus: MessageBus = bus
        self.task_store: TaskStore = task_store
        self.protocols: TeamProtocols = protocols
        self.workdir: Path = Path(workdir).resolve()
        self._llm_caller: Optional[Callable[..., Any]] = llm_caller
        self.teammates: Dict[str, TeammateRuntime] = {}
        self._lock = threading.RLock()

    def spawn(
        self,
        name: str,
        role: str,
        task_id: Optional[str] = None,
        require_plan: bool = False,
    ) -> TeammateRuntime:
        """派生一名新常驻队友（若同名队友已处于 `SHUTDOWN` 状态则安全回收后重新上线）。"""
        old_thread: Optional[threading.Thread] = None
        with self._lock:
            existing = self.teammates.get(name)
            if existing is not None:
                if existing.status != "SHUTDOWN":
                    raise ValueError(f"已存在同名队友: {name} (当前状态: {existing.status})")
                # Why: 同名队友已正式 SHUTDOWN 后，允许在同一会话中重新 spawn 复用代号，
                # 避免用户连续发起多轮团队任务时被旧的离线实例永久阻塞。
                existing.stop_event.set()
                old_thread = existing.thread

        # 在锁外等待旧线程彻底收尾，避免持锁 join 引发死锁
        if old_thread is not None and old_thread is not threading.current_thread() and old_thread.is_alive():
            old_thread.join(timeout=1.5)

        with self._lock:
            existing = self.teammates.get(name)
            if existing is not None and existing.status != "SHUTDOWN":
                raise ValueError(f"已存在同名队友: {name} (当前状态: {existing.status})")

            if task_id:
                if not self.task_store.exists(task_id):
                    raise ValueError(f"任务不存在: {task_id}")
                claim_res = self.task_store.claim(task_id, owner=name)
                if "已认领" not in claim_res:
                    raise ValueError(f"无法为队友分配初始任务: {claim_res}")

            if existing is not None:
                # 清理旧离线实例遗留的未读邮箱，防止新上线的同名队友误读历史指令
                self.bus.read_inbox(name)

            mate = TeammateRuntime(
                name=name,
                role=role,
                bus=self.bus,
                task_store=self.task_store,
                protocols=self.protocols,
                workdir=self.workdir,
                initial_task_id=task_id,
                require_plan=require_plan,
                llm_caller=self._llm_caller,
            )

            if task_id:
                t = self.task_store.load(task_id)
                mate.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[指派初始任务 {t.id}]\n"
                            f"标题: {t.subject}\n"
                            f"描述: {t.description}\n"
                            f"请着手进行开发，完成后调用 complete_task。"
                        ),
                    }
                )

            self.teammates[name] = mate
            mate.start()
            return mate

    def get(self, name: str) -> Optional[TeammateRuntime]:
        """按名称查找已派生的队友实例。"""
        with self._lock:
            return self.teammates.get(name)

    def list_all(self) -> List[TeammateRuntime]:
        """列出所有已派生的团队成员列表。"""
        with self._lock:
            return list(self.teammates.values())

    def shutdown_all(self) -> None:
        """停止并回收所有运行中的队友线程，防止进程或测试夹具退出后残留工作线程。"""
        with self._lock:
            mates = list(self.teammates.values())
            for mate in mates:
                mate.stop_event.set()
                if mate.status != "SHUTDOWN":
                    mate.status = "SHUTDOWN"

        # 唤醒可能在 MessageBus.wait_for_messages() 中阻塞的队友线程，然后等待其安全退出
        with self.bus._changed:
            self.bus._changed.notify_all()

        current = threading.current_thread()
        for mate in mates:
            if mate.thread and mate.thread is not current and mate.thread.is_alive():
                mate.thread.join(timeout=1.5)
