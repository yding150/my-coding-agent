"""
test_agent_teams.py - M6.2 常驻队友运行时、团队工具与主循环收件箱唤醒单元测试
"""

import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import coding_agent.cli as cli_mod
import coding_agent.runtime as runtime_mod
from coding_agent.hooks import HookEvent, HookManager
from coding_agent.tasks import TaskStore
from coding_agent.teams import (
    MessageBus,
    TeamManager,
    TeamProtocols,
    TeammateRuntime,
    register_team_tools,
)
from coding_agent.teams.__main__ import main as teams_demo_main
from coding_agent.tools import ToolManager


def _ns_msg(content=None, tool_calls=None):
    """构造离线主循环与队友测试用模拟大模型返回消息。"""
    tc_objs = []
    for tc in tool_calls or []:
        args = tc["arguments"]
        tc_objs.append(
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                ),
            )
        )
    return SimpleNamespace(content=content, tool_calls=tc_objs)


class TestAgentTeamsSystem(unittest.TestCase):
    """测试公共层 MessageBus / TeamProtocols / TeammateRuntime / TeamManager 与工具注册。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name).resolve()
        self.bus = MessageBus(workdir=self.test_dir)
        self.tasks_dir = self.test_dir / ".agent" / "tasks"
        self.tasks = TaskStore(directory=self.tasks_dir, workdir=self.test_dir)
        self.protocols = TeamProtocols()
        self.team_mgr = TeamManager(
            bus=self.bus,
            task_store=self.tasks,
            protocols=self.protocols,
            workdir=self.test_dir,
        )

    def tearDown(self):
        self.team_mgr.shutdown_all()
        self.temp_dir.cleanup()

    def test_message_bus_delivery_and_consume(self):
        """测试 1：消息总线发送、唤醒与消费。"""
        self.assertFalse(self.bus.peek("lead"))

        self.bus.send("alice", "lead", "任务已完成", msg_type="result")
        self.assertTrue(self.bus.peek("lead"))

        inbox = self.bus.read_inbox("lead")
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["from"], "alice")
        self.assertEqual(inbox[0]["content"], "任务已完成")
        self.assertEqual(inbox[0]["type"], "result")
        self.assertFalse(self.bus.peek("lead"))

    def test_plan_approval_gate_interception(self):
        """测试 2：计划审批闸门对写操作的精准拦截与放行。"""
        agent = "bob"
        self.protocols.set_plan_requirement(agent, required=True)

        allowed, _ = self.protocols.is_tool_allowed(agent, "read_file")
        self.assertTrue(allowed)

        allowed_w, reason_w = self.protocols.is_tool_allowed(agent, "write_file")
        self.assertFalse(allowed_w)
        self.assertIn("拦截", reason_w)

        allowed_s, _ = self.protocols.is_tool_allowed(agent, "run_shell")
        self.assertFalse(allowed_s)

        req = self.protocols.submit_plan(agent, "采用 SQLite 重构")
        self.assertEqual(req.status, "pending")
        self.assertEqual(self.protocols.get_plan_gate(agent), "pending")
        self.assertFalse(self.protocols.is_tool_allowed(agent, "write_file")[0])

        self.protocols.approve_plan(req.request_id, approve=True)
        self.assertEqual(self.protocols.get_plan_gate(agent), "approved")
        self.assertTrue(self.protocols.is_tool_allowed(agent, "write_file")[0])
        self.assertTrue(self.protocols.is_tool_allowed(agent, "run_shell")[0])

    def test_shutdown_protocol_handshake(self):
        """测试 3：平滑关机握手协议。"""
        req = self.protocols.create_shutdown_request("charlie")
        self.assertEqual(req.status, "pending")

        success = self.protocols.acknowledge_shutdown(req.request_id, "charlie")
        self.assertTrue(success)
        self.assertEqual(req.status, "approved")

    def test_auto_claim_ready_tasks(self):
        """测试 4：空闲队友在 IDLE 状态下自动扫描并原子认领任务。"""
        task = self.tasks.create("优化缓存模块", "实现 Redis 缓存")
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.owner)

        mate = TeammateRuntime(
            name="david",
            role="性能优化专家",
            bus=self.bus,
            task_store=self.tasks,
            protocols=self.protocols,
            workdir=self.test_dir,
        )
        self.assertEqual(mate.status, "IDLE")

        with redirect_stdout(io.StringIO()):
            mate._idle_wait_and_scan()

        reloaded = self.tasks.load(task.id)
        self.assertEqual(reloaded.status, "in_progress")
        self.assertEqual(reloaded.owner, "david")
        self.assertEqual(mate.status, "WORK")
        self.assertEqual(mate.current_task_id, task.id)

    def test_team_tools_registration(self):
        """测试 5：Lead 团队工具注册与 Schema 导出。"""
        tools = ToolManager()
        register_team_tools(tools, self.team_mgr, self.bus, self.protocols)

        schemas = tools.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        for tool_name in (
            "spawn_teammate",
            "send_message",
            "request_plan",
            "approve_plan",
            "request_shutdown",
            "list_teammates",
            "read_inbox",
        ):
            self.assertIn(tool_name, names)

        # 验证 read_inbox 为原子消费（读取后清空队列）
        self.bus.send("alice", "lead", "计划请求正文", msg_type="plan_approval_request", metadata={"request_id": "req_1"})
        first_read = tools.execute("read_inbox", {})
        self.assertIn("计划请求正文", first_read)
        self.assertFalse(self.bus.peek("lead"))
        self.assertIn("暂无未读消息", tools.execute("read_inbox", {}))

    def test_respawn_after_shutdown_and_reject_send_to_shutdown_teammate(self):
        """队友 SHUTDOWN 后：send_message 必须拦截提示已离线，且允许 spawn_teammate 重新上线同名队友。"""
        tools = ToolManager()
        register_team_tools(tools, self.team_mgr, self.bus, self.protocols)
        self.team_mgr._llm_caller = lambda messages, tools: _ns_msg(content="收到并完成")

        with redirect_stdout(io.StringIO()):
            first = self.team_mgr.spawn("alice", "巡检员", require_plan=True)
            first.stop_event.set()
            first.status = "SHUTDOWN"
            if first.thread is not None:
                first.thread.join(timeout=1.5)

        # 未重新 spawn 前，在 WORK/IDLE 状态禁止重名；SHUTDOWN 后 send_message/request_plan 必须报错拒绝
        send_err = tools.execute("send_message", {"to_agent": "alice", "content": "继续干活"})
        self.assertIn("已关机离线", send_err)
        self.assertIn("SHUTDOWN", send_err)
        self.assertFalse(self.bus.peek("alice"))

        plan_err = tools.execute("request_plan", {"teammate": "alice"})
        self.assertIn("已关机离线", plan_err)
        self.assertIn("已关机离线", tools.execute("list_teammates", {}))

        # 预置一条给旧实例的陈旧信件，验证重新 spawn 时应清空旧邮箱并重置闸门
        self.bus.send("lead", "alice", "陈旧指令", msg_type="message")
        with redirect_stdout(io.StringIO()):
            second = self.team_mgr.spawn("alice", "新巡检员", require_plan=True)

        self.assertIsNot(second, first)
        self.assertEqual(self.protocols.get_plan_gate("alice"), "required")
        # 仍处于在线状态时不能重复 spawn 同名队友
        with self.assertRaises(ValueError):
            self.team_mgr.spawn("alice", "冲突队友")
        with redirect_stdout(io.StringIO()):
            self.team_mgr.shutdown_all()

    def test_teammate_message_window_pruning_safety(self):
        """测试 6：长程上下文滑动修剪的协议安全性（杜绝孤立 role: tool 引发 400 错误）。"""
        mate = TeammateRuntime(
            name="eve",
            role="协议专家",
            bus=self.bus,
            task_store=self.tasks,
            protocols=self.protocols,
            workdir=self.test_dir,
        )
        messages = [{"role": "system", "content": "sys"}]
        for i in range(12):
            asst_obj = MagicMock()
            asst_obj.role = "assistant"
            asst_obj.content = f"thinking {i}"
            asst_obj.tool_calls = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ]
            messages.append(asst_obj)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"result {i}",
                }
            )
        mate.messages = messages
        self.assertGreater(len(mate.messages), 20)

        with patch("coding_agent.teams.runtime.call_llm") as mock_call:
            mock_call.return_value = MagicMock(tool_calls=None, content="done")
            with redirect_stdout(io.StringIO()):
                mate._execute_work_turn()

        first_after_sys = mate.messages[1]
        self.assertNotEqual(mate._get_role(first_after_sys), "tool")
        for idx in range(1, len(mate.messages)):
            if mate._get_role(mate.messages[idx]) == "tool":
                prev = mate.messages[idx - 1]
                self.assertEqual(mate._get_role(prev), "assistant")
                call_ids = [
                    c["id"] if isinstance(c, dict) else c.id
                    for c in (prev.get("tool_calls") if isinstance(prev, dict) else getattr(prev, "tool_calls", [])) or []
                ]
                self.assertIn(mate.messages[idx].get("tool_call_id"), call_ids)


class TestAgentTeamsRuntimeIntegration(unittest.TestCase):
    """验证 M6.2 主循环工具装配、收件箱唤醒、审批闭环与 CLI 退出清理集成。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        runtime_mod.stop_runtime_threads()
        self.temp_dir.cleanup()

    def test_parent_tools_mounts_team_tools_while_shared_tools_and_subagent_stay_clean(self):
        """parent_tools 必须挂载团队 6 工具且纳入 PLANNING_TOOLS；shared_tools 与子代理不得继承。"""
        team_tools = {
            "spawn_teammate",
            "send_message",
            "request_plan",
            "approve_plan",
            "request_shutdown",
            "list_teammates",
            "read_inbox",
        }
        parent_names = set(runtime_mod.parent_tools.list_tools())
        default_names = set(runtime_mod.default_tools.list_tools())
        shared_names = set(runtime_mod.shared_tools.list_tools())
        sub_names = set(runtime_mod.default_subagent_runner.tools.list_tools())

        self.assertEqual(parent_names, default_names)
        self.assertTrue(team_tools.issubset(parent_names))
        self.assertTrue(shared_names.isdisjoint(team_tools))
        self.assertTrue(sub_names.isdisjoint(team_tools))
        self.assertIn("spawn_teammate", runtime_mod.PLANNING_TOOLS)

        prompt_text = runtime_mod.build_system_prompt(workdir=self.workdir)
        self.assertIn("spawn_teammate", prompt_text)
        self.assertIn("approve_plan", prompt_text)

    def test_format_team_events_and_team_event_turn_skips_user_prompt_hook(self):
        """format_team_events 应渲染结构化摘要，`is_team_event=True` 不误触发 UserPromptSubmit。"""
        formatted = runtime_mod.format_team_events(
            [
                {
                    "from": "alice",
                    "type": "plan_approval_request",
                    "content": "方案内容",
                    "metadata": {"request_id": "req_11112222"},
                },
                {
                    "from": "alice",
                    "type": "result",
                    "content": "完成认证重构",
                    "metadata": {"task_id": "task_aaaa1111"},
                },
                {"from": "alice", "type": "idle_notification", "content": "idle", "metadata": {}},
                {"from": "alice", "type": "shutdown_response", "content": "bye", "metadata": {}},
            ]
        )
        self.assertIn("计划待审", formatted)
        self.assertIn("req_11112222", formatted)
        self.assertIn("成果回报", formatted)
        self.assertIn("状态就绪", formatted)
        self.assertIn("关机下线", formatted)

        hooks = HookManager()
        prompt_events = []

        @hooks.on(HookEvent.USER_PROMPT_SUBMIT)
        def _on_submit(text: str) -> None:
            prompt_events.append(text)

        reply = runtime_mod.run_agent_turn(
            formatted,
            tools=runtime_mod.create_default_tools(workdir=self.workdir),
            hooks=hooks,
            messages=[],
            llm_caller=lambda messages, tools: _ns_msg(content="已获悉团队动态"),
            workdir=self.workdir,
            is_team_event=True,
            verbose=False,
        )
        self.assertEqual(reply, "已获悉团队动态")
        self.assertEqual(prompt_events, [])

    def test_end_to_end_team_spawn_plan_gate_inbox_wakeup_and_shutdown(self):
        """端到端闭环：派发 require_plan 队友 -> 闸门拦截 -> submit_plan -> 唤醒 Lead 审批 -> 写入完结 -> 关机。"""
        task_store = TaskStore(self.workdir / ".agent" / "tasks", workdir=self.workdir)
        bus = MessageBus(workdir=self.workdir)
        protocols = TeamProtocols()
        task = task_store.create("交付模块文档", "生成 GUIDE.md")

        mate_step = 0

        def mate_llm(messages, tools):
            nonlocal mate_step
            mate_step += 1
            if mate_step == 1:
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "m1",
                            "name": "write_file",
                            "arguments": {"path": "GUIDE.md", "content": "blocked\n"},
                        }
                    ]
                )
            if mate_step == 2:
                self.assertIn("操作被团队计划闸门拦截", messages[-1]["content"])
                return _ns_msg(
                    tool_calls=[
                        {"id": "m2", "name": "submit_plan", "arguments": {"plan": "写入 GUIDE.md"}}
                    ]
                )
            if mate_step == 3:
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "m3",
                            "name": "write_file",
                            "arguments": {"path": "GUIDE.md", "content": "APPROVED_WRITE\n"},
                        },
                        {
                            "id": "m4",
                            "name": "complete_task",
                            "arguments": {"task_id": task.id},
                        },
                    ]
                )
            return _ns_msg(content="GUIDE.md 已写入完毕。")

        team_mgr = TeamManager(
            bus=bus,
            task_store=task_store,
            protocols=protocols,
            workdir=self.workdir,
            llm_caller=mate_llm,
        )
        lead_tools = runtime_mod.create_default_tools(
            workdir=self.workdir,
            task_store=task_store,
            team_mgr=team_mgr,
            bus=bus,
            protocols=protocols,
        )
        lead_messages = []
        lead_lock = threading.RLock()
        stop_event = threading.Event()
        shutdown_finished = threading.Event()

        def lead_runner(prompt: str, *, is_scheduled: bool = False, is_team_event: bool = False):
            def lead_llm(messages, tools):
                is_initial_event = messages[-1].get("role") == "user"
                if "计划待审" in prompt and is_initial_event:
                    req_id = protocols.plan_request_ids["alice"]
                    return _ns_msg(
                        tool_calls=[
                            {
                                "id": "la",
                                "name": "approve_plan",
                                "arguments": {"request_id": req_id, "approve": True, "feedback": "放行"},
                            }
                        ]
                    )
                if "成果回报" in prompt and is_initial_event:
                    return _ns_msg(
                        tool_calls=[
                            {"id": "ls", "name": "request_shutdown", "arguments": {"teammate": "alice"}}
                        ]
                    )
                if "关机下线" in prompt:
                    shutdown_finished.set()
                    return _ns_msg(content="队友已安全下线")
                return _ns_msg(content="继续跟进")

            return runtime_mod.run_agent_turn(
                prompt,
                tools=lead_tools,
                hooks=HookManager(),
                messages=lead_messages,
                llm_caller=lead_llm,
                workdir=self.workdir,
                is_scheduled=is_scheduled,
                is_team_event=is_team_event,
                verbose=False,
            )

        consumer = threading.Thread(
            target=runtime_mod.queue_processor_loop,
            args=(stop_event,),
            kwargs={"bus": bus, "lock": lead_lock, "turn_runner": lead_runner, "poll_interval": 0.03},
            daemon=True,
        )
        consumer.start()
        try:
            with lead_lock:
                with redirect_stdout(io.StringIO()):
                    spawn_res = lead_tools.execute(
                        "spawn_teammate",
                        {"name": "alice", "role": "文档专家", "task_id": task.id, "require_plan": True},
                    )
            self.assertIn("成功启动持久队友 'alice'", spawn_res)
            self.assertTrue(shutdown_finished.wait(timeout=8.0), "团队审批-完工-关机闭环超时")
        finally:
            stop_event.set()
            team_mgr.shutdown_all()
            consumer.join(timeout=1.5)

        self.assertEqual((self.workdir / "GUIDE.md").read_text(encoding="utf-8").strip(), "APPROVED_WRITE")
        self.assertEqual(task_store.load(task.id).status, "completed")
        self.assertEqual(team_mgr.get("alice").status, "SHUTDOWN")
        self.assertIsNone(team_mgr.get("alice").current_task_id)

    def test_foreground_turn_consumes_lead_inbox_without_duplicate_wakeup(self):
        """前台回合内若持续轮询完成全部协作与关机，收件箱应在回合内被消费，退出后不再触发二次总结。"""
        task_store = TaskStore(self.workdir / ".agent" / "tasks", workdir=self.workdir)
        bus = MessageBus(workdir=self.workdir)
        protocols = TeamProtocols()
        task = task_store.create("前台同步闭环", "创建 DONE.txt")
        mate_done = threading.Event()
        wake_calls = []

        def mate_llm(messages, tools):
            return _ns_msg(content="已收到并完成。")

        team_mgr = TeamManager(
            bus=bus,
            task_store=task_store,
            protocols=protocols,
            workdir=self.workdir,
            llm_caller=mate_llm,
        )
        lead_tools = runtime_mod.create_default_tools(
            workdir=self.workdir,
            task_store=task_store,
            team_mgr=team_mgr,
            bus=bus,
            protocols=protocols,
        )

        step = 0

        def fg_llm(messages, tools):
            nonlocal step
            step += 1
            if step == 1:
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "s1",
                            "name": "spawn_teammate",
                            "arguments": {"name": "alice", "role": "助手", "task_id": task.id},
                        }
                    ]
                )
            if step == 2:
                # 模拟 alice 在回合进行中向 Lead 发来成果与空闲消息
                bus.send("alice", "lead", "任务 DONE", msg_type="result", metadata={"task_id": task.id})
                bus.send("alice", "lead", "idle", msg_type="idle_notification")
                return _ns_msg(
                    tool_calls=[
                        {"id": "s2", "name": "request_shutdown", "arguments": {"teammate": "alice"}}
                    ]
                )
            if step == 3:
                # 模拟队友很快返回 shutdown_response，同时验证消息已通过 _inject_team_events 注入上下文
                bus.send("alice", "lead", "已安全离线", msg_type="shutdown_response")
                return _ns_msg(tool_calls=[{"id": "s3", "name": "list_teammates", "arguments": {}}])
            # 第 4 次请求 LLM 前，第 3 步入箱的 shutdown_response 也应已被原子消费并注入
            joined = "\n".join(str(m.get("content", "")) for m in messages)
            self.assertIn("成果回报", joined)
            self.assertIn("关机下线", joined)
            mate_done.set()
            return _ns_msg(content="单次总结：任务与关机均已全部完成。")

        lead_messages = []
        try:
            with redirect_stdout(io.StringIO()):
                reply = runtime_mod.run_agent_turn(
                    "启动 alice 并完成总结",
                    tools=lead_tools,
                    hooks=HookManager(),
                    messages=lead_messages,
                    llm_caller=fg_llm,
                    workdir=self.workdir,
                    verbose=False,
                )

            self.assertTrue(mate_done.is_set())
            self.assertEqual(reply, "单次总结：任务与关机均已全部完成。")
            # 回合结束时 Lead 收件箱必须已被完全消费，绝不遗留给 queue_processor_loop 二次唤醒
            self.assertFalse(bus.peek("lead"))

            stop_event = threading.Event()
            consumer = threading.Thread(
                target=runtime_mod.queue_processor_loop,
                args=(stop_event,),
                kwargs={"bus": bus, "turn_runner": lambda p, **kw: wake_calls.append(p), "poll_interval": 0.02},
                daemon=True,
            )
            consumer.start()
            try:
                time.sleep(0.15)
            finally:
                stop_event.set()
                consumer.join(timeout=1.0)

            self.assertEqual(wake_calls, [])
        finally:
            team_mgr.shutdown_all()

    def test_team_event_turn_multi_step_consumes_inbox_without_duplicate_wakeup(self):
        """团队事件唤醒回合内若连续多轮跟进完审批、核验与关机，必须清空收件箱不再二次唤醒。"""
        task_store = TaskStore(self.workdir / ".agent" / "tasks", workdir=self.workdir)
        bus = MessageBus(workdir=self.workdir)
        protocols = TeamProtocols()
        task = task_store.create("事件回合完整跟进", "生成 OUT.txt")
        wake_calls = []

        team_mgr = TeamManager(
            bus=bus,
            task_store=task_store,
            protocols=protocols,
            workdir=self.workdir,
            llm_caller=lambda messages, tools: _ns_msg(content="已处理"),
        )
        lead_tools = runtime_mod.create_default_tools(
            workdir=self.workdir,
            task_store=task_store,
            team_mgr=team_mgr,
            bus=bus,
            protocols=protocols,
        )

        step = 0

        def event_lead_llm(messages, tools):
            nonlocal step
            step += 1
            if step == 1:
                return _ns_msg(tool_calls=[{"id": "e1", "name": "list_teammates", "arguments": {}}])
            if step == 2:
                bus.send("alice", "lead", "成果完成", msg_type="result", metadata={"task_id": task.id})
                bus.send("alice", "lead", "idle", msg_type="idle_notification")
                return _ns_msg(tool_calls=[{"id": "e2", "name": "list_tasks", "arguments": {}}])
            if step == 3:
                bus.send("alice", "lead", "已安全离线", msg_type="shutdown_response")
                return _ns_msg(tool_calls=[{"id": "e3", "name": "list_teammates", "arguments": {}}])
            joined = "\n".join(str(m.get("content", "")) for m in messages)
            self.assertIn("成果回报", joined)
            self.assertIn("关机下线", joined)
            return _ns_msg(content="事件回合已一次性完成全部验收与关机汇总。")

        try:
            with redirect_stdout(io.StringIO()):
                reply = runtime_mod.run_agent_turn(
                    "[Team events 团队动态事件]:\n  - 计划待审: 队友 'alice' 提交了方案",
                    tools=lead_tools,
                    hooks=HookManager(),
                    messages=[],
                    llm_caller=event_lead_llm,
                    workdir=self.workdir,
                    is_team_event=True,
                    verbose=False,
                )

            self.assertEqual(reply, "事件回合已一次性完成全部验收与关机汇总。")
            self.assertFalse(bus.peek("lead"))

            stop_event = threading.Event()
            consumer = threading.Thread(
                target=runtime_mod.queue_processor_loop,
                args=(stop_event,),
                kwargs={"bus": bus, "turn_runner": lambda p, **kw: wake_calls.append(p), "poll_interval": 0.02},
                daemon=True,
            )
            consumer.start()
            try:
                time.sleep(0.15)
            finally:
                stop_event.set()
                consumer.join(timeout=1.0)

            self.assertEqual(wake_calls, [])
        finally:
            team_mgr.shutdown_all()

    def test_cli_exits_clean_up_default_team_manager(self):
        """CLI 单次模式与 REPL 退出时必须触发 default_team_manager.shutdown_all()。"""
        with (
            patch.object(cli_mod.default_team_manager, "shutdown_all") as mock_team_stop,
            patch.object(cli_mod.default_bg_manager, "cleanup_all") as mock_bg_clean,
            patch.object(cli_mod, "run_agent_turn", return_value="ok"),
        ):
            with redirect_stdout(io.StringIO()):
                code = cli_mod.main(["查询状态"])
        self.assertEqual(code, 0)
        mock_team_stop.assert_called_once()
        mock_bg_clean.assert_called_once()

        with (
            patch.object(cli_mod, "start_runtime_threads") as mock_start,
            patch.object(cli_mod, "stop_runtime_threads") as mock_stop,
            patch.object(cli_mod.default_team_manager, "shutdown_all") as mock_team_stop2,
            patch.object(cli_mod.default_bg_manager, "cleanup_all") as mock_bg_clean2,
            patch("builtins.input", side_effect=["exit"]),
        ):
            with redirect_stdout(io.StringIO()):
                code2 = cli_mod.run_interactive_repl()
        self.assertEqual(code2, 0)
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_team_stop2.assert_called_once()
        mock_bg_clean2.assert_called_once()

    def test_teams_demo_main_full_pipeline_runs_cleanly(self):
        """M6.2 扩展后的离线演示入口 `python -m coding_agent.teams` 应完整跑通两部分并返回 0。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = teams_demo_main()

        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("[全链路校验] shutdown闭环=True", out)
        self.assertIn("任务状态=completed", out)
        self.assertIn("文件产物='TEAM_OK'", out)


if __name__ == "__main__":
    unittest.main()
