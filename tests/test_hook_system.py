"""
test_hook_system.py - M2.1 生命周期钩子管理器与审批门禁单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coding_agent.hooks.__main__ as hooks_cli_mod
import coding_agent.runtime as runtime_mod
from coding_agent.hooks import (
    HookEvent,
    HookManager,
    create_permission_hook,
    large_output_warning_hook,
    register_default_hooks,
    session_summary_hook,
)


class TestHookSystem(unittest.TestCase):
    """测试 HookManager 优先级调度、拦截机制与内置钩子库"""

    def test_priority_ordering(self):
        """测试钩子严格按照 priority 数值升序执行（数值越小越先触发）"""
        mgr = HookManager()
        execution_order = []

        # 故意乱序注册
        mgr.register("test_event", lambda: execution_order.append("p20"), priority=20, name="low")
        mgr.register("test_event", lambda: execution_order.append("p0"), priority=0, name="highest")
        mgr.register("test_event", lambda: execution_order.append("p10"), priority=10, name="medium")

        mgr.trigger("test_event")
        self.assertEqual(execution_order, ["p0", "p10", "p20"])

    def test_interception_short_circuit(self):
        """测试返回非 None 时成功拦截并短路中断后续钩子"""
        mgr = HookManager()
        order = []

        def audit_log():
            order.append("audit_recorded")
            return None  # 放行

        def permission_block():
            order.append("permission_denied")
            return "拒绝执行：命中安全规则"  # 拦截信号

        def should_not_run():
            order.append("unreachable")
            return None

        mgr.register(HookEvent.PRE_TOOL_USE, audit_log, priority=0)
        mgr.register(HookEvent.PRE_TOOL_USE, permission_block, priority=10)
        mgr.register(HookEvent.PRE_TOOL_USE, should_not_run, priority=20)

        result = mgr.trigger(HookEvent.PRE_TOOL_USE)

        self.assertEqual(result, "拒绝执行：命中安全规则")
        self.assertEqual(order, ["audit_recorded", "permission_denied"])

    def test_decorator_registration(self):
        """测试 @hooks.on(...) 装饰器注册语法"""
        mgr = HookManager()
        triggered = []

        @mgr.on(HookEvent.USER_PROMPT_SUBMIT, priority=5)
        def my_listener(prompt: str):
            triggered.append(prompt)

        mgr.trigger(HookEvent.USER_PROMPT_SUBMIT, "你好 Agent")
        self.assertEqual(triggered, ["你好 Agent"])

    def test_permission_hook_detection(self):
        """测试安全命令与非 Shell 工具自动放行"""
        perm_hook = create_permission_hook(auto_approve=False)

        safe_res = perm_hook("run_shell", {"command": "dir"})
        self.assertIsNone(safe_res)

        read_res = perm_hook("read_file", {"path": "test.txt"})
        self.assertIsNone(read_res)

    def test_permission_hook_interactive_allow_and_deny(self):
        """测试危险指令触发 HITL 审批门禁时的授权放行、拒绝拦截与默认回车拒绝"""
        perm_hook = create_permission_hook(auto_approve=False)
        dangerous_args = {"command": "Remove-Item -Recurse ./tmp"}

        # 1. 输入无效字符后再输入 y -> 授权放行
        with patch("builtins.input", side_effect=["maybe", "y"]), redirect_stdout(io.StringIO()):
            self.assertIsNone(perm_hook("run_shell", dangerous_args))

        # 2. 输入 n -> 拦截并返回拒绝提示
        with patch("builtins.input", return_value="n"), redirect_stdout(io.StringIO()):
            blocked = perm_hook("run_shell", dangerous_args)
        self.assertIsNotNone(blocked)
        self.assertIn("拒绝了此高危操作", str(blocked))

        # 3. 直接回车（[y/N] 默认 N）或 EOFError -> 安全拒绝
        with patch("builtins.input", return_value=""), redirect_stdout(io.StringIO()):
            blocked_empty = perm_hook("run_shell", {"command": "rmdir /s /q dist"})
        self.assertIn("拒绝了此高危操作", str(blocked_empty))

        with patch("builtins.input", side_effect=EOFError), redirect_stdout(io.StringIO()):
            blocked_eof = perm_hook("run_shell", {"command": "del /f important.txt"})
        self.assertIn("拒绝了此高危操作", str(blocked_eof))

    def test_register_default_hooks(self):
        """测试一键挂载默认钩子集合及优先级顺序"""
        mgr = HookManager()
        register_default_hooks(mgr, auto_approve=True)

        pre_hooks = mgr.list_hooks(HookEvent.PRE_TOOL_USE)
        self.assertIn("tool_logger", pre_hooks)
        self.assertIn("permission_gate", pre_hooks)
        self.assertLess(pre_hooks.index("tool_logger"), pre_hooks.index("permission_gate"))

        # auto_approve=True 时高危指令也应放行
        with redirect_stdout(io.StringIO()):
            res = mgr.trigger(HookEvent.PRE_TOOL_USE, "run_shell", {"command": "format D:"})
        self.assertIsNone(res)

    def test_trigger_all_clone_and_exception_isolation(self):
        """测试 trigger_all 广播收集、clone 隔离以及异常钩子隔离"""
        mgr = HookManager()

        def broken_hook():
            raise RuntimeError("boom")

        mgr.register(HookEvent.STOP, broken_hook, priority=0, name="broken")
        mgr.register(HookEvent.STOP, lambda: "ok", priority=10, name="healthy")

        with redirect_stdout(io.StringIO()):
            results = mgr.trigger_all(HookEvent.STOP)
        self.assertEqual(results, [None, "ok"])

        cloned = mgr.clone()
        cloned.register(HookEvent.STOP, lambda: "extra", priority=20, name="extra")
        self.assertNotIn("extra", mgr.list_hooks(HookEvent.STOP))
        self.assertIn("extra", cloned.list_hooks(HookEvent.STOP))

    def test_builtin_output_warning_summary_and_demo_cli(self):
        """测试超长输出警告、会话统计以及 hooks 演示入口"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            large_output_warning_hook("read_file", {}, "x" * 2500)
            session_summary_hook([{"role": "user"}, {"role": "tool"}, {"role": "tool"}])
        out = buf.getvalue()
        self.assertIn("2500 字符", out)
        self.assertIn("共发生 2 次工具调用", out)

        demo_buf = io.StringIO()
        with redirect_stdout(demo_buf):
            exit_code = hooks_cli_mod.main(["--choice", "n"])
        self.assertEqual(exit_code, 0)
        self.assertIn("[Permission 审批]", demo_buf.getvalue())
        self.assertIn("拒绝了此高危操作", demo_buf.getvalue())


class TestHookRuntimeIntegration(unittest.TestCase):
    """测试 HookManager 接入 runtime.py 主循环后的端到端行为"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.tools = runtime_mod.create_default_tools(workdir=self.workdir)
        self.messages = []

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _fake_msg(content=None, tool_calls=None):
        tc_list = []
        for tc in tool_calls or []:
            tc_list.append(
                SimpleNamespace(
                    id=tc["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=json.dumps(tc["arguments"], ensure_ascii=False),
                    ),
                )
            )
        return SimpleNamespace(content=content, tool_calls=tc_list)

    def test_runtime_blocks_dangerous_command_and_feeds_refusal_to_llm(self):
        """主循环遇到危险指令时应触发 HITL 门禁，拒绝后不执行命令并将原因回填模型"""
        hooks = runtime_mod.create_default_hooks(auto_approve=False)
        events_seen = []
        hooks.register(HookEvent.USER_PROMPT_SUBMIT, lambda p: events_seen.append(("submit", p)), priority=0)
        hooks.register(
            HookEvent.POST_TOOL_USE,
            lambda fn, args, out: events_seen.append(("post", fn, str(out))),
            priority=0,
        )

        # 预置一个文件，验证若拒绝执行 Remove-Item，文件不会被删除
        victim = self.workdir / "keep.txt"
        victim.write_text("safe", encoding="utf-8")

        step = 0

        def scripted_llm(messages, tools):
            nonlocal step
            step += 1
            if step == 1:
                return self._fake_msg(
                    tool_calls=[
                        {
                            "id": "tc_del",
                            "name": "run_shell",
                            "arguments": {"command": "Remove-Item -Recurse keep.txt"},
                        }
                    ]
                )
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(messages[-1]["tool_call_id"], "tc_del")
            self.assertIn("操作被钩子拦截拒绝", messages[-1]["content"])
            return self._fake_msg(content="已取消危险删除操作。")

        with patch("builtins.input", return_value="n"), redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "删除 keep.txt",
                tools=self.tools,
                hooks=hooks,
                messages=self.messages,
                llm_caller=scripted_llm,
                workdir=self.workdir,
                verbose=False,
            )

        self.assertEqual(reply, "已取消危险删除操作。")
        self.assertTrue(victim.exists())
        self.assertEqual(events_seen[0], ("submit", "删除 keep.txt"))
        # 被 PreToolUse 短路拦截时不应触发工具执行与 PostToolUse
        self.assertFalse(any(evt[0] == "post" for evt in events_seen))

    def test_runtime_stop_hook_can_force_continuation(self):
        """Stop 切面返回非 None 指令时应阻止退出并自动注回 user 消息续轮"""
        hooks = HookManager()
        stop_calls = 0

        def gate_stop(messages):
            nonlocal stop_calls
            stop_calls += 1
            if stop_calls == 1:
                return "请先补充检查结果再结束。"
            return None

        hooks.register(HookEvent.STOP, gate_stop, priority=10)

        turn = 0

        def scripted_llm(messages, tools):
            nonlocal turn
            turn += 1
            if turn == 1:
                return self._fake_msg(content="初版回答")
            self.assertEqual(messages[-1], {"role": "user", "content": "请先补充检查结果再结束。"})
            return self._fake_msg(content="已补充检查，最终完成。")

        reply = runtime_mod.run_agent_turn(
            "执行检查",
            tools=self.tools,
            hooks=hooks,
            messages=self.messages,
            llm_caller=scripted_llm,
            workdir=self.workdir,
            verbose=False,
        )
        self.assertEqual(turn, 2)
        self.assertEqual(stop_calls, 2)
        self.assertEqual(reply, "已补充检查，最终完成。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
