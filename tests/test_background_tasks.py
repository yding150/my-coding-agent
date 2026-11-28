"""
test_background_tasks.py - M5.1 异步后台任务管理 (`coding_agent.background`) 与主循环集成单元测试
"""

import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coding_agent.cli as cli_mod
import coding_agent.runtime as runtime_mod
from coding_agent.background import BackgroundTaskManager, register_background_tools
from coding_agent.background.__main__ import main as background_demo_main
from coding_agent.hooks import HookManager
from coding_agent.tools import ToolManager, register_core_tools


def _ns_msg(content=None, tool_calls=None):
    """构造离线主循环测试用模拟大模型返回消息。"""
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


class TestBackgroundTaskSystem(unittest.TestCase):
    """测试 BackgroundTaskManager 核心行为、通知队列语义及后台工具分发。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.bg_mgr = BackgroundTaskManager(workdir=self.workdir)
        self.tools = ToolManager()
        register_core_tools(self.tools, workdir=self.workdir)
        register_background_tools(self.tools, self.bg_mgr)

    def tearDown(self):
        self.bg_mgr.cleanup_all()
        self.temp_dir.cleanup()

    def test_start_and_immediate_return(self):
        """后台启动慢命令应立即返回 bg_xxxx，主线程不被阻塞。"""
        t0 = time.time()
        if sys.platform == "win32":
            cmd = "Start-Sleep -Milliseconds 1500; Write-Output 'Done Sleep'"
        else:
            cmd = "sleep 1.5; echo 'Done Sleep'"

        with redirect_stdout(io.StringIO()):
            task_id = self.bg_mgr.start(cmd)
        elapsed = time.time() - t0

        self.assertTrue(task_id.startswith("bg_"))
        self.assertLess(elapsed, 0.5, "启动后台任务耗时过长，违反了不阻塞主线程的原则")

        detail = self.bg_mgr.get_task(task_id)
        self.assertIsNotNone(detail)
        self.assertEqual(detail["status"], "running")

    def test_async_completion_and_notification_collection(self):
        """后台任务执行完毕后，能够收集到规范的 `<task_notification>` XML 通知且仅消费一次。"""
        if sys.platform == "win32":
            cmd = "Write-Output 'Hello Async Background'"
        else:
            cmd = "echo 'Hello Async Background'"

        with redirect_stdout(io.StringIO()):
            task_id = self.bg_mgr.start(cmd)

        completed = False
        detail = None
        for _ in range(80):
            detail = self.bg_mgr.get_task(task_id)
            if detail and detail["status"] in ("completed", "failed"):
                completed = True
                break
            time.sleep(0.1)

        self.assertTrue(completed, "后台任务未在预期时间内完成")
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["exit_code"], 0)
        self.assertIn("Hello Async Background", detail["output"])

        notifications = self.bg_mgr.collect_notifications()
        self.assertEqual(len(notifications), 1)
        self.assertIn("<task_notification>", notifications[0])
        self.assertIn(f"<task_id>{task_id}</task_id>", notifications[0])
        self.assertIn("<status>completed</status>", notifications[0])
        self.assertIn("Hello Async Background", notifications[0])

        # 验证再次收集时队列已被清空（防重复通知）
        self.assertEqual(self.bg_mgr.collect_notifications(), [])

    def test_kill_task(self):
        """强制终止长时间运行的后台任务后状态应更新为 killed。"""
        cmd = "Start-Sleep -Seconds 30" if sys.platform == "win32" else "sleep 30"

        with redirect_stdout(io.StringIO()):
            task_id = self.bg_mgr.start(cmd)
        detail = self.bg_mgr.get_task(task_id)
        self.assertEqual(detail["status"], "running")

        with redirect_stdout(io.StringIO()):
            killed = self.bg_mgr.kill_task(task_id)
        self.assertTrue(killed)

        detail_after = self.bg_mgr.get_task(task_id)
        self.assertEqual(detail_after["status"], "killed")

    def test_failed_command_handling(self):
        """非 0 退出码命令应正确捕获并标记为 failed。"""
        if sys.platform == "win32":
            cmd = "Write-Error 'Something Went Wrong'; exit 1"
        else:
            cmd = "echo 'Something Went Wrong' >&2; exit 1"

        with redirect_stdout(io.StringIO()):
            task_id = self.bg_mgr.start(cmd)

        detail = None
        for _ in range(80):
            detail = self.bg_mgr.get_task(task_id)
            if detail and detail["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["status"], "failed")
        self.assertNotEqual(detail["exit_code"], 0)

    def test_inject_notifications_messages_structure(self):
        """验证通知安全并入 messages 的 OpenAI 角色规则（尾部 user 合并 vs tool 后另起 user）。"""
        self.bg_mgr.tasks["bg_9999"] = {
            "task_id": "bg_9999",
            "command": "npm install",
            "status": "completed",
            "exit_code": 0,
        }
        self.bg_mgr.results["bg_9999"] = "added 12 packages in 3s"
        self.bg_mgr._ready_queue.append("bg_9999")

        # 情况 A: 消息列表末尾是 user 消息
        messages_user_tail = [{"role": "user", "content": "请帮我检查项目状态"}]
        with redirect_stdout(io.StringIO()):
            count = self.bg_mgr.inject_notifications(messages_user_tail)
        self.assertEqual(count, 1)
        self.assertEqual(len(messages_user_tail), 1)
        self.assertIn("请帮我检查项目状态", messages_user_tail[0]["content"])
        self.assertIn("<task_id>bg_9999</task_id>", messages_user_tail[0]["content"])

        self.bg_mgr.tasks["bg_8888"] = {
            "task_id": "bg_8888",
            "command": "cargo test",
            "status": "completed",
            "exit_code": 0,
        }
        self.bg_mgr.results["bg_8888"] = "test result: ok."
        self.bg_mgr._ready_queue.append("bg_8888")

        # 情况 B: 消息列表末尾是 tool 消息，必须新开一条 user 消息
        messages_tool_tail = [
            {"role": "assistant", "content": "好的，正在读取文件。"},
            {"role": "tool", "tool_call_id": "call_123", "content": "file content"},
        ]
        with redirect_stdout(io.StringIO()):
            count2 = self.bg_mgr.inject_notifications(messages_tool_tail)
        self.assertEqual(count2, 1)
        self.assertEqual(len(messages_tool_tail), 3)
        self.assertEqual(messages_tool_tail[-1]["role"], "user")
        self.assertIn("【系统后台任务完成通知】", messages_tool_tail[-1]["content"])
        self.assertIn("<task_id>bg_8888</task_id>", messages_tool_tail[-1]["content"])

    def test_enhanced_run_shell_and_management_tools(self):
        """验证增强版 run_shell 与 check_bg_task/list_bg_tasks/kill_bg_task 工具封装。"""
        with redirect_stdout(io.StringIO()):
            sync_res = self.tools.execute("run_shell", {"command": "Write-Output 'Sync Call'"})
        self.assertIn("Sync Call", sync_res)

        sleep_cmd = "Start-Sleep -Seconds 20" if sys.platform == "win32" else "sleep 20"
        with redirect_stdout(io.StringIO()):
            async_res = self.tools.execute(
                "run_shell", {"command": sleep_cmd, "run_in_background": True}
            )
            self.assertIn("[后台任务 bg_", async_res)
            self.assertIn("已成功启动", async_res)

            list_res = self.tools.execute("list_bg_tasks", {})
            self.assertIn("bg_0001", list_res)
            self.assertIn("running", list_res)

            check_res = self.tools.execute("check_bg_task", {"task_id": "bg_0001"})
            self.assertIn("当前状态: running", check_res)

            kill_res = self.tools.execute("kill_bg_task", {"task_id": "bg_0001"})
            self.assertIn("已成功强制终止后台任务 'bg_0001'", kill_res)


class TestBackgroundRuntimeIntegration(unittest.TestCase):
    """验证 M5.1 与 runtime.py 主循环及 cli.py 退出清理的端到端集成。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.bg_mgr = BackgroundTaskManager(workdir=self.workdir)

    def tearDown(self):
        self.bg_mgr.cleanup_all()
        self.temp_dir.cleanup()

    def test_parent_tools_mounts_background_tools_while_shared_tools_and_subagent_stay_clean(self):
        """parent_tools 必须挂载后台管理 3 工具并增强 run_shell，shared_tools 与子代理不得继承。"""
        bg_tools = {"check_bg_task", "kill_bg_task", "list_bg_tasks"}
        parent_names = set(runtime_mod.parent_tools.list_tools())
        default_names = set(runtime_mod.default_tools.list_tools())
        shared_names = set(runtime_mod.shared_tools.list_tools())
        sub_names = set(runtime_mod.default_subagent_runner.tools.list_tools())

        self.assertEqual(parent_names, default_names)
        self.assertTrue(bg_tools.issubset(parent_names))
        self.assertTrue(shared_names.isdisjoint(bg_tools))
        self.assertTrue(sub_names.isdisjoint(bg_tools))

        parent_schemas = {
            s["function"]["name"]: s["function"] for s in runtime_mod.parent_tools.get_schemas()
        }
        shared_schemas = {
            s["function"]["name"]: s["function"] for s in runtime_mod.shared_tools.get_schemas()
        }
        self.assertIn("run_in_background", parent_schemas["run_shell"]["parameters"]["properties"])
        self.assertNotIn("run_in_background", shared_schemas["run_shell"]["parameters"]["properties"])

        prompt_text = runtime_mod.build_system_prompt(workdir=self.workdir)
        self.assertIn("run_in_background", prompt_text)
        self.assertIn("<task_notification>", prompt_text)

    def test_runtime_loop_injects_task_notification_after_tool_message(self):
        """主循环每轮调用 LLM 前应调用 BG_MANAGER.inject_notifications，且在 tool 消息后另起 user 消息。"""
        tools = runtime_mod.create_default_tools(workdir=self.workdir, bg_mgr=self.bg_mgr)
        (self.workdir / "alpha.txt").write_text("alpha ok\n", encoding="utf-8")
        messages = runtime_mod.create_session_messages(workdir=self.workdir)

        turn = 0
        observed_roles_on_turn2 = []

        def scripted_llm(messages, tools):
            nonlocal turn
            turn += 1
            if turn == 1:
                cmd = (
                    "Start-Sleep -Milliseconds 150; Write-Output 'ASYNC_TOKEN_77'"
                    if sys.platform == "win32"
                    else "sleep 0.15; echo 'ASYNC_TOKEN_77'"
                )
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "call_bg",
                            "name": "run_shell",
                            "arguments": {"command": cmd, "run_in_background": True},
                        }
                    ]
                )
            if turn == 2:
                # 在返回 tool(call_bg) 后等待后台进程就绪，下一轮循环头即会自动注入通知到新的 user 消息
                deadline = time.time() + 5.0
                while self.bg_mgr.has_running_tasks() and time.time() < deadline:
                    time.sleep(0.05)
                return _ns_msg(
                    tool_calls=[
                        {"id": "call_fg", "name": "read_file", "arguments": {"path": "alpha.txt"}}
                    ]
                )
            observed_roles_on_turn2.extend(m.get("role") for m in messages)
            return _ns_msg(content="已汇总后台结果与前台读取。")

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "启动后台任务并在同会话观察就绪通知",
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=scripted_llm,
                workdir=self.workdir,
                verbose=False,
            )

        self.assertEqual(reply, "已汇总后台结果与前台读取。")
        # 确认第 2 轮工具调用完毕后、第 3 轮模型前，在 tool 消息之后以新 user 消息注入了 <task_notification>
        self.assertTrue(
            any(
                m.get("role") == "user" and "<task_notification>" in str(m.get("content", ""))
                for m in messages
            )
        )
        self.assertIn("ASYNC_TOKEN_77", json.dumps(messages, ensure_ascii=False))
        # 验证没有破坏 tool -> tool_call_id 配对：最后一个通知 user 出现在 tool 消息之后
        notif_idx = next(
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user" and "<task_notification>" in str(m.get("content", ""))
        )
        self.assertEqual(messages[notif_idx - 1]["role"], "tool")
        self.assertEqual(messages[-1]["role"], "assistant")

    def test_cli_exits_clean_up_default_bg_manager(self):
        """CLI 单次模式与交互式 REPL 退出时必须触发 default_bg_manager.cleanup_all()。"""
        with patch.object(cli_mod.default_bg_manager, "cleanup_all") as mock_cleanup:
            with patch.object(cli_mod, "run_agent_turn", return_value="ok"):
                with redirect_stdout(io.StringIO()):
                    code = cli_mod.main(["查看版本"])
            self.assertEqual(code, 0)
            self.assertEqual(mock_cleanup.call_count, 1)

            with patch("builtins.input", side_effect=["exit"]):
                with redirect_stdout(io.StringIO()):
                    code2 = cli_mod.run_interactive_repl()
            self.assertEqual(code2, 0)
            self.assertEqual(mock_cleanup.call_count, 2)

    def test_background_demo_main_runs_cleanly(self):
        """M5.1 离线演示入口 `python -m coding_agent.background` 应正常跑通并返回 0。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = background_demo_main()

        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("[Background 后台任务启动]", out)
        self.assertIn("[Background 通知注入]", out)
        self.assertIn("<task_notification>", out)
        self.assertIn("BG_BUILD_DONE", out)


if __name__ == "__main__":
    unittest.main()
