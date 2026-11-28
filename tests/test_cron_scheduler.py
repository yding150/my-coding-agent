"""
test_cron_scheduler.py - M5.2 Cron 定时调度引擎 (`coding_agent.cron`) 与并发唤醒线程单元测试
"""

import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coding_agent.cli as cli_mod
import coding_agent.runtime as runtime_mod
from coding_agent.cron import (
    CronSchedulerManager,
    cron_matches,
    register_cron_tools,
    validate_cron,
)
from coding_agent.cron.__main__ import main as cron_demo_main
from coding_agent.hooks import HookEvent, HookManager
from coding_agent.tools import ToolManager


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


class TestCronSchedulerSystem(unittest.TestCase):
    """测试 Cron 表达式校验/匹配、原子持久化、去重巡检、At-Least-Once 状态机与工具注册。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name).resolve()
        self.mgr = CronSchedulerManager(workdir=self.test_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_cron_syntax(self):
        """五段式 Cron 表达式合法与非法语法校验。"""
        self.assertIsNone(validate_cron("* * * * *"))
        self.assertIsNone(validate_cron("*/5 * * * *"))
        self.assertIsNone(validate_cron("0 9 * * 1-5"))
        self.assertIsNone(validate_cron("0,30 8-18 1,15 * 0"))
        self.assertIsNone(validate_cron("59 23 31 12 6"))

        self.assertIn("包含 5 个字段", validate_cron("* * * *"))
        self.assertIn("包含 5 个字段", validate_cron("* * * * * *"))
        self.assertIn("超出合法取值边界", validate_cron("60 * * * *"))
        self.assertIn("超出合法取值边界", validate_cron("* 24 * * *"))
        self.assertIn("超出合法取值边界", validate_cron("* * 0 * *"))
        self.assertIn("超出合法取值边界", validate_cron("* * 32 * *"))
        self.assertIn("超出合法取值边界", validate_cron("* * * 13 *"))
        self.assertIn("超出合法取值边界", validate_cron("* * * * 7"))
        self.assertIn("起始值大于结束值", validate_cron("10-5 * * * *"))
        self.assertIn("无效的步长", validate_cron("*/0 * * * *"))

    def test_cron_matches_engine(self):
        """五段式时间匹配引擎精度测试（含星期与步长）。"""
        # 2026-11-16 (周一) 14:30:00 -> Cron weekday = (0+1)%7 = 1
        moment = datetime(2026, 11, 16, 14, 30, 0)

        self.assertTrue(cron_matches("* * * * *", moment))
        self.assertTrue(cron_matches("30 14 * * *", moment))
        self.assertTrue(cron_matches("*/5 * * * *", moment))
        self.assertTrue(cron_matches("*/10 14 * * *", moment))
        self.assertTrue(cron_matches("30 14 16 11 1", moment))
        self.assertTrue(cron_matches("20-40 10-18 * * 1-5", moment))
        self.assertTrue(cron_matches("0,15,30,45 * * * *", moment))

        self.assertFalse(cron_matches("31 14 * * *", moment))
        self.assertFalse(cron_matches("30 15 * * *", moment))
        self.assertFalse(cron_matches("30 14 * * 0", moment))
        self.assertFalse(cron_matches("*/7 * * * *", moment))

    def test_schedule_and_persistence_under_dot_agent(self):
        """任务注册原子写入 `.agent/scheduled_tasks.json`，支持跨实例恢复且拒绝越界路径。"""
        with redirect_stdout(io.StringIO()):
            job = self.mgr.schedule(
                cron="0 9 * * *",
                prompt="执行日常构建测试",
                recurring=True,
                durable=True,
            )

        self.assertTrue(job.id.startswith("cron_"))
        self.assertEqual(job.cron, "0 9 * * *")
        self.assertEqual(self.mgr.durable_file, self.test_dir / ".agent" / "scheduled_tasks.json")
        self.assertTrue(self.mgr.durable_file.exists())

        content = json.loads(self.mgr.durable_file.read_text(encoding="utf-8"))
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["id"], job.id)
        self.assertEqual(content[0]["prompt"], "执行日常构建测试")

        with redirect_stdout(io.StringIO()):
            new_mgr = CronSchedulerManager(workdir=self.test_dir)
        recovered_job = new_mgr.get_job(job.id)
        self.assertIsNotNone(recovered_job)
        self.assertEqual(recovered_job.prompt, "执行日常构建测试")
        self.assertEqual(recovered_job.cron, "0 9 * * *")

        with self.assertRaises(ValueError):
            CronSchedulerManager(workdir=self.test_dir, file_name="../../outside.json")

    def test_cancel_job(self):
        """取消任务并同步清理持久化文件。"""
        with redirect_stdout(io.StringIO()):
            job = self.mgr.schedule(cron="*/10 * * * *", prompt="临时任务")
            self.assertEqual(len(self.mgr.list_jobs()), 1)
            success = self.mgr.cancel(job.id)

        self.assertTrue(success)
        self.assertEqual(len(self.mgr.list_jobs()), 0)

        content = json.loads(self.mgr.durable_file.read_text(encoding="utf-8"))
        self.assertEqual(len(content), 0)
        self.assertFalse(self.mgr.cancel("cron_not_exist"))

    def test_poll_due_jobs_and_deduplication(self):
        """到期触发入队及同一分钟内 `last_fired` + `pending_delivery` 防重锁。"""
        moment = datetime(2026, 11, 16, 9, 0, 0)
        with redirect_stdout(io.StringIO()):
            job = self.mgr.schedule(cron="0 9 * * *", prompt="早上9点跑测试", recurring=True)
            due1 = self.mgr.poll_due_jobs(moment)

        self.assertEqual(len(due1), 1)
        self.assertEqual(due1[0].id, job.id)
        self.assertTrue(job.pending_delivery)
        self.assertEqual(job.last_fired, "2026-11-16 09:00")
        self.assertTrue(self.mgr.has_queued_jobs())

        due2 = self.mgr.poll_due_jobs(moment)
        self.assertEqual(len(due2), 0)
        self.assertEqual(len(self.mgr.cron_queue), 1)

    def test_acknowledge_and_restore_lifecycle(self):
        """消费队列、确认完成 (`acknowledge_jobs`) 与异常回滚 (`restore_jobs`) 状态机。"""
        with redirect_stdout(io.StringIO()):
            job_recur = self.mgr.schedule("0 9 * * *", "周期任务", recurring=True)
            job_once = self.mgr.schedule("0 9 * * *", "单次任务", recurring=False)
            self.mgr.poll_due_jobs(datetime(2026, 11, 16, 9, 0, 0))

        consumed = self.mgr.consume_queue()
        self.assertEqual(len(consumed), 2)
        self.assertFalse(self.mgr.has_queued_jobs())

        self.mgr.restore_jobs(consumed)
        self.assertTrue(self.mgr.has_queued_jobs())
        self.assertEqual(len(self.mgr.cron_queue), 2)

        consumed_again = self.mgr.consume_queue()
        self.mgr.acknowledge_jobs(consumed_again)

        self.assertIsNone(self.mgr.get_job(job_once.id))
        recur_after = self.mgr.get_job(job_recur.id)
        self.assertIsNotNone(recur_after)
        self.assertFalse(recur_after.pending_delivery)

    def test_tool_registration_and_execution(self):
        """挂载 `schedule_cron` / `list_crons` / `cancel_cron` 工具及执行闭环。"""
        tools = ToolManager()
        register_cron_tools(tools, self.mgr)

        schema_names = [s["function"]["name"] for s in tools.get_schemas()]
        self.assertEqual(set(schema_names), {"schedule_cron", "list_crons", "cancel_cron"})

        with redirect_stdout(io.StringIO()):
            res = tools.execute(
                "schedule_cron",
                {"cron": "*/15 * * * *", "prompt": "定时检查服务健康状态", "recurring": True},
            )
        self.assertIn("定时任务注册成功", res)
        self.assertIn("cron_", res)

        list_res = tools.execute("list_crons", {})
        self.assertIn("定时任务一览看板", list_res)
        self.assertIn("定时检查服务健康状态", list_res)
        self.assertIn("等待调度", list_res)

        job_id = self.mgr.list_jobs()[0].id
        with redirect_stdout(io.StringIO()):
            cancel_res = tools.execute("cancel_cron", {"job_id": job_id})
        self.assertIn("已成功取消定时任务", cancel_res)

    def test_list_crons_shows_pending_delivery_state(self):
        """任务触发但尚未 ACK 时，`list_crons` 必须清晰展示交付执行中状态防止误判。"""
        tools = ToolManager()
        register_cron_tools(tools, self.mgr)

        with redirect_stdout(io.StringIO()):
            once_job = self.mgr.schedule("* * * * *", "单次巡检", recurring=False)
            recur_job = self.mgr.schedule("* * * * *", "周期巡检", recurring=True)
            self.mgr.poll_due_jobs(datetime(2026, 11, 28, 16, 20, 0))

        board = tools.execute("list_crons", {})
        self.assertIn(once_job.id, board)
        self.assertIn("单次任务将在本轮完成后自动核销，不会重复触发", board)
        self.assertIn(recur_job.id, board)
        self.assertIn("本轮完成后解锁等待下次调度", board)

        consumed = self.mgr.consume_queue()
        self.mgr.acknowledge_jobs(consumed)
        board_after = tools.execute("list_crons", {})
        self.assertNotIn(once_job.id, board_after)
        self.assertIn(recur_job.id, board_after)
        self.assertIn("等待调度", board_after)


class TestCronRuntimeIntegration(unittest.TestCase):
    """验证 M5.2 Cron 工具分层、`agent_lock` 互斥以及后台空闲唤醒线程集成。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.cron_mgr = CronSchedulerManager(workdir=self.workdir)

    def tearDown(self):
        runtime_mod.stop_runtime_threads()
        self.temp_dir.cleanup()

    def test_parent_tools_mounts_cron_tools_while_shared_tools_and_subagent_stay_clean(self):
        """parent_tools 必须挂载 Cron 3 工具，shared_tools 与同步子代理禁止继承。"""
        cron_tools = {"schedule_cron", "list_crons", "cancel_cron"}
        parent_names = set(runtime_mod.parent_tools.list_tools())
        default_names = set(runtime_mod.default_tools.list_tools())
        shared_names = set(runtime_mod.shared_tools.list_tools())
        sub_names = set(runtime_mod.default_subagent_runner.tools.list_tools())

        self.assertEqual(parent_names, default_names)
        self.assertTrue(cron_tools.issubset(parent_names))
        self.assertTrue(shared_names.isdisjoint(cron_tools))
        self.assertTrue(sub_names.isdisjoint(cron_tools))

        prompt_text = runtime_mod.build_system_prompt(workdir=self.workdir)
        self.assertIn("schedule_cron", prompt_text)
        self.assertIn(".agent/scheduled_tasks.json", prompt_text)

    def test_scheduled_turn_skips_user_prompt_hook_and_adds_cron_prefix(self):
        """`is_scheduled=True` 时不得触发 `UserPromptSubmit` 钩子，并应附加闹钟前缀。"""
        hooks = HookManager()
        user_prompt_events = []

        @hooks.on(HookEvent.USER_PROMPT_SUBMIT)
        def _capture_prompt(text: str) -> None:
            user_prompt_events.append(text)

        messages = []
        reply = runtime_mod.run_agent_turn(
            "检查日志状态",
            tools=runtime_mod.create_default_tools(workdir=self.workdir, cron_mgr=self.cron_mgr),
            hooks=hooks,
            messages=messages,
            llm_caller=lambda messages, tools: _ns_msg(content="定时巡检完成"),
            workdir=self.workdir,
            is_scheduled=True,
            verbose=False,
        )

        self.assertEqual(reply, "定时巡检完成")
        self.assertEqual(user_prompt_events, [])
        self.assertEqual(messages[1]["role"], "user")
        self.assertTrue(messages[1]["content"].startswith("⏰ [定时调度任务触发] "))

    def test_agent_lock_defers_queue_wakeup_until_foreground_idle(self):
        """前台持有 `agent_lock` 时 `queue_processor_loop` 必须非阻塞退让；释放后立即唤醒并 ACK。"""
        with redirect_stdout(io.StringIO()):
            job = self.cron_mgr.schedule("* * * * *", "后台巡检任务", recurring=False)
            self.cron_mgr.poll_due_jobs(datetime.now())

        test_lock = threading.RLock()
        stop_event = threading.Event()
        invoked_prompts = []
        done_event = threading.Event()

        def fake_runner(prompt: str, *, is_scheduled: bool = False):
            invoked_prompts.append((prompt, is_scheduled))
            done_event.set()
            return "done"

        test_lock.acquire()
        thread = threading.Thread(
            target=runtime_mod.queue_processor_loop,
            args=(stop_event,),
            kwargs={
                "cron_mgr": self.cron_mgr,
                "lock": test_lock,
                "turn_runner": fake_runner,
                "poll_interval": 0.03,
            },
            daemon=True,
        )
        thread.start()
        try:
            time.sleep(0.15)
            # 前台仍持有互斥锁时绝对不能唤醒执行
            self.assertEqual(invoked_prompts, [])
            self.assertTrue(self.cron_mgr.has_queued_jobs())
        finally:
            test_lock.release()

        self.assertTrue(done_event.wait(timeout=2.0), "释放锁后后台唤醒线程未按预期消费任务")
        stop_event.set()
        thread.join(timeout=1.0)

        self.assertEqual(invoked_prompts, [("后台巡检任务", True)])
        # 一次性任务被 acknowledge_jobs 销毁
        self.assertIsNone(self.cron_mgr.get_job(job.id))
        self.assertFalse(self.cron_mgr.has_queued_jobs())

    def test_queue_processor_restores_job_when_runner_fails(self):
        """`queue_processor_loop` 调用模型抛异常时必须调用 `restore_jobs` 回滚放回队列。"""
        with redirect_stdout(io.StringIO()):
            job = self.cron_mgr.schedule("* * * * *", "易失败定时任务", recurring=True)
            self.cron_mgr.poll_due_jobs(datetime.now())

        test_lock = threading.RLock()
        stop_event = threading.Event()
        attempted = threading.Event()

        def failing_runner(prompt: str, *, is_scheduled: bool = False):
            attempted.set()
            raise RuntimeError("transient network failure")

        thread = threading.Thread(
            target=runtime_mod.queue_processor_loop,
            args=(stop_event,),
            kwargs={
                "cron_mgr": self.cron_mgr,
                "lock": test_lock,
                "turn_runner": failing_runner,
                "poll_interval": 0.03,
            },
            daemon=True,
        )
        thread.start()
        try:
            self.assertTrue(attempted.wait(timeout=2.0))
            time.sleep(0.05)
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

        self.assertTrue(self.cron_mgr.has_queued_jobs())
        restored = self.cron_mgr.get_job(job.id)
        self.assertIsNotNone(restored)
        self.assertTrue(restored.pending_delivery)
        # 确保异常后互斥锁仍被 finally 正确释放
        self.assertTrue(test_lock.acquire(blocking=False))
        test_lock.release()

    def test_start_and_stop_runtime_threads_and_cli_repl_lifecycle(self):
        """`start_runtime_threads`/`stop_runtime_threads` 及 CLI REPL 退出应安全启停所有线程。"""
        threads = runtime_mod.start_runtime_threads(
            cron_mgr=self.cron_mgr,
            scheduler_interval=0.05,
            processor_interval=0.05,
        )
        self.assertEqual([t.name for t in threads], ["cron-scheduler", "queue-processor"])
        self.assertTrue(all(t.is_alive() for t in threads))

        runtime_mod.stop_runtime_threads()
        self.assertEqual(runtime_mod.runtime_threads, [])
        self.assertTrue(all(not t.is_alive() for t in threads))

        # 验证 cli.run_interactive_repl 启动并正常退出时调用 start_runtime_threads / stop_runtime_threads
        with (
            patch.object(cli_mod, "start_runtime_threads") as mock_start,
            patch.object(cli_mod, "stop_runtime_threads") as mock_stop,
            patch.object(cli_mod.default_bg_manager, "cleanup_all") as mock_bg_clean,
            patch("builtins.input", side_effect=["exit"]),
        ):
            with redirect_stdout(io.StringIO()):
                code = cli_mod.run_interactive_repl()

        self.assertEqual(code, 0)
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_bg_clean.assert_called_once()

    def test_cron_demo_main_runs_cleanly(self):
        """M5.2 离线演示入口 `python -m coding_agent.cron` 应跑通并返回 0。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cron_demo_main()

        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("[Cron 任务注册]", out)
        self.assertIn("[互斥锁退让校验] 前台持锁期间是否提前触发唤醒: False", out)
        self.assertIn("[空闲唤醒校验] 前台释放锁后唤醒成功: True", out)
        self.assertIn("[一次性任务自动销毁校验] 剩余活跃 Cron 数: 0", out)


if __name__ == "__main__":
    unittest.main()
