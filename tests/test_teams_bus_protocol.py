"""
test_teams_bus_protocol.py - M6.1 离线 JSONL 邮箱 (`MessageBus`) 与团队审批协议 (`TeamProtocols`) 单元测试
"""

import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from coding_agent.teams import (
    DEFAULT_MAILBOX_DIR,
    PLAN_GATED_TOOLS,
    MessageBus,
    TeamProtocols,
)
from coding_agent.teams.__main__ import main as teams_demo_main


class TestMessageBus(unittest.TestCase):
    """测试基于 `.agent/mailboxes/*.jsonl` 的离线消息总线收发、并发阻塞唤醒与安全边界。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.bus = MessageBus(workdir=self.workdir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mailbox_converges_under_dot_agent_and_blocks_escape(self):
        """默认邮箱目录必须收敛于 `.agent/mailboxes/` 且懒创建，越界路径和非法 Agent 名应被拦截。"""
        self.assertEqual(DEFAULT_MAILBOX_DIR, ".agent/mailboxes")
        self.assertEqual(self.bus.mailbox_root, self.workdir / ".agent" / "mailboxes")
        self.assertFalse(self.bus.mailbox_root.exists())

        with self.assertRaises(ValueError):
            MessageBus(workdir=self.workdir, mailbox_dir="../outside_mailboxes")

        with self.assertRaises(ValueError):
            self.bus.send("lead", "../../..", "invalid agent")

        with self.assertRaises(ValueError):
            self.bus.peek("")

    def test_send_peek_read_and_partial_limit_consume(self):
        """验证单条/批量 JSONL 写入、`peek` 检测、`limit` 分批消费与队列清空语义。"""
        self.assertFalse(self.bus.peek("alice"))

        self.bus.send("lead", "alice", "第一条任务", msg_type="task_assignment", metadata={"task_id": "t1"})
        self.bus.send("bob", "alice", "第二条协同", msg_type="message")
        self.bus.send("lead", "alice", "第三条补充", msg_type="message")

        self.assertTrue(self.bus.peek("alice"))
        mailbox_file = self.bus.mailbox_root / "alice.jsonl"
        self.assertTrue(mailbox_file.is_file())

        # 按 limit=2 分批消费，应精确写回剩余 1 条未读消息
        first_batch = self.bus.read_inbox("alice", limit=2)
        self.assertEqual([m["content"] for m in first_batch], ["第一条任务", "第二条协同"])
        self.assertEqual(first_batch[0]["metadata"], {"task_id": "t1"})
        self.assertTrue(self.bus.peek("alice"))

        second_batch = self.bus.read_inbox("alice")
        self.assertEqual([m["content"] for m in second_batch], ["第三条补充"])
        self.assertFalse(self.bus.peek("alice"))
        self.assertFalse(mailbox_file.exists())

    def test_read_inbox_tolerates_corrupted_jsonl_lines(self):
        """收件箱中混入空行或损坏 JSON 行时应容错跳过并清空，避免整箱卡死。"""
        self.bus.mailbox_root.mkdir(parents=True, exist_ok=True)
        path = self.bus.mailbox_root / "charlie.jsonl"
        valid_msg = json.dumps({"from": "lead", "to": "charlie", "content": "ok"}, ensure_ascii=False)
        path.write_text(f"\n{{bad json line\n{valid_msg}\n", encoding="utf-8")

        inbox = self.bus.read_inbox("charlie")
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["content"], "ok")
        self.assertFalse(self.bus.peek("charlie"))

    def test_wait_for_messages_thread_wakeup_and_timeout(self):
        """测试空闲接收线程通过条件变量阻塞等待，来信立即唤醒且超时安全返回空列表。"""
        self.assertEqual(self.bus.wait_for_messages("david", timeout=0.05), [])

        received = []
        ready_to_wait = threading.Event()

        def consumer():
            ready_to_wait.set()
            received.extend(self.bus.wait_for_messages("david", timeout=2.0))

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        self.assertTrue(ready_to_wait.wait(timeout=1.0))
        time.sleep(0.05)

        self.bus.send("lead", "david", "立即唤醒工作", msg_type="wakeup")
        t.join(timeout=2.0)

        self.assertFalse(t.is_alive())
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["content"], "立即唤醒工作")
        self.assertFalse(self.bus.peek("david"))

    def test_concurrent_multi_producer_writes_preserve_all_messages(self):
        """多线程并发向同一收件箱写入时，RLock 必须保证零消息丢失与零 JSON 交错损坏。"""
        workers = []
        for wid in range(5):
            def _send_batch(idx=wid):
                for seq in range(20):
                    self.bus.send(f"worker_{idx}", "lead", f"msg_{idx}_{seq}")

            th = threading.Thread(target=_send_batch, daemon=True)
            workers.append(th)
            th.start()

        for th in workers:
            th.join(timeout=5.0)

        inbox = self.bus.read_inbox("lead")
        self.assertEqual(len(inbox), 100)


class TestTeamProtocols(unittest.TestCase):
    """测试 TeamProtocols 计划审批闸门与平滑关机协议状态流转。"""

    def setUp(self):
        self.protocols = TeamProtocols()

    def test_plan_approval_gate_full_state_transitions(self):
        """验证未提交/待审批/已驳回均拦截修改工具，批准后解锁，取消约束后放行。"""
        agent = "bob"
        self.assertEqual(self.protocols.get_plan_gate(agent), "not_required")
        self.assertTrue(self.protocols.is_tool_allowed(agent, "write_file")[0])

        # 开启计划审批要求后，只读工具放行、修改型工具全拦截
        self.protocols.set_plan_requirement(agent, required=True)
        self.assertEqual(self.protocols.get_plan_gate(agent), "required")
        self.assertTrue(self.protocols.is_tool_allowed(agent, "read_file")[0])
        self.assertTrue(self.protocols.is_tool_allowed(agent, "glob_files")[0])

        for gated in PLAN_GATED_TOOLS:
            allowed, reason = self.protocols.is_tool_allowed(agent, gated)
            self.assertFalse(allowed)
            self.assertIn("拦截", reason)

        # 提交方案 -> pending（仍拦截）
        req1 = self.protocols.submit_plan(agent, "方案1：直接改表结构", task_id="task_12345678")
        self.assertEqual(req1.status, "pending")
        self.assertEqual(req1.task_id, "task_12345678")
        self.assertEqual(self.protocols.get_plan_gate(agent), "pending")
        self.assertEqual(self.protocols.plan_request_ids[agent], req1.request_id)
        self.assertFalse(self.protocols.is_tool_allowed(agent, "edit_file")[0])

        # 驳回方案 -> rejected（仍拦截并记录反馈）
        rejected = self.protocols.approve_plan(req1.request_id, approve=False, feedback="需要迁移脚本")
        self.assertIsNotNone(rejected)
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.feedback, "需要迁移脚本")
        self.assertEqual(self.protocols.get_plan_gate(agent), "rejected")
        self.assertFalse(self.protocols.is_tool_allowed(agent, "run_shell")[0])

        # 重新提交并批准 -> approved（全部修改工具放行）
        req2 = self.protocols.submit_plan(agent, "方案2：双写渐进迁移", task_id="task_12345678")
        approved = self.protocols.approve_plan(req2.request_id, approve=True, feedback="LGTM")
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.feedback, "LGTM")
        self.assertEqual(self.protocols.get_plan_gate(agent), "approved")
        for gated in PLAN_GATED_TOOLS:
            self.assertTrue(self.protocols.is_tool_allowed(agent, gated)[0])

        # 取消要求后恢复 not_required 并清除绑定 req_id
        self.protocols.set_plan_requirement(agent, required=False)
        self.assertEqual(self.protocols.get_plan_gate(agent), "not_required")
        self.assertNotIn(agent, self.protocols.plan_request_ids)

    def test_shutdown_protocol_and_mismatched_requests(self):
        """验证关机请求校验目标 Agent 身份且 `approve_plan` 不误认关机请求。"""
        shutdown_req = self.protocols.create_shutdown_request("alice")
        self.assertEqual(shutdown_req.type, "shutdown")
        self.assertEqual(shutdown_req.status, "pending")

        # 类型不匹配时 approve_plan 返回 None；不存在 ID 返回 None
        self.assertIsNone(self.protocols.approve_plan(shutdown_req.request_id, approve=True))
        self.assertIsNone(self.protocols.approve_plan("req_missing", approve=True))

        # 其他 Agent 冒名确认关机必须被拒绝
        self.assertFalse(self.protocols.acknowledge_shutdown(shutdown_req.request_id, "bob"))
        self.assertEqual(shutdown_req.status, "pending")

        # 目标 Agent 确认成功
        self.assertTrue(self.protocols.acknowledge_shutdown(shutdown_req.request_id, "alice"))
        self.assertEqual(shutdown_req.status, "approved")

    def test_teams_demo_main_runs_cleanly(self):
        """M6.1 离线消息投递与协议审批演示入口 `python -m coding_agent.teams` 应正常跑通。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = teams_demo_main()

        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("收到 2 条消息", out)
        self.assertIn("[消费即清空校验] peek('alice') = False", out)
        self.assertIn("[计划闸门-批准后] 状态=approved, write_file 放行=True", out)
        self.assertIn("[平滑关机握手]", out)
        self.assertIn("ack=True, status=approved", out)


if __name__ == "__main__":
    unittest.main()
