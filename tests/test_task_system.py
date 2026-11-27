"""
test_task_system.py - M4.2 持久 DAG 任务图系统 (`TaskStore`) 与主循环集成单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coding_agent.runtime as runtime_mod
from coding_agent.hooks import HookManager
from coding_agent.subagent import SubagentRunner
from coding_agent.tasks import TaskStore, register_task_tools
from coding_agent.tasks.__main__ import main as tasks_demo_main
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


class TestTaskSystem(unittest.TestCase):
    """测试公共层 TaskStore 与 6 个任务图工具的核心不变量。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.tasks_dir = self.workdir / ".agent" / "tasks"
        self.store = TaskStore(self.tasks_dir, workdir=self.workdir)
        self.tools = ToolManager()
        register_task_tools(self.tools, self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_registration_and_schemas(self):
        """测试 6 个任务工具注册及两阶段建图 Schema 约束。"""
        expected = {
            "create_task",
            "update_task",
            "list_tasks",
            "get_task",
            "claim_task",
            "complete_task",
        }
        self.assertEqual(set(self.tools.list_tools()), expected)
        self.assertFalse(self.tasks_dir.exists())

        schemas = {s["function"]["name"]: s["function"] for s in self.tools.get_schemas()}
        create_params = schemas["create_task"]["parameters"]
        update_params = schemas["update_task"]["parameters"]

        self.assertNotIn("blockedBy", create_params["properties"])
        self.assertIs(create_params["additionalProperties"], False)
        self.assertEqual(update_params["required"], ["task_id", "addBlockedBy"])
        self.assertIs(update_params["additionalProperties"], False)

    def test_two_phase_dag_claim_complete_and_unblock(self):
        """测试两阶段建图、依赖门禁、owner 校验及完成后自动解锁下游。"""
        schema = self.store.create("创建数据库表", "设计 users 与 tasks 表")
        api = self.store.create("编写后端 API")
        tests = self.store.create("补充集成测试")
        docs = self.store.create("编写接口文档")

        self.store.update_dependencies(api.id, [schema.id])
        self.store.update_dependencies(tests.id, [api.id])
        self.store.update_dependencies(docs.id, [schema.id])

        # 未完成 schema 前无法认领 api
        with redirect_stdout(io.StringIO()):
            blocked_msg = self.store.claim(api.id)
        self.assertIn("Blocked by", blocked_msg)
        self.assertIn(schema.id, blocked_msg)

        # 认领并完成 schema，应同时解锁 api 与 docs（但不解锁 tests）
        with redirect_stdout(io.StringIO()):
            self.assertIn("已认领任务", self.store.claim(schema.id))
            complete_msg = self.store.complete(schema.id)
        self.assertIn("已解锁下游任务", complete_msg)
        self.assertIn("编写后端 API", complete_msg)
        self.assertIn("编写接口文档", complete_msg)
        self.assertNotIn("补充集成测试", complete_msg)

        # 认领 api 并校验 owner 归属权
        with redirect_stdout(io.StringIO()):
            self.assertIn("已认领任务", self.store.claim(api.id))
            wrong_owner = self.store.complete(api.id, owner="other")
        self.assertIn("无权完成", wrong_owner)

        # 正确 owner 完成后应解锁 tests
        with redirect_stdout(io.StringIO()):
            api_done = self.store.complete(api.id)
        self.assertIn("补充集成测试", api_done)
        self.assertEqual(self.store.load(api.id).status, "completed")

        # 管理员覆盖校验：队友认领的任务，总指挥 (lead/agent) 在 allow_admin 下可直接验收完成
        with redirect_stdout(io.StringIO()):
            self.store.claim(tests.id, owner="teammate-worker")
            self.assertIn("无权完成", self.store.complete(tests.id, owner="lead", allow_admin=False))
            admin_done = self.store.complete(tests.id, owner="lead", allow_admin=True)
        self.assertIn("已完成任务", admin_done)
        self.assertEqual(self.store.load(tests.id).status, "completed")

    def test_update_atomic_validation_cycle_and_idempotency(self):
        """测试 update_dependencies 原子校验（防部分写入）、判环、幂等性与状态锁。"""
        t1 = self.store.create("步骤一")
        t2 = self.store.create("步骤二")
        t3 = self.store.create("步骤三")

        # 含不存在依赖时整批拒绝，不发生部分写入
        bad_res = self.tools.execute(
            "update_task",
            {"task_id": t2.id, "addBlockedBy": [t1.id, "task_00000000"]},
        )
        self.assertIn("未找到前置依赖任务", bad_res)
        self.assertEqual(self.store.load(t2.id).blockedBy, [])

        # 自依赖拒绝
        self_res = self.tools.execute(
            "update_task", {"task_id": t2.id, "addBlockedBy": [t2.id]}
        )
        self.assertIn("不能依赖自身", self_res)

        # 重复依赖幂等
        self.store.update_dependencies(t2.id, [t1.id, t1.id])
        self.store.update_dependencies(t2.id, [t1.id])
        self.assertEqual(self.store.load(t2.id).blockedBy, [t1.id])

        # 构造环路 t1 -> t3 -> t2 -> t1 应被拒绝
        self.store.update_dependencies(t3.id, [t2.id])
        cycle_res = self.tools.execute(
            "update_task", {"task_id": t1.id, "addBlockedBy": [t3.id]}
        )
        self.assertIn("检测到依赖环路", cycle_res)
        self.assertEqual(self.store.load(t1.id).blockedBy, [])

        # 已认领任务不允许再改依赖
        with redirect_stdout(io.StringIO()):
            self.store.claim(t1.id)
        started_res = self.tools.execute(
            "update_task", {"task_id": t1.id, "addBlockedBy": [t2.id]}
        )
        self.assertIn("pending 且无人认领", started_res)

    def test_id_collision_retry_and_cross_session_recovery(self):
        """测试随机 ID 冲突重试（排他写入）与新实例跨会话恢复。"""
        with patch(
            "coding_agent.tasks.store.secrets.token_hex",
            side_effect=["deadbeef", "deadbeef", "cafebabe"],
        ):
            first = self.store.create("任务 A", "描述 A")
            second = self.store.create("任务 B", "描述 B")

        self.assertEqual(first.id, "task_deadbeef")
        self.assertEqual(second.id, "task_cafebabe")

        # 使用全新的 TaskStore 实例模拟跨会话重启恢复
        restarted_store = TaskStore(self.tasks_dir, workdir=self.workdir)
        tasks = restarted_store.list()
        self.assertEqual([t.subject for t in tasks], ["任务 B", "任务 A"])
        detail = json.loads(restarted_store.get_detail(first.id))
        self.assertEqual(detail["description"], "描述 A")
        self.assertIn(first.id, restarted_store.render_list())

    def test_invalid_id_and_workspace_escape_defense(self):
        """测试非法 ID 拦截与工作区外越界目录/符号链接防护。"""
        invalid_res = self.tools.execute("get_task", {"task_id": "../outside"})
        self.assertIn("非法的任务 ID", invalid_res)

        with tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir).resolve()

            # 1. 直接指向工作区外部目录时应被拦截
            direct_escape = TaskStore(outside_path, workdir=self.workdir)
            with self.assertRaises(ValueError):
                direct_escape.create("越界任务")
            self.assertEqual(list(outside_path.iterdir()), [])

            # 2. 若系统允许创建符号链接，额外校验符号链接逃逸防护
            escape_root = self.workdir / ".escape_tasks"
            try:
                escape_root.symlink_to(outside_path, target_is_directory=True)
            except OSError:
                return

            escaped_store = TaskStore(escape_root, workdir=self.workdir)
            with self.assertRaises(ValueError):
                escaped_store.create("符号链接越界任务")
            self.assertEqual(list(outside_path.iterdir()), [])


class TestTaskRuntimeIntegration(unittest.TestCase):
    """验证 M4.2 持久 DAG 任务图与 runtime.py 主循环集成。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parent_tools_mounts_dag_tools_while_shared_tools_and_subagent_stay_clean(self):
        """parent_tools 必须挂载 DAG 6 工具，shared_tools 与子代理不可继承以避免越权改图。"""
        dag_tools = {"create_task", "update_task", "list_tasks", "get_task", "claim_task", "complete_task"}
        parent_names = set(runtime_mod.parent_tools.list_tools())
        default_names = set(runtime_mod.default_tools.list_tools())
        shared_names = set(runtime_mod.shared_tools.list_tools())
        sub_names = set(runtime_mod.default_subagent_runner.tools.list_tools())

        self.assertEqual(parent_names, default_names)
        self.assertTrue(dag_tools.issubset(parent_names))
        self.assertTrue(shared_names.isdisjoint(dag_tools))
        self.assertTrue(sub_names.isdisjoint(dag_tools))
        for tool_name in ("create_task", "update_task", "claim_task", "complete_task"):
            self.assertIn(tool_name, runtime_mod.PLANNING_TOOLS)

        prompt_text = runtime_mod.build_system_prompt(workdir=self.workdir)
        self.assertIn("双轨计划分工规范", prompt_text)
        self.assertIn(".agent/tasks/", prompt_text)
        self.assertIn("create_task", prompt_text)

    def test_runtime_executes_dag_task_graph_and_subagent_collaboration(self):
        """主循环应能在 .agent/tasks/ 建立依赖任务图、委派子代理调研并自动解锁完成下游。"""
        (self.workdir / "README.md").write_text("PROJECT_VER=2.0.0\n", encoding="utf-8")
        tasks_dir = self.workdir / ".agent" / "tasks"
        store = runtime_mod.create_default_task_store(workdir=self.workdir)

        shared = runtime_mod.create_shared_tools(workdir=self.workdir)
        sub_step = 0

        def sub_llm(messages, tools):
            nonlocal sub_step
            sub_step += 1
            if sub_step == 1:
                return _ns_msg(
                    tool_calls=[{"id": "sub_1", "name": "read_file", "arguments": {"path": "README.md"}}]
                )
            return _ns_msg(content=f"调研完毕：{messages[-1]['content'].strip()}")

        runner = SubagentRunner(shared_tools=shared, workdir=self.workdir, llm_caller=sub_llm)
        parent = runtime_mod.create_default_tools(
            workdir=self.workdir,
            task_store=store,
            subagent_runner=runner,
        )

        lead_turn = 0
        created_ids = []
        planning_state = {"rounds_since_planning": 4}

        def lead_llm(messages, tools):
            nonlocal lead_turn
            lead_turn += 1
            if lead_turn == 1:
                return _ns_msg(
                    tool_calls=[
                        {"id": "t_1", "name": "create_task", "arguments": {"subject": "调研版本", "description": "查 README.md"}},
                        {"id": "t_2", "name": "create_task", "arguments": {"subject": "汇总报告", "description": "依赖调研结论"}},
                    ]
                )
            if lead_turn == 2:
                # 从工具输出里提取真实分配的 2 个 task_xxxxxxxx ID
                import re

                for msg in messages:
                    if msg.get("role") == "tool":
                        found = re.findall(r"task_[0-9a-f]{8}", msg["content"])
                        for tid in found:
                            if tid not in created_ids:
                                created_ids.append(tid)
                self.assertEqual(len(created_ids), 2)
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "u_1",
                            "name": "update_task",
                            "arguments": {"task_id": created_ids[1], "addBlockedBy": [created_ids[0]]},
                        }
                    ]
                )
            if lead_turn == 3:
                return _ns_msg(
                    tool_calls=[
                        {"id": "c_1", "name": "claim_task", "arguments": {"task_id": created_ids[0]}},
                        {
                            "id": "sub_call",
                            "name": "task",
                            "arguments": {"description": "读取版本文件", "prompt": "读取 README.md"},
                        },
                    ]
                )
            if lead_turn == 4:
                return _ns_msg(
                    tool_calls=[
                        {"id": "done_1", "name": "complete_task", "arguments": {"task_id": created_ids[0]}},
                    ]
                )
            if lead_turn == 5:
                return _ns_msg(
                    tool_calls=[
                        {"id": "c_2", "name": "claim_task", "arguments": {"task_id": created_ids[1]}},
                        {"id": "done_2", "name": "complete_task", "arguments": {"task_id": created_ids[1]}},
                    ]
                )
            return _ns_msg(content="双任务依赖图与子代理协同全部完成。")

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "创建两阶段依赖任务并委派子代理读取 README.md 后完成",
                tools=parent,
                hooks=HookManager(),
                messages=[],
                llm_caller=lead_llm,
                workdir=self.workdir,
                planning_state=planning_state,
                verbose=False,
            )

        self.assertEqual(reply, "双任务依赖图与子代理协同全部完成。")
        self.assertEqual(planning_state["rounds_since_planning"], 0)
        persisted = sorted(tasks_dir.glob("task_*.json"))
        self.assertEqual(len(persisted), 2)

        reloaded = TaskStore(tasks_dir, workdir=self.workdir)
        t_first = reloaded.load(created_ids[0])
        t_second = reloaded.load(created_ids[1])
        self.assertEqual(t_first.status, "completed")
        self.assertEqual(t_second.status, "completed")
        self.assertEqual(t_second.blockedBy, [created_ids[0]])

    def test_tasks_demo_main_runs_cleanly(self):
        """M4.2 离线演示入口 `python -m coding_agent.tasks` 应跑通并返回 0。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = tasks_demo_main()

        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("[系统守卫提示]", out)
        self.assertIn("Blocked by", out)
        self.assertIn("已解锁下游任务", out)
        self.assertIn("3 个 JSON 文件保存在", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
