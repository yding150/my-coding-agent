"""
test_integrated_harness.py - 双轨计划与双轨代理协同联合验证测试集
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from coding_agent.subagent.runner import SubagentRunner
from coding_agent.subagent.tools import register_subagent_tools
from coding_agent.tasks import TaskStore, TodoManager, register_task_tools, register_todo_tools
from coding_agent.tools.manager import ToolManager


class TestIntegratedHarness(unittest.TestCase):
    """测试双轨计划分工、双轨代理分工、一次性隔离子代理与防递归边界。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.shared_tools = ToolManager()

        @self.shared_tools.register(
            name="read_file",
            description="read file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        def dummy_read(path: str) -> str:
            return f"content of {path}"

        @self.shared_tools.register(
            name="task",
            description="subagent tool",
            parameters={"type": "object", "properties": {"prompt": {"type": "string"}}},
        )
        def dummy_task(prompt: str) -> str:
            return "nested task"

        @self.shared_tools.register(
            name="spawn_teammate",
            description="team tool",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        def dummy_spawn(name: str) -> str:
            return "spawned"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_todo_manager_basic_and_validation(self):
        """测试 TodoManager 计划第一轨：整表覆盖更新与状态校验。"""
        mgr = TodoManager()
        todos = [
            {"content": "第一步：分析依赖", "status": "completed"},
            {"content": "第二步：编写核心代码", "status": "in_progress"},
            {"content": "第三步：运行单元测试", "status": "pending"},
        ]
        res = mgr.update(todos)
        self.assertIn("会话待办清单更新", res)
        self.assertIn("[x] 第一步：分析依赖", res)
        self.assertIn("[>] 第二步：编写核心代码", res)
        self.assertIn("[ ] 第三步：运行单元测试", res)
        self.assertIn("1/3 已完成", res)

        # 校验：同一时间多个 in_progress 应抛出异常
        bad_todos = [
            {"content": "任务1", "status": "in_progress"},
            {"content": "任务2", "status": "in_progress"},
        ]
        with self.assertRaises(ValueError) as ctx:
            mgr.update(bad_todos)
        self.assertIn("只允许有 1 个任务处于 'in_progress'", str(ctx.exception))

    def test_subagent_anti_recursion_and_forbidden_tools(self):
        """测试子代理防递归边界：必须严格剔除 task、spawn_teammate、compact 等委派工具。"""
        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir)

        # 允许基础工具存在
        self.assertIn("read_file", runner.tools._tools)

        # 严格禁止任何再委派工具进入子代理工具集
        self.assertNotIn("task", runner.tools._tools)
        self.assertNotIn("spawn_teammate", runner.tools._tools)
        self.assertNotIn("compact", runner.tools._tools)

    @patch("coding_agent.subagent.runner.call_llm")
    def test_subagent_direct_answer(self, mock_call_llm):
        """测试一次性子代理直接给出答案（无工具调用）并返回精炼文本。"""
        mock_resp = MagicMock()
        mock_resp.tool_calls = []
        mock_resp.content = "经调研，本项目使用的是 pytest 测试框架。"
        mock_call_llm.return_value = mock_resp

        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir)
        with redirect_stdout(io.StringIO()):
            res = runner.run(prompt="查看测试框架", description="调研单测")

        self.assertEqual(res, "经调研，本项目使用的是 pytest 测试框架。")
        self.assertEqual(mock_call_llm.call_count, 1)

    @patch("coding_agent.subagent.runner.call_llm")
    def test_subagent_multi_turn_tool_execution(self, mock_call_llm):
        """测试一次性子代理多轮工具调用，最后产出总结。"""
        # 第一轮：模型调用 read_file 工具
        mock_call = MagicMock()
        mock_call.id = "call_01"
        mock_call.function.name = "read_file"
        mock_call.function.arguments = '{"path": "test.py"}'

        mock_resp_1 = MagicMock()
        mock_resp_1.tool_calls = [mock_call]
        mock_resp_1.content = None

        # 第二轮：模型根据工具返回的内容给出最终总结
        mock_resp_2 = MagicMock()
        mock_resp_2.tool_calls = []
        mock_resp_2.content = "发现 test.py 中定义了 3 个关键测试类。"

        mock_call_llm.side_effect = [mock_resp_1, mock_resp_2]

        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir)
        with redirect_stdout(io.StringIO()):
            res = runner.run(prompt="分析测试文件")

        self.assertEqual(res, "发现 test.py 中定义了 3 个关键测试类。")
        self.assertEqual(mock_call_llm.call_count, 2)

    def test_register_subagent_tools_integration(self):
        """测试向 Lead ToolManager 挂载双轨计划 (todo_write + DAG TaskStore 6工具) 与双轨代理 (task + spawn_teammate)。"""
        lead_tools = ToolManager()
        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir)
        todo_mgr = TodoManager()
        task_store = TaskStore(self.workdir / ".agent" / "tasks", workdir=self.workdir)

        # 注册双轨计划：计划第一轨 - todo_write
        register_todo_tools(lead_tools, todo_mgr)

        # 注册双轨计划：计划第二轨 - 真实 TaskStore DAG 6 工具
        register_task_tools(lead_tools, task_store)

        # 注册双轨代理：代理第一轨 - 一次性 task
        register_subagent_tools(lead_tools, runner)

        # 注册双轨代理：代理第二轨 - 持久 spawn_teammate（M6.2 迁入前用桩位对齐验证）
        @lead_tools.register(
            name="spawn_teammate",
            description="持久队友",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        def dummy_spawn(name: str) -> str:
            return f"teammate {name} online"

        schemas = lead_tools.get_schemas()
        names = [s["function"]["name"] for s in schemas]

        # 确认双轨计划工具同时存在于 Lead 工具池中
        self.assertIn("todo_write", names)
        for dag_tool in ("create_task", "update_task", "list_tasks", "get_task", "claim_task", "complete_task"):
            self.assertIn(dag_tool, names)

        # 确认双轨代理工具同时存在于 Lead 工具池中
        self.assertIn("task", names)
        self.assertIn("spawn_teammate", names)


if __name__ == "__main__":
    unittest.main()
