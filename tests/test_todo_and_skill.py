"""
test_todo_and_skill.py - M2.3 TodoManager、SkillLoader 与主循环计划守卫单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import coding_agent.runtime as runtime_mod
import coding_agent.tasks.__main__ as tasks_cli_mod
from coding_agent.config import SKILLS_DIR
from coding_agent.hooks import HookManager
from coding_agent.skills import SkillLoader, register_skill_tools
from coding_agent.tools import TodoManager, ToolManager, register_todo_tools


class TestTodoAndSkill(unittest.TestCase):
    """测试公共层 TodoManager 与 SkillLoader"""

    def test_todo_manager_lifecycle(self):
        """测试任务清单的状态更新、校验与看板渲染"""
        todo_mgr = TodoManager()

        todos_payload = [
            {"content": "第1步：调研需求", "status": "completed"},
            {"content": "第2步：编写代码", "status": "in_progress"},
            {"content": "第3步：自动化测试", "status": "pending"},
        ]
        with redirect_stdout(io.StringIO()):
            res = todo_mgr.update(todos_payload)
        self.assertIn("任务清单更新", res)
        self.assertIn("会话待办清单更新", res)
        self.assertIn("[x] 第1步：调研需求", res)
        self.assertIn("[>] 第2步：编写代码", res)
        self.assertIn("[ ] 第3步：自动化测试", res)
        self.assertFalse(todo_mgr.is_all_completed())
        self.assertEqual(len(todo_mgr.get_items()), 3)

        # 测试多个 in_progress 约束
        bad_payload = [
            {"content": "任务1", "status": "in_progress"},
            {"content": "任务2", "status": "in_progress"},
        ]
        with self.assertRaises(ValueError):
            todo_mgr.update(bad_payload)

    def test_todo_manager_string_parsing_and_validation_edges(self):
        """测试 JSON/字面量字符串兼容、空列表与越界数量校验"""
        todo_mgr = TodoManager()
        with redirect_stdout(io.StringIO()):
            res_json = todo_mgr.update('[{"content": "步骤A", "status": "completed"}]')
        self.assertIn("[x] 步骤A", res_json)
        self.assertTrue(todo_mgr.is_all_completed())

        with self.assertRaises(ValueError):
            todo_mgr.update("not valid list")
        with self.assertRaises(ValueError):
            todo_mgr.update([{"content": f"t{i}", "status": "pending"} for i in range(21)])
        with self.assertRaises(ValueError):
            todo_mgr.update([{"content": "", "status": "pending"}])
        with self.assertRaises(ValueError):
            todo_mgr.update([{"content": "t1", "status": "unknown_status"}])

    def test_register_todo_tools(self):
        """测试 todo_write 工具挂载与 execute() 执行（含校验失败自愈回填）"""
        tools = ToolManager()
        _, todo_mgr = register_todo_tools(tools)
        self.assertTrue(tools.has("todo_write"))

        with redirect_stdout(io.StringIO()):
            exec_res = tools.execute(
                "todo_write",
                {"todos": [{"content": "单测任务", "status": "completed"}]},
            )
        self.assertIn("[x] 单测任务", exec_res)
        self.assertTrue(todo_mgr.is_all_completed())

        # 通过 ToolManager 调用触发校验异常时，应返回可读错误提示而非崩溃
        err_res = tools.execute(
            "todo_write",
            {
                "todos": [
                    {"content": "A", "status": "in_progress"},
                    {"content": "B", "status": "in_progress"},
                ]
            },
        )
        self.assertIn("任务清单更新校验失败", err_res)

    def test_skill_loader(self):
        """测试技能加载器对 skills 目录的扫描与按需加载"""
        loader = SkillLoader(SKILLS_DIR)

        self.assertIn("code-review", loader.list_skills())
        catalog_text = loader.catalog()
        self.assertIn("code-review", catalog_text)

        content = loader.load("code-review")
        self.assertIn("代码审查技能规范", content)

        tools = ToolManager()
        register_skill_tools(tools, loader)
        self.assertTrue(tools.has("load_skill"))

        tool_res = tools.execute("load_skill", {"name": "code-review"})
        self.assertIn("代码审查技能规范", tool_res)


class TestTodoRuntimeIntegration(unittest.TestCase):
    """测试 todo_write 与连续未更新计划守卫接入 runtime.py 主循环"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()

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

    def test_runtime_todo_write_and_planning_guard_trigger(self):
        """连续 5 轮未调用 todo_write 应注入守卫提醒，调用 todo_write 后应清零计数"""
        (self.workdir / "app.py").write_text("x = 1\n", encoding="utf-8")
        todo_mgr = TodoManager()
        tools = runtime_mod.create_default_tools(workdir=self.workdir, todo_mgr=todo_mgr)
        messages = []
        planning_state = {"rounds_since_planning": 0}
        step = 0

        def scripted_llm(messages, tools):
            nonlocal step
            step += 1
            self.assertIn("todo_write", {t["function"]["name"] for t in tools})

            # 前 5 轮仅执行 read_file，不规划
            if step <= 5:
                return self._fake_msg(
                    tool_calls=[
                        {"id": f"tc_r_{step}", "name": "read_file", "arguments": {"path": "app.py"}}
                    ]
                )

            # 第 6 轮必须收到连续 5 轮未更新计划的系统守卫提示，且计数已清零
            if step == 6:
                self.assertEqual(messages[-1]["role"], "user")
                self.assertIn("[系统守卫提示]", messages[-1]["content"])
                self.assertEqual(planning_state["rounds_since_planning"], 0)
                return self._fake_msg(
                    tool_calls=[
                        {
                            "id": "tc_todo",
                            "name": "todo_write",
                            "arguments": {
                                "todos": [
                                    {"content": "阅读 app.py", "status": "completed"},
                                    {"content": "输出总结", "status": "in_progress"},
                                ]
                            },
                        }
                    ]
                )

            return self._fake_msg(content="已更新待办清单并完成处理。")

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "执行多步任务",
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=scripted_llm,
                workdir=self.workdir,
                todo_mgr=todo_mgr,
                planning_state=planning_state,
                verbose=False,
            )

        self.assertEqual(reply, "已更新待办清单并完成处理。")
        self.assertEqual(planning_state["rounds_since_planning"], 0)
        self.assertEqual(len(todo_mgr.get_items()), 2)

    def test_planning_state_persists_across_turns_and_demo_cli(self):
        """多次独立 run_agent_turn 共享 planning_state 时，累计满 5 轮也应触发守卫；Demo CLI 跑通"""
        (self.workdir / "a.txt").write_text("hello\n", encoding="utf-8")
        todo_mgr = TodoManager()
        tools = runtime_mod.create_default_tools(workdir=self.workdir, todo_mgr=todo_mgr)
        messages = []
        planning_state = {"rounds_since_planning": 0}

        def one_read_turn(messages, tools):
            last = messages[-1]
            if last["role"] == "user" and "[系统守卫提示]" not in str(last.get("content", "")):
                return self._fake_msg(
                    tool_calls=[
                        {"id": f"tc_{len(messages)}", "name": "read_file", "arguments": {"path": "a.txt"}}
                    ]
                )
            return self._fake_msg(content="读完")

        with redirect_stdout(io.StringIO()):
            for i in range(5):
                runtime_mod.run_agent_turn(
                    f"第{i + 1}次读取",
                    tools=tools,
                    hooks=HookManager(),
                    messages=messages,
                    llm_caller=one_read_turn,
                    workdir=self.workdir,
                    planning_state=planning_state,
                    verbose=False,
                )

        # 第 5 次完成后触发守卫并清零，且守卫消息已落在 messages 尾部
        self.assertEqual(planning_state["rounds_since_planning"], 0)
        self.assertTrue(any("[系统守卫提示]" in str(m.get("content", "")) for m in messages))

        demo_buf = io.StringIO()
        with redirect_stdout(demo_buf):
            code = tasks_cli_mod.main()
        self.assertEqual(code, 0)
        self.assertIn("[系统守卫提示]", demo_buf.getvalue())
        self.assertIn("[会话待办清单更新 / 任务清单更新]", demo_buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
