"""
test_subagent_system.py - M4.1 一次性隔离子代理 (`coding_agent.subagent`) 与主循环集成单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import coding_agent.runtime as runtime_mod
from coding_agent.hooks import HookManager
from coding_agent.subagent import (
    FORBIDDEN_SUBAGENT_TOOLS,
    SubagentRunner,
    register_subagent_tools,
)
from coding_agent.subagent.__main__ import main as subagent_demo_main
from coding_agent.tools import ToolManager, register_core_tools


def _ns_msg(content=None, tool_calls=None):
    """构造轻量 SimpleNamespace 模拟大模型返回消息。"""
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


def _mock_tc(tc_id: str, name: str, arguments: dict) -> MagicMock:
    """构造未显式设置 `.type` 的 MagicMock 工具调用对象，验证序列化兼容性。"""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments, ensure_ascii=False)
    return tc


class TestSubagentRunner(unittest.TestCase):
    """验证 SubagentRunner 隔离边界、抗递归剥离、工具执行与异常兜底。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.shared_tools = register_core_tools(ToolManager(), workdir=self.workdir)

        # 刻意向传入容器额外挂载 4 个禁止工具，验证 SubagentRunner 克隆后彻底剥离且不污染原容器
        for forbidden in FORBIDDEN_SUBAGENT_TOOLS:

            @self.shared_tools.register(
                name=forbidden,
                description=f"mock forbidden tool {forbidden}",
                parameters={"type": "object", "properties": {}},
            )
            def _forbidden_stub() -> str:
                return "should not run"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_anti_recursion_strips_forbidden_tools_without_mutating_source(self):
        """子代理必须自动剥离 task/spawn_teammate/compact/connect_mcp，且不能影响传入的 shared_tools。"""
        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir)

        sub_tool_names = set(runner.tools.list_tools())
        src_tool_names = set(self.shared_tools.list_tools())

        for forbidden in FORBIDDEN_SUBAGENT_TOOLS:
            self.assertNotIn(forbidden, sub_tool_names)
            self.assertIn(forbidden, src_tool_names)
        self.assertIn("read_file", sub_tool_names)
        self.assertIn("run_shell", sub_tool_names)

    def test_direct_text_answer_includes_workdir_and_description(self):
        """无工具调用时应把工作区与子任务描述注入系统提示词并直接返回文本摘要。"""
        observed_calls = []

        def fake_llm(messages, tools):
            observed_calls.append((messages, tools))
            return _ns_msg(content="核心结论：项目包含 src 与 tests 目录。")

        runner = SubagentRunner(
            shared_tools=self.shared_tools,
            workdir=self.workdir,
            llm_caller=fake_llm,
        )
        with redirect_stdout(io.StringIO()):
            result = runner.run(prompt="列出目录结构", description="目录摸底")

        self.assertEqual(result, "核心结论：项目包含 src 与 tests 目录。")
        self.assertEqual(len(observed_calls), 1)
        sent_messages, sent_schemas = observed_calls[0]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertIn(str(self.workdir), sent_messages[0]["content"])
        self.assertIn("目录摸底", sent_messages[0]["content"])
        self.assertEqual(sent_messages[1], {"role": "user", "content": "列出目录结构"})

        schema_names = {item["function"]["name"] for item in sent_schemas}
        for forbidden in FORBIDDEN_SUBAGENT_TOOLS:
            self.assertNotIn(forbidden, schema_names)

    @patch("coding_agent.subagent.runner.call_llm")
    def test_multi_turn_tool_execution_with_magic_mock_and_patched_call_llm(self, mock_call_llm):
        """未传自定义 llm_caller 时使用默认 call_llm；MagicMock 响应也能完成多轮工具调用闭环。"""
        (self.workdir / "alpha.txt").write_text("SECRET_TOKEN_42\n", encoding="utf-8")

        resp_turn_1 = MagicMock()
        resp_turn_1.tool_calls = [_mock_tc("sub_tc_1", "read_file", {"path": "alpha.txt"})]

        resp_turn_2 = MagicMock()
        resp_turn_2.tool_calls = None
        resp_turn_2.content = "已完成读取，秘钥为 SECRET_TOKEN_42。"

        mock_call_llm.side_effect = [resp_turn_1, resp_turn_2]

        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir)
        with redirect_stdout(io.StringIO()):
            result = runner.run(prompt="读取 alpha.txt 提取关键口令", description="提取口令")

        self.assertEqual(result, "已完成读取，秘钥为 SECRET_TOKEN_42。")
        self.assertEqual(mock_call_llm.call_count, 2)

        # 验证第 2 轮喂给大模型的 sub_messages 已严格回填 read_file 执行输出
        second_call_msgs = mock_call_llm.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_call_msgs if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "sub_tc_1")
        self.assertIn("SECRET_TOKEN_42", tool_msgs[0]["content"])

    def test_malformed_json_arguments_returns_error_message_to_model(self):
        """模型生成非法 JSON 参数时不应崩溃，应将报错作为 tool 消息反馈给模型重试。"""
        turn = 0

        def flaky_llm(messages, tools):
            nonlocal turn
            turn += 1
            if turn == 1:
                return _ns_msg(
                    tool_calls=[
                        {"id": "bad_1", "name": "read_file", "arguments": "{bad json"}
                    ]
                )
            tool_reply = messages[-1]["content"]
            return _ns_msg(content=f"已收到错误提示并结束：{tool_reply}")

        runner = SubagentRunner(
            shared_tools=self.shared_tools,
            workdir=self.workdir,
            llm_caller=flaky_llm,
        )
        with redirect_stdout(io.StringIO()):
            result = runner.run(prompt="测试非法 JSON 处理")

        self.assertIn("JSON 解析失败", result)

    def test_max_turns_limit_returns_graceful_notice(self):
        """子代理一直调用工具不收敛时，应在达到 max_turns 后安全返回提示文本。"""

        def endless_tool_llm(messages, tools):
            return _ns_msg(
                tool_calls=[
                    {"id": "loop_1", "name": "glob_files", "arguments": {"pattern": "*"}}
                ]
            )

        runner = SubagentRunner(
            shared_tools=self.shared_tools,
            workdir=self.workdir,
            max_turns=2,
            llm_caller=endless_tool_llm,
        )
        with redirect_stdout(io.StringIO()):
            result = runner.run(prompt="无限循环测试")

        self.assertIn("2 轮", result)
        self.assertIn("上限", result)

    def test_llm_exception_is_caught_and_returned_as_string(self):
        """子代理内部调用大模型抛出异常时，必须转化为错误信息返回主循环，不可炸毁主会话。"""

        def broken_llm(messages, tools):
            raise RuntimeError("mock upstream timeout")

        runner = SubagentRunner(
            shared_tools=self.shared_tools,
            workdir=self.workdir,
            llm_caller=broken_llm,
        )
        with redirect_stdout(io.StringIO()):
            result = runner.run(prompt="触发网络异常")

        self.assertIn("RuntimeError", result)
        self.assertIn("mock upstream timeout", result)

    def test_register_subagent_tools_exposes_task_tool(self):
        """register_subagent_tools 应向目标 ToolManager 挂载 `task` 委派工具并正确转发入参。"""
        seen = {}

        def fake_llm(messages, tools):
            seen["prompt"] = messages[1]["content"]
            seen["system"] = messages[0]["content"]
            return _ns_msg(content="task 委派成功")

        runner = SubagentRunner(shared_tools=self.shared_tools, workdir=self.workdir, llm_caller=fake_llm)
        parent = ToolManager()
        register_subagent_tools(parent, runner)

        self.assertEqual(parent.list_tools(), ["task"])
        with redirect_stdout(io.StringIO()):
            out = parent.execute("task", {"prompt": "分析模块", "description": "结构检查"})
        self.assertEqual(out, "task 委派成功")
        self.assertEqual(seen["prompt"], "分析模块")
        self.assertIn("结构检查", seen["system"])


class TestSubagentRuntimeIntegration(unittest.TestCase):
    """验证 M4.1 工具分层、`PLANNING_TOOLS` 扩展以及 `runtime.run_agent_turn` 端到端集成。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_shared_tools_and_parent_tools_split(self):
        """shared_tools 不应包含 task/compact；parent_tools 与 default_tools 必须同时包含两者。"""
        shared_names = set(runtime_mod.shared_tools.list_tools())
        parent_names = set(runtime_mod.parent_tools.list_tools())
        default_names = set(runtime_mod.default_tools.list_tools())

        self.assertEqual(parent_names, default_names)
        self.assertNotIn("task", shared_names)
        self.assertNotIn("compact", shared_names)
        self.assertIn("read_file", shared_names)
        self.assertIn("load_skill", shared_names)
        self.assertIn("todo_write", shared_names)

        self.assertIn("task", parent_names)
        self.assertIn("compact", parent_names)
        self.assertTrue(shared_names.issubset(parent_names))
        self.assertNotIn("task", runtime_mod.default_subagent_runner.tools.list_tools())
        self.assertNotIn("compact", runtime_mod.default_subagent_runner.tools.list_tools())

    def test_planning_tools_includes_task_and_resets_planning_counter(self):
        """PLANNING_TOOLS 必须包含 `task`，主代理委派子代理后应清零连续未规划轮数。"""
        self.assertIn("task", runtime_mod.PLANNING_TOOLS)
        self.assertIn("todo_write", runtime_mod.PLANNING_TOOLS)

        shared = runtime_mod.create_shared_tools(workdir=self.workdir)
        runner = SubagentRunner(
            shared_tools=shared,
            workdir=self.workdir,
            llm_caller=lambda messages, tools: _ns_msg(content="子代理已完成扫描"),
        )
        parent = runtime_mod.create_default_tools(workdir=self.workdir, subagent_runner=runner)

        turn = 0

        def lead_llm(messages, tools):
            nonlocal turn
            turn += 1
            if turn == 1:
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "call_task_1",
                            "name": "task",
                            "arguments": {"prompt": "扫描项目", "description": "扫描测试"},
                        }
                    ]
                )
            return _ns_msg(content="已获取扫描结果。")

        planning_state = {"rounds_since_planning": 4}
        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "请帮我调研",
                tools=parent,
                hooks=HookManager(),
                messages=[],
                llm_caller=lead_llm,
                workdir=self.workdir,
                planning_state=planning_state,
                verbose=False,
            )

        self.assertEqual(reply, "已获取扫描结果。")
        self.assertEqual(planning_state["rounds_since_planning"], 0)

    def test_runtime_delegates_exploration_to_isolated_subagent(self):
        """主循环经 `task` 委派多步文件阅读时，中间工具噪声不得混入父会话 `messages`。"""
        (self.workdir / "notes.md").write_text("NOISY_LINE_1\nTARGET_FACT_99\n", encoding="utf-8")

        shared = runtime_mod.create_shared_tools(workdir=self.workdir)
        sub_turn = 0
        sub_observed_schemas = []

        def sub_llm(messages, tools):
            nonlocal sub_turn
            sub_turn += 1
            sub_observed_schemas.append({t["function"]["name"] for t in tools})
            if sub_turn == 1:
                return _ns_msg(
                    tool_calls=[
                        {"id": "sub_read_1", "name": "read_file", "arguments": {"path": "notes.md"}}
                    ]
                )
            self.assertIn("TARGET_FACT_99", messages[-1]["content"])
            return _ns_msg(content="精炼摘要：关键事实为 TARGET_FACT_99。")

        runner = SubagentRunner(shared_tools=shared, workdir=self.workdir, llm_caller=sub_llm)
        parent = runtime_mod.create_default_tools(workdir=self.workdir, subagent_runner=runner)

        lead_turn = 0

        def lead_llm(messages, tools):
            nonlocal lead_turn
            lead_turn += 1
            if lead_turn == 1:
                return _ns_msg(
                    tool_calls=[
                        {
                            "id": "lead_task_1",
                            "name": "task",
                            "arguments": {
                                "description": "调研 notes.md",
                                "prompt": "读取 notes.md 并总结关键事实",
                            },
                        }
                    ]
                )
            return _ns_msg(content=f"最终汇报 -> {messages[-1]['content']}")

        parent_messages = runtime_mod.create_session_messages(workdir=self.workdir)
        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "帮我查一下 notes.md 的关键事实",
                tools=parent,
                hooks=HookManager(),
                messages=parent_messages,
                llm_caller=lead_llm,
                workdir=self.workdir,
                verbose=False,
            )

        self.assertEqual(reply, "最终汇报 -> 精炼摘要：关键事实为 TARGET_FACT_99。")
        # 父会话应恰好只有: system + user + assistant(task) + tool(summary) + assistant(final) = 5 条
        self.assertEqual(len(parent_messages), 5)
        serialized_parent = json.dumps(parent_messages, ensure_ascii=False)
        self.assertNotIn("NOISY_LINE_1", serialized_parent)
        self.assertNotIn("sub_read_1", serialized_parent)
        self.assertIn("精炼摘要：关键事实为 TARGET_FACT_99。", serialized_parent)
        # 确保子代理在模型视角中也完全看不见 task/compact
        self.assertNotIn("task", sub_observed_schemas[0])
        self.assertNotIn("compact", sub_observed_schemas[0])

    def test_subagent_demo_main_runs_cleanly(self):
        """M4.1 CLI 离线演示入口 `python -m coding_agent.subagent` 应正常执行并返回 0。"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = subagent_demo_main()

        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("[Subagent 启动]", output)
        self.assertIn("[Subagent 完成]", output)
        self.assertIn("父会话最终消息条数: 5 条", output)


if __name__ == "__main__":
    unittest.main()
