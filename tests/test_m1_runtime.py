"""
test_m1_runtime.py - M1.3 最小单 Agent 主循环与 CLI 入口单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coding_agent.cli as cli_mod
import coding_agent.runtime as runtime_mod


def _fake_msg(content=None, tool_calls=None):
    """构造测试用模拟模型返回消息。"""
    tc_list = []
    for tc in tool_calls or []:
        tc_list.append(
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc["arguments"]
                    if isinstance(tc["arguments"], str)
                    else json.dumps(tc["arguments"], ensure_ascii=False),
                ),
            )
        )
    return SimpleNamespace(content=content, tool_calls=tc_list)


class TestM1RuntimeLoop(unittest.TestCase):
    """测试 runtime.py 的核心 while True 单 Agent 工具循环"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.tools = runtime_mod.create_default_tools(workdir=self.workdir)
        self.messages = []

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_direct_answer_initializes_system_and_returns_text(self):
        """无工具调用时，应自动初始化 system 消息并直接返回模型回答"""

        def fake_llm(messages, tools):
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[-1], {"role": "user", "content": "你好"})
            self.assertEqual(len(tools), 5)
            return _fake_msg(content="你好，我可以帮你读写代码。")

        reply = runtime_mod.run_agent_turn(
            "你好",
            tools=self.tools,
            messages=self.messages,
            llm_caller=fake_llm,
            workdir=self.workdir,
            verbose=False,
        )
        self.assertEqual(reply, "你好，我可以帮你读写代码。")
        self.assertEqual([m["role"] for m in self.messages], ["system", "user", "assistant"])

    def test_multi_turn_tool_calling_loop_creates_and_reads_file(self):
        """测试多轮工具调用 -> tool 结果回注 -> 最终文本回复闭环"""
        (self.workdir / "existing.py").write_text("print('ok')\n", encoding="utf-8")
        observed_tool_msgs = []
        turn_idx = 0

        def scripted_llm(messages, tools):
            nonlocal turn_idx
            turn_idx += 1
            if turn_idx == 1:
                return _fake_msg(
                    content="先查看目录文件",
                    tool_calls=[
                        {"id": "tc_1", "name": "glob_files", "arguments": {"pattern": "**/*"}}
                    ],
                )
            if turn_idx == 2:
                # 第 2 轮应能看到第 1 轮 glob_files 的真实输出
                last_msg = messages[-1]
                self.assertEqual(last_msg["role"], "tool")
                self.assertEqual(last_msg["tool_call_id"], "tc_1")
                self.assertIn("existing.py", last_msg["content"])
                observed_tool_msgs.append(last_msg["content"])
                return _fake_msg(
                    content=None,
                    tool_calls=[
                        {
                            "id": "tc_2",
                            "name": "write_file",
                            "arguments": {"path": "notes/test.txt", "content": "M1.3 OK"},
                        },
                        {
                            "id": "tc_3",
                            "name": "read_file",
                            "arguments": {"path": "notes/test.txt"},
                        },
                    ],
                )

            # 第 3 轮应看到 write_file 与 read_file 的真实工具输出
            self.assertEqual(messages[-2]["tool_call_id"], "tc_2")
            self.assertEqual(messages[-1]["tool_call_id"], "tc_3")
            self.assertEqual(messages[-1]["content"], "M1.3 OK")
            return _fake_msg(content="已列出文件并创建了 notes/test.txt。")

        reply = runtime_mod.run_agent_turn(
            "列出当前目录文件并创建一个测试文件",
            tools=self.tools,
            messages=self.messages,
            llm_caller=scripted_llm,
            workdir=self.workdir,
            verbose=False,
        )

        self.assertEqual(reply, "已列出文件并创建了 notes/test.txt。")
        self.assertEqual(turn_idx, 3)
        self.assertTrue((self.workdir / "notes" / "test.txt").is_file())
        self.assertEqual((self.workdir / "notes" / "test.txt").read_text(encoding="utf-8"), "M1.3 OK")
        self.assertEqual(len(observed_tool_msgs), 1)

    def test_invalid_json_arguments_still_emits_paired_tool_message(self):
        """工具参数 JSON 畸形时应回填错误 tool 消息（保持协议配对）并让模型自愈重试"""
        turn_idx = 0

        def scripted_llm(messages, tools):
            nonlocal turn_idx
            turn_idx += 1
            if turn_idx == 1:
                return _fake_msg(
                    content=None,
                    tool_calls=[{"id": "tc_bad", "name": "read_file", "arguments": "{bad json"}],
                )
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(messages[-1]["tool_call_id"], "tc_bad")
            self.assertIn("JSON 解析失败", messages[-1]["content"])
            return _fake_msg(content="抱歉，参数格式有误，已停止。")

        reply = runtime_mod.run_agent_turn(
            "读取不存在的文件",
            tools=self.tools,
            messages=self.messages,
            llm_caller=scripted_llm,
            workdir=self.workdir,
            verbose=False,
        )
        self.assertIn("已停止", reply)
        self.assertEqual(turn_idx, 2)

    def test_max_turns_guard_stops_infinite_loop(self):
        """达到 max_turns 上限时必须熔断退出，防止死循环"""
        call_count = 0

        def endless_tool_llm(messages, tools):
            nonlocal call_count
            call_count += 1
            return _fake_msg(
                content=None,
                tool_calls=[{"id": f"tc_{call_count}", "name": "glob_files", "arguments": {"pattern": "*"}}],
            )

        reply = runtime_mod.run_agent_turn(
            "一直搜索",
            tools=self.tools,
            messages=self.messages,
            llm_caller=endless_tool_llm,
            workdir=self.workdir,
            max_turns=2,
            verbose=False,
        )
        self.assertEqual(call_count, 2)
        self.assertIn("MAX_TURNS=2", reply)


class TestM1Cli(unittest.TestCase):
    """测试 cli.py 的 demo、单次指令、--help 与交互式 REPL 异常处理"""

    def test_cli_demo_mode(self):
        """uv run cagent demo 应完整跑通离线 3 轮工具闭环演示"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_mod.main(["demo"])
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("M1.3 离线 Demo 模式", out)
        self.assertIn("hello_agent.txt", out)
        self.assertIn("Demo 演示结束", out)

    def test_cli_single_prompt_success_and_error(self):
        """测试单次指令正常执行与异常处理退出码"""
        with patch.object(cli_mod, "run_agent_turn", return_value="完成") as mock_turn:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli_mod.main(["列出", "文件"])
            self.assertEqual(code, 0)
            mock_turn.assert_called_once_with("列出 文件")

        with patch.object(cli_mod, "run_agent_turn", side_effect=ValueError("缺少 API Key")):
            err_buf = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err_buf):
                code = cli_mod.main(["测试指令"])
            self.assertEqual(code, 1)
            self.assertIn("缺少 API Key", err_buf.getvalue())

    def test_cli_repl_rolls_back_on_error_and_exits(self):
        """交互式 REPL 中单轮调用失败应回滚会话上下文，输入 exit 正常退出"""
        original_messages = list(cli_mod.session_messages)
        cli_mod.session_messages.clear()
        cli_mod.session_messages.append({"role": "system", "content": "sys"})

        def failing_turn(prompt):
            cli_mod.session_messages.append({"role": "user", "content": prompt})
            raise RuntimeError("模拟网络超时")

        try:
            with (
                patch("builtins.input", side_effect=["第一句", "exit"]),
                patch.object(cli_mod, "run_agent_turn", side_effect=failing_turn),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                code = cli_mod.main([])
            self.assertEqual(code, 0)
            # 失败轮次的 user 消息应已回滚
            self.assertEqual(cli_mod.session_messages, [{"role": "system", "content": "sys"}])
        finally:
            cli_mod.session_messages[:] = original_messages


if __name__ == "__main__":
    unittest.main(verbosity=2)
