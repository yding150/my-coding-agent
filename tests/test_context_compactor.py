"""
test_context_compactor.py - M3.1 ContextCompactor 四步压缩管线与主循环集成单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

import coding_agent.context.__main__ as context_cli_mod
import coding_agent.runtime as runtime_mod
from coding_agent.config import build_agent_paths
from coding_agent.context import ContextCompactor
from coding_agent.hooks import HookManager


class TestContextCompactor(unittest.TestCase):
    """测试公共层 ContextCompactor 四步分层压缩"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        paths = build_agent_paths(workdir=Path(self.temp_dir.name))
        self.paths = paths
        self.compactor = ContextCompactor(
            transcript_dir=paths.transcript_dir,
            tool_results_dir=paths.tool_results_dir,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_layer1_tool_result_budget(self):
        """测试 Layer 1：超大工具结果转存至 .agent/tool-results 并保留 2000 字符预览"""
        huge_text = "A" * 25000
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user request"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_shell"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": huge_text},
        ]
        res = self.compactor.tool_result_budget(messages)
        content = res[-1]["content"]
        self.assertTrue(content.startswith("<persisted-output>"))
        self.assertIn("完整输出已落盘保存至", content)
        self.assertLess(len(content), 3000)
        self.assertTrue((self.paths.tool_results_dir / "c1.txt").is_file())

    def test_layer2_snip_compact_and_pairing(self):
        """测试 Layer 2：滑动窗口剪裁与 OpenAI tool_calls 配对保护"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial user"},
            {"role": "assistant", "content": "initial reply"},
        ]
        for i in range(2, 26):
            messages.append(
                {
                    "role": "assistant",
                    "content": f"turn {i}",
                    "tool_calls": [{"id": f"c_{i}", "type": "function", "function": {"name": "read"}}],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"c_{i}", "content": f"result {i}"})

        self.assertGreaterEqual(len(messages), 50)
        with redirect_stdout(io.StringIO()):
            res = self.compactor.snip_compact(messages, trigger=50, target=30)
        self.assertLessEqual(len(res), 32)
        self.assertIn("系统提示：中间", res[3]["content"])
        self.assertGreaterEqual(len(list(self.paths.transcript_dir.glob("transcript_*.jsonl"))), 1)

    def test_layer3_micro_compact(self):
        """测试 Layer 3：微紧凑将已读旧工具输出置换为路径"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "req"},
        ]
        for i in range(10):
            messages.append({"role": "assistant", "content": f"msg {i}"})
            messages.append({"role": "tool", "tool_call_id": f"c_{i}", "content": "X" * 300})

        res = self.compactor.micro_compact(messages, target_chars=1000)
        earlier_tools = [
            m for m in res if m.get("role") == "tool" and "[Earlier tool result" in str(m.get("content"))
        ]
        self.assertGreater(len(earlier_tools), 0)

    def test_layer4_compact_history_and_prepare_escalation(self):
        """测试 Layer 4：compact_history 事实性摘要重构与 prepare 自动升级触发"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "重构认证模块"},
            {"role": "assistant", "content": "收到"},
            {"role": "tool", "tool_call_id": "t_last", "content": "Z" * 500},
        ]
        fake_summary = SimpleNamespace(content="已确定目标为重构认证模块，关键文件 auth.py。")

        with (
            patch("coding_agent.context.compactor.call_llm", return_value=fake_summary),
            patch.object(ContextCompactor, "CONTEXT_CHAR_LIMIT", 200),
            redirect_stdout(io.StringIO()),
        ):
            rebuilt = self.compactor.prepare(messages, active_request="重构认证模块")

        self.assertEqual(len(rebuilt), 2)
        self.assertEqual(rebuilt[0]["role"], "system")
        self.assertIn("[Compacted 历史压缩通知]", rebuilt[1]["content"])
        self.assertIn("重构认证模块", rebuilt[1]["content"])
        self.assertGreaterEqual(len(list(self.paths.transcript_dir.glob("transcript_*.jsonl"))), 1)

    def test_mixed_chat_completion_message_objects(self):
        """测试混入 OpenAI ChatCompletionMessage 对象时全管线兼容性"""

        def make_assistant_msg(call_id: str) -> ChatCompletionMessage:
            return ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id=call_id,
                        type="function",
                        function=Function(name="run_shell", arguments='{"command": "dir"}'),
                    )
                ],
            )

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "帮我查看目录"},
        ]
        for i in range(6):
            messages.append(make_assistant_msg(f"call_{i}"))
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "Y" * 300})

        prepared = self.compactor.prepare(messages, active_request="帮我查看目录")
        self.assertEqual(len(prepared), len(messages))

        micro = self.compactor.micro_compact(prepared, target_chars=500)
        earlier_tools = [
            m
            for m in micro
            if isinstance(m, dict) and m.get("role") == "tool" and "[Earlier tool result" in str(m.get("content"))
        ]
        self.assertGreater(len(earlier_tools), 0)

        long_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "start"},
            ChatCompletionMessage(role="assistant", content="ok"),
        ]
        for i in range(25):
            long_messages.append(make_assistant_msg(f"s_{i}"))
            long_messages.append({"role": "tool", "tool_call_id": f"s_{i}", "content": f"res {i}"})

        with redirect_stdout(io.StringIO()):
            snipped = self.compactor.snip_compact(long_messages, trigger=50, target=30)
            self.assertLessEqual(len(snipped), 32)
            self.assertIn("系统提示：中间", snipped[3]["content"])

            rebuilt = self.compactor.reactive_compact(long_messages, active_request="start")
            self.assertEqual(rebuilt[0]["role"], "system")
            self.assertNotEqual(self.compactor._msg_get(rebuilt[2], "role"), "tool")


class TestContextRuntimeIntegration(unittest.TestCase):
    """测试 ContextCompactor 与 compact 工具接入 runtime.py 主循环"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = build_agent_paths(workdir=Path(self.temp_dir.name))
        self.compactor = ContextCompactor(
            transcript_dir=self.paths.transcript_dir,
            tool_results_dir=self.paths.tool_results_dir,
        )
        self.tools = runtime_mod.create_default_tools(workdir=self.paths.workdir)
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

    def test_runtime_compact_tool_archives_to_dot_agent_transcripts(self):
        """主循环主动调用 compact 工具后，应在 .agent/transcripts/ 生成归档并裁剪上下文"""
        step = 0

        def scripted_llm(messages, tools):
            nonlocal step
            step += 1
            self.assertIn("compact", {t["function"]["name"] for t in tools})
            if step == 1:
                return self._fake_msg(
                    tool_calls=[{"id": "tc_cmp", "name": "compact", "arguments": {}}]
                )
            self.assertTrue(
                any("[Emergency Compact:" in str(m.get("content", "")) for m in messages)
            )
            return self._fake_msg(content="已完成主动压缩归档。")

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "压缩会话历史",
                tools=self.tools,
                hooks=HookManager(),
                messages=self.messages,
                llm_caller=scripted_llm,
                workdir=self.paths.workdir,
                compactor=self.compactor,
                verbose=False,
            )

        self.assertEqual(reply, "已完成主动压缩归档。")
        transcripts = list(self.paths.transcript_dir.glob("transcript_*.jsonl"))
        self.assertEqual(len(transcripts), 1)
        self.assertTrue(transcripts[0].is_relative_to(self.paths.agent_dir))

    def test_runtime_reactive_compact_recovers_from_prompt_too_long(self):
        """主循环遇到 prompt_too_long / context_length_exceeded 时应应急压缩并自动重试"""
        calls = 0

        def flaky_llm(messages, tools):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Error code: 400 - context_length_exceeded: too many tokens")
            self.assertTrue(
                any("[Emergency Compact:" in str(m.get("content", "")) for m in messages)
            )
            return self._fake_msg(content="压缩自愈后恢复正常回复。")

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "继续分析超长日志",
                tools=self.tools,
                hooks=HookManager(),
                messages=self.messages,
                llm_caller=flaky_llm,
                workdir=self.paths.workdir,
                compactor=self.compactor,
                verbose=False,
            )

        self.assertEqual(calls, 2)
        self.assertEqual(reply, "压缩自愈后恢复正常回复。")
        self.assertEqual(len(list(self.paths.transcript_dir.glob("transcript_*.jsonl"))), 1)

    def test_context_demo_cli_runs_cleanly(self):
        """uv run python -m coding_agent.context 演示入口应完整跑通四步压缩与归档"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = context_cli_mod.main()
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Layer 1 tool_result_budget", out)
        self.assertIn("Layer 4 compact_history", out)
        self.assertIn(".agent/transcripts/", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
