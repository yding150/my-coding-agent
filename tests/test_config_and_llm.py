"""
test_config_and_llm.py - M1.1 配置底座与 LLM 适配层单元测试
"""

import importlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import coding_agent.config as config_mod
import coding_agent.llm.client as llm_client_mod


class TestConfigPaths(unittest.TestCase):
    """测试 config.py 的 .agent/ 运行态路径收拢与环境变量读取"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runtime_paths_converged_under_dot_agent(self):
        """所有运行态目录必须统一收拢于 <workdir>/.agent/ 下，且构造时不提前落盘"""
        paths = config_mod.build_agent_paths(workdir=self.workdir)

        self.assertEqual(paths.workdir, self.workdir)
        self.assertEqual(paths.agent_dir, self.workdir / ".agent")
        self.assertEqual(paths.transcript_dir, self.workdir / ".agent" / "transcripts")
        self.assertEqual(paths.tool_results_dir, self.workdir / ".agent" / "tool-results")
        self.assertEqual(paths.memory_dir, self.workdir / ".agent" / "memory")
        self.assertEqual(paths.tasks_dir, self.workdir / ".agent" / "tasks")
        self.assertEqual(paths.mailbox_dir, self.workdir / ".agent" / "mailboxes")
        self.assertEqual(paths.workflow_runtime_dir, self.workdir / ".agent" / "workflows")

        # 构造路径对象不应产生磁盘目录副作用
        self.assertFalse(paths.agent_dir.exists())

        # 显式调用 ensure_runtime_dirs 后应全部就绪
        paths.ensure_runtime_dirs()
        self.assertTrue(paths.agent_dir.is_dir())
        for runtime_dir in paths.runtime_dirs():
            self.assertTrue(runtime_dir.is_dir())
            self.assertTrue(runtime_dir.is_relative_to(paths.agent_dir))

    def test_skills_dir_resolution_prefers_workdir_then_fallback(self):
        """工作区存在 skills/ 时优先使用，否则回退到项目内置 skills/ 目录"""
        fake_root = self.workdir / "repo_root"
        (fake_root / "skills").mkdir(parents=True)
        ws = self.workdir / "target_ws"
        ws.mkdir()

        fallback_paths = config_mod.build_agent_paths(workdir=ws, project_root=fake_root)
        self.assertEqual(fallback_paths.skills_dir, (fake_root / "skills").resolve())

        # 若同名 skills 是普通文件而非目录，也应回退到内置 skills/
        (ws / "skills").write_text("not a dir", encoding="utf-8")
        file_fallback_paths = config_mod.build_agent_paths(workdir=ws, project_root=fake_root)
        self.assertEqual(file_fallback_paths.skills_dir, (fake_root / "skills").resolve())
        (ws / "skills").unlink()

        (ws / "skills").mkdir()
        local_paths = config_mod.build_agent_paths(workdir=ws, project_root=fake_root)
        self.assertEqual(local_paths.skills_dir, (ws / "skills").resolve())

    def test_env_helpers_defaults_and_overrides(self):
        """测试 MAX_TURNS / GOAL_BLOCK_CAP / GOAL_EVALUATOR_MODEL_ID 读取与容错"""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config_mod.get_max_turns(), 25)
            self.assertEqual(config_mod.get_goal_block_cap(), 8)
            self.assertIsNone(config_mod.get_goal_evaluator_model())

        with patch.dict(
            os.environ,
            {
                "MAX_TURNS": "40",
                "GOAL_BLOCK_CAP": "12",
                "GOAL_EVALUATOR_MODEL_ID": "gpt-4o-mini",
            },
            clear=True,
        ):
            self.assertEqual(config_mod.get_max_turns(), 40)
            self.assertEqual(config_mod.get_goal_block_cap(), 12)
            self.assertEqual(config_mod.get_goal_evaluator_model(), "gpt-4o-mini")

        # 空串或非法数字安全回退默认值
        with patch.dict(
            os.environ,
            {"MAX_TURNS": "not-an-int", "GOAL_BLOCK_CAP": "-3"},
            clear=True,
        ):
            self.assertEqual(config_mod.get_max_turns(), 25)
            self.assertEqual(config_mod.get_goal_block_cap(), 8)

    def test_config_summary_masks_secret_and_cli_runs(self):
        """自检报告与 CLI 入口应脱敏展示 API_KEY 且输出 .agent/ 路径"""
        paths = config_mod.build_agent_paths(workdir=self.workdir)
        with patch.object(llm_client_mod, "API_KEY", "sk-secret-1234567890"):
            summary = config_mod.format_config_summary(paths)
            self.assertNotIn("sk-secret-1234567890", summary)
            self.assertIn("sk-***7890", summary)
            self.assertIn(".agent", summary)

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            exit_code = config_mod.main([])
        self.assertEqual(exit_code, 0)
        self.assertIn("Coding Agent 配置与运行态路径自检", stdout_buf.getvalue())

    def test_config_cli_ping_success_and_failure(self):
        """验证 python -m coding_agent.config --ping 成功与异常退出码"""
        stdout_buf = io.StringIO()
        with patch("coding_agent.llm.call_llm", return_value=SimpleNamespace(content="OK")):
            with redirect_stdout(stdout_buf):
                exit_code = config_mod.main(["--ping"])
        self.assertEqual(exit_code, 0)
        self.assertIn("成功！模型回复: 'OK'", stdout_buf.getvalue())

        stderr_buf = io.StringIO()
        with patch("coding_agent.llm.call_llm", side_effect=RuntimeError("network down")):
            with redirect_stdout(io.StringIO()), patch("sys.stderr", stderr_buf):
                fail_code = config_mod.main(["--ping"])
        self.assertEqual(fail_code, 1)
        self.assertIn("network down", stderr_buf.getvalue())


class TestLlmClient(unittest.TestCase):
    """测试 coding_agent.llm.client 的配置优先级、延迟校验与调用参数转发"""

    def tearDown(self):
        # 还原全局 client 状态，避免影响后续测试
        importlib.reload(llm_client_mod)

    def test_infer_default_model(self):
        """测试根据 base_url 自动推导默认模型名称"""
        self.assertEqual(llm_client_mod._infer_default_model("https://api.deepseek.com"), "deepseek-chat")
        self.assertEqual(llm_client_mod._infer_default_model("https://open.bigmodel.cn/api/paas/v4"), "glm-4")
        self.assertEqual(llm_client_mod._infer_default_model("https://api.openai.com/v1"), "gpt-4o")
        self.assertEqual(llm_client_mod._infer_default_model(None), "gpt-4o")

    def test_env_priority_and_no_import_crash_without_api_key(self):
        """无 API_KEY 时允许安全 import，但在真正调用时抛出 ValueError；有配置时按优先级读取"""
        with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, {}, clear=True):
            reloaded = importlib.reload(llm_client_mod)
            self.assertIsNone(reloaded.API_KEY)
            self.assertIsNone(reloaded.client)
            self.assertEqual(reloaded.DEFAULT_MODEL, "gpt-4o")
            with self.assertRaises(ValueError):
                reloaded.call_llm([{"role": "user", "content": "hi"}])

        env_vars = {
            "LLM_API_KEY": "llm-key",
            "DEEPSEEK_API_KEY": "ds-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "ZHIPU_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
            "LLM_MODEL_ID": "custom-model-v1",
        }
        with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, env_vars, clear=True):
            with patch("openai.OpenAI") as mock_openai_cls:
                reloaded = importlib.reload(llm_client_mod)
                self.assertEqual(reloaded.API_KEY, "llm-key")
                self.assertEqual(reloaded.BASE_URL, "https://api.deepseek.com")
                self.assertEqual(reloaded.DEFAULT_MODEL, "custom-model-v1")
                mock_openai_cls.assert_called_once_with(
                    api_key="llm-key",
                    base_url="https://api.deepseek.com",
                )

    def test_call_llm_and_call_llm_response_forwarding(self):
        """验证 call_llm / call_llm_response 正确转发参数，且空 tools 列表不传递"""
        fake_msg = SimpleNamespace(content="OK", tool_calls=None)
        fake_resp = SimpleNamespace(id="resp_001", choices=[SimpleNamespace(message=fake_msg)])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_resp

        messages = [{"role": "user", "content": "hello"}]
        tools_schema = [{"type": "function", "function": {"name": "read_file"}}]

        with patch.object(llm_client_mod, "client", mock_client):
            # 1. 不传 tools 或传空列表 []，不应向 SDK 传递 tools 字段
            msg = llm_client_mod.call_llm(messages=messages, tools=[], temperature=0.2)
            self.assertIs(msg, fake_msg)
            _, kwargs1 = mock_client.chat.completions.create.call_args
            self.assertEqual(kwargs1["model"], llm_client_mod.DEFAULT_MODEL)
            self.assertEqual(kwargs1["messages"], messages)
            self.assertEqual(kwargs1["temperature"], 0.2)
            self.assertNotIn("tools", kwargs1)

            # 2. 传入非空 tools 与显式 model 时，应完整透传
            resp = llm_client_mod.call_llm_response(
                messages=messages,
                tools=tools_schema,
                model="deepseek-reasoner",
            )
            self.assertIs(resp, fake_resp)
            _, kwargs2 = mock_client.chat.completions.create.call_args
            self.assertEqual(kwargs2["model"], "deepseek-reasoner")
            self.assertEqual(kwargs2["tools"], tools_schema)

    def test_client_main_prints_error_and_handles_none_content(self):
        """验证 llm.client.main() 异常提示详情输出与 None content 安全兜底"""
        stdout_buf = io.StringIO()
        with (
            patch("builtins.input", side_effect=["你好", "再见", "exit"]),
            patch.object(
                llm_client_mod,
                "call_llm",
                side_effect=[RuntimeError("bad gateway"), SimpleNamespace(content=None)],
            ),
            redirect_stdout(stdout_buf),
        ):
            llm_client_mod.main()

        output = stdout_buf.getvalue()
        self.assertIn("调用失败：bad gateway", output)
        self.assertIn("助手回答：", output)
        self.assertIn("已退出对话。", output)


if __name__ == "__main__":
    unittest.main()

