"""
test_memory_system.py - M3.2 MemoryManager 跨会话持久记忆与主循环集成单元测试
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import coding_agent.memory.__main__ as memory_cli_mod
import coding_agent.runtime as runtime_mod
from coding_agent.config import build_agent_paths
from coding_agent.hooks import HookManager
from coding_agent.memory import MemoryManager, memory_slug


class TestMemoryManager(unittest.TestCase):
    """测试公共层 MemoryManager 的存储、召回、提取与整理能力"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = build_agent_paths(workdir=Path(self.temp_dir.name))
        self.workdir = self.paths.workdir
        self.memory_dir = self.paths.memory_dir

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_storage_index_and_path_safety(self):
        """测试记忆文件写盘（.agent/memory）、MEMORY.md 索引重建及路径逃逸拦截"""
        mgr = MemoryManager(self.memory_dir, workdir=self.workdir)
        path = mgr.write_memory_file(
            name="indent-preference",
            mem_type="user",
            description="用户偏好使用 Tab 缩进",
            body="所有代码缩进优先使用 Tab，而不是空格。",
        )
        self.assertTrue(path.exists())
        self.assertTrue(path.is_relative_to(self.paths.agent_dir))
        index_text = mgr.read_memory_index()
        self.assertIn("[indent-preference](indent-preference.md)", index_text)
        self.assertIn("用户偏好使用 Tab 缩进", index_text)

        # 验证路径穿越与索引覆写被拒绝
        with self.assertRaises(ValueError):
            mgr.memory_path("../escape.md")
        with self.assertRaises(ValueError):
            mgr.memory_path("MEMORY.md")

    def test_recall_selection_and_keyword_fallback(self):
        """测试模型召回选择及模型失败时的中英文关键词兜底匹配"""

        def ok_llm(messages, **kwargs):
            return SimpleNamespace(role="assistant", content="[1]")

        mgr = MemoryManager(self.memory_dir, workdir=self.workdir, llm_caller=ok_llm)
        mgr.write_memory_file(
            "db-testing", "feedback", "不要 mock 数据库测试", "集成测试连接真实测试库"
        )
        mgr.write_memory_file(
            "tab-indent", "user", "代码缩进偏好 Tab", "用户偏好用 Tab 进行代码缩进"
        )

        msgs = [
            {"role": "user", "content": "<reminder>忽略此系统提醒</reminder>"},
            {"role": "user", "content": "请问我偏好什么缩进风格？"},
        ]
        selected = mgr.select_relevant_memories(msgs)
        self.assertEqual(selected, ["tab-indent.md"])

        recalled_json = mgr.load_memories(msgs)
        loaded = json.loads(recalled_json)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["source"], "tab-indent.md")

        # 模型调用抛错时自动退化为关键词二元切片匹配
        def fail_llm(messages, **kwargs):
            raise RuntimeError("模拟网络超时")

        fallback_mgr = MemoryManager(
            self.memory_dir, workdir=self.workdir, llm_caller=fail_llm
        )
        fallback_selected = fallback_mgr.select_relevant_memories(
            [{"role": "user", "content": "缩进风格设置"}]
        )
        self.assertEqual(fallback_selected, ["tab-indent.md"])

    def test_extract_durability_gate_and_dedup(self):
        """测试回合后提取的持久性门禁、临时标志词拦截与重复去重"""
        payload = [
            {
                "name": "use-tabs",
                "type": "user",
                "scope": "persistent",
                "description": "使用 Tab 缩进",
                "body": "用户要求长期使用 Tab 缩进。",
            },
            {
                "name": "no-files-now",
                "type": "feedback",
                "scope": "current_task",
                "description": "当前任务禁止建文件",
                "body": "不要创建新文件。",
            },
            {
                "name": "temp-rule-disguised",
                "type": "feedback",
                "scope": "persistent",
                "description": "本次会话跳过单测",
                "body": "本次会话暂时跳过耗时测试。",
            },
        ]

        def mock_extract_llm(messages, **kwargs):
            return SimpleNamespace(
                role="assistant", content=json.dumps(payload, ensure_ascii=False)
            )

        mgr = MemoryManager(
            self.memory_dir, workdir=self.workdir, llm_caller=mock_extract_llm
        )
        messages = [
            {"role": "user", "content": "记住我喜欢用 Tab 缩进，但本次会话不要创建文件"},
            SimpleNamespace(role="assistant", content="好的，已记录您的长期偏好。"),
        ]

        with redirect_stdout(io.StringIO()):
            stored_first = mgr.extract_memories(messages)
        self.assertEqual(stored_first, 1)
        records = mgr.list_memory_files()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "use-tabs")

        # 再次执行相同提取应当被去重过滤（返回 0）
        stored_second = mgr.extract_memories(messages)
        self.assertEqual(stored_second, 0)

    def test_consolidate_and_atomic_rollback(self):
        """测试达标自动合并整理，以及写盘异常时的快照原子回滚"""
        consolidated_output = [
            {
                "name": "merged-rule",
                "type": "project",
                "description": "合并后的稳定项目规范",
                "body": "统一采用中文注释与严格增量演进。",
            }
        ]

        def mock_consolidate_llm(messages, **kwargs):
            return SimpleNamespace(
                role="assistant",
                content=json.dumps(consolidated_output, ensure_ascii=False),
            )

        mgr = MemoryManager(
            self.memory_dir,
            workdir=self.workdir,
            llm_caller=mock_consolidate_llm,
            consolidate_threshold=3,
        )
        mgr.write_memory_file("rule-a", "project", "规范 A", "正文 A")
        mgr.write_memory_file("rule-b", "project", "规范 B", "正文 B")
        mgr.write_memory_file("rule-c", "project", "规范 C", "正文 C")
        self.assertEqual(len(mgr.list_memory_files()), 3)

        original_doc = mgr.memory_document

        def broken_document(*args, **kwargs):
            raise IOError("模拟磁盘写入中断")

        mgr.memory_document = broken_document  # type: ignore[method-assign]
        with redirect_stdout(io.StringIO()):
            rolled_back_count = mgr.consolidate_memories()
        self.assertEqual(rolled_back_count, 0)
        self.assertEqual(len(mgr.list_memory_files()), 3)

        mgr.memory_document = original_doc  # type: ignore[method-assign]
        with redirect_stdout(io.StringIO()):
            merged_count = mgr.consolidate_memories()
        self.assertEqual(merged_count, 1)
        records = mgr.list_memory_files()
        self.assertEqual([r["name"] for r in records], ["merged-rule"])
        self.assertIn("merged-rule", mgr.read_memory_index())
        self.assertEqual(memory_slug("Use Pytest!"), "use-pytest")


class TestMemoryRuntimeIntegration(unittest.TestCase):
    """测试 MemoryManager 接入 runtime.py 后的跨会话提取与自动召回"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = build_agent_paths(workdir=Path(self.temp_dir.name))
        self.tools = runtime_mod.create_default_tools(workdir=self.paths.workdir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runtime_cross_session_recall_and_extraction(self):
        """会话 1 提炼偏好落盘至 .agent/memory，会话 2 启动时应自动召回注入 System Prompt"""

        def session1_mem_llm(messages, **kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    [
                        {
                            "name": "pytest-preference",
                            "type": "user",
                            "scope": "persistent",
                            "description": "用户习惯用 pytest 跑测试",
                            "body": "执行单元测试统一优先使用 pytest。",
                        }
                    ],
                    ensure_ascii=False,
                )
            )

        mem1 = MemoryManager(self.paths.memory_dir, workdir=self.paths.workdir, llm_caller=session1_mem_llm)
        msgs1 = []
        with redirect_stdout(io.StringIO()):
            runtime_mod.run_agent_turn(
                "我习惯用 pytest 跑测试",
                tools=self.tools,
                hooks=HookManager(),
                messages=msgs1,
                llm_caller=lambda messages, tools: SimpleNamespace(content="好的，记住了。", tool_calls=[]),
                workdir=self.paths.workdir,
                memory_mgr=mem1,
                verbose=False,
            )

        self.assertTrue((self.paths.memory_dir / "pytest-preference.md").is_file())
        self.assertIn("pytest-preference", mem1.read_memory_index())

        # 新会话（空 messages）启动
        mem2 = MemoryManager(
            self.paths.memory_dir,
            workdir=self.paths.workdir,
            llm_caller=lambda messages, **kw: SimpleNamespace(content="[0]"),
        )
        msgs2 = []
        seen_system_prompts = []

        def session2_llm(messages, tools):
            seen_system_prompts.append(messages[0]["content"])
            return SimpleNamespace(content="将使用 pytest 运行测试。", tool_calls=[])

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "运行单元测试",
                tools=self.tools,
                hooks=HookManager(),
                messages=msgs2,
                llm_caller=session2_llm,
                workdir=self.paths.workdir,
                memory_mgr=mem2,
                verbose=False,
            )

        self.assertEqual(reply, "将使用 pytest 运行测试。")
        self.assertIn("【历史跨会话记忆（自动激活）】", seen_system_prompts[0])
        self.assertIn("pytest", seen_system_prompts[0])

    def test_runtime_triggers_consolidate_when_threshold_reached(self):
        """回合结束提取记忆达到整理阈值时，应自动触发 consolidate_memories 归并"""
        mem_calls = 0

        def mock_mem_llm(messages, **kwargs):
            nonlocal mem_calls
            mem_calls += 1
            prompt_text = messages[0]["content"]
            if "只提取在未来新会话" in prompt_text:
                return SimpleNamespace(
                    content=json.dumps(
                        [
                            {
                                "name": "rule-two",
                                "type": "project",
                                "scope": "persistent",
                                "description": "新项目规范二",
                                "body": "代码需附带中文 docstring。",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            if "任务要求：合并重复条目" in prompt_text:
                return SimpleNamespace(
                    content=json.dumps(
                        [
                            {
                                "name": "merged-rules",
                                "type": "project",
                                "description": "合并后的项目规范",
                                "body": "统一规范汇总。",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            return SimpleNamespace(content="[]")

        mem = MemoryManager(
            self.paths.memory_dir,
            workdir=self.paths.workdir,
            llm_caller=mock_mem_llm,
            consolidate_threshold=2,
        )
        mem.write_memory_file("rule-one", "project", "旧规范一", "旧规范内容。")

        with redirect_stdout(io.StringIO()):
            runtime_mod.run_agent_turn(
                "以后代码都要带中文 docstring",
                tools=self.tools,
                hooks=HookManager(),
                messages=[],
                llm_caller=lambda messages, tools: SimpleNamespace(content="了解。", tool_calls=[]),
                workdir=self.paths.workdir,
                memory_mgr=mem,
                verbose=False,
            )

        records = mem.list_memory_files()
        self.assertEqual([r["name"] for r in records], ["merged-rules"])
        self.assertGreaterEqual(mem_calls, 2)

    def test_memory_demo_cli_runs_cleanly(self):
        """uv run python -m coding_agent.memory 演示入口应跑通提取与新会话召回"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = memory_cli_mod.main()
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("test-runner-pytest", out)
        self.assertIn("【历史跨会话记忆（自动激活）】", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
