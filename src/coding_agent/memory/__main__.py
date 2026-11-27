"""
__main__.py - `coding_agent.memory` 跨会话记忆提取与新会话自动召回演示入口

用法：
    uv run python -m coding_agent.memory
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import build_agent_paths, configure_utf8_console
from coding_agent.hooks import HookManager
from coding_agent.memory import MemoryManager
from coding_agent.runtime import create_default_tools, run_agent_turn


def _mock_msg(content: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=[])


def main() -> int:
    """演示“第 1 次会话提炼偏好落盘至 `.agent/memory/` -> 第 2 次新会话自动召回注入 System Prompt”。"""
    configure_utf8_console()
    print("=" * 70)
    print("演示：M3.2 跨会话记忆提取与新会话 System Prompt 自动召回")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        paths = build_agent_paths(workdir=Path(tmp))

        def extract_llm(messages: List[Dict[str, Any]], **kwargs: Any):
            return _mock_msg(
                content=json.dumps(
                    [
                        {
                            "name": "test-runner-pytest",
                            "type": "user",
                            "scope": "persistent",
                            "description": "用户习惯使用 pytest 运行单元测试",
                            "body": "执行 Python 测试时优先使用 pytest。",
                        }
                    ],
                    ensure_ascii=False,
                )
            )

        # 会话 1：用户表达长期偏好，退出后自动提炼落盘
        print("\n[会话 1] 用户输入偏好并结束会话:")
        mem1 = MemoryManager(paths.memory_dir, workdir=paths.workdir, llm_caller=extract_llm)
        tools1 = create_default_tools(workdir=paths.workdir)
        msgs1: List[Dict[str, Any]] = []
        run_agent_turn(
            "请记住，我平时习惯用 pytest 跑测试。",
            tools=tools1,
            hooks=HookManager(),
            messages=msgs1,
            llm_caller=lambda messages, tools: _mock_msg("好的，我已记下您偏好使用 pytest 运行测试。"),
            workdir=paths.workdir,
            memory_mgr=mem1,
            verbose=True,
        )
        print(f"[磁盘检查] MEMORY.md 索引内容:\n{mem1.read_memory_index()}")

        # 会话 2：全新空消息列表启动，检查新会话 System Prompt 已自动召回该记忆
        print("\n[会话 2] 开启全新会话，验证 System Prompt 自动召回历史记忆:")
        mem2 = MemoryManager(
            paths.memory_dir,
            workdir=paths.workdir,
            llm_caller=lambda messages, **kw: _mock_msg("[0]"),
        )
        tools2 = create_default_tools(workdir=paths.workdir)
        msgs2: List[Dict[str, Any]] = []
        run_agent_turn(
            "帮我运行一下这个项目的测试",
            tools=tools2,
            hooks=HookManager(),
            messages=msgs2,
            llm_caller=lambda messages, tools: _mock_msg(
                "已从跨会话记忆得知您习惯使用 pytest，准备用 pytest 运行测试。"
            ),
            workdir=paths.workdir,
            memory_mgr=mem2,
            verbose=True,
        )
        print("\n[新会话 System Prompt 召回片段验证]:")
        sys_prompt = msgs2[0]["content"]
        idx = sys_prompt.find("【历史跨会话记忆")
        print(sys_prompt[idx:] if idx >= 0 else "(未找到记忆片段)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
