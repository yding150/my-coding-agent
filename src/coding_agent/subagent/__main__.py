"""
__main__.py - `coding_agent.subagent` 一次性隔离子代理与主循环集成演示入口

用法：
    uv run python -m coding_agent.subagent
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import configure_utf8_console
from coding_agent.hooks import HookManager
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    create_shared_tools,
    run_agent_turn,
)
from coding_agent.subagent import SubagentRunner


def _mock_msg(content: Optional[str] = None, tool_calls: Optional[List[Dict[str, Any]]] = None):
    """构造离线演示用模拟模型消息。"""
    tc_objs = []
    for tc in tool_calls or []:
        tc_objs.append(
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=json.dumps(tc["arguments"], ensure_ascii=False),
                ),
            )
        )
    return SimpleNamespace(content=content, tool_calls=tc_objs)


def main() -> int:
    """演示主循环通过 `task` 工具派发一次性子代理执行只读调研并仅回收精炼摘要。"""
    configure_utf8_console()
    print("=" * 70)
    print("演示：M4.1 一次性隔离子代理 (SubagentRunner) 与 runtime.py 主循环集成")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        (demo_workdir / "pyproject.toml").write_text(
            '[project]\nname = "demo-repo"\ndependencies = ["pytest>=9.0"]\n',
            encoding="utf-8",
        )

        shared = create_shared_tools(workdir=demo_workdir)
        sub_step = 0

        def subagent_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal sub_step
            sub_step += 1
            if sub_step == 1:
                return _mock_msg(
                    tool_calls=[
                        {"id": "sub_1", "name": "read_file", "arguments": {"path": "pyproject.toml"}}
                    ]
                )
            content = messages[-1]["content"]
            return _mock_msg(content=f"调研结论：项目依赖配置如下 -> {content.strip()}")

        runner = SubagentRunner(shared_tools=shared, workdir=demo_workdir, llm_caller=subagent_llm)
        print(f"[防递归校验] shared_tools 工具: {shared.list_tools()}")
        print(f"[防递归校验] 子代理可用工具: {runner.tools.list_tools()}")

        parent = create_default_tools(workdir=demo_workdir, subagent_runner=runner)
        print(f"[工具分层校验] parent_tools 工具: {parent.list_tools()}")

        parent_msgs = create_session_messages(workdir=demo_workdir)
        lead_step = 0

        def lead_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal lead_step
            lead_step += 1
            if lead_step == 1:
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "lead_1",
                            "name": "task",
                            "arguments": {
                                "description": "只读调研项目依赖配置",
                                "prompt": "读取 pyproject.toml 并汇报项目名称与依赖",
                            },
                        }
                    ]
                )
            return _mock_msg(content=f"主 Agent 已收到子代理汇报：{messages[-1]['content']}")

        run_agent_turn(
            "帮我调研当前项目的依赖配置",
            tools=parent,
            hooks=HookManager(),
            messages=parent_msgs,
            llm_caller=lead_llm,
            workdir=demo_workdir,
            verbose=True,
        )
        print(f"\n[上下文隔离校验] 父会话最终消息条数: {len(parent_msgs)} 条（未混入子代理内部 read_file 过程）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
