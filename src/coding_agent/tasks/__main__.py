"""
__main__.py - `coding_agent.tasks` 会话 Todo 清单与连续未规划守卫演示入口

用法：
    uv run python -m coding_agent.tasks
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
    run_agent_turn,
)
from coding_agent.tasks import TodoManager


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
    """演示 `todo_write` 更新清单以及连续 5 轮未规划触发 `rounds_since_planning` 守卫提醒。"""
    configure_utf8_console()
    print("=" * 65)
    print("演示：M2.3 会话级 Todo 清单与连续未更新计划守卫")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        (demo_workdir / "sample.py").write_text("print('hello')\n", encoding="utf-8")

        todo_mgr = TodoManager()
        tools = create_default_tools(workdir=demo_workdir, todo_mgr=todo_mgr)
        messages = create_session_messages(workdir=demo_workdir)
        step = 0

        def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step
            step += 1

            # 前 5 轮故意只读文件而不调用 todo_write，触发第 5 轮后的系统守卫提醒
            if step <= 5:
                return _mock_msg(
                    tool_calls=[
                        {"id": f"tc_read_{step}", "name": "read_file", "arguments": {"path": "sample.py"}}
                    ]
                )

            # 第 6 轮模型收到守卫提示后，立即调用 todo_write 更新清单
            if step == 6:
                guard_msg = messages[-1]["content"]
                print(f"\n[第 6 轮模型已看到守卫提示]: {guard_msg}")
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "tc_todo_1",
                            "name": "todo_write",
                            "arguments": {
                                "todos": [
                                    {"content": "第1步：检查 sample.py", "status": "completed"},
                                    {"content": "第2步：汇总代码状态", "status": "in_progress"},
                                ]
                            },
                        }
                    ]
                )

            return _mock_msg(
                content="已响应系统守卫提醒并更新 Todo 清单，全部演示步骤完成。",
                tool_calls=None,
            )

        run_agent_turn(
            "检查工作区并跟踪任务进度",
            tools=tools,
            hooks=HookManager(),
            messages=messages,
            llm_caller=scripted_llm,
            workdir=demo_workdir,
            todo_mgr=todo_mgr,
            verbose=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
