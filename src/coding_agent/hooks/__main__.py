"""
__main__.py - `coding_agent.hooks` 生命周期钩子与主循环危险指令拦截演示入口

用法：
    uv run python -m coding_agent.hooks              # 在真实 run_agent_turn 中交互式输入 [y/N]
    uv run python -m coding_agent.hooks --choice n   # 非交互演示主循环危险指令拦截与模型退避
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

from coding_agent.config import configure_utf8_console
from coding_agent.hooks import HookEvent, HookManager, register_default_hooks
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    run_agent_turn,
)


def _mock_msg(content: Optional[str] = None, tool_calls: Optional[List[Dict[str, Any]]] = None):
    """构造用于演示主循环钩子拦截的模拟模型响应消息。"""
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """在 `run_agent_turn` 主循环中演示危险 Shell 指令 `[y/N]` 审批门禁拦截与模型纠错闭环。"""
    configure_utf8_console()
    parser = argparse.ArgumentParser(
        prog="python -m coding_agent.hooks",
        description="演示 runtime.py 主循环接入 HookManager 后的审计日志与危险指令 [y/N] 拦截。",
    )
    parser.add_argument(
        "--choice",
        choices=["y", "n"],
        default=None,
        help="可选：预设审批输入（用于非交互自动化验证）；不传则由终端真实读取 [y/N]。",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 65)
    print("演示：M2.1 HookManager 接入 runtime.py 主循环与危险指令 [y/N] 拦截")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        demo_tools = create_default_tools(workdir=demo_workdir)
        demo_hooks = register_default_hooks(HookManager(), auto_approve=False)
        demo_messages = create_session_messages(workdir=demo_workdir)

        step = 0

        def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step
            step += 1
            if step == 1:
                return _mock_msg(
                    content="我将先尝试清理临时目录。",
                    tool_calls=[
                        {
                            "id": "call_danger_1",
                            "name": "run_shell",
                            "arguments": {"command": "Remove-Item -Recurse ./tmp"},
                        }
                    ],
                )
            # 观察工具被 Hook 拦截后回填进 messages[-1] 的真实拒绝原因
            last_tool_msg = messages[-1]["content"]
            return _mock_msg(
                content=f"已收到门禁反馈，停止危险操作并改用安全方案。门禁回执：{last_tool_msg}",
                tool_calls=None,
            )

        def _execute() -> str:
            return run_agent_turn(
                "清理临时目录 ./tmp",
                tools=demo_tools,
                hooks=demo_hooks,
                messages=demo_messages,
                llm_caller=scripted_llm,
                workdir=demo_workdir,
                verbose=True,
            )

        if args.choice is not None:
            with patch("builtins.input", return_value=args.choice):
                _execute()
        else:
            _execute()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
