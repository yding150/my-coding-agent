"""
__main__.py - `coding_agent.background` 异步后台任务与通知注入离线演示入口

用法：
    uv run python -m coding_agent.background
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.background import BackgroundTaskManager
from coding_agent.config import configure_utf8_console
from coding_agent.hooks import HookManager
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    run_agent_turn,
)


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
    """演示 `run_shell(run_in_background=True)` 非阻塞派发与 `<task_notification>` 自动注入。"""
    configure_utf8_console()
    print("=" * 70)
    print("演示：M5.1 异步后台任务管理 (BackgroundTaskManager) 与 runtime.py 集成")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        (demo_workdir / "notes.txt").write_text("background demo parallel work ok\n", encoding="utf-8")

        bg_mgr = BackgroundTaskManager(workdir=demo_workdir)
        try:
            tools = create_default_tools(workdir=demo_workdir, bg_mgr=bg_mgr)
            messages = create_session_messages(workdir=demo_workdir)
            bg_cmd = (
                "Start-Sleep -Milliseconds 350; Write-Output 'BG_BUILD_DONE'"
                if sys.platform == "win32"
                else "sleep 0.35; echo 'BG_BUILD_DONE'"
            )

            step = 0

            def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
                nonlocal step
                step += 1
                if step == 1:
                    return _mock_msg(
                        tool_calls=[
                            {
                                "id": "bg_call_1",
                                "name": "run_shell",
                                "arguments": {"command": bg_cmd, "run_in_background": True},
                            }
                        ]
                    )
                if step == 2:
                    # 模拟主循环利用等待间隙先读取本地文件，同时等待后台进程完成进入就绪队列
                    return _mock_msg(
                        tool_calls=[
                            {"id": "fg_read_1", "name": "read_file", "arguments": {"path": "notes.txt"}}
                        ]
                    )
                if step == 3:
                    # 第 2 轮工具结束后等待一小会儿，模拟多步骤间隙后台任务完成；第 3 轮模型前将注入通知
                    deadline = time.time() + 5.0
                    while bg_mgr.has_running_tasks() and time.time() < deadline:
                        time.sleep(0.05)
                    # 注意：inject_notifications 已在本轮 llm_caller 调用之前执行完毕，模型直接看到通知！
                    return _mock_msg(content="已同时完成前台读取与后台构建任务。")
                return _mock_msg(content="已确认收到 <task_notification> 并总结后台输出。")

            # 为了让第 3 轮 llm_caller 前就绪队列已有结果，我们在第 2 轮 read_file 执行完成或间隙等待：
            # 更直观的做法是分两次 run_agent_turn：回合 1 启动后台 + 前台读取；等待就绪后回合 2 自动注入通知！
            step = 0

            def turn1_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
                nonlocal step
                step += 1
                if step == 1:
                    return _mock_msg(
                        tool_calls=[
                            {
                                "id": "bg_call_1",
                                "name": "run_shell",
                                "arguments": {"command": bg_cmd, "run_in_background": True},
                            }
                        ]
                    )
                if step == 2:
                    return _mock_msg(
                        tool_calls=[
                            {"id": "fg_read_1", "name": "read_file", "arguments": {"path": "notes.txt"}}
                        ]
                    )
                return _mock_msg(content="后台任务已在独立线程启动，当前已完成前台文件读取。")

            run_agent_turn(
                "请在后台运行慢命令，同时先读取 notes.txt",
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=turn1_llm,
                workdir=demo_workdir,
                bg_mgr=bg_mgr,
                verbose=True,
            )

            deadline = time.time() + 5.0
            while bg_mgr.has_running_tasks() and time.time() < deadline:
                time.sleep(0.05)

            def turn2_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
                latest_user = messages[-1]["content"]
                print(f"\n[第 2 回合模型已接收到注入的后台通知]:\n{latest_user}")
                return _mock_msg(content="已收到后台任务完成通知并确认输出为 BG_BUILD_DONE。")

            run_agent_turn(
                "查看刚刚的进度",
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=turn2_llm,
                workdir=demo_workdir,
                bg_mgr=bg_mgr,
                verbose=True,
            )
        finally:
            bg_mgr.cleanup_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
