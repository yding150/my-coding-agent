"""
__main__.py - `coding_agent.context` 四步上下文压缩演进与主循环 `compact` 归档演示入口

用法：
    uv run python -m coding_agent.context
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import build_agent_paths, configure_utf8_console
from coding_agent.context import ContextCompactor
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
    """演示四步压缩管线（对照 scratch_demo_compact.py）及主循环调用 `compact` 落盘 `.agent/transcripts`。"""
    configure_utf8_console()
    print("=" * 70)
    print("演示：M3.1 ContextCompactor 四步压缩管线与 .agent/ 转录归档")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        paths = build_agent_paths(workdir=Path(tmp))
        compactor = ContextCompactor(
            transcript_dir=paths.transcript_dir,
            tool_results_dir=paths.tool_results_dir,
            llm_caller=lambda msgs, **kw: _mock_msg(
                content="事实摘要：已确认目标为重构认证模块，已查阅 auth_module_v2~v25。"
            ),
        )

        # 1. 构造带 30,000+ 字符大输出的消息历史，演示 Layer 1 ~ Layer 4 演进
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "你是一个资深编程助手。"},
            {"role": "user", "content": "请帮我重构认证模块。"},
            {"role": "assistant", "content": "好的，先查看日志。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_001", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "ERROR Database connection failed\n" * 800,
            },
        ]
        print(f"\n[初始状态] 消息数: {len(messages)} 条, 总字符: {compactor.estimate_chars(messages)}")

        messages = compactor.tool_result_budget(messages)
        print(f"[Layer 1 tool_result_budget] 转存大日志后总字符: {compactor.estimate_chars(messages)}")

        for i in range(2, 26):
            cid = f"call_{i:03d}"
            messages.append(
                {
                    "role": "assistant",
                    "content": f"第 {i} 步排查...",
                    "tool_calls": [
                        {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": f"// File {i}: auth_v{i}.py\n" + ("token_check = True\n" * 12),
                }
            )
        print(f"[对话膨胀] 累积消息数: {len(messages)} 条, 总字符: {compactor.estimate_chars(messages)}")

        messages = compactor.snip_compact(messages, trigger=50, target=30)
        print(f"[Layer 2 snip_compact] 裁剪后消息数: {len(messages)} 条")

        messages = compactor.micro_compact(messages, target_chars=3000)
        print(f"[Layer 3 micro_compact] 置换旧工具结果后字符: {compactor.estimate_chars(messages)}")

        messages = compactor.compact_history(messages, active_request="重构认证模块")
        print(f"[Layer 4 compact_history] 重构后消息数: {len(messages)} 条")

        # 2. 在 run_agent_turn 主循环中演示主动调用 compact 工具并检查 .agent/transcripts/
        print("\n" + "-" * 70)
        print("[主循环集成演示] 在 run_agent_turn 中调用 `compact` 触发转录归档:")
        demo_tools = create_default_tools(workdir=paths.workdir)
        demo_msgs = create_session_messages(workdir=paths.workdir)
        step = 0

        def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step
            step += 1
            if step == 1:
                return _mock_msg(
                    tool_calls=[{"id": "tc_cmp", "name": "compact", "arguments": {}}]
                )
            return _mock_msg(content="已完成主动压缩与归档。")

        run_agent_turn(
            "帮我压缩当前会话历史",
            tools=demo_tools,
            hooks=HookManager(),
            messages=demo_msgs,
            llm_caller=scripted_llm,
            workdir=paths.workdir,
            compactor=compactor,
            verbose=True,
        )
        archives = list(paths.transcript_dir.glob("transcript_*.jsonl"))
        print(f"[磁盘归档校验] `.agent/transcripts/` 共生成 {len(archives)} 份转录文件: {[p.name for p in archives]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
