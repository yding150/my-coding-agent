"""
cli.py - Coding Agent 命令行主程序入口（M1.3 基础交互版）

架构角色：
    对外暴露终端入口 `main()`（对应 `pyproject.toml` 中的 `cagent` 与 `coding-agent` 命令），
    支持三种使用模式：
    1. `cagent demo`：无需 API Key 的离线闭环演示（直观观察 `while True` 工具循环全过程）；
    2. `cagent "<指令>"`：单次执行一条自然语言指令后退出；
    3. `cagent`（无参数）：进入交互式多轮 REPL 对话模式。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from coding_agent.config import WORKDIR, configure_utf8_console
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    default_bg_manager,
    default_skill_loader,
    default_tools,
    run_agent_turn,
    session_messages,
)


def _make_mock_message(content: Optional[str] = None, tool_calls: Optional[List[Dict[str, Any]]] = None):
    """构造用于离线 Demo 的模拟 ChatCompletionMessage 对象。"""
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


def run_demo_mode() -> int:
    """
    运行 M1.3 离线闭环演示模式（无需真实 API Key）。

    演示脚本会在系统临时沙箱目录中模拟一次典型的 3 步闭环：
        第 1 轮：Mock 模型调用 `glob_files` 列出文件；
        第 2 轮：Mock 模型调用 `write_file` 创建测试文件并读取校验；
        第 3 轮：Mock 模型结合真实工具输出给出最终中文总结。
    """
    configure_utf8_console()
    print("=" * 65)
    print("【M1.3 离线 Demo 模式】: 演示单 Agent 工具闭环 (One Loop)")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        demo_tools = create_default_tools(workdir=demo_workdir)
        demo_messages = create_session_messages(workdir=demo_workdir)

        # 预置一个初始示例文件，便于模型先“看见”目录再创建新文件
        (demo_workdir / "README.txt").write_text("M1 Demo Workspace\n", encoding="utf-8")

        step_index = 0

        def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step_index
            step_index += 1
            print(f"\n--- [循环第 {step_index} 轮] 模型收到 {len(messages)} 条消息、{len(tools)} 个工具 ---")

            if step_index == 1:
                return _make_mock_message(
                    content="我先列出当前目录文件，再创建测试文件。",
                    tool_calls=[
                        {"id": "call_demo_1", "name": "glob_files", "arguments": {"pattern": "**/*"}}
                    ],
                )
            if step_index == 2:
                return _make_mock_message(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_demo_2",
                            "name": "write_file",
                            "arguments": {"path": "hello_agent.txt", "content": "Created by M1.3 Agent Loop!\n"},
                        },
                        {
                            "id": "call_demo_3",
                            "name": "read_file",
                            "arguments": {"path": "hello_agent.txt"},
                        },
                    ],
                )
            return _make_mock_message(
                content=(
                    "已完成操作：\n"
                    "1. 列出工作区已有文件 `README.txt`；\n"
                    "2. 创建测试文件 `hello_agent.txt`，并已回读确认内容为 `Created by M1.3 Agent Loop!`。"
                ),
                tool_calls=None,
            )

        user_prompt = "列出当前目录文件并创建一个测试文件 hello_agent.txt"
        print(f"用户指令: {user_prompt}")
        print(f"演示沙箱: {demo_workdir} (退出后自动清理)")
        try:
            run_agent_turn(
                user_prompt,
                tools=demo_tools,
                messages=demo_messages,
                llm_caller=scripted_llm,
                workdir=demo_workdir,
                verbose=True,
            )
        finally:
            demo_bg = getattr(demo_tools, "_bg_manager", None)
            if demo_bg is not None:
                demo_bg.cleanup_all()

    print("\n" + "=" * 65)
    print("Demo 演示结束：共经历 3 轮模型调用完成“列目录 -> 写文件 -> 校验 -> 总结”闭环。")
    print("=" * 65)
    return 0


def run_interactive_repl() -> int:
    """启动交互式多轮命令行 REPL。"""
    configure_utf8_console()
    print("=" * 65)
    print("Coding Agent 交互式命令行已启动")
    print(f"当前工作区: {WORKDIR}")
    print(f"已挂载工具: {', '.join(default_tools.list_tools())}")
    print(f"可用技能  : {', '.join(default_skill_loader.list_skills()) or '(无)'}")
    print("快捷命令  : 输入 'q' / 'exit' / 'quit' 退出，或运行 `cagent demo` 查看离线演示")
    print("=" * 65)

    try:
        while True:
            try:
                user_input = input("\n用户输入>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n会话已终止。")
                break

            if not user_input:
                continue
            if user_input.lower() in {"q", "exit", "quit"}:
                print("再见！")
                break

            snapshot_len = len(session_messages)
            try:
                run_agent_turn(user_input)
            except Exception as exc:
                # Why: 本轮如果因 API Key 未配置或网络超时失败，将 session_messages 截断回本轮开始前，
                # 避免留下孤立的 user 或未闭合的 assistant(tool_calls) 污染后续交互。
                del session_messages[snapshot_len:]
                print(f"\n[执行失败]: {exc}", file=sys.stderr)
    finally:
        default_bg_manager.cleanup_all()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Coding Agent 命令行主入口（挂载至 `cagent` 与 `coding-agent` 脚本命令）。

    :param argv: 命令行参数列表；默认读取 `sys.argv[1:]`。
    :return: 进程退出码（0 表示正常，非 0 表示执行异常）。
    """
    configure_utf8_console()
    parser = argparse.ArgumentParser(
        prog="cagent",
        description="Coding Agent 命令行工具（M1.3 单 Agent 工具闭环版）",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="单次执行指令；传入 `demo` 运行离线模拟演示；留空进入交互式 REPL。",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.prompt:
        return run_interactive_repl()

    if len(args.prompt) == 1 and args.prompt[0].lower() == "demo":
        return run_demo_mode()

    single_prompt = " ".join(args.prompt).strip()
    print("=" * 60)
    print(f"工作区: {WORKDIR}")
    print(f"正在执行单次指令: {single_prompt}")
    print("=" * 60)
    try:
        run_agent_turn(single_prompt)
    except Exception as exc:
        print(f"\n[执行失败]: {exc}", file=sys.stderr)
        return 1
    finally:
        default_bg_manager.cleanup_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
