"""
__main__.py - `coding_agent.tools` 离线工具注册与核心工具执行演示入口

用法：
    uv run python -m coding_agent.tools
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from coding_agent.config import configure_utf8_console
from coding_agent.tools import ToolManager, register_core_tools


def main() -> int:
    """演示 ToolManager 注册、参数错误自愈、OpenAI Schema 导出与 5 大核心工具沙箱执行。"""
    configure_utf8_console()
    print("=" * 60)
    print("演示：ToolManager 工具注册、自愈分发与 5 大核心工具")
    print("=" * 60)

    # 1. 基础注册与异常自愈（对应 _legacy_ref/test/demo_tool_register.py）
    tools = ToolManager()
    print(f"[阶段 1] 初始状态工具箱: {tools.list_tools()} (数量: {len(tools)})")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp).resolve()
        register_core_tools(tools, workdir=workdir)
        print(f"[阶段 2] 挂载核心工具后: {tools.list_tools()}")
        print(
            "[阶段 3] 写入/读取/编辑/搜索沙箱演示:\n"
            f"  - write_file : {tools.execute('write_file', {'path': 'demo/hello.txt', 'content': 'Hello M1.2'})}\n"
            f"  - edit_file  : {tools.execute('edit_file', {'path': 'demo/hello.txt', 'old_text': 'M1.2', 'new_text': 'ToolManager'})}\n"
            f"  - read_file  : {tools.execute('read_file', {'path': 'demo/hello.txt'})}\n"
            f"  - glob_files : {tools.execute('glob_files', {'pattern': '**/*.txt'})}\n"
            f"  - 越界拦截   : {tools.execute('read_file', {'path': '../../outside.txt'})}"
        )

    print(f"[阶段 4] 缺参自愈: {tools.execute('read_file', {})}")
    print(f"[阶段 5] 未知工具: {tools.execute('unknown_tool', {})}")
    print("[阶段 6] 导出的 OpenAI Schema (前 2 项示例):")
    print(json.dumps(tools.get_schemas()[:2], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
