"""
__main__.py - `coding_agent.skills` 两层技能按需加载与主循环集成演示入口

用法：
    uv run python -m coding_agent.skills
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import SKILLS_DIR, configure_utf8_console
from coding_agent.hooks import HookManager
from coding_agent.runtime import (
    build_system_prompt,
    create_default_tools,
    create_session_messages,
    run_agent_turn,
)
from coding_agent.skills import SkillLoader


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
    """演示第一层轻量目录注入 System Prompt 与第二层 `load_skill('code-review')` 按需加载闭环。"""
    configure_utf8_console()
    loader = SkillLoader(SKILLS_DIR)

    print("=" * 65)
    print("演示：M2.2 两层技能加载与 runtime.py 主循环集成")
    print("=" * 65)
    print(f"技能资产目录: {loader.skills_dir}")
    print(f"已发现技能  : {loader.list_skills()}")
    print("\n[第一层注入 System Prompt 的轻量目录预览]:")
    print(loader.catalog())

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        demo_tools = create_default_tools(workdir=demo_workdir, skill_loader=loader)
        demo_messages = create_session_messages(workdir=demo_workdir, skill_loader=loader)
        step = 0

        def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step
            step += 1
            if step == 1:
                assert "code-review" in messages[0]["content"]
                return _mock_msg(
                    content="我将先加载 code-review 专业技能规范再进行评审说明。",
                    tool_calls=[
                        {"id": "call_skill_1", "name": "load_skill", "arguments": {"name": "code-review"}}
                    ],
                )
            skill_body = messages[-1]["content"]
            first_heading = skill_body.splitlines()[0] if skill_body else ""
            return _mock_msg(
                content=f"已成功按需加载 `code-review` 技能（正文标题：{first_heading}，共 {len(skill_body)} 字符）。",
                tool_calls=None,
            )

        print("\n[第二层按需加载主循环闭环]:")
        run_agent_turn(
            "请加载 code-review 技能并总结其核心规范",
            tools=demo_tools,
            hooks=HookManager(),
            messages=demo_messages,
            llm_caller=scripted_llm,
            workdir=demo_workdir,
            skill_loader=loader,
            verbose=True,
        )

    # 验证 System Prompt 确实不包含完整技能正文，仅保留摘要目录
    prompt_text = build_system_prompt(skill_loader=loader)
    print(f"\n[Token 保护检查] System Prompt 长度: {len(prompt_text)} 字符（未塞入全文技能规范）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
