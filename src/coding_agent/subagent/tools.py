"""
tools.py - 一次性隔离子代理工具注册器（Subagent Tool Registration）

架构角色：
    向主 Agent（Lead）的工具管理器注册 `task` 委派工具，使主循环可以将独立探索、
    只读调研或大文本检索任务下放至 `SubagentRunner` 执行并仅回收精炼摘要。
"""

from __future__ import annotations

from typing import Optional

from coding_agent.subagent.runner import SubagentRunner
from coding_agent.tools.manager import ToolManager


def register_subagent_tools(tool_mgr: ToolManager, runner: SubagentRunner) -> ToolManager:
    """
    向 `ToolManager` 注册一次性隔离子代理委派工具 `task`。

    :param tool_mgr: 父级（Lead）工具管理器，切勿传入子代理自身的 `runner.tools`。
    :param runner: 预先绑定 `shared_tools` 的 `SubagentRunner` 实例。
    :return: 注册完成后的原 `ToolManager` 实例。
    """

    @tool_mgr.register(
        name="task",
        description=(
            "在全新的隔离对话上下文中运行一次性轻量子代理（Ephemeral Subagent），"
            "处理独立的探索、调研或单点操作，并直接返回其最终结论文本。"
            "适用场景：探查文件结构、读取大文本寻找关键信息、运行临时验证等。"
            "注意：子代理的大量中间工具调用和长输出将被丢弃，不会污染你的当前主会话上下文。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "委派给子代理的具体任务执行指令与详细要求",
                },
                "description": {
                    "type": "string",
                    "description": "子任务的简短一句话描述（如 '调研测试框架版本'）",
                },
            },
            "required": ["prompt"],
        },
    )
    def run_subagent_task(prompt: str, description: Optional[str] = None) -> str:
        return runner.run(prompt=prompt, description=description)

    return tool_mgr
