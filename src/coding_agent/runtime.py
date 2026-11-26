"""
runtime.py - Coding Agent 最小单 Agent 主循环驱动引擎（M1.3 基础底座）

架构角色：
    连接“大模型大脑 (`coding_agent.llm`)”与“本地工具手脚 (`coding_agent.tools`)”的核心驱动引擎，
    实现经典的 ReAct / Tool-Calling `while True` 单 Agent 闭环（对应原教学项目 `s02` 能力）。
    后续 M2 ~ M9 的切面钩子、技能、上下文压缩、长期记忆、子代理、后台任务、团队协同、
    MCP 与 Goal Stop Hook 都将在此循环基础上逐层挂载。

解决的核心痛点：
    1. 将大模型纯文本对话升级为“思考 -> 调用工具 -> 观察真实输出 -> 纠错/继续 -> 总结回复”的多轮闭环；
    2. 严格遵守 OpenAI `assistant(tool_calls)` 与 `tool(tool_call_id)` 的消息配对协议；
    3. 通过 `MAX_AGENT_TURNS` 轮次上限熔断，防止模型连续空转调用工具造成死循环或高额 Token 消耗。

暴露的核心接口：
    - `build_system_prompt`: 生成绑定当前工作区的系统提示词
    - `create_default_tools`: 创建并挂载 5 大核心工具的 `ToolManager`
    - `run_agent_turn`: 执行单次用户提问驱动的多轮工具调用闭环
    - `create_session_messages` / `session_messages`: 会话上下文容器
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from coding_agent.config import MAX_AGENT_TURNS, WORKDIR, configure_utf8_console
from coding_agent.llm import call_llm
from coding_agent.tools import ToolManager, register_core_tools

configure_utf8_console()

# 默认会话上下文与默认工具管理器（供交互式 REPL 直接复用；M1 阶段保持简单直观，P2 再抽离为 AgentSession）
session_messages: List[Dict[str, Any]] = []
default_tools: ToolManager = register_core_tools(ToolManager(), workdir=WORKDIR)


def build_system_prompt(workdir: Optional[Path] = None) -> str:
    """
    构建最小单 Agent 系统提示词（System Prompt）。

    :param workdir: 当前绑定的沙箱工作区路径，默认使用 `WORKDIR`。
    :return: 告知模型角色、工作区位置与工具使用原则的系统提示词文本。
    """
    base = (workdir or WORKDIR).resolve()
    return (
        f"你是一个位于工作区 {base} 的务实 Coding Agent（编程助手）。\n"
        f"你可以使用内置工具读写文件、搜索目录和执行命令行指令来解决用户的工程问题。\n\n"
        f"【基本工作准则】：\n"
        f"1. 行动优先：查看文件、列目录、修改代码或验证结果时，直接调用工具获取真实信息，不要凭空猜测；\n"
        f"2. 路径约束：所有文件操作均相对于工作区 {base} 进行；\n"
        f"3. 实事求是：若工具返回报错，请根据错误信息调整参数重试，并在最终回答中如实汇报结果。"
    )


def create_default_tools(workdir: Optional[Path] = None) -> ToolManager:
    """创建并装配好 5 大核心工具的 `ToolManager` 实例（便于测试或隔离会话使用）。"""
    return register_core_tools(ToolManager(), workdir=workdir or WORKDIR)


def create_session_messages(workdir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """初始化一条仅包含 `system` 提示词的新对话历史列表。"""
    return [{"role": "system", "content": build_system_prompt(workdir=workdir)}]


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """统一读取 dict 键值或对象属性，兼容真实 OpenAI 对象与测试 Mock 对象。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _serialize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """将模型返回的 `tool_calls` 列表规范化为纯 JSON 可序列化字典列表。"""
    serialized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        fn = _get_attr_or_key(tc, "function")
        serialized.append(
            {
                "id": _get_attr_or_key(tc, "id", ""),
                "type": _get_attr_or_key(tc, "type", "function"),
                "function": {
                    "name": _get_attr_or_key(fn, "name", ""),
                    "arguments": _get_attr_or_key(fn, "arguments", "") or "{}",
                },
            }
        )
    return serialized


def _assistant_message_to_dict(response_msg: Any) -> Dict[str, Any]:
    """
    将模型返回的消息体转换为纯净的 OpenAI `assistant` 消息字典。

    Why:
        直接将 `ChatCompletionMessage` Pydantic 对象塞回 `messages` 传给部分第三方
        OpenAI 兼容接口（如 DeepSeek/智谱等）时，可能会序列化出 `refusal`、`annotations`
        等模型未识别的字段引发 400 错误；转成标准字典最稳妥，也方便后续持久化与日志排查。
    """
    content = _get_attr_or_key(response_msg, "content", None)
    tool_calls = _get_attr_or_key(response_msg, "tool_calls", None)

    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": content if content is not None else "",
    }
    if tool_calls:
        msg["tool_calls"] = _serialize_tool_calls(tool_calls)
    return msg


def _parse_tool_arguments(raw_args: Any) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    解析模型生成的工具参数 JSON 字符串。

    :param raw_args: 模型返回的原始 `arguments`（通常为 JSON 字符串，也兼容已是 dict 的 Mock）。
    :return: `(parsed_dict, error_message)`；解析成功时 `error_message` 为 None。
    """
    if isinstance(raw_args, dict):
        return raw_args, None
    if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
        return {}, None
    if not isinstance(raw_args, str):
        return None, f"错误: 工具参数格式异常，期望 JSON 字符串，实际为 {type(raw_args).__name__}"

    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return None, f"错误: 工具参数 JSON 解析失败 ({exc})。请输出合法的 JSON 对象。"

    if not isinstance(parsed, dict):
        return None, f"错误: 工具参数必须是 JSON 对象 (object)，实际收到 {type(parsed).__name__}"
    return parsed, None


def run_agent_turn(
    prompt: str,
    *,
    tools: Optional[ToolManager] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    llm_caller: Callable[..., Any] = call_llm,
    workdir: Optional[Path] = None,
    max_turns: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """
    执行一次用户输入触发的单 Agent 闭环（核心 `while True` 工具循环）。

    运行流程：
        1. 确保会话顶部存在 `system` 提示词，并追加本次 `user` 消息；
        2. 循环调用大模型（携带 `tools.get_schemas()`）；
        3. 若模型**未发起** `tool_calls`，说明它认为已经完成任务，输出并直接返回最终文本；
        4. 若模型**发起了** `tool_calls`，依次解析参数、通过 `tools.execute()` 执行本地工具，
           并把每条执行结果封装为 `role="tool"` 消息（带匹配 `tool_call_id`）回填进 `messages`；
        5. 回到步骤 2 让模型“看见”工具结果，直到它给出最终回复或触发 `max_turns` 上限保护。

    :param prompt: 用户本轮输入的自然语言指令。
    :param tools: 工具管理器，默认使用绑定 `WORKDIR` 的 `default_tools`。
    :param messages: 对话消息列表（原地追加更新），默认使用模块级 `session_messages`。
    :param llm_caller: 模型调用函数（默认 `call_llm`，测试或 Demo 可传入 Mock 函数）。
    :param workdir: 工作区根目录（仅在初始化首条 system 消息时生效）。
    :param max_turns: 单次用户提问最多允许的模型调用轮数，默认取 `MAX_AGENT_TURNS`。
    :param verbose: 是否向终端打印工具调用与助手回答过程。
    :return: 模型最终输出的回复文本（若超限则返回提示文本）。
    """
    active_tools = tools if tools is not None else default_tools
    active_messages = messages if messages is not None else session_messages
    turn_limit = max_turns if max_turns is not None else MAX_AGENT_TURNS

    # 1. 确保上下文第 0 条始终是 system prompt（避免首次对话缺少行为约束）
    if not active_messages or active_messages[0].get("role") != "system":
        active_messages.insert(0, {"role": "system", "content": build_system_prompt(workdir=workdir)})

    # 2. 记录本轮用户输入；若调用大模型前或首轮调用即抛异常，由调用方决定是否回滚
    active_messages.append({"role": "user", "content": prompt})

    turns_count = 0
    while True:
        turns_count += 1
        # Why: 必须限制单次对话的最大循环轮数，防止模型陷入“反复调工具不收敛”的死循环
        if turns_count > turn_limit:
            limit_msg = f"⚠️ 已达到单次调用最大轮数限制 (MAX_TURNS={turn_limit})，强制结束本轮并交还控制权。"
            if verbose:
                print(f"\n{limit_msg}")
            return limit_msg

        # 3. 携带当前工具的 OpenAI Schema 调用大模型
        schemas = active_tools.get_schemas()
        response_msg = llm_caller(messages=active_messages, tools=schemas)

        # 4. 无论模型是否调用工具，都必须先把这条 assistant 消息追加进对话历史
        assistant_dict = _assistant_message_to_dict(response_msg)
        active_messages.append(assistant_dict)

        tool_calls = assistant_dict.get("tool_calls") or []

        # 5. 退出条件：模型没有发起任何 tool_calls，表示本轮推理结束，直接返回文本回复
        if not tool_calls:
            final_text = str(assistant_dict.get("content") or "")
            if verbose and final_text:
                print(f"\n助手回答:\n{final_text}")
            return final_text

        # 6. 执行模型请求的每一个工具，并严格按 tool_call_id 配对回填 role="tool" 消息
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            parsed_args, parse_err = _parse_tool_arguments(raw_args)

            if parse_err is not None:
                # Why: 即使 JSON 解析失败，也必须回填一条对应的 role="tool" 消息，
                # 否则下一轮调用 OpenAI API 会因“存在未响应的 tool_call_id”而报 400 错误。
                tool_output = parse_err
                if verbose:
                    print(f"\n[工具参数异常]: {fn_name} -> {tool_output}")
            else:
                if verbose:
                    print(f"\n[调用工具]: {fn_name}({parsed_args})")
                tool_output = active_tools.execute(fn_name, parsed_args)

            if verbose:
                preview = tool_output[:400] + ("..." if len(tool_output) > 400 else "")
                print(f"[工具输出]:\n{preview}")

            active_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                }
            )
