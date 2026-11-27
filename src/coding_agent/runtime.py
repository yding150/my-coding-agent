"""
runtime.py - Coding Agent 单 Agent 主循环驱动引擎（已集成 M2 Hook/Skill/Todo + M3.1 Context + M3.2 Memory）

架构角色：
    连接“大模型大脑 (`coding_agent.llm`)”、“本地工具手脚 (`coding_agent.tools`)”、
    “生命周期切面门禁 (`coding_agent.hooks`)”、“按需专业技能 (`coding_agent.skills`)”、
    “会话待办清单 (`coding_agent.tasks`)”、“四步上下文压缩 (`coding_agent.context`)”
    与“跨会话持久记忆 (`coding_agent.memory`)”的核心驱动引擎。

解决的核心痛点：
    1. 将大模型纯文本对话升级为“思考 -> 切面审批/执行工具 -> 观察真实输出 -> 总结回复”的多轮闭环；
    2. 通过 `SkillLoader` 两层加载机制在不撑爆 System Prompt 的前提下赋予 Agent 领域专家技能；
    3. 通过 `todo_write` 整表覆盖待办清单与 `rounds_since_planning` 守卫提醒对抗长链条思维漂移；
    4. 通过 `ContextCompactor` 四步压缩与 `reactive_compact` 防止上下文超限崩溃，支持 `compact` 主动归档至 `.agent/transcripts/`；
    5. 通过 `MemoryManager` 在会话启动前按需召回 `.agent/memory/` 历史偏好并在结束时提炼落盘；
    6. 严格遵守 OpenAI `assistant(tool_calls)` 与 `tool(tool_call_id)` 的消息配对协议，并受 `MAX_AGENT_TURNS` 保护。

暴露的核心接口：
    - `build_system_prompt`: 生成绑定工作区、轻量技能目录、上下文规则、规划指引与召回记忆的系统提示词
    - `create_default_tools` / `create_default_hooks` / `create_default_compactor` / `create_default_memory_manager`
    - `execute_tool`: 经由 `PreToolUse` / `PostToolUse` 切面包装的安全工具执行函数
    - `run_agent_turn`: 执行单次用户提问驱动的多轮工具调用闭环
    - `create_session_messages` / `session_messages` / `default_*`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from coding_agent.config import (
    MAX_AGENT_TURNS,
    MEMORY_DIR,
    SKILLS_DIR,
    TOOL_RESULTS_DIR,
    TRANSCRIPT_DIR,
    WORKDIR,
    configure_utf8_console,
)
from coding_agent.context import ContextCompactor
from coding_agent.hooks import HookEvent, HookManager, register_default_hooks
from coding_agent.llm import call_llm
from coding_agent.memory import MemoryManager
from coding_agent.skills import SkillLoader, register_skill_tools
from coding_agent.tasks import TodoManager, register_todo_tools
from coding_agent.tools import ToolManager, register_core_tools

configure_utf8_console()

# 默认技能加载器、会话 Todo 管理器、上下文压缩器、跨会话记忆管理器与会话上下文（P2 再收拢为 AgentSession）
default_skill_loader: SkillLoader = SkillLoader(SKILLS_DIR)
default_todo_mgr: TodoManager = TodoManager()
default_compactor: ContextCompactor = ContextCompactor(TRANSCRIPT_DIR, TOOL_RESULTS_DIR)
default_memory_mgr: MemoryManager = MemoryManager(MEMORY_DIR, workdir=WORKDIR)
session_messages: List[Dict[str, Any]] = []
rounds_since_planning: int = 0
PLANNING_GUARD_THRESHOLD: int = 5
PLANNING_TOOLS: tuple[str, ...] = ("todo_write",)


def build_system_prompt(
    workdir: Optional[Path] = None,
    skill_loader: Optional[SkillLoader] = None,
    relevant_memories: str = "",
) -> str:
    """
    构建包含工作区约束、第一层轻量技能目录、规划规范、压缩说明与历史相关记忆的系统提示词。

    :param workdir: 当前绑定的沙箱工作区路径，默认使用 `WORKDIR`。
    :param skill_loader: 技能加载器实例，默认使用 `default_skill_loader`。
    :param relevant_memories: 本轮按需召回的相关跨会话记忆 JSON 字符串（空串时不追加段落）。
    :return: 完整的系统提示词文本。
    """
    base = (workdir or WORKDIR).resolve()
    active_loader = skill_loader if skill_loader is not None else default_skill_loader
    sections = [
        (
            f"你是一个位于工作区 {base} 的务实 Coding Agent（编程助手）。\n"
            f"你可以使用内置工具读写文件、搜索目录、执行命令行指令、维护待办清单、压缩长上下文以及按需加载专业技能来解决用户的工程问题。\n\n"
            f"【可用专业技能目录（轻量元数据）】：\n"
            f"{active_loader.catalog()}\n\n"
            f"【基本工作准则】：\n"
            f"1. 行动优先：查看文件、列目录、修改代码或验证结果时，直接调用工具获取真实信息，不要凭空猜测；\n"
            f"2. 规划防漂移：面对多步骤复杂任务，使用 `todo_write` 维护当前会话待办清单，保持焦点不偏离目标；\n"
            f"3. 按需加载技能：若任务命中上方某项专业技能领域，请先调用 `load_skill` 读取完整操作规范后再执行；\n"
            f"4. 上下文管理：遇到包含 `[Compacted]` 或 `[Emergency Compact]` 的历史压缩通知时，专注当前最新目标；上下文过长时可调用 `compact` 归档历史；\n"
            f"5. 路径约束：所有文件操作均相对于工作区 {base} 进行；\n"
            f"6. 实事求是：若工具返回报错，请根据错误信息调整参数重试，并在最终回答中如实汇报结果。"
        )
    ]

    if relevant_memories:
        sections.append(
            "【历史跨会话记忆（自动激活）】：\n"
            "以下是与当前任务高度相关的历史知识与用户偏好，请优先遵循：\n"
            f"{relevant_memories}"
        )

    return "\n\n".join(sections)


def create_default_compactor(
    transcript_dir: Optional[Path] = None,
    tool_results_dir: Optional[Path] = None,
) -> ContextCompactor:
    """创建绑定 `.agent/transcripts` 与 `.agent/tool-results` 的 `ContextCompactor` 实例。"""
    return ContextCompactor(
        transcript_dir=transcript_dir or TRANSCRIPT_DIR,
        tool_results_dir=tool_results_dir or TOOL_RESULTS_DIR,
    )


def create_default_memory_manager(
    memory_dir: Optional[Path] = None,
    workdir: Optional[Path] = None,
    llm_caller: Optional[Callable[..., Any]] = None,
) -> MemoryManager:
    """创建绑定 `.agent/memory` 的 `MemoryManager` 实例。"""
    base = (workdir or WORKDIR).resolve()
    return MemoryManager(
        memory_dir=memory_dir or MEMORY_DIR,
        workdir=base,
        llm_caller=llm_caller,
    )


def create_default_tools(
    workdir: Optional[Path] = None,
    skill_loader: Optional[SkillLoader] = None,
    todo_mgr: Optional[TodoManager] = None,
) -> ToolManager:
    """创建并装配好 5 大核心工具、`load_skill`、`todo_write` 与 `compact` 的 `ToolManager` 实例。"""
    manager = register_core_tools(ToolManager(), workdir=workdir or WORKDIR)
    active_loader = skill_loader if skill_loader is not None else default_skill_loader
    active_todo = todo_mgr if todo_mgr is not None else default_todo_mgr
    register_skill_tools(manager, active_loader)
    register_todo_tools(manager, active_todo)

    @manager.register(
        name="compact",
        description="主动触发上下文压缩归档。将过往对话转存至 .agent/transcripts/ 转录文件，并裁剪冗余历史。",
        parameters={"type": "object", "properties": {}},
    )
    def run_compact() -> str:
        return "已收到压缩请求。当前轮次工具执行完毕后，系统将立即触发上下文历史归档与裁剪！"

    return manager


def create_default_hooks(auto_approve: bool = False) -> HookManager:
    """创建并装配好默认生命周期钩子链（审计日志 + 危险指令门禁 + 超长输出警告 + 会话统计）。"""
    return register_default_hooks(HookManager(), auto_approve=auto_approve)


default_tools: ToolManager = create_default_tools(
    workdir=WORKDIR,
    skill_loader=default_skill_loader,
    todo_mgr=default_todo_mgr,
)
default_hooks: HookManager = create_default_hooks()


def execute_tool(
    fn_name: str,
    args: Dict[str, Any],
    tool_mgr: ToolManager,
    hook_mgr: Optional[HookManager] = None,
) -> str:
    """
    经由 `PreToolUse` 与 `PostToolUse` 生命周期切面包装的统一工具执行器。

    :param fn_name: 待调用的工具名。
    :param args: 解析后的工具参数字典。
    :param tool_mgr: 目标工具管理器。
    :param hook_mgr: 生命周期钩子管理器；默认使用全局 `default_hooks`。
    :return: 工具真实输出或被前置切面拦截的拒绝原因文本。
    """
    active_hooks = hook_mgr if hook_mgr is not None else default_hooks
    blocked_reason = active_hooks.trigger(HookEvent.PRE_TOOL_USE, fn_name, args)
    if blocked_reason is not None:
        return f"操作被钩子拦截拒绝: {blocked_reason}"

    output = tool_mgr.execute(fn_name, args)
    active_hooks.trigger(HookEvent.POST_TOOL_USE, fn_name, args, output)
    return str(output)


def create_session_messages(
    workdir: Optional[Path] = None,
    skill_loader: Optional[SkillLoader] = None,
) -> List[Dict[str, Any]]:
    """初始化一条仅包含 `system` 提示词（含轻量技能目录）的新对话历史列表。"""
    return [{"role": "system", "content": build_system_prompt(workdir=workdir, skill_loader=skill_loader)}]


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
    hooks: Optional[HookManager] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    llm_caller: Callable[..., Any] = call_llm,
    workdir: Optional[Path] = None,
    skill_loader: Optional[SkillLoader] = None,
    todo_mgr: Optional[TodoManager] = None,
    compactor: Optional[ContextCompactor] = None,
    memory_mgr: Optional[MemoryManager] = None,
    planning_state: Optional[Dict[str, int]] = None,
    max_turns: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """
    执行一次用户输入触发的单 Agent 闭环（核心 `while True` 工具循环 + Hook + Skill + Todo + Context + Memory）。

    运行流程：
        1. 触发 `UserPromptSubmit` 切面并追加 `user` 消息；
        2. 会话启动前按需召回跨会话记忆并刷新顶部 `system` 提示词；
        3. 每次调用大模型前执行 `ContextCompactor.prepare()` 四步压缩管线；
        4. 调用大模型（携带 `tools.get_schemas()`），遇上下文超长异常则执行 `reactive_compact` 自愈重试；
        5. 若模型**未发起** `tool_calls`，触发 `Stop` 切面（若 `Stop` 返回强行续轮指令则注入并继续，
           否则提炼/整理跨会话记忆并返回最终文本）；
        6. 若模型**发起了** `tool_calls`，通过 `execute_tool()` 依次经历 `PreToolUse`、`PostToolUse`，
           按 `tool_call_id` 配对回填 `role="tool"`；
        7. 执行连续未规划守卫（`rounds_since_planning >= 5`）以及主动 `compact` 归档裁剪；
        8. 回到步骤 3 让模型“看见”工具结果，直到给出最终回复或触发 `max_turns` 上限保护。

    :param prompt: 用户本轮输入的自然语言指令。
    :param tools: 工具管理器，默认使用绑定 `WORKDIR` 的 `default_tools`。
    :param hooks: 生命周期钩子管理器，默认使用 `default_hooks`。
    :param messages: 对话消息列表（原地追加更新），默认使用模块级 `session_messages`。
    :param llm_caller: 模型调用函数（默认 `call_llm`，测试或 Demo 可传入 Mock 函数）。
    :param workdir: 工作区根目录（用于生成或刷新 system 消息）。
    :param skill_loader: 可选技能加载器（用于生成或刷新 system 消息中的轻量技能目录）。
    :param todo_mgr: 可选外部 Todo 管理器（仅作语义标注；若需绑定工具请通过 `create_default_tools(..., todo_mgr=...)` 传入）。
    :param compactor: 可选上下文压缩器；默认使用绑定 `.agent/` 目录的 `default_compactor`。
    :param memory_mgr: 可选跨会话记忆管理器；使用默认 `call_llm` 时默认启用 `default_memory_mgr`。
    :param planning_state: 可选的连续未规划计数字典 `{"rounds_since_planning": int}`；
        传入自定义 `messages` 时默认隔离新建，未传 `messages` 时默认使用模块级计数器。
    :param max_turns: 单次用户提问最多允许的模型调用轮数，默认取 `MAX_AGENT_TURNS`。
    :param verbose: 是否向终端打印工具调用与助手回答过程。
    :return: 模型最终输出的回复文本（若超限则返回提示文本）。
    """
    global rounds_since_planning

    # 显式保留 todo_mgr 语义入参供未来会话重构对齐，当前由 tools 中的 todo_write 闭包持有
    _ = todo_mgr
    active_tools = tools if tools is not None else default_tools
    active_hooks = hooks if hooks is not None else default_hooks
    active_compactor = compactor if compactor is not None else default_compactor
    # Why: 当外部传入自定义 Mock llm_caller（离线单测或 Demo）且未显式传入 memory_mgr 时，
    # 跳过默认记忆管理器，防止离线测试向真实大模型发起网络请求或写入当前仓库 .agent/memory/。
    active_memory: Optional[MemoryManager]
    if memory_mgr is not None:
        active_memory = memory_mgr
    elif llm_caller is call_llm:
        active_memory = default_memory_mgr
    else:
        active_memory = None

    use_default_session = messages is None
    active_messages = messages if messages is not None else session_messages
    turn_limit = max_turns if max_turns is not None else MAX_AGENT_TURNS
    active_user_request = prompt

    # Why: 当使用默认全局 REPL 会话时延续跨轮次计数器；当测试传入隔离 messages 时使用独立计数器避免串扰
    if planning_state is not None:
        local_planning = planning_state
    elif use_default_session:
        local_planning = {"rounds_since_planning": rounds_since_planning}
    else:
        local_planning = {"rounds_since_planning": 0}

    # 1. 触发 UserPromptSubmit 前置切面并追加本轮用户输入
    active_hooks.trigger(HookEvent.USER_PROMPT_SUBMIT, prompt)
    active_messages.append({"role": "user", "content": prompt})

    # 2. 会话前相关记忆召回并刷新第 0 条 system 提示词（此时消息列表已含最新 user 提问，便于按需检索）
    relevant_memories = active_memory.load_memories(active_messages) if active_memory is not None else ""
    refreshed_system = build_system_prompt(
        workdir=workdir,
        skill_loader=skill_loader,
        relevant_memories=relevant_memories,
    )
    if active_messages and active_messages[0].get("role") == "system":
        active_messages[0]["content"] = refreshed_system
    else:
        active_messages.insert(0, {"role": "system", "content": refreshed_system})

    turns_count = 0
    try:
        while True:
            turns_count += 1
            # Why: 必须限制单次对话的最大循环轮数，防止模型陷入“反复调工具不收敛”的死循环
            if turns_count > turn_limit:
                limit_msg = f"⚠️ 已达到单次调用最大轮数限制 (MAX_TURNS={turn_limit})，强制结束本轮并交还控制权。"
                if verbose:
                    print(f"\n{limit_msg}")
                active_hooks.trigger(HookEvent.STOP, active_messages)
                return limit_msg

            # 3. 执行四步上下文压缩管线（使用切片原地替换，确保调用方持有的列表引用同步更新）
            prepared = active_compactor.prepare(active_messages, active_request=active_user_request)
            if prepared is not active_messages:
                active_messages[:] = prepared

            # 4. 携带当前工具的 OpenAI Schema 调用大模型，遇上下文超长时触发响应式应急压缩重试
            schemas = active_tools.get_schemas()
            try:
                response_msg = llm_caller(messages=active_messages, tools=schemas)
            except Exception as exc:
                err_text = str(exc).lower()
                if any(k in err_text for k in ("prompt_too_long", "context_length_exceeded", "too many tokens")):
                    rebuilt = active_compactor.reactive_compact(
                        active_messages,
                        active_request=active_user_request,
                    )
                    if rebuilt is not active_messages:
                        active_messages[:] = rebuilt
                    response_msg = llm_caller(messages=active_messages, tools=schemas)
                else:
                    raise

            # 5. 无论模型是否调用工具，都必须先把这条 assistant 消息追加进对话历史
            assistant_dict = _assistant_message_to_dict(response_msg)
            active_messages.append(assistant_dict)

            tool_calls = assistant_dict.get("tool_calls") or []

            # 6. 退出条件：模型没有发起任何 tool_calls，触发 Stop 切面裁决后结束或续轮
            if not tool_calls:
                final_text = str(assistant_dict.get("content") or "")
                if verbose and final_text:
                    print(f"\n助手回答:\n{final_text}")

                force_continue = active_hooks.trigger(HookEvent.STOP, active_messages)
                if force_continue:
                    # Why: 允许 Stop 切面通过返回非 None 提示文本阻止模型提前结束（为 M9 Goal 闭环预留接口）
                    active_messages.append({"role": "user", "content": str(force_continue)})
                    continue

                if active_memory is not None:
                    if active_memory.extract_memories(active_messages):
                        active_memory.consolidate_memories()
                return final_text

            # 7. 经过 PreToolUse / PostToolUse 切面执行每个工具，并按 tool_call_id 配对回填 role="tool"
            used_planning_in_this_round = False
            compact_requested = False

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                parsed_args, parse_err = _parse_tool_arguments(raw_args)

                if fn_name in PLANNING_TOOLS:
                    used_planning_in_this_round = True
                if fn_name == "compact":
                    compact_requested = True

                if parse_err is not None:
                    # Why: 即使 JSON 解析失败，也必须回填一条对应的 role="tool" 消息，
                    # 否则下一轮调用 OpenAI API 会因“存在未响应的 tool_call_id”而报 400 错误。
                    tool_output = parse_err
                    if verbose:
                        print(f"\n[工具参数异常]: {fn_name} -> {tool_output}")
                else:
                    if verbose:
                        print(f"\n[调用工具]: {fn_name}({parsed_args})")
                    tool_output = execute_tool(fn_name, parsed_args, active_tools, hook_mgr=active_hooks)

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

            # 8. 连续未更新计划守卫：一旦连续 5 轮工具调用均未规划，注入轻量纠偏提醒并清零计数
            if used_planning_in_this_round:
                local_planning["rounds_since_planning"] = 0
            else:
                local_planning["rounds_since_planning"] += 1

            if local_planning["rounds_since_planning"] >= PLANNING_GUARD_THRESHOLD:
                nudge = (
                    f"[系统守卫提示]: 您已经连续 {PLANNING_GUARD_THRESHOLD} 轮未更新任务计划。"
                    "请使用 todo_write 梳理或更新当前会话待办步骤，防止长链条执行偏离目标。"
                )
                active_messages.append({"role": "user", "content": nudge})
                local_planning["rounds_since_planning"] = 0

            # 9. 若模型主动调用了 compact 工具，在本轮工具回填后立即执行历史归档与裁剪
            if compact_requested:
                if verbose:
                    print("\n🧹 [Context 压缩] 主动压缩生效，正在执行会话历史归档与裁剪...")
                rebuilt = active_compactor.reactive_compact(
                    active_messages,
                    active_request=active_user_request,
                )
                if rebuilt is not active_messages:
                    active_messages[:] = rebuilt
    finally:
        if use_default_session and planning_state is None:
            rounds_since_planning = local_planning["rounds_since_planning"]
