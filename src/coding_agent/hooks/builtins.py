"""
builtins.py - 内置通用生命周期钩子库

架构角色：
    提供开箱即用的标准生命周期切面处理器，在 M2.3 接入主循环后覆盖工具调用审计、
    高危命令行人机交互审批（HITL）、超长输出预警与会话工具调用量统计。

解决的核心痛点：
    1. 高危操作不可逆：在 `run_shell` 执行破坏性命令前通过终端 `[y/N]` 确认门禁人工把关；
    2. 调试黑盒：所有工具调用无论最终是否被门禁拒绝，均在 `priority=0` 提前打印审计日志；
    3. 上下文隐形膨胀：在 `PostToolUse` 阶段提示超大返回值，提醒观察上下文开销。

暴露的核心接口：
    - `DEFAULT_DANGEROUS_PATTERNS`: 默认高危命令正则模式库
    - `tool_logger_hook`: 前置审计日志钩子 (`priority=0`)
    - `create_permission_hook`: HITL 人机交互安全门禁钩子工厂
    - `large_output_warning_hook`: 后置超长输出警告钩子
    - `session_summary_hook`: 终止阶段工具调用统计钩子
    - `register_default_hooks`: 一键挂载默认标准切面链
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import re
from typing import Callable, Iterator, List, Optional

from coding_agent.config import prompt_console_input
from coding_agent.hooks.manager import HookEvent, HookManager

_TURN_TOOL_COUNT: ContextVar[Optional[int]] = ContextVar(
    "coding_agent_turn_tool_count",
    default=None,
)


@contextmanager
def turn_tool_summary_scope(count: int) -> Iterator[None]:
    """
    为 `HookEvent.STOP` 注入当前 `run_agent_turn` 的精确工具调用次数。

    Why:
        多轮 REPL 会话中 `session_messages` 会跨轮累积，且 `snip_compact` 会裁剪旧消息；
        若仅靠统计 `messages` 里的 `role == "tool"` 条数，会出现跨轮虚高或压缩后突然变小的问题。
    """
    token = _TURN_TOOL_COUNT.set(count)
    try:
        yield
    finally:
        _TURN_TOOL_COUNT.reset(token)

# 默认内置的高危系统指令正则模式库（主要覆盖 Windows PowerShell / CMD 与破坏性语句）
DEFAULT_DANGEROUS_PATTERNS: List[str] = [
    r"\brmdir\s+/[sS]\b",
    r"\bdel\s+/[fF]\b",
    # Why: 仅匹配处于命令动词位置的 Windows format（或紧跟盘符参数）及 PowerShell Format-Volume，
    # 避免把常见的 `Format-List` / `Format-Table` 或 `--format` 参数误判为磁盘格式化命令。
    r"(?:^|[;&|]\s*)format(?:\s|$)|(?<![-\w.])format\s+[A-Za-z]:|\bFormat-Volume\b",
    r"\bRemove-Item\b.*-Recurse",
    r">>\s*system\.ini",
    r">\s*C:\\Windows",
    r"\bDrop\s+Database\b",
]


def tool_logger_hook(fn_name: str, args: dict) -> None:
    """
    【前置切面】打印工具调用的审计日志（应以 `priority=0` 优先执行，确保被拦截时也留痕）。
    """
    print(f"\n🔧 \033[94m[Hook 审计日志]\033[0m 正在调用工具 -> {fn_name}")
    args_str = json.dumps(args, ensure_ascii=False)
    if len(args_str) > 120:
        args_str = args_str[:120] + "... (已截断)"
    print(f"   参数: {args_str}")


def create_permission_hook(
    dangerous_patterns: Optional[List[str]] = None,
    auto_approve: bool = False,
) -> Callable[[str, dict], Optional[str]]:
    """
    【前置切面工厂】构造带人机交互审批（HITL）的安全门禁钩子。

    Why:
        高危命令一旦执行可能造成不可逆的文件删除或系统损坏，不能完全依赖模型自律。
        返回拒绝理由字符串会触发 `HookManager.trigger()` 短路，并将原因回填给大模型
        促使其改用更安全方案。

    :param dangerous_patterns: 自定义高危正则表达式列表，未传时使用 `DEFAULT_DANGEROUS_PATTERNS`。
    :param auto_approve: 为 True 时自动放行（仅用于非交互调试场景，不阻塞控制台输入）。
    :return: 适用于 `PreToolUse` 的钩子函数 `(fn_name, args) -> Optional[str]`。
    """
    patterns = dangerous_patterns if dangerous_patterns is not None else DEFAULT_DANGEROUS_PATTERNS

    def permission_hook(fn_name: str, args: dict) -> Optional[str]:
        is_dangerous = False
        warning_desc = ""

        # 当前门禁重点拦截命令行工具中的破坏性指令模式
        if fn_name == "run_shell":
            cmd = str(args.get("command", ""))
            for pattern in patterns:
                if re.search(pattern, cmd, re.IGNORECASE):
                    is_dangerous = True
                    warning_desc = f"命中高危系统指令模式: {pattern}"
                    break

        if not is_dangerous:
            return None

        print("\n" + "=" * 50)
        print("⚠️  \033[91m[Permission 审批] 发现潜在破坏性操作!\033[0m")
        print(f"操作类型: {fn_name}")
        print(f"参数详情: {json.dumps(args, ensure_ascii=False)}")
        print(f"拦截原因: {warning_desc}")
        print("=" * 50)

        if auto_approve:
            print("[Permission 审批] 调试模式：已自动放行高危操作。")
            return None

        while True:
            try:
                choice = prompt_console_input("是否允许继续执行此危险操作？[y/N]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                # 非交互终端或 Ctrl+C 中断一律按安全兜底拒绝
                choice = "n"

            if choice in ("y", "yes"):
                print("[Permission 审批] 用户已授权，放行操作。")
                return None
            if choice in ("n", "no", ""):
                print("[Permission 审批] 用户拒绝了该操作，已将拒绝信号反馈给模型。")
                return "用户在人机交互卡点拒绝了此高危操作，请勿重试相同危险命令，尝试其它安全方案。"
            print("输入无效，请输入 'y' 确认授权，或 'n'（直接回车）拒绝。")

    return permission_hook


def large_output_warning_hook(
    fn_name: str,
    args: dict,
    output: str,
    threshold: int = 2000,
) -> None:
    """
    【后置切面】监控单次工具调用的返回字符量，提醒超大输出对上下文空间的占用。
    """
    output_len = len(str(output))
    if output_len > threshold:
        print(f"⚠️  \033[33m[Hook 监控提示]\033[0m 工具 {fn_name} 输出了 {output_len} 字符，较占上下文空间。")


def session_summary_hook(messages: list) -> None:
    """
    【终止切面】在单轮对话或会话终止时，统计本轮发生的 `tool` 调用次数。
    """
    turn_count = _TURN_TOOL_COUNT.get()
    tool_count = (
        turn_count
        if turn_count is not None
        else sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
    )
    print(f"\n📊 \033[92m[会话生命周期总结]\033[0m 本轮共发生 {tool_count} 次工具调用。")


def register_default_hooks(
    manager: HookManager,
    dangerous_patterns: Optional[List[str]] = None,
    auto_approve: bool = False,
) -> HookManager:
    """
    一键为 `HookManager` 挂载标准生命周期钩子链：
    1. `PreToolUse`: `tool_logger_hook` (`priority=0`，确保先记录审计日志)
    2. `PreToolUse`: `permission_hook` (`priority=10`，审计后再进行危险指令 HITL 审批)
    3. `PostToolUse`: `large_output_warning_hook` (`priority=10`)
    4. `Stop`: `session_summary_hook` (`priority=10`)
    """
    # Why: 必须保证审计日志 priority=0 先于权限拦截 priority=10，这样即便被拦截拒绝也有完整日志
    manager.register(HookEvent.PRE_TOOL_USE, tool_logger_hook, priority=0, name="tool_logger")

    perm_hook = create_permission_hook(
        dangerous_patterns=dangerous_patterns,
        auto_approve=auto_approve,
    )
    manager.register(HookEvent.PRE_TOOL_USE, perm_hook, priority=10, name="permission_gate")

    manager.register(HookEvent.POST_TOOL_USE, large_output_warning_hook, priority=10, name="output_warning")
    manager.register(HookEvent.STOP, session_summary_hook, priority=10, name="session_summary")

    return manager
