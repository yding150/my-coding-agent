"""
tools.py - 异步后台任务工具封装与 `run_shell` 增强注册器

架构角色：
    为父级 `ToolManager` 升级原有 `run_shell`（追加 `run_in_background: bool` 参数）
    并挂载 `check_bg_task`、`kill_bg_task`、`list_bg_tasks` 三大管理查询工具。

解决的核心痛点：
    1. 保持同步/异步统一心智：模型仍只面对同一个 `run_shell` 工具名，通过显式传入
       `run_in_background=true` 即可把重型构建或测试无缝转入后台执行；
    2. 作用域安全隔离：仅对 `parent_tools` 增强补丁而不修改 `shared_tools`，从而保证
       一次性同步子代理不会错误启动需要跨轮次回调的后台守护进程。

暴露的核心接口：
    - `patch_run_shell_with_background`: 增强 `run_shell` 支持 `run_in_background`
    - `register_background_tools`: 一键升级 `run_shell` 并挂载 3 个后台管理工具
"""

from __future__ import annotations

from coding_agent.background.manager import BackgroundTaskManager
from coding_agent.tools.manager import ToolManager


def patch_run_shell_with_background(
    tool_mgr: ToolManager,
    bg_mgr: BackgroundTaskManager,
) -> ToolManager:
    """
    用支持 `run_in_background` 参数的增强版本覆盖原有的 `run_shell` 工具。

    :param tool_mgr: 目标父级 `ToolManager` 实例。
    :param bg_mgr: 对应的 `BackgroundTaskManager` 实例。
    :return: 完成补丁后的原 `ToolManager` 实例。
    """
    tool_def = tool_mgr.get("run_shell")
    original_handler = tool_def.func if tool_def else None

    def _enhanced_run_shell(command: str, run_in_background: bool = False) -> str:
        if run_in_background:
            try:
                task_id = bg_mgr.start(command)
                return (
                    f"[后台任务 {task_id} 已成功启动]\n"
                    f"命令行 '{command}' 正在后台独立执行。\n"
                    f"你可以继续执行其他不依赖此命令的后续操作。当该任务执行完毕后，系统会在后续轮次中自动注入 <task_notification> 通知。"
                    f"如需主动查询状态或查看日志，亦可随时调用 check_bg_task(task_id='{task_id}')。"
                )
            except Exception as exc:
                return f"启动后台任务失败: {exc}"

        if original_handler is not None:
            return original_handler(command=command)
        from coding_agent.tools.core_tools import run_shell

        return run_shell(command=command, workdir=bg_mgr.workdir)

    tool_mgr.register(
        _enhanced_run_shell,
        name="run_shell",
        description=(
            "在本地运行命令行指令（PowerShell）。"
            "对于可能耗时较长、阻塞主循环的操作（例如：依赖安装、全量构建、自动化测试、运行长期监控等），"
            "强烈建议显式设置参数 run_in_background=true，让任务在后台异步执行，避免 Agent 被阻塞。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "待执行的命令行指令内容",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "是否在后台异步运行。长耗时、重型构建或服务启动任务请设置为 true，普通快速查询保持 false。",
                },
            },
            "required": ["command"],
        },
    )
    return tool_mgr


def register_background_tools(
    tool_mgr: ToolManager,
    bg_mgr: BackgroundTaskManager,
) -> ToolManager:
    """
    向 `ToolManager` 升级增强 `run_shell` 并挂载后台任务查询/终止管理工：
    - `check_bg_task`
    - `kill_bg_task`
    - `list_bg_tasks`
    """
    patch_run_shell_with_background(tool_mgr, bg_mgr)

    @tool_mgr.register(
        name="check_bg_task",
        description="主动查询指定后台任务的实时执行状态（running/completed/failed/killed）与最新输出日志。",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "后台任务 ID，例如 'bg_0001'"}
            },
            "required": ["task_id"],
        },
    )
    def run_check_bg_task(task_id: str) -> str:
        detail = bg_mgr.get_task(task_id)
        if not detail:
            return f"未找到 ID 为 '{task_id}' 的后台任务。"

        status = detail.get("status")
        cmd = detail.get("command")
        exit_code = detail.get("exit_code")
        output = detail.get("output", "")

        return (
            f"【后台任务状态看板】\n"
            f"- 任务 ID: {task_id}\n"
            f"- 当前状态: {status}\n"
            f"- 执行命令: {cmd}\n"
            f"- 退出码: {exit_code if exit_code is not None else '运行中'}\n"
            f"- 输出内容（前 2000 字符）:\n{output[:2000]}"
        )

    @tool_mgr.register(
        name="kill_bg_task",
        description="强行终止一个正在后台运行的异常或死循环任务，释放系统资源。",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "要终止的后台任务 ID，例如 'bg_0001'"}
            },
            "required": ["task_id"],
        },
    )
    def run_kill_bg_task(task_id: str) -> str:
        success = bg_mgr.kill_task(task_id)
        if success:
            return f"已成功强制终止后台任务 '{task_id}'。"
        return f"终止失败：未找到后台任务 '{task_id}' 或该任务已不在运行状态。"

    @tool_mgr.register(
        name="list_bg_tasks",
        description="列出当前会话中所有的后台任务及其当前运行状态。",
        parameters={"type": "object", "properties": {}},
    )
    def run_list_bg_tasks() -> str:
        tasks = bg_mgr.list_tasks()
        if not tasks:
            return "当前没有任何后台任务。"

        lines = ["【后台任务一览表】:"]
        for t in tasks:
            tid = t.get("task_id")
            st = t.get("status")
            cmd = t.get("command", "")[:40]
            lines.append(f"  - [{st}] {tid}: {cmd}")
        return "\n".join(lines)

    return tool_mgr
