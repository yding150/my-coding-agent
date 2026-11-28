"""
tools.py - Cron 定时调度管理工具封装

架构角色：
    将 `CronSchedulerManager` 的能力封装并挂载为标准 OpenAI Function Calling 工具：
    - `schedule_cron`: 注册基于五段式 Unix Cron 表达式的周期或单次定时任务；
    - `list_crons`: 列出所有活跃的定时调度任务；
    - `cancel_cron`: 根据任务 ID 注销指定的定时任务。

暴露的核心函数：
    - `register_cron_tools`: 一键向 `ToolManager` 挂载 3 大 Cron 管理工具
"""

from __future__ import annotations

from coding_agent.cron.manager import CronSchedulerManager
from coding_agent.tools.manager import ToolManager


def register_cron_tools(tool_mgr: ToolManager, cron_mgr: CronSchedulerManager) -> ToolManager:
    """
    向 `ToolManager` 挂载 3 大定时调度管理工具（`schedule_cron`、`list_crons`、`cancel_cron`）。

    :param tool_mgr: 目标父级 `ToolManager` 实例（切勿暴露给同步一次性子代理）。
    :param cron_mgr: 绑定的 `CronSchedulerManager` 实例。
    :return: 注册完成后的原 `ToolManager` 实例。
    """

    @tool_mgr.register(
        name="schedule_cron",
        description=(
            "注册一个基于标准 5 段式 Unix Cron 表达式的定时任务。\n"
            "当指定时间到达且主 Agent 空闲时，系统会自动将该 prompt 作为一条新的唤醒任务启动执行。\n"
            "【Cron 表达式规范: 分(0-59) 时(0-23) 日(1-31) 月(1-12) 星期(0-6, 0为周日)】\n"
            "常用示例：\n"
            "- '*/5 * * * *' : 每 5 分钟执行一次\n"
            "- '0 9 * * *'   : 每天早上 09:00 执行一次\n"
            "- '0 18 * * 1-5': 每周一至周五下午 18:00 执行\n"
            "- '30 23 * * 0' : 每周日晚上 23:30 执行"
        ),
        parameters={
            "type": "object",
            "properties": {
                "cron": {
                    "type": "string",
                    "description": "5 段式标准 Cron 表达式，如 '*/5 * * * *' 或 '0 9 * * *'",
                },
                "prompt": {
                    "type": "string",
                    "description": "到期触发时交给 Agent 执行的任务提示词指令",
                },
                "recurring": {
                    "type": "boolean",
                    "description": "是否为周期性任务。默认为 true（持续循环触发）；若为 false 则单次触发后自动销毁。",
                },
                "durable": {
                    "type": "boolean",
                    "description": "是否持久化到磁盘 (.agent/scheduled_tasks.json)。默认为 true（重启保留）；若为 false 仅当前会话有效。",
                },
            },
            "required": ["cron", "prompt"],
        },
    )
    def run_schedule_cron(
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> str:
        try:
            job = cron_mgr.schedule(
                cron=cron,
                prompt=prompt,
                recurring=recurring,
                durable=durable,
            )
            mode_str = "周期循环" if job.recurring else "单次执行"
            persist_str = "持久化保存" if job.durable else "会话内存"
            return (
                f"定时任务注册成功！\n"
                f"- 任务 ID: {job.id}\n"
                f"- 调度规则: {job.cron}\n"
                f"- 执行指令: {job.prompt}\n"
                f"- 属性: [{mode_str}, {persist_str}]\n"
                f"系统将在匹配时刻且会话空闲时自动唤醒 Agent 执行该任务。"
            )
        except Exception as exc:
            return f"定时任务注册失败: {exc}"

    @tool_mgr.register(
        name="list_crons",
        description="列出当前所有已注册的定时调度任务列表及其配置与交付状态详情。",
        parameters={"type": "object", "properties": {}},
    )
    def run_list_crons() -> str:
        jobs = cron_mgr.list_jobs()
        if not jobs:
            return "当前没有任何已注册的定时任务。"

        lines = ["【定时任务一览看板】:"]
        for j in jobs:
            mode_str = "周期循环" if j.recurring else "单次执行"
            persist_str = "持久化" if j.durable else "仅内存"
            # Why: 定时唤醒回合运行期间 pending_delivery 仍为 True，直到本轮结束后才执行 acknowledge_jobs；
            # 显式标出“交付执行中”，避免大模型在回合内查询看板时误判单次任务未销毁或将重复触发。
            if j.pending_delivery:
                if j.recurring:
                    delivery_str = "交付执行中（本轮完成后解锁等待下次调度，同分钟不重复触发）"
                else:
                    delivery_str = "交付执行中（单次任务将在本轮完成后自动核销，不会重复触发）"
            else:
                delivery_str = "等待调度"
            lines.append(
                f"  - [{j.id}] '{j.cron}' -> {j.prompt[:60]} [{mode_str}, {persist_str}, {delivery_str}]"
            )
        return "\n".join(lines)

    @tool_mgr.register(
        name="cancel_cron",
        description="根据任务 ID 取消并注销指定的定时任务。",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "待取消的定时任务 ID，例如 'cron_1a2b3c4d'",
                }
            },
            "required": ["job_id"],
        },
    )
    def run_cancel_cron(job_id: str) -> str:
        success = cron_mgr.cancel(job_id)
        if success:
            return f"已成功取消定时任务 '{job_id}'。"
        return f"取消失败：未找到 ID 为 '{job_id}' 的定时任务。"

    return tool_mgr
