"""
todo.py - 会话级轻量内存待办清单管理器（In-Memory Todo Planning）

架构角色：
    承担双轨计划体系中的“第一轨：轻量会话内存清单”，维护单 Agent 在当前轮次/会话内的
    短期执行步骤清单；与 M4.2 将迁入的“第二轨：持久 DAG 任务图 (`TaskStore`)”互补。

解决的核心痛点：
    1. 对抗长工具链中的注意力漂移：采用“全量整表覆盖”更新机制，强迫模型每次更新进度时
       重温完整步骤序列，避免漏步或忘记原始目标；
    2. 约束单焦点执行：严格限制同一时刻最多只有 1 个任务处于 `in_progress` 状态，
       促使模型按顺序推进而非同时开启一堆半成品；
    3. 零磁盘 I/O 开销：纯内存维护，兼容 JSON 字符串或 Python 字面量参数输入。

暴露的核心接口：
    - `TODO_TOOL_SCHEMA`: `todo_write` 标准 OpenAI JSON Schema 描述
    - `TodoManager`: 待办清单校验、整表更新与终端看板渲染管理器
    - `register_todo_tools`: 一键将 `todo_write` 工具挂载至 `ToolManager`
"""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from coding_agent.tools.manager import ToolManager

# todo_write 标准 OpenAI JSON Schema
TODO_TOOL_SCHEMA: Dict[str, Any] = {
    "description": (
        "创建或更新当前会话内的轻量内存待办清单（计划第一轨，状态流转: pending -> in_progress -> completed）。"
        "采用全量整表覆盖更新，同一时刻最多仅允许 1 个任务处于 'in_progress' 状态。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "完整的待办清单列表（整表覆盖），每项需包含 content 描述与 status 状态",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "任务的具体内容描述",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "任务状态：pending(待办), in_progress(进行中), completed(已完成)",
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
}


class TodoManager:
    """
    轻量会话内存待办清单管理器。

    生命周期：
        在会话启动时随运行时创建，全程仅存在于内存中；每次调用 `update()` 都会对传入清单
        进行合法性、数量上限（<= 20）及单 `in_progress` 约束校验，随后覆盖旧清单并渲染看板。
    """

    def __init__(self) -> None:
        self.items: List[Dict[str, str]] = []

    def update(self, todos: Any) -> str:
        """
        全量覆盖更新并校验会话待办清单列表。

        :param todos: 待办项列表，兼容 JSON 字符串或 Python 字面量字符串输入。
        :return: 格式化渲染后的进度看板字符串。
        :raises ValueError: 当格式非法、超过 20 项、状态无效或同时存在多个 `in_progress` 时抛出。
        """
        # Why: 部分开源/兼容模型偶尔会把 array 参数二次序列化成字符串，这里做容错解析提升鲁棒性
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError("todos 必须是一个合法的 JSON 列表或数组") from exc

        if not isinstance(todos, list):
            raise ValueError("todos 必须是列表类型")
        if len(todos) > 20:
            raise ValueError("一次最多允许维护 20 个子任务")

        validated: List[Dict[str, str]] = []
        in_progress_count = 0

        for index, item in enumerate(todos):
            if not isinstance(item, dict):
                raise ValueError(f"第 {index + 1} 个任务项必须是对象类型")

            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()

            if not content:
                raise ValueError(f"第 {index + 1} 个任务项缺少 content 内容描述")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(
                    f"第 {index + 1} 个任务状态 '{status}' 不合法，仅支持 pending/in_progress/completed"
                )

            if status == "in_progress":
                in_progress_count += 1

            validated.append({"content": content, "status": status})

        # Why: 强制单任务进行中约束，防止模型把所有步骤一次性标记为进行中导致执行失焦
        if in_progress_count > 1:
            raise ValueError("同一时间内只允许有 1 个任务处于 'in_progress' 进行中状态")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """渲染当前待办清单进度看板字符串并输出提示到终端。"""
        if not self.items:
            return "当前暂无待办任务。"

        status_icons = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
        }
        # 标题同时兼容历史单测断言的“会话待办清单更新”与“任务清单更新”关键字
        lines = ["\n📋 [会话待办清单更新 / 任务清单更新]"]

        completed_count = 0
        for item in self.items:
            status = item["status"]
            if status == "completed":
                completed_count += 1
            icon = status_icons.get(status, "[?]")
            lines.append(f"  {icon} {item['content']}")

        lines.append(f"\n  (进度: {completed_count}/{len(self.items)} 已完成)")
        rendered_str = "\n".join(lines)
        print(rendered_str)
        return rendered_str

    def get_items(self) -> List[Dict[str, str]]:
        """获取当前任务项列表的浅拷贝。"""
        return list(self.items)

    def is_all_completed(self) -> bool:
        """判断当前清单是否非空且所有任务项均已完成。"""
        return bool(self.items) and all(item["status"] == "completed" for item in self.items)


def register_todo_tools(
    manager: ToolManager,
    todo_mgr: Optional[TodoManager] = None,
) -> Tuple[ToolManager, TodoManager]:
    """
    一键将 `todo_write` 工具挂载到指定的 `ToolManager` 中。

    :param manager: 目标工具管理器。
    :param todo_mgr: 可选外部 `TodoManager` 实例；未传时自动新建。
    :return: `(manager, todo_mgr)` 元组，既兼容解包获取管理器，也兼容直接传参挂载。
    """
    mgr = todo_mgr or TodoManager()

    def run_todo_write(todos: Any) -> str:
        try:
            return mgr.update(todos)
        except ValueError as exc:
            # Why: 将校验错误转为可读文本回填给模型（例如同时标了两个 in_progress），促使其自行修正
            return f"任务清单更新校验失败: {exc}"

    manager.register(
        run_todo_write,
        name="todo_write",
        description=TODO_TOOL_SCHEMA["description"],
        parameters=TODO_TOOL_SCHEMA["parameters"],
    )
    return manager, mgr
