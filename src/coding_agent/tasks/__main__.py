"""
__main__.py - `coding_agent.tasks` 双轨计划系统（会话 Todo + 持久 DAG 任务图）离线演示入口

用法：
    uv run python -m coding_agent.tasks
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import configure_utf8_console
from coding_agent.hooks import HookManager
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    run_agent_turn,
)
from coding_agent.tasks import TaskStore, TodoManager


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


def _demo_track1_todo_guard() -> None:
    """演示第一轨：`todo_write` 更新清单与连续 5 轮未规划触发 `rounds_since_planning` 守卫提醒。"""
    print("=" * 70)
    print("演示第一轨：会话级 Todo 清单 (todo_write) 与连续未更新计划守卫")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        (demo_workdir / "sample.py").write_text("print('hello')\n", encoding="utf-8")

        todo_mgr = TodoManager()
        tools = create_default_tools(workdir=demo_workdir, todo_mgr=todo_mgr)
        messages = create_session_messages(workdir=demo_workdir)
        step = 0

        def scripted_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step
            step += 1

            # 前 5 轮故意只读文件而不调用规划工具，触发第 5 轮后的系统守卫提醒
            if step <= 5:
                return _mock_msg(
                    tool_calls=[
                        {"id": f"tc_read_{step}", "name": "read_file", "arguments": {"path": "sample.py"}}
                    ]
                )

            if step == 6:
                guard_msg = messages[-1]["content"]
                print(f"\n[第 6 轮模型已看到守卫提示]: {guard_msg}")
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "tc_todo_1",
                            "name": "todo_write",
                            "arguments": {
                                "todos": [
                                    {"content": "第1步：检查 sample.py", "status": "completed"},
                                    {"content": "第2步：汇总代码状态", "status": "in_progress"},
                                ]
                            },
                        }
                    ]
                )

            return _mock_msg(
                content="已响应系统守卫提醒并更新 Todo 清单，第一轨演示完成。",
                tool_calls=None,
            )

        run_agent_turn(
            "检查工作区并跟踪任务进度",
            tools=tools,
            hooks=HookManager(),
            messages=messages,
            llm_caller=scripted_llm,
            workdir=demo_workdir,
            todo_mgr=todo_mgr,
            verbose=True,
        )


def _demo_track2_dag_task_store() -> None:
    """演示第二轨：持久 DAG 任务图两阶段建图、阻塞拦截、完成自动解锁与主循环集成。"""
    print("\n" + "=" * 70)
    print("演示第二轨：M4.2 持久 DAG 任务依赖图 (TaskStore) 与 runtime.py 主循环集成")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        tasks_dir = demo_workdir / ".agent" / "tasks"
        store = TaskStore(tasks_dir, workdir=demo_workdir)

        t1 = store.create("梳理数据库模型", "设计 Task 与 Subtask 字段")
        t2 = store.create("实现 API 路由", "依赖数据库模型落表")
        store.update_dependencies(t2.id, [t1.id])

        print(f"[阻塞校验] 提前认领 t2 -> {store.claim(t2.id)}")
        print(f"[认领 t1] -> {store.claim(t1.id)}")
        print(f"[完成 t1 并解锁下游] -> {store.complete(t1.id)}")

        agent_store = TaskStore(tasks_dir, workdir=demo_workdir)
        parent = create_default_tools(workdir=demo_workdir, task_store=agent_store)
        messages = create_session_messages(workdir=demo_workdir)
        step = 0

        def fake_lead_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal step
            step += 1
            if step == 1:
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "dag_1",
                            "name": "create_task",
                            "arguments": {"subject": "编写端到端验收测试", "description": "验证 DAG 全流程"},
                        }
                    ]
                )
            if step == 2:
                return _mock_msg(tool_calls=[{"id": "dag_2", "name": "list_tasks", "arguments": {}}])
            return _mock_msg(content=f"当前持久任务图如下：\n{messages[-1]['content']}")

        reply = run_agent_turn(
            "帮我追加一个端到端验收任务并列出任务图",
            tools=parent,
            hooks=HookManager(),
            messages=messages,
            llm_caller=fake_lead_llm,
            workdir=demo_workdir,
            verbose=True,
        )
        print(f"\n[最终任务图输出]\n{reply}")
        print(f"[磁盘文件数] {len(list(tasks_dir.glob('task_*.json')))} 个 JSON 文件保存在 {tasks_dir}")


def main() -> int:
    """运行双轨计划（Todo 清单 + 持久 DAG 任务图）完整离线演示。"""
    configure_utf8_console()
    _demo_track1_todo_guard()
    _demo_track2_dag_task_store()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
