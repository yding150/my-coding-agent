"""
__main__.py - `coding_agent.teams` 多 Agent 团队协同与主循环收件箱唤醒离线演示入口

用法：
    uv run python -m coding_agent.teams
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import configure_utf8_console
from coding_agent.hooks import HookManager
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    queue_processor_loop,
    run_agent_turn,
)
from coding_agent.tasks import TaskStore
from coding_agent.teams import MessageBus, TeamManager, TeamProtocols


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


def _demo_part1_bus_and_protocol() -> None:
    """演示 M6.1：JSONL 邮箱线程唤醒与计划审批闸门状态机流转。"""
    print("=" * 70)
    print("演示第一部分：M6.1 离线 JSONL 邮箱 (MessageBus) 与团队控制协议 (TeamProtocols)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        bus = MessageBus(workdir=demo_workdir)
        protocols = TeamProtocols()

        delivered = []

        def _producer() -> None:
            time.sleep(0.08)
            bus.send(
                from_agent="lead",
                to_agent="alice",
                content="请先只读摸底 auth 模块并提交修改方案",
                msg_type="task_assignment",
                metadata={"priority": "high"},
            )
            bus.send(
                from_agent="bob",
                to_agent="alice",
                content="已将测试夹具路径整理好",
                msg_type="message",
            )

        sender_thread = threading.Thread(target=_producer, daemon=True)
        sender_thread.start()

        print(f"[邮箱位置] 收件箱目录收敛于: {bus.mailbox_root}")
        print("[阻塞等待] alice 正在 wait_for_messages() 休眠等待来信...")
        delivered.extend(bus.wait_for_messages("alice", timeout=2.0))
        sender_thread.join(timeout=1.0)

        print(f"[即时唤醒收信] 收到 {len(delivered)} 条消息:")
        for item in delivered:
            print(f"  - from={item['from']} type={item['type']} content={item['content']!r}")
        print(f"[消费即清空校验] peek('alice') = {bus.peek('alice')}")

        protocols.set_plan_requirement("alice", required=True)
        ok_read, _ = protocols.is_tool_allowed("alice", "read_file")
        ok_write_before, reason_before = protocols.is_tool_allowed("alice", "write_file")
        print(f"\n[计划闸门-审批前] read_file 放行={ok_read}, write_file 放行={ok_write_before}")
        print(f"  拦截原因: {reason_before}")

        req1 = protocols.submit_plan("alice", "方案v1：直接覆盖认证逻辑", task_id="task_a1b2c3d4")
        protocols.approve_plan(req1.request_id, approve=False, feedback="请补充向后兼容方案")
        print(
            f"[计划闸门-驳回后] 状态={protocols.get_plan_gate('alice')}, "
            f"write_file 放行={protocols.is_tool_allowed('alice', 'write_file')[0]}"
        )

        req2 = protocols.submit_plan("alice", "方案v2：保留旧 Token 校验并新增单元测试", task_id="task_a1b2c3d4")
        protocols.approve_plan(req2.request_id, approve=True, feedback="方案严谨，准予施工")
        print(
            f"[计划闸门-批准后] 状态={protocols.get_plan_gate('alice')}, "
            f"write_file 放行={protocols.is_tool_allowed('alice', 'write_file')[0]}, "
            f"feedback={req2.feedback!r}"
        )

        shutdown_req = protocols.create_shutdown_request("alice")
        ack_ok = protocols.acknowledge_shutdown(shutdown_req.request_id, "alice")
        print(f"[平滑关机握手] req_id={shutdown_req.request_id}, ack={ack_ok}, status={shutdown_req.status}")


def _demo_part2_teammate_and_lead_wakeup() -> None:
    """演示 M6.2：常驻队友线程提交计划、Lead 收件箱自动唤醒审批并解锁写入施工、平滑关机。"""
    print("\n" + "=" * 70)
    print("演示第二部分：M6.2 常驻队友线程与 Lead 收件箱自动唤醒闭环")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        task_store = TaskStore(demo_workdir / ".agent" / "tasks", workdir=demo_workdir)
        bus = MessageBus(workdir=demo_workdir)
        protocols = TeamProtocols()
        task = task_store.create("编写配置说明", "生成 TEAM_NOTE.md")

        mate_step = 0

        def mate_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal mate_step
            mate_step += 1
            if mate_step == 1:
                # 第 1 步尝试写文件 -> 触发计划审批闸门拦截
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "m_w1",
                            "name": "write_file",
                            "arguments": {"path": "TEAM_NOTE.md", "content": "draft\n"},
                        }
                    ]
                )
            if mate_step == 2:
                # 看到被拦截提示后调用 submit_plan 提交方案并挂起等审批
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "m_plan",
                            "name": "submit_plan",
                            "arguments": {"plan": "创建 TEAM_NOTE.md 并标注团队协同验证完成"},
                        }
                    ]
                )
            if mate_step == 3:
                # 获批并被 Lead 回信唤醒后，再次调用 write_file 成功写盘并调用 complete_task
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "m_w2",
                            "name": "write_file",
                            "arguments": {"path": "TEAM_NOTE.md", "content": "TEAM_OK\n"},
                        },
                        {
                            "id": "m_done",
                            "name": "complete_task",
                            "arguments": {"task_id": task.id},
                        },
                    ]
                )
            return _mock_msg(content="已获批并写入 TEAM_NOTE.md，任务完成。")

        team_mgr = TeamManager(
            bus=bus,
            task_store=task_store,
            protocols=protocols,
            workdir=demo_workdir,
            llm_caller=mate_llm,
        )
        lead_tools = create_default_tools(
            workdir=demo_workdir,
            task_store=task_store,
            team_mgr=team_mgr,
            bus=bus,
            protocols=protocols,
        )
        lead_messages = create_session_messages(workdir=demo_workdir)
        lead_lock = threading.RLock()
        stop_event = threading.Event()
        team_done = threading.Event()

        def lead_event_runner(prompt: str, *, is_scheduled: bool = False, is_team_event: bool = False):
            def lead_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
                latest = messages[-1]["content"]
                is_initial_event = messages[-1].get("role") == "user"
                # 1. 若收到计划审批请求，提取 req_id 调用 approve_plan 放行
                if "plan_approval_request" in prompt or "请求ID:" in prompt:
                    if is_initial_event:
                        req_id = protocols.plan_request_ids.get("alice", "")
                        return _mock_msg(
                            tool_calls=[
                                {
                                    "id": "lead_app",
                                    "name": "approve_plan",
                                    "arguments": {
                                        "request_id": req_id,
                                        "approve": True,
                                        "feedback": "方案合理，准予写入",
                                    },
                                }
                            ]
                        )
                    return _mock_msg(content="已批准 alice 的计划方案，等待执行完成汇报。")

                # 2. 收到队友成果汇报后，发送平滑关机指令结束演示
                if "成果回报" in prompt:
                    if is_initial_event:
                        return _mock_msg(
                            tool_calls=[
                                {
                                    "id": "lead_sd",
                                    "name": "request_shutdown",
                                    "arguments": {"teammate": "alice"},
                                }
                            ]
                        )
                    return _mock_msg(content="已收到 alice 成果汇报并发出平滑关机请求。")

                if "关机下线" in prompt:
                    team_done.set()
                    return _mock_msg(content="alice 已平滑下线，团队协同闭环完成。")

                return _mock_msg(content=f"已记录事件：{latest[:40]}")

            return run_agent_turn(
                prompt,
                tools=lead_tools,
                hooks=HookManager(),
                messages=lead_messages,
                llm_caller=lead_llm,
                workdir=demo_workdir,
                is_scheduled=is_scheduled,
                is_team_event=is_team_event,
                verbose=True,
            )

        consumer = threading.Thread(
            target=queue_processor_loop,
            args=(stop_event,),
            kwargs={
                "bus": bus,
                "lock": lead_lock,
                "turn_runner": lead_event_runner,
                "poll_interval": 0.05,
            },
            daemon=True,
        )
        consumer.start()

        try:
            with lead_lock:
                lead_tools.execute(
                    "spawn_teammate",
                    {
                        "name": "alice",
                        "role": "配置开发工程师",
                        "task_id": task.id,
                        "require_plan": True,
                    },
                )
            finished = team_done.wait(timeout=8.0)
            note_content = (demo_workdir / "TEAM_NOTE.md").read_text(encoding="utf-8").strip()
            print(
                f"\n[全链路校验] shutdown闭环={finished}, "
                f"任务状态={task_store.load(task.id).status}, "
                f"文件产物={note_content!r}"
            )
        finally:
            stop_event.set()
            team_mgr.shutdown_all()
            consumer.join(timeout=1.5)


def main() -> int:
    """运行 M6.1 与 M6.2 团队通信、审批闸门及常驻队友唤醒完整离线演示。"""
    configure_utf8_console()
    _demo_part1_bus_and_protocol()
    _demo_part2_teammate_and_lead_wakeup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
