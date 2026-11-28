"""
__main__.py - `coding_agent.cron` 定时调度与并发唤醒线程离线演示入口

用法：
    uv run python -m coding_agent.cron
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from coding_agent.config import configure_utf8_console
from coding_agent.cron import CronSchedulerManager
from coding_agent.hooks import HookManager
from coding_agent.runtime import (
    create_default_tools,
    create_session_messages,
    queue_processor_loop,
    run_agent_turn,
)


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
    """演示 `schedule_cron` 注册、`.agent/scheduled_tasks.json` 持久化与 `agent_lock` 空闲唤醒闭环。"""
    configure_utf8_console()
    print("=" * 70)
    print("演示：M5.2 Cron 定时调度引擎与后台空闲唤醒 (queue_processor_loop + agent_lock)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        cron_mgr = CronSchedulerManager(workdir=demo_workdir)
        tools = create_default_tools(workdir=demo_workdir, cron_mgr=cron_mgr)
        messages = create_session_messages(workdir=demo_workdir)
        demo_lock = threading.RLock()
        stop_event = threading.Event()
        wakeup_done = threading.Event()

        reg_step = 0

        def register_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
            nonlocal reg_step
            reg_step += 1
            if reg_step == 1:
                return _mock_msg(
                    tool_calls=[
                        {
                            "id": "cron_call_1",
                            "name": "schedule_cron",
                            "arguments": {
                                "cron": "* * * * *",
                                "prompt": "巡检工作区健康状态并报告",
                                "recurring": False,
                                "durable": True,
                            },
                        }
                    ]
                )
            return _mock_msg(content="已成功注册一次性定时任务（将落盘至 .agent/scheduled_tasks.json）。")

        with demo_lock:
            run_agent_turn(
                "帮我注册一个每分钟巡检的一次性定时任务",
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=register_llm,
                workdir=demo_workdir,
                cron_mgr=cron_mgr,
                lock=demo_lock,
                verbose=True,
            )

        print(f"[持久化校验] 磁盘文件存在: {cron_mgr.durable_file.exists()} -> {cron_mgr.durable_file}")

        # 触发到期巡检并将就绪任务交由后台消费者线程异步空闲唤醒
        due = cron_mgr.poll_due_jobs(datetime.now())
        print(f"[触发校验] 到期入队任务数: {len(due)}")

        def wakeup_runner(prompt: str, *, is_scheduled: bool = False):
            def woke_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
                return _mock_msg(content="巡检完成：所有本地模块均健康运行。")

            res = run_agent_turn(
                prompt,
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=woke_llm,
                workdir=demo_workdir,
                cron_mgr=cron_mgr,
                is_scheduled=is_scheduled,
                verbose=True,
            )
            wakeup_done.set()
            return res

        # 刻意短暂占住 demo_lock，验证后台消费者在会话忙碌时退让，释放后立即空闲唤醒
        demo_lock.acquire()
        worker = threading.Thread(
            target=queue_processor_loop,
            args=(stop_event,),
            kwargs={
                "cron_mgr": cron_mgr,
                "lock": demo_lock,
                "turn_runner": wakeup_runner,
                "poll_interval": 0.05,
            },
            daemon=True,
        )
        worker.start()
        time.sleep(0.12)
        print(f"[互斥锁退让校验] 前台持锁期间是否提前触发唤醒: {wakeup_done.is_set()}")
        demo_lock.release()

        triggered = wakeup_done.wait(timeout=3.0)
        stop_event.set()
        worker.join(timeout=1.0)

        print(f"[空闲唤醒校验] 前台释放锁后唤醒成功: {triggered}")
        print(f"[一次性任务自动销毁校验] 剩余活跃 Cron 数: {len(cron_mgr.list_jobs())}")

    return 0 if triggered and len(cron_mgr.list_jobs()) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
