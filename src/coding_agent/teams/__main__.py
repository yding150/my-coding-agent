"""
__main__.py - `coding_agent.teams` 离线 JSONL 邮箱与团队控制协议演示入口

用法：
    uv run python -m coding_agent.teams
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from coding_agent.config import configure_utf8_console
from coding_agent.teams import MessageBus, TeamProtocols


def main() -> int:
    """演示 `.agent/mailboxes/` 线程安全收发阻塞唤醒，以及计划审批闸门与平滑关机握手流转。"""
    configure_utf8_console()
    print("=" * 70)
    print("演示：M6.1 离线 JSONL 邮箱 (MessageBus) 与团队控制协议 (TeamProtocols)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        demo_workdir = Path(tmp).resolve()
        bus = MessageBus(workdir=demo_workdir)
        protocols = TeamProtocols()

        # 1. 多线程邮箱并发投递与条件变量低功耗唤醒演示
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

        # 2. TeamProtocols 计划审批闸门与平滑关机状态流转演示
        protocols.set_plan_requirement("alice", required=True)
        ok_read, _ = protocols.is_tool_allowed("alice", "read_file")
        ok_write_before, reason_before = protocols.is_tool_allowed("alice", "write_file")
        print(f"\n[计划闸门-审批前] read_file 放行={ok_read}, write_file 放行={ok_write_before}")
        print(f"  拦截原因: {reason_before}")

        req1 = protocols.submit_plan("alice", "方案v1：直接覆盖认证逻辑", task_id="task_a1b2c3d4")
        protocols.approve_plan(req1.request_id, approve=False, feedback="请补充向后兼容方案")
        print(f"[计划闸门-驳回后] 状态={protocols.get_plan_gate('alice')}, write_file 放行={protocols.is_tool_allowed('alice', 'write_file')[0]}")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
