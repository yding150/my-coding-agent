"""
bus.py - 基于 JSONL 文件收件箱的团队异步消息总线（MessageBus）

架构角色：
    为多 Agent 团队协同提供持久化、跨线程且零上下文污染的异步信箱通信基础设施，
    所有队友与 Lead 各自拥有独立的 `<workdir>/.agent/mailboxes/<agent>.jsonl` 收件箱。

解决的核心痛点：
    1. 上下文解耦隔离：通信过程不直接注入对方大模型 `messages[]`，防止长对话互相污染；
    2. 线程安全与低功耗唤醒：基于共享 `RLock` 的 `threading.Condition` 条件变量，
       支持空闲队友阻塞休眠等待，收到新信件瞬间唤醒；
    3. 消费即清空（Queue Semantics）：`read_inbox()` 原子读取后删除文件（按 `limit` 截断时
       仅写回剩余行），确保每条消息不重不漏；
    4. 运行态收敛与路径安全：邮箱目录默认收敛在 `.agent/mailboxes`，并拦截非法 Agent 名与越界目录。

暴露的核心类：
    - `DEFAULT_MAILBOX_DIR`: 默认邮箱相对路径（`".agent/mailboxes"`）
    - `MessageBus`: JSONL 收件箱读写、条件变量通知与阻塞等待总线
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MAILBOX_DIR: str = ".agent/mailboxes"


class MessageBus:
    """
    基于 `.agent/mailboxes/*.jsonl` 文件收件箱的跨 Agent 异步消息总线。

    生命周期：
        随运行时或测试工作区初始化，构造时不提前创建磁盘目录，仅在首次 `send()` 时懒创建。
    """

    def __init__(
        self,
        workdir: Optional[Path] = None,
        mailbox_dir: str | Path = DEFAULT_MAILBOX_DIR,
    ) -> None:
        self.workdir: Path = (workdir or Path.cwd()).resolve()
        candidate = Path(mailbox_dir)
        resolved_root = (
            candidate.resolve() if candidate.is_absolute() else (self.workdir / candidate).resolve()
        )
        # Why: 限制邮箱根目录位于工作区以内，防止传入 `../` 导致任意目录文件读写或删除
        if not resolved_root.is_relative_to(self.workdir):
            raise ValueError(f"邮箱目录超出工作区安全边界: {mailbox_dir}")
        self.mailbox_root: Path = resolved_root
        self._lock = threading.RLock()
        # Why: Condition 复用同一把 RLock，使得 send() 在发信后能安全 notify_all()，
        # 而 wait_for_messages() 在持锁状态下可重入调用 peek()/read_inbox() 且不发生自死锁。
        self._changed = threading.Condition(self._lock)

    def _path(self, agent: str) -> Path:
        """校验 Agent 名称并返回安全的 JSONL 收件箱路径。"""
        safe_name = "".join(c for c in agent if c.isalnum() or c in ("-", "_"))
        if not safe_name:
            raise ValueError(f"非法的 Agent 收件箱名称: {agent!r}")
        path = (self.mailbox_root / f"{safe_name}.jsonl").resolve()
        if not path.is_relative_to(self.mailbox_root):
            raise ValueError(f"非法的 Agent 收件箱路径: {agent!r}")
        return path

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        向目标 Agent 的 JSONL 收件箱追加一条消息，并唤醒所有等待中的接收线程。

        :param from_agent: 发送方标识（如 `"lead"`、`"alice"`）。
        :param to_agent: 接收方标识。
        :param content: 消息正文。
        :param msg_type: 控制/业务消息类型（如 `"message"`、`"result"`、`"plan_approval_request"` 等）。
        :param metadata: 附加结构化元数据（如 `request_id`、`task_id`）。
        :return: 已写入的消息字典副本。
        """
        msg: Dict[str, Any] = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "metadata": dict(metadata) if metadata else {},
            "timestamp": time.time(),
        }

        with self._changed:
            self.mailbox_root.mkdir(parents=True, exist_ok=True)
            path = self._path(to_agent)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self._changed.notify_all()

        return msg

    def peek(self, agent: str) -> bool:
        """非破坏性检查指定 Agent 是否有未读消息。"""
        with self._lock:
            path = self._path(agent)
            return path.is_file() and path.stat().st_size > 0

    def read_inbox(self, agent: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        读取并清空指定 Agent 的收件箱（支持通过 `limit` 分批消费）。

        :param agent: 目标 Agent 名称。
        :param limit: 可选单次最大读取条数；未读完的后续行会安全写回原收件箱文件。
        :return: 解析成功的未读消息字典列表。
        """
        with self._lock:
            path = self._path(agent)
            if not path.is_file():
                return []

            messages: List[Dict[str, Any]] = []
            remaining_lines: List[str] = []
            try:
                content = path.read_text(encoding="utf-8")
                lines = [line for line in content.splitlines() if line.strip()]
                for idx, line in enumerate(lines):
                    if limit is not None and len(messages) >= limit:
                        remaining_lines.extend(lines[idx:])
                        break
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 容错跳过偶发损坏行，避免整箱卡死
                        continue
            finally:
                try:
                    if remaining_lines:
                        path.write_text("\n".join(remaining_lines) + "\n", encoding="utf-8")
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass

            return messages

    def wait_for_messages(self, agent: str, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        阻塞等待消息到达；若已有未读消息立即消费返回，否则休眠直到通知或超时。

        :param agent: 目标 Agent 名称。
        :param timeout: 最长等待秒数；为 `None` 时持续阻塞直到收到消息。
        :return: 读取到的消息列表；超时无消息则返回空列表 `[]`。
        """
        # Why: 使用单调时钟 monotonic 计算剩余截止时间，防止系统校时跳变或伪唤醒导致提前/永久等待
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while not self.peek(agent):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return []
                self._changed.wait(remaining)
            return self.read_inbox(agent)
