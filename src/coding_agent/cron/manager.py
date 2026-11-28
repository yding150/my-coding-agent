"""
manager.py - 持久化 Cron 定时调度管理器（CronSchedulerManager）

架构角色：
    作为 Coding Agent 运行时的定时任务调度中枢，负责任务注册、取消、分钟级到期巡检、
    `.agent/scheduled_tasks.json` 原子持久化恢复以及 At-Least-Once 交付状态机。

解决的核心痛点：
    1. 运行态目录收敛：持久任务文件收拢保存至 `<workdir>/.agent/scheduled_tasks.json`（懒创建），
       使用同目录临时文件 + `os.replace()` 原子写盘，杜绝断电或崩溃破坏 JSON；
    2. 防同分钟重复触发：结合 `last_fired` 分钟戳与 `pending_delivery` 锁标记，即使巡检线程
       每秒轮询也不会在同一分钟内重复入队；
    3. At-Least-Once 交付与失败回滚：`consume_queue()` -> 执行 `run_agent_turn` -> 成功则
       `acknowledge_jobs()`（单次任务自动销毁，周期任务释放锁），异常则 `restore_jobs()` 放回队列；
    4. 内存与磁盘事务一致性：注册/取消/确认若写盘失败则自动回滚内存结构。

暴露的核心类：
    - `CronSchedulerManager`: Cron 任务注册、持久化与队列状态管理器
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from coding_agent.cron.job import CronJob, cron_matches, validate_cron

DEFAULT_SCHEDULED_TASKS_REL: str = ".agent/scheduled_tasks.json"


class CronSchedulerManager:
    """
    定时任务调度管理器，负责任务注册、时间匹配轮询、原子文件持久化与交付队列状态机。

    生命周期：
        随运行时或测试工作区实例化，启动时自动从 `.agent/scheduled_tasks.json` 恢复持久任务，
        并将未完成交付的任务重新放回 `cron_queue` 等待消费者线程投递。
    """

    def __init__(
        self,
        workdir: Optional[Path] = None,
        file_name: str = DEFAULT_SCHEDULED_TASKS_REL,
        durable_file: Optional[Path] = None,
    ) -> None:
        self.workdir: Path = (workdir or Path.cwd()).resolve()
        candidate = Path(durable_file) if durable_file is not None else (self.workdir / file_name)
        resolved = candidate.resolve()
        # Why: 约束持久调度文件路径不得逃逸目标工作区根目录，阻断恶意参数触发越界写盘
        if not resolved.is_relative_to(self.workdir):
            raise ValueError(f"定时任务持久化文件路径超出工作区边界: {candidate}")
        self.durable_file: Path = resolved

        self.scheduled_jobs: Dict[str, CronJob] = {}
        self.cron_queue: List[CronJob] = []
        # Why: schedule/poll/acknowledge/save 等均可能在同一调度线程内嵌套持锁，使用 RLock 防自死锁
        self._lock = threading.RLock()

        # 初始化时自动从磁盘加载持久化任务（不存在时保持惰性不提前创建文件）
        self.load_durable_jobs()

    def _generate_id(self) -> str:
        """生成随机全局唯一 ID，形如 `cron_a1b2c3d4`。"""
        for _ in range(100):
            jid = f"cron_{secrets.token_hex(4)}"
            if jid not in self.scheduled_jobs:
                return jid
        raise RuntimeError("无法为定时任务生成唯一 ID，碰撞达到上限")

    def schedule(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> CronJob:
        """
        注册一项新的定时任务。

        :param cron: 五段式标准 Cron 表达式。
        :param prompt: 到期唤醒 Agent 执行的任务指令。
        :param recurring: True 为周期性任务，False 为单次任务（消费确认后自动销毁）。
        :param durable: True 时持久化落盘至 `.agent/scheduled_tasks.json`。
        :raises ValueError: Cron 表达式非法或 prompt 为空时抛出。
        """
        err = validate_cron(cron)
        if err:
            raise ValueError(f"Cron 表达式不合法: {err}")

        if not prompt or not prompt.strip():
            raise ValueError("定时任务 prompt 内容不能为空")

        with self._lock:
            job = CronJob(
                id=self._generate_id(),
                cron=cron.strip(),
                prompt=prompt.strip(),
                recurring=recurring,
                durable=durable,
            )
            self.scheduled_jobs[job.id] = job

            if durable:
                try:
                    self.save_durable_jobs()
                except Exception:
                    self.scheduled_jobs.pop(job.id, None)
                    raise

            print(f"  \033[36m[Cron 任务注册]\033[0m {job.id}: '{job.cron}' -> {job.prompt[:50]}")
            return job

    def cancel(self, job_id: str) -> bool:
        """取消并移除指定的定时任务，并同步更新持久化文件。"""
        with self._lock:
            job = self.scheduled_jobs.get(job_id)
            if not job:
                return False

            old_queue = list(self.cron_queue)
            self.scheduled_jobs.pop(job_id, None)
            self.cron_queue = [q for q in self.cron_queue if q.id != job_id]

            if job.durable:
                try:
                    self.save_durable_jobs()
                except Exception:
                    # 写盘失败时原子回滚内存状态，保证视图与磁盘一致
                    self.scheduled_jobs[job_id] = job
                    self.cron_queue = old_queue
                    raise

            print(f"  \033[33m[Cron 任务取消]\033[0m 已成功移除任务 {job_id}")
            return True

    def list_jobs(self) -> List[CronJob]:
        """列出当前所有活跃的定时任务。"""
        with self._lock:
            return list(self.scheduled_jobs.values())

    def get_job(self, job_id: str) -> Optional[CronJob]:
        """获取指定 ID 的任务对象。"""
        with self._lock:
            return self.scheduled_jobs.get(job_id)

    def poll_due_jobs(self, moment: Optional[datetime] = None) -> List[CronJob]:
        """巡检并收集所有到达触发时刻的定时任务，推入交付就绪队列。"""
        now = moment or datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        newly_due: List[CronJob] = []

        with self._lock:
            for job in list(self.scheduled_jobs.values()):
                # Why: 双重防重门禁——已在队列等待交付（pending_delivery）或当前分钟已触发过，严禁重复入队
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue

                if cron_matches(job.cron, now):
                    old_pending = job.pending_delivery
                    old_last_fired = job.last_fired

                    job.pending_delivery = True
                    job.last_fired = minute_marker

                    if job.durable:
                        try:
                            self.save_durable_jobs()
                        except Exception as exc:
                            job.pending_delivery = old_pending
                            job.last_fired = old_last_fired
                            print(f"  \033[31m[Cron 入队持久化失败]\033[0m {job.id}: {exc}")
                            continue

                    self.cron_queue.append(job)
                    newly_due.append(job)
                    print(f"  \033[32m[Cron 任务到期触发]\033[0m {job.id}: '{job.prompt[:50]}'")

        return newly_due

    def has_queued_jobs(self) -> bool:
        """检查是否有等待交付给 Agent 的定时任务。"""
        with self._lock:
            return len(self.cron_queue) > 0

    def consume_queue(self) -> List[CronJob]:
        """提取当前所有就绪任务并清空队列。"""
        with self._lock:
            jobs = list(self.cron_queue)
            self.cron_queue.clear()
            return jobs

    def acknowledge_jobs(self, jobs: List[CronJob]) -> None:
        """
        确认大模型已成功接收并响应了定时任务：
        - 一次性任务 (`recurring=False`)：从调度表中彻底销毁；
        - 周期性任务 (`recurring=True`)：清除 `pending_delivery` 标记，等待下一次匹配。
        """
        with self._lock:
            removed_jobs: List[CronJob] = []
            updated_jobs: List[Tuple[CronJob, bool]] = []

            for delivered in jobs:
                current = self.scheduled_jobs.get(delivered.id)
                if not current:
                    continue

                updated_jobs.append((current, current.pending_delivery))
                if current.recurring:
                    current.pending_delivery = False
                else:
                    removed_jobs.append(current)
                    self.scheduled_jobs.pop(current.id, None)

            need_save = any(j.durable for j, _ in updated_jobs) or any(j.durable for j in removed_jobs)
            if need_save:
                try:
                    self.save_durable_jobs()
                except Exception:
                    for rj in removed_jobs:
                        self.scheduled_jobs[rj.id] = rj
                    for cj, old_pending in updated_jobs:
                        cj.pending_delivery = old_pending
                    raise

    def restore_jobs(self, jobs: List[CronJob]) -> None:
        """当调用大模型或派发出现网络/系统异常时，将任务安全回滚放回交付队列。"""
        with self._lock:
            existing_ids = {q.id for q in self.cron_queue}
            for j in jobs:
                current = self.scheduled_jobs.get(j.id)
                if current:
                    current.pending_delivery = True
                    if current.id not in existing_ids:
                        self.cron_queue.append(current)
                        existing_ids.add(current.id)

    def poll(self, moment: Optional[datetime] = None) -> List[CronJob]:
        """别名兼容方法，等价于 `poll_due_jobs`。"""
        return self.poll_due_jobs(moment)

    def get_pending_prompt(self, timeout: float = 0.0) -> Optional[str]:
        """别名兼容方法：若队列中有就绪任务，弹出并返回其 `prompt`。"""
        _ = timeout
        with self._lock:
            if self.cron_queue:
                job = self.cron_queue.pop(0)
                self._compat_delivering = job
                return job.prompt
        return None

    def restore_pending_prompt(self, prompt: str) -> None:
        """别名兼容方法：执行未成功时安全放回队列。"""
        with self._lock:
            job = getattr(self, "_compat_delivering", None)
            if job and job.prompt == prompt:
                self.restore_jobs([job])
                self._compat_delivering = None

    def save_durable_jobs(self) -> None:
        """
        将所有 `durable=True` 的定时任务原子写入 `.agent/scheduled_tasks.json`。

        采用同目录临时文件 + `os.replace()` 机制，彻底避免中途断电或崩溃损坏原文件。
        """
        with self._lock:
            self.durable_file.parent.mkdir(parents=True, exist_ok=True)
            durable_list = [
                job.to_dict()
                for job in self.scheduled_jobs.values()
                if job.durable
            ]
            tmp_file = self.durable_file.with_name(
                f"{self.durable_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                tmp_file.write_text(
                    json.dumps(durable_list, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp_file, self.durable_file)
            finally:
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass

    def load_durable_jobs(self) -> None:
        """从磁盘加载持久化任务，并恢复崩溃前未交付的 `pending_delivery` 队列项。"""
        if not self.durable_file.exists():
            return

        try:
            content = self.durable_file.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            if not isinstance(data, list):
                print(f"  \033[33m[Cron 恢复警告]\033[0m {self.durable_file.name} 格式非列表，已跳过")
                return
        except Exception as exc:
            print(f"  \033[33m[Cron 恢复警告]\033[0m 读取 {self.durable_file.name} 失败: {exc}")
            return

        loaded_count = 0
        with self._lock:
            for item in data:
                try:
                    job = CronJob.from_dict(item)
                    if validate_cron(job.cron):
                        continue
                    if not job.id.startswith("cron_") or not job.prompt.strip():
                        continue
                    self.scheduled_jobs[job.id] = job
                    if job.pending_delivery:
                        self.cron_queue.append(job)
                    loaded_count += 1
                except Exception:
                    continue

        if loaded_count > 0:
            print(f"  \033[36m[Cron 任务恢复]\033[0m 已成功恢复 {loaded_count} 个持久化定时任务。")
