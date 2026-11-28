"""
manager.py - 异步后台任务管理器（BackgroundTaskManager）

架构角色：
    作为 Coding Agent 运行时的异步命令行任务调度中心，把长耗时命令（构建、依赖安装、
    测试套件等）移出阻塞的主循环，使主 Agent 可在等待期间继续并行推进其他工作，
    并在任务结束后的下一轮大模型推理前自动回注 `<task_notification>` 通知。

解决的核心痛点：
    1. 非阻塞后台派发：`start()` 在 `RLock` 保护下通过子进程 + 守护监控线程立即返回 `bg_xxxx`，
       彻底避免长命令长时间冻结单线程工具循环；
    2. 协议安全通知注入：`inject_notifications()` 严守 OpenAI 消息角色时序规范——当最后一条
       消息为 `user` 时无缝并入正文，当最后一条为 `tool` 或 `assistant` 时自动另起新 `user` 消息；
    3. 全生命周期资源兜底：保留最大输出截断、超时自动清理整棵进程树、主动强杀与 `atexit`
       退出自动清理机制，防止 Windows/Linux 产生孤儿进程及管道泄漏。

暴露的核心类：
    - `BackgroundTaskManager`: 异步后台子进程生命周期管理器与通知注入器
"""

from __future__ import annotations

import atexit
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class BackgroundTaskManager:
    """
    异步后台任务管理器，负责长耗时命令行进程的后台异步执行、状态维护与事件通知收集。

    生命周期：
        随运行时初始化或按工作区创建，构造时注册 `atexit` 清理兜底；
        命令行工具或主循环关闭时应显式调用 `cleanup_all()` 终止所有未决子进程。
    """

    def __init__(self, workdir: Optional[Path] = None, max_output_chars: int = 50000) -> None:
        self.workdir: Path = (workdir or Path.cwd()).resolve()
        self.max_output_chars: int = max_output_chars

        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, str] = {}
        self._ready_queue: List[str] = []
        self._processes: Dict[str, subprocess.Popen[str]] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._counter: int = 0
        # Why: cleanup_all() 与 kill_task() 均会操作 _processes / tasks 并可能在同一线程内嵌套持锁，
        # 使用可重入 RLock 可防止未来重构复用时发生同线程自死锁。
        self._lock = threading.RLock()

        # 注册退出清理钩子，避免进程异常退出后遗留后台孤儿子进程
        atexit.register(self.cleanup_all)

    def start(self, command: str, workdir: Optional[Path] = None, timeout: int = 300) -> str:
        """
        在后台启动一个命令行任务，并立即返回任务 ID（如 `bg_0001`）。

        :param command: 待执行的 Shell 命令行文本。
        :param workdir: 工作区目录；默认使用管理器绑定的 `workdir`。
        :param timeout: 超时强杀秒数，默认 300 秒。
        :return: 唯一后台任务 ID。
        :raises ValueError: 当命令内容为空时抛出。
        :raises RuntimeError: 当底层子进程创建失败时抛出。
        """
        if not command or not command.strip():
            raise ValueError("后台执行的命令内容不能为空")

        cwd = (workdir or self.workdir).resolve()

        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"

            # 跨平台 Shell 指令包装
            if sys.platform == "win32":
                args = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
                creationflags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
                    else 0
                )
            else:
                args = ["bash", "-c", command]
                creationflags = 0

            try:
                proc: subprocess.Popen[str] = subprocess.Popen(
                    args,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # Why: 关闭标准输入流，防止后台 CLI 命令等待交互确认时无限挂死
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
            except Exception as exc:
                raise RuntimeError(f"启动后台子进程失败: {exc}") from exc

            self._processes[task_id] = proc
            self.tasks[task_id] = {
                "task_id": task_id,
                "command": command,
                "status": "running",
                "start_time": time.time(),
                "end_time": None,
                "exit_code": None,
            }

            worker = threading.Thread(
                target=self._worker,
                args=(task_id, proc, timeout),
                name=f"Worker-{task_id}",
                daemon=True,
            )
            self._threads[task_id] = worker
            worker.start()

        print(f"  \033[36m[Background 后台任务启动]\033[0m {task_id}: {command[:60]}...")
        return task_id

    def _worker(self, task_id: str, proc: subprocess.Popen[str], timeout: int) -> None:
        """后台监控线程：等待进程结束、读取输出并推入就绪队列。"""
        output = ""
        exit_code: Optional[int] = None
        status = "completed"

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            raw_out = (stdout or "") + (stderr or "")
            output = raw_out.strip()
            exit_code = proc.returncode
            if exit_code != 0:
                status = "failed"
        except subprocess.TimeoutExpired:
            self._terminate_proc(proc)
            output = f"执行超时错误: 任务运行超过 {timeout} 秒已被系统强制终止。"
            exit_code = -1
            status = "failed"
        except Exception as exc:
            output = f"后台任务执行发生未捕获异常: {type(exc).__name__}: {exc}"
            exit_code = -1
            status = "failed"
        finally:
            if len(output) > self.max_output_chars:
                output = (
                    output[: self.max_output_chars]
                    + f"\n... (输出已截断，仅保留前 {self.max_output_chars} 字符)"
                )

            final_status = status
            with self._lock:
                task = self.tasks.get(task_id)
                if task:
                    # Why: 若外层 kill_task() 已抢先标记 killed 并写入终止说明，
                    # 必须保留 killed 状态与原因，禁止被线程收尾覆盖为 failed/completed。
                    if task["status"] == "killed":
                        final_status = "killed"
                    else:
                        task["status"] = status
                        final_status = status
                        self.results[task_id] = output or "(命令执行完成，无输出内容)"
                    task["end_time"] = time.time()
                    if task["exit_code"] is None:
                        task["exit_code"] = exit_code

                if task_id not in self._ready_queue:
                    self._ready_queue.append(task_id)
                self._processes.pop(task_id, None)
                self._threads.pop(task_id, None)

            status_color = (
                "\033[32m"
                if final_status == "completed"
                else ("\033[33m" if final_status == "killed" else "\033[31m")
            )
            print(f"  {status_color}[Background 后台任务就绪]\033[0m {task_id} -> 状态: {final_status}")

    def collect_notifications(self) -> List[str]:
        """
        从就绪队列中拉取所有已完成的后台任务，并生成结构化 `<task_notification>` XML 通知。

        该方法每次调用后会自动清空已拉取队列，保证每条完成通知只注入一次。
        """
        with self._lock:
            if not self._ready_queue:
                return []

            collected_ids = list(self._ready_queue)
            self._ready_queue.clear()

            notifications: List[str] = []
            for tid in collected_ids:
                task = self.tasks.get(tid, {})
                result = self.results.get(tid, "")
                summary = result[:1000] + ("..." if len(result) > 1000 else "")

                block = (
                    f"<task_notification>\n"
                    f"  <task_id>{tid}</task_id>\n"
                    f"  <status>{task.get('status', 'unknown')}</status>\n"
                    f"  <command>{task.get('command', '')}</command>\n"
                    f"  <exit_code>{task.get('exit_code')}</exit_code>\n"
                    f"  <summary>{summary}</summary>\n"
                    f"</task_notification>"
                )
                notifications.append(block)

            return notifications

    def inject_notifications(self, messages: List[Dict[str, Any]]) -> int:
        """
        将所有已完成后台任务的通知安全注入到对话历史中，并返回注入的通知数量。

        :param messages: 当前会话消息列表（原地修改）。
        :return: 本次注入上下文的通知条数。
        """
        notifications = self.collect_notifications()
        if not notifications:
            return 0

        notification_text = "【系统后台任务完成通知】\n" + "\n\n".join(notifications)

        if not messages:
            messages.append({"role": "user", "content": notification_text})
            return len(notifications)

        last_msg = messages[-1]

        # Why: 若末尾本就是 user 消息，直接无缝拼接正文，既能防止连续两条 user 消息触发部分网关报错，
        # 也能在同一用户轮次直接补充异步到达的后台结果。
        if isinstance(last_msg, dict) and last_msg.get("role") == "user":
            orig_content = last_msg.get("content", "")
            if isinstance(orig_content, str):
                last_msg["content"] = f"{orig_content}\n\n{notification_text}"
            elif isinstance(orig_content, list):
                orig_content.append({"type": "text", "text": notification_text})
            else:
                last_msg["content"] = f"{str(orig_content)}\n\n{notification_text}"
        else:
            # Why: 若最后一条消息为 tool（工具回填结果）或 assistant，OpenAI 协议严禁篡改或把通知塞进
            # tool 块后紧接另一条非配对消息，因此必须另起一条新的 user 消息才能合法承载后台通知。
            messages.append({"role": "user", "content": notification_text})

        print(f"🔔 \033[36m[Background 通知注入] 已将 {len(notifications)} 个就绪后台任务通知并入上下文\033[0m")
        return len(notifications)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取指定后台任务的元信息与当前可见执行输出。"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            detail = dict(task)
            detail["output"] = self.results.get(task_id, "(任务仍在运行中，暂无最终输出)")
            return detail

    def list_tasks(self) -> List[Dict[str, Any]]:
        """获取所有后台任务的摘要列表。"""
        with self._lock:
            return [dict(t) for t in self.tasks.values()]

    def has_running_tasks(self) -> bool:
        """检查当前是否有仍在执行中的后台任务。"""
        with self._lock:
            return any(t.get("status") == "running" for t in self.tasks.values())

    def kill_task(self, task_id: str) -> bool:
        """主动强制终止一个正在运行的后台任务。"""
        with self._lock:
            proc = self._processes.get(task_id)
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task["status"] != "running":
                return True

            task["status"] = "killed"
            task["end_time"] = time.time()
            task["exit_code"] = -9

            if proc:
                self._terminate_proc(proc)
                self._processes.pop(task_id, None)

            self.results[task_id] = "(任务已被用户或 Agent 强制终止)"
            if task_id not in self._ready_queue:
                self._ready_queue.append(task_id)

            print(f"  \033[33m[Background 任务终止]\033[0m {task_id} 已被强行中止。")
            return True

    def _terminate_proc(self, proc: subprocess.Popen[str]) -> None:
        """安全终止子进程及整棵子孙进程树（兼容跨平台）。"""
        if proc.poll() is not None:
            return
        if sys.platform == "win32":
            # Why: Windows 下 PowerShell 启动的子命令会形成进程树；若仅 kill 父壳进程，
            # 子孙进程仍持有 stdout/stderr 管道写端，导致监控线程 communicate() 卡住。
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=0.5)
        except Exception:
            pass

    def cleanup_all(self) -> None:
        """清理所有正在运行的后台子进程并等待监控线程退出，防止孤儿进程与目录句柄泄漏。"""
        with self._lock:
            for tid, proc in list(self._processes.items()):
                task = self.tasks.get(tid)
                if task and task.get("status") == "running":
                    task["status"] = "killed"
                    task["exit_code"] = -9
                try:
                    self._terminate_proc(proc)
                except Exception:
                    pass
            self._processes.clear()
            threads_to_join = list(self._threads.values())

        # Why: 必须在释放 self._lock 后再 join 工作线程，否则刚被杀死的子进程触发 _worker finally
        # 块请求同一把锁时会发生死锁；等待其收尾可确保 Windows 彻底释放子进程持有的工作目录句柄。
        current = threading.current_thread()
        for worker in threads_to_join:
            if worker is not current and worker.is_alive():
                worker.join(timeout=1.0)
