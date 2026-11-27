"""
store.py - 持久化 DAG 任务依赖图系统（TaskStore）与任务图工具集

架构角色：
    承担双轨计划体系中的“第二轨：跨会话持久化 DAG 任务依赖图”，将 Task1 (`TodoManager`)
    的内存短期步骤升级为“磁盘 JSON 持久化（收拢于 `.agent/tasks/`）+ 有向无环依赖图（DAG）
    + 认领/完成状态机”的可协调任务系统，并在 `runtime.py` 中直接挂载至父级 `parent_tools`。

解决的核心痛点：
    1. 跨会话断点恢复：每条任务写入 `.agent/tasks/task_<8位hex>.json`，上下文压缩或重启后仍可
       通过 `list_tasks` / `get_task` 读取完整验收说明与依赖状态；
    2. 两阶段建图防幻觉（Two-Phase DAG）：禁止在 `create_task` 中猜测未知 ID 填写依赖，
       必须先创建节点获取真实随机 ID，再通过 `update_task` 全量预校验 + DFS 判环建立边；
    3. 原子门禁与自动解锁：认领前检查前置 `blockedBy` 是否全部 `completed`，完成后校验
       `owner` 归属权（支持 Lead/Agent 终审）并自动计算汇报新解锁的下游任务；
    4. 安全路径隔离：严格校验 `TASK_ID_PATTERN` 及工作区 `is_relative_to` 边界，阻断路径穿越与软链接逃逸。

暴露的核心接口：
    - 常量与实体：`TASK_ID_PATTERN`, `TASK_STATUSES`, `Task`
    - 核心存储器：`TaskStore`
    - 工具注册函数：`register_task_tools`（挂载 `create_task`, `update_task`, `list_tasks`, `get_task`, `claim_task`, `complete_task`）
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from coding_agent.tools.manager import ToolManager

TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")
TASK_STATUSES = ("pending", "in_progress", "completed")


@dataclass
class Task:
    """单条持久化任务实体。"""

    id: str  # 形如 "task_a1b2c3d4"（task_ + 8位小写十六进制）
    subject: str  # 任务简短标题
    description: str  # 详细验收要求（跨会话恢复时通过 get_task 读取）
    status: str  # "pending" | "in_progress" | "completed"
    owner: Optional[str]  # 认领该任务的 Agent 标识，未认领为 None
    blockedBy: List[str]  # 前置阻塞任务 ID 列表


class TaskStore:
    """
    持久化 DAG 任务图存储器。

    生命周期：
        在运行时启动或工作区初始化时构造（构造阶段不主动建目录，首次落盘时按需懒创建），
        为父代理及后续常驻团队提供任务创建、依赖维护、状态流转与下游解锁检测能力。
    """

    def __init__(self, directory: Path, workdir: Optional[Path] = None) -> None:
        self.directory: Path = Path(directory)
        self.workdir: Path = (workdir or self.directory.parent).resolve()

    # ----------------------------------------------------
    # 1. 安全边界与基础读写（Storage & Path Guard）
    # ----------------------------------------------------
    def _root(self, create: bool = False) -> Path:
        """校验任务存储目录未逃逸工作区边界（防符号链接穿越），按需创建目录。"""
        root = self.directory.resolve()
        # Why: 先 resolve() 再判定 is_relative_to，可同时拦截 `../` 越界与目录软链接逃逸
        if not root.is_relative_to(self.workdir):
            raise ValueError("任务存储目录超出工作区安全边界")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def _path(self, task_id: str, create_root: bool = False) -> Path:
        """校验 `task_id` 格式并返回安全的 JSON 文件路径。"""
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"非法的任务 ID: {task_id!r}")
        root = self._root(create=create_root)
        path = (root / f"{task_id}.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"非法的任务 ID: {task_id!r}")
        return path

    def exists(self, task_id: str) -> bool:
        """判断指定 `task_id` 对应的任务文件是否存在。"""
        return self._path(task_id).is_file()

    def save(self, task: Task) -> None:
        """将任务实体持久化写入对应 JSON 文件。"""
        self._path(task.id, create_root=True).write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, task_id: str) -> Task:
        """读取并校验单个任务文件。"""
        data = json.loads(self._path(task_id).read_text(encoding="utf-8"))
        task = Task(**data)
        if task.id != task_id:
            raise ValueError(f"任务文件内的 ID ({task.id}) 与文件名 ({task_id}) 不一致")
        if task.status not in TASK_STATUSES:
            raise ValueError(f"非法的任务状态: {task.status}")
        return task

    def list(self) -> List[Task]:
        """按文件名排序列出所有已持久化的任务。"""
        if not self.directory.exists():
            return []
        root = self._root()
        return [self.load(path.stem) for path in sorted(root.glob("task_*.json"))]

    # ----------------------------------------------------
    # 2. 两阶段建图（Create & Update Dependencies）
    # ----------------------------------------------------
    def create(self, subject: str, description: str = "") -> Task:
        """创建新任务节点，采用排他写入（`'x'` 模式）避免随机 ID 碰撞覆写。"""
        clean_subject = subject.strip()
        if not clean_subject:
            raise ValueError("任务标题 (subject) 不能为空")

        self._root(create=True)
        for _ in range(100):
            task = Task(
                id=f"task_{secrets.token_hex(4)}",
                subject=clean_subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            try:
                # Why: 使用 "x" 排他创建模式而非先 exists 再 write，消除并发/碰撞窗口覆写已有任务文件
                with self._path(task.id, create_root=True).open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(asdict(task), handle, ensure_ascii=False, indent=2)
                return task
            except FileExistsError:
                continue
        raise RuntimeError("无法分配唯一的任务 ID（重试次数耗尽）")

    def _depends_on(self, task_id: str, target_id: str) -> bool:
        """DFS 判断 `task_id` 是否直接或传递依赖于 `target_id`（用于依赖判环）。"""
        pending = [task_id]
        visited = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self.load(current).blockedBy)
        return False

    def update_dependencies(
        self, task_id: str, add_blocked_by: List[str]
    ) -> Task:
        """
        为任务添加前置依赖（`addBlockedBy`）。

        采用“先全量校验、后原子保存”策略，失败时绝不产生部分写入。
        """
        if not isinstance(add_blocked_by, list):
            raise ValueError("addBlockedBy 必须是任务 ID 列表")

        task = self.load(task_id)
        # Why: 一旦任务被认领进入执行或已完成，前置依赖拓扑就已冻结，禁止中途改图破坏状态机
        if task.status != "pending" or task.owner is not None:
            raise ValueError(
                f"任务 {task_id} 仅在 pending 且无人认领状态下允许更新依赖"
            )

        dependencies = list(dict.fromkeys(add_blocked_by))
        # Why: 在修改 task.blockedBy 前完成全部合法性与 DFS 判环检查，保证任一异常都不脏写磁盘
        for dependency in dependencies:
            if dependency == task_id:
                raise ValueError("任务不能依赖自身")
            if not self.exists(dependency):
                raise ValueError(f"未找到前置依赖任务: {dependency}")
            if dependency not in task.blockedBy and self._depends_on(
                dependency, task_id
            ):
                raise ValueError(f"检测到依赖环路: {task_id} -> {dependency}")

        task.blockedBy.extend(
            dependency
            for dependency in dependencies
            if dependency not in task.blockedBy
        )
        self.save(task)
        return task

    # ----------------------------------------------------
    # 3. 依赖检查、认领与完成解锁（DAG Gate & Lifecycle）
    # ----------------------------------------------------
    def incomplete_dependencies(self, task: Task) -> List[str]:
        """返回尚未处于 `completed` 状态的前置依赖 ID 列表。"""
        incomplete: List[str] = []
        for dependency in task.blockedBy:
            try:
                if self.load(dependency).status != "completed":
                    incomplete.append(dependency)
            except (FileNotFoundError, ValueError):
                incomplete.append(dependency)
        return incomplete

    def can_start(self, task_id: str) -> bool:
        """判断指定任务的所有前置依赖是否均已完成。"""
        return not self.incomplete_dependencies(self.load(task_id))

    def claim(self, task_id: str, owner: str = "agent") -> str:
        """认领任务：仅当状态为 `pending` 且所有 `blockedBy` 均已 `completed` 时放行。"""
        task = self.load(task_id)
        if task.status != "pending":
            return f"任务 {task_id} 当前状态为 {task.status}，无法认领"

        dependencies = self.incomplete_dependencies(task)
        if dependencies:
            return f"任务 {task_id} 尚被前置任务阻塞 (Blocked by): {dependencies}"

        task.owner = owner
        task.status = "in_progress"
        self.save(task)
        print(f"  \033[36m[Task 认领]\033[0m {task.subject} -> in_progress (owner: {owner})")
        return f"已认领任务 {task.id} ({task.subject})"

    def complete(
        self,
        task_id: str,
        owner: str = "agent",
        allow_admin: bool = False,
    ) -> str:
        """完成任务：校验状态与 `owner` 归属，并统计刚刚被解锁的下游任务。"""
        task = self.load(task_id)
        if task.status not in ("in_progress", "pending"):
            return f"任务 {task_id} 当前状态为 {task.status}，无法标记完成"

        is_admin = allow_admin and owner in ("agent", "lead")
        if task.owner != owner and not is_admin:
            return f"任务 {task_id} 当前归属 {task.owner}，而非 {owner}，无权完成"

        # Why: 在更新状态前先采集原本已可执行（ready_before）的带依赖任务集合，
        # 这样完成后即可精准识别出“因本次完成而刚刚新解锁”的下游节点，避免重复播报。
        ready_before = {
            candidate.id
            for candidate in self.list()
            if candidate.status == "pending"
            and candidate.blockedBy
            and self.can_start(candidate.id)
        }

        task.status = "completed"
        if task.owner is None:
            task.owner = owner
        self.save(task)

        unblocked = [
            candidate.subject
            for candidate in self.list()
            if candidate.status == "pending"
            and candidate.blockedBy
            and candidate.id not in ready_before
            and self.can_start(candidate.id)
        ]

        print(f"  \033[32m[Task 完成]\033[0m {task.subject}")
        message = f"已完成任务 {task.id} ({task.subject})"
        if unblocked:
            joined = ", ".join(unblocked)
            message += f"\n已解锁下游任务 (Unblocked): {joined}"
            print(f"  \033[33m[Task 解锁]\033[0m {joined}")
        return message

    # ----------------------------------------------------
    # 4. 格式化展示（Inspect & List）
    # ----------------------------------------------------
    def get_detail(self, task_id: str) -> str:
        """返回单个任务的完整 JSON 详情（用于跨会话恢复上下文）。"""
        return json.dumps(asdict(self.load(task_id)), ensure_ascii=False, indent=2)

    def render_list(self) -> str:
        """格式化输出当前所有任务的状态、归属与阻塞关系。"""
        tasks = self.list()
        if not tasks:
            return "当前暂无持久化任务。请使用 create_task 创建任务节点。"

        status_markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        lines: List[str] = []
        for task in tasks:
            marker = status_markers.get(task.status, "[?]")
            deps = (
                f" (blockedBy: {', '.join(task.blockedBy)})"
                if task.blockedBy
                else ""
            )
            owner = f" [owner: {task.owner}]" if task.owner else ""
            lines.append(
                f"{marker} {task.id}: {task.subject} [{task.status}]{owner}{deps}"
            )
        return "\n".join(lines)


def register_task_tools(
    manager: ToolManager,
    task_store: TaskStore,
    default_owner: str = "agent",
) -> ToolManager:
    """
    一键将持久化任务图系统的 6 个核心工具挂载到指定的 `ToolManager` 中：
    - `create_task`
    - `update_task`
    - `list_tasks`
    - `get_task`
    - `claim_task`
    - `complete_task`
    """

    def run_create_task(subject: str, description: str = "") -> str:
        task = task_store.create(subject, description)
        print(f"  \033[36m[Task 创建]\033[0m {task.id}: {task.subject}")
        return f"已创建任务 {task.id}: {task.subject}"

    def run_update_task(task_id: str, addBlockedBy: List[str]) -> str:
        task = task_store.update_dependencies(task_id, addBlockedBy)
        deps = ", ".join(task.blockedBy) or "(无)"
        print(f"  \033[36m[Task 依赖]\033[0m {task.subject} blockedBy: {deps}")
        return f"已更新任务 {task.id} 的前置依赖 blockedBy: {deps}"

    def run_list_tasks() -> str:
        return task_store.render_list()

    def run_get_task(task_id: str) -> str:
        return task_store.get_detail(task_id)

    def run_claim_task(task_id: str) -> str:
        return task_store.claim(task_id, owner=default_owner)

    def run_complete_task(task_id: str) -> str:
        # 主代理（default_owner 为 agent 或 lead）具备协调管理权限，可对团队任务进行终审验收完成
        return task_store.complete(
            task_id,
            owner=default_owner,
            allow_admin=(default_owner in ("agent", "lead")),
        )

    # Why: create_task Schema 显式禁止 additionalProperties 且不提供 blockedBy 字段，
    # 强迫模型走“先创建拿真实 task_xxxxxxxx ID，再 update_task 连线”的两阶段建图范式。
    manager.register(
        run_create_task,
        name="create_task",
        description="创建一个持久化任务节点并返回运行时生成的唯一任务 ID（task_xxxxxxxx）。",
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "简短明确的任务标题"},
                "description": {"type": "string", "description": "任务详细说明与验收要求"},
            },
            "required": ["subject"],
            "additionalProperties": False,
        },
    )
    manager.register(
        run_update_task,
        name="update_task",
        description="使用 create_task 返回的真实任务 ID，为指定任务添加前置依赖（addBlockedBy）。",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "pattern": r"^task_[0-9a-f]{8}$",
                    "description": "待添加依赖的目标任务 ID",
                },
                "addBlockedBy": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": r"^task_[0-9a-f]{8}$",
                    },
                    "minItems": 1,
                    "description": "前置阻塞任务的 ID 列表",
                },
            },
            "required": ["task_id", "addBlockedBy"],
            "additionalProperties": False,
        },
    )
    manager.register(
        run_list_tasks,
        name="list_tasks",
        description="列出 .agent/tasks/ 中所有持久化任务及其状态、归属者与前置依赖关系。",
        parameters={"type": "object", "properties": {}},
    )
    manager.register(
        run_get_task,
        name="get_task",
        description="根据任务 ID 读取完整任务 JSON 详情（含 description 等）。",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "要查询的任务 ID"},
            },
            "required": ["task_id"],
        },
    )
    manager.register(
        run_claim_task,
        name="claim_task",
        description="认领一个前置依赖已全部完成的 pending 任务（状态流转为 in_progress）。",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "要认领的任务 ID"},
            },
            "required": ["task_id"],
        },
    )
    manager.register(
        run_complete_task,
        name="complete_task",
        description="将当前已认领的任务标记为 completed，并自动检查解锁下游依赖任务。",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "已完成的任务 ID"},
            },
            "required": ["task_id"],
        },
    )
    return manager
