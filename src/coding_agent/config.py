"""
config.py - 统一路径与环境配置底座模块

架构角色：
    作为整个 `coding_agent` 包的配置单一事实来源（Single Source of Truth），
    集中解析工作区目录 (`WORKDIR`)、内置技能目录以及 `.agent/` 下的全部运行态目录，
    并统一加载 `.env` 环境变量与运行时开关参数。

解决的核心痛点：
    本模块将所有运行态持久化与临时目录收拢到 `<WORKDIR>/.agent/` 下，避免污染业务代码仓库，
    同时通过 `build_agent_paths(workdir)` 支持测试隔离与未来会话实例化扩展。

暴露的核心对象：
    - `AgentPaths`: 运行态路径集合数据类（不可变）
    - `build_agent_paths`: 按指定工作区生成路径集合
    - `DEFAULT_PATHS` 与模块级路径常量 (`WORKDIR`, `AGENT_DIR`, `SKILLS_DIR`, `TASKS_DIR` 等)
    - `get_max_turns`, `get_goal_block_cap`, `get_goal_evaluator_model`
    - `format_config_summary`, `main`（命令行配置自检入口）
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

# 在导入配置阶段即加载 .env（override=False，避免覆盖测试或终端显式注入的环境变量）
load_dotenv(override=False)

# 本项目源码根目录：<repo>/src/coding_agent/config.py -> 上溯三级到 <repo>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DEFAULT_MAX_TURNS: int = 25
DEFAULT_GOAL_BLOCK_CAP: int = 8


def configure_utf8_console() -> None:
    """
    在 Windows 控制台下显式适配 UTF-8 标准输出/错误编码。

    Why:
        Windows 默认终端代码页常为 GBK，输出中文提示词、Emoji 或工具结果时极易抛出
        `UnicodeEncodeError`。测试框架（如 pytest）接管流对象时可能不支持 `reconfigure`，
        因此需要容错捕获。
    """
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


configure_utf8_console()


@dataclass(frozen=True)
class AgentPaths:
    """
    Coding Agent 工作区与运行态目录路径集合。

    设计说明：
        仅描述路径映射关系，构造实例时**不会**主动创建磁盘目录，
        由各业务模块按需懒创建（以兼容测试中“未写入前目录尚不存在”的断言）。
    """

    project_root: Path
    workdir: Path
    agent_dir: Path
    skills_dir: Path
    transcript_dir: Path
    tool_results_dir: Path
    memory_dir: Path
    tasks_dir: Path
    mailbox_dir: Path
    workflow_runtime_dir: Path

    def runtime_dirs(self) -> tuple[Path, ...]:
        """返回 `.agent/` 下的全部运行态子目录元组，便于批量巡检或初始化。"""
        return (
            self.transcript_dir,
            self.tool_results_dir,
            self.memory_dir,
            self.tasks_dir,
            self.mailbox_dir,
            self.workflow_runtime_dir,
        )

    def ensure_runtime_dirs(self) -> None:
        """显式创建 `.agent/` 及其全部运行态子目录。"""
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.runtime_dirs():
            directory.mkdir(parents=True, exist_ok=True)


def build_agent_paths(
    workdir: Optional[Path | str] = None,
    project_root: Optional[Path | str] = None,
) -> AgentPaths:
    """
    根据目标工作区构建统一收拢到 `.agent/` 的路径集合。

    :param workdir: 目标工作区路径；为 None 时默认使用当前工作目录 `Path.cwd()`。
    :param project_root: 项目仓库根目录；为 None 时默认使用本包所在的 `PROJECT_ROOT`。
    :return: 解析为绝对路径的 `AgentPaths` 实例。
    """
    resolved_root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    resolved_workdir = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
    agent_dir = resolved_workdir / ".agent"

    # 若工作区自身存在 skills/ 目录则优先使用，否则回退到随包分发的内置 skills/ 资产目录
    local_skills = resolved_workdir / "skills"
    skills_dir = local_skills if local_skills.is_dir() else (resolved_root / "skills")

    return AgentPaths(
        project_root=resolved_root,
        workdir=resolved_workdir,
        agent_dir=agent_dir,
        skills_dir=skills_dir,
        transcript_dir=agent_dir / "transcripts",
        tool_results_dir=agent_dir / "tool-results",
        memory_dir=agent_dir / "memory",
        tasks_dir=agent_dir / "tasks",
        mailbox_dir=agent_dir / "mailboxes",
        workflow_runtime_dir=agent_dir / "workflows",
    )


def _read_positive_int_env(name: str, default: int) -> int:
    """读取正整数环境变量；未配置、空字符串或非正整数时安全回退默认值。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def get_max_turns(default: int = DEFAULT_MAX_TURNS) -> int:
    """获取单次 Agent Turn 允许的最大工具循环轮数 (`MAX_TURNS`)。"""
    return _read_positive_int_env("MAX_TURNS", default)


def get_goal_block_cap(default: int = DEFAULT_GOAL_BLOCK_CAP) -> int:
    """获取 Goal Stop Hook 连续拦截熔断上限 (`GOAL_BLOCK_CAP`)。"""
    return _read_positive_int_env("GOAL_BLOCK_CAP", default)


def get_goal_evaluator_model() -> Optional[str]:
    """获取独立目标评估器指定模型 (`GOAL_EVALUATOR_MODEL_ID`)，未配置时返回 None。"""
    value = (os.getenv("GOAL_EVALUATOR_MODEL_ID") or "").strip()
    return value or None


# 模块级默认实例与路径常量，供 M1~M9 增量移植直接引用
DEFAULT_PATHS: AgentPaths = build_agent_paths()
WORKDIR: Path = DEFAULT_PATHS.workdir
AGENT_DIR: Path = DEFAULT_PATHS.agent_dir
SKILLS_DIR: Path = DEFAULT_PATHS.skills_dir
TRANSCRIPT_DIR: Path = DEFAULT_PATHS.transcript_dir
TOOL_RESULTS_DIR: Path = DEFAULT_PATHS.tool_results_dir
MEMORY_DIR: Path = DEFAULT_PATHS.memory_dir
TASKS_DIR: Path = DEFAULT_PATHS.tasks_dir
MAILBOX_DIR: Path = DEFAULT_PATHS.mailbox_dir
WORKFLOW_RUNTIME_DIR: Path = DEFAULT_PATHS.workflow_runtime_dir
SCHEDULED_TASKS_FILE: Path = AGENT_DIR / "scheduled_tasks.json"

MAX_AGENT_TURNS: int = get_max_turns()
GOAL_BLOCK_CAP: int = get_goal_block_cap()
GOAL_EVALUATOR_MODEL_ID: Optional[str] = get_goal_evaluator_model()


def _mask_secret(value: Optional[str]) -> str:
    """脱敏显示密钥配置，避免自检输出泄露完整 API Key。"""
    if not value:
        return "<未配置>"
    if len(value) <= 8:
        return f"***(已配置, 长度={len(value)})"
    return f"{value[:3]}***{value[-4:]}"


def format_config_summary(paths: Optional[AgentPaths] = None) -> str:
    """
    生成可读的配置与路径自检报告（包含工作区、`.agent/` 子目录与 LLM 环境配置）。

    :param paths: 可选路径集合，默认使用当前 `build_agent_paths()` 结果。
    :return: 格式化后的多行中文自检报告文本。
    """
    # 延迟导入 llm.client 以避免循环依赖，同时动态反映当前模块级配置状态
    from coding_agent.llm import client as llm_client

    active_paths = paths or build_agent_paths()
    lines = [
        "=" * 60,
        "Coding Agent 配置与运行态路径自检",
        "=" * 60,
        f"项目根目录 (PROJECT_ROOT) : {active_paths.project_root}",
        f"当前工作区 (WORKDIR)      : {active_paths.workdir}",
        f"运行态根目录 (AGENT_DIR)  : {active_paths.agent_dir}",
        f"技能资产目录 (SKILLS_DIR) : {active_paths.skills_dir}",
        "-" * 60,
        "运行态子目录（统一收拢于 .agent/ 下）:",
        f"  - transcripts      : {active_paths.transcript_dir}",
        f"  - tool-results     : {active_paths.tool_results_dir}",
        f"  - memory           : {active_paths.memory_dir}",
        f"  - tasks            : {active_paths.tasks_dir}",
        f"  - mailboxes        : {active_paths.mailbox_dir}",
        f"  - workflows        : {active_paths.workflow_runtime_dir}",
        "-" * 60,
        "大模型与运行时环境变量:",
        f"  - API_KEY          : {_mask_secret(llm_client.API_KEY)}",
        f"  - BASE_URL         : {llm_client.BASE_URL or '<OpenAI 默认地址>'}",
        f"  - DEFAULT_MODEL    : {llm_client.DEFAULT_MODEL}",
        f"  - EVALUATOR_MODEL  : {get_goal_evaluator_model() or '<复用主模型>'}",
        f"  - MAX_TURNS        : {get_max_turns()}",
        f"  - GOAL_BLOCK_CAP   : {get_goal_block_cap()}",
        "=" * 60,
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    M1.1 配置自检命令行入口：`uv run python -m coding_agent.config [--ping]`。

    :param argv: 命令行参数列表，默认从 `sys.argv[1:]` 读取。
    :return: 进程退出码（0 表示成功，非 0 表示连通性探测失败）。
    """
    parser = argparse.ArgumentParser(
        prog="python -m coding_agent.config",
        description="检查 Coding Agent 的 .agent/ 运行态路径收拢与 LLM 环境变量配置。",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="若已配置有效 API Key，发送一条极简测试消息验证模型接口连通性。",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(format_config_summary())

    if args.ping:
        from coding_agent.llm import call_llm

        print("\n[LLM 连通性自检] 正在发送最小探活请求...")
        try:
            msg = call_llm(
                messages=[{"role": "user", "content": "请仅回复 OK"}],
                max_tokens=16,
            )
            reply = (getattr(msg, "content", None) or "").strip()
            print(f"[LLM 连通性自检] 成功！模型回复: {reply!r}")
        except Exception as exc:
            print(f"[LLM 连通性自检] 失败: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
