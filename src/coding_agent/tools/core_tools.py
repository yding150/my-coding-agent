"""
core_tools.py - 核心基础行动力工具库

架构角色：
    提供 Coding Agent 感知与操作本地代码仓库的 5 大最通用基础工具：
    `run_shell`、`read_file`、`write_file`、`edit_file`、`glob_files`，
    以及保障工作区沙箱安全边界的路径解析函数 `_safe_path`。

解决的核心痛点：
    1. 统一把文件读写与目录匹配约束在工作区 (`WORKDIR`) 之内，拦截 `../` 或越界路径访问；
    2. 对命令行执行增加超时强杀进程树保护，防止子进程常驻挂起导致主循环永久阻塞；
    3. 对文件读取与文件列表检索增加行/条数折叠保护，避免单次大输出撑爆模型上下文。

暴露的核心接口：
    - 路径安全：`_safe_path`, `_resolve_workdir`
    - 5 大核心工具：`run_shell`, `read_file`, `write_file`, `edit_file`, `glob_files`
    - Schema 与挂载：`CORE_SCHEMAS`, `register_core_tools`
"""

from __future__ import annotations

import glob as g
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from coding_agent.config import configure_utf8_console
from coding_agent.tools.manager import ToolManager

configure_utf8_console()


def _resolve_workdir(workdir: Optional[Path] = None) -> Path:
    """解析当前生效的工作区绝对基准路径（未显式传入时默认当前工作目录）。"""
    return (workdir or Path.cwd()).resolve()


def _safe_path(p: str, workdir: Optional[Path] = None) -> Path:
    """
    严格校验并解析用户/模型传入的路径，杜绝路径穿透（Path Traversal）越界逃逸。

    Why:
        先对 `base` 与 `base / p` 执行 `.resolve()` 消除 `..` 与符号链接，再使用
        `Path.is_relative_to(base)` 而非字符串前缀匹配，避免 `/tmp/work_secret`
        伪造绕过 `/tmp/work` 边界检查。

    :param p: 模型或用户传入的相对（或工作区内）路径字符串。
    :param workdir: 目标工作区根目录，默认当前工作目录。
    :return: 校验通过后的绝对 `Path` 对象。
    :raises ValueError: 当解析后的目标路径不在工作区根目录之下时抛出。
    """
    base = _resolve_workdir(workdir)
    target = (base / p).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"访问被拒绝: 检测到路径越界逃逸 '{p}'")
    return target


def run_shell(command: str, workdir: Optional[Path] = None, timeout: int = 120) -> str:
    """
    在工作区目录下执行 PowerShell 命令行指令，带超时熔断与进程树清理。

    Why:
        1. 传入 `stdin=subprocess.DEVNULL`：防止某些交互式 CLI 子命令向用户终端请求输入而无限挂起；
        2. 超时时在 Windows 下使用 `taskkill /F /T` 清理整棵进程树：因为单纯 `proc.kill()`
           仅杀死父壳进程，子孙进程仍可能继续持有 stdout/stderr 管道句柄导致 `communicate()` 卡死。

    :param command: 待执行的 PowerShell 命令文本。
    :param workdir: 命令执行的工作目录，默认当前目录。
    :param timeout: 超时秒数，默认 120 秒。
    :return: 命令输出文本或超时/异常说明。
    """
    cwd = _resolve_workdir(workdir)
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            out = (stdout or "") + (stderr or "")
            return out.strip() if out.strip() else "(命令执行成功，无输出内容)"
        except subprocess.TimeoutExpired:
            # 强行终止包含子孙进程的完整进程树，防止后台服务继承管道句柄导致挂起
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    proc.kill()
            else:
                proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=2)
                out = (stdout or "") + (stderr or "")
            except Exception:
                out = ""
            return f"执行超时错误: 命令执行超过 {timeout} 秒已强行中断。" + (
                f"\n部分输出: {out}" if out else ""
            )
    except Exception as e:
        return f"命令行执行异常: {str(e)}"


def read_file(path: str, limit: Optional[int] = 100, workdir: Optional[Path] = None) -> str:
    """
    读取工作区内文本文件内容，支持最大行数折叠保护。

    Why:
        默认截断前 `limit` 行，避免模型误读日志文件或大型生成物时一次性耗尽上下文窗口。

    :param path: 文件相对路径。
    :param limit: 最多返回行数（默认 100 行）；传 None 或 0 表示不限制。
    :param workdir: 工作区根目录。
    :return: 文件文本内容或错误说明。
    """
    try:
        file_path = _safe_path(path, workdir=workdir)
        if not file_path.is_file():
            return f"错误: 文件不存在 '{path}'"
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... (剩余 {len(lines) - limit} 行已省略)"]
        return "\n".join(lines)
    except Exception as e:
        return f"读取文件失败: {str(e)}"


def write_file(path: str, content: str, workdir: Optional[Path] = None) -> str:
    """
    创建或覆盖写入工作区内文件（自动递归创建缺失父目录）。

    :param path: 待写入的文件相对路径。
    :param content: 完整文件内容文本。
    :param workdir: 工作区根目录。
    :return: 写入成功提示或失败原因。
    """
    try:
        file_path = _safe_path(path, workdir=workdir)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"成功写入文件: {path} (共 {len(content)} 字符)"
    except Exception as e:
        return f"写入文件失败: {str(e)}"


def edit_file(path: str, old_text: str, new_text: str, workdir: Optional[Path] = None) -> str:
    """
    在工作区内文件中精确定位首处匹配旧文本并替换为新文本。

    :param path: 目标文件相对路径。
    :param old_text: 待替换的原文本片段。
    :param new_text: 替换后的新文本片段。
    :param workdir: 工作区根目录。
    :return: 更新结果或未匹配/异常提示。
    """
    try:
        file_path = _safe_path(path, workdir=workdir)
        if not file_path.is_file():
            return f"错误: 待编辑的文件不存在 '{path}'"
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"错误: 在文件 '{path}' 中未找到匹配的原文本，未能执行替换。"
        new_content = text.replace(old_text, new_text, 1)
        file_path.write_text(new_content, encoding="utf-8")
        return f"成功更新文件: {path}"
    except Exception as e:
        return f"编辑失败: {str(e)}"


def glob_files(pattern: str, workdir: Optional[Path] = None) -> str:
    """
    按通配符模式检索工作区内文件列表（支持 `**` 递归，带 100 条上限折叠）。

    :param pattern: 通配符模式（如 `'**/*.py'`）。
    :param workdir: 工作区根目录。
    :return: 换行拼接的相对路径列表或提示文本。
    """
    cwd = _resolve_workdir(workdir)
    try:
        matches = sorted(
            {
                match
                for match in g.glob(pattern, root_dir=str(cwd), recursive=True)
                # 二次校验符号链接解析后仍处于工作区内，防止目录搜索越界
                if (cwd / match).resolve().is_relative_to(cwd)
            }
        )
        shown = matches[:100]
        if len(matches) > 100:
            shown.append(f"... (共找到 {len(matches)} 个文件，已省略超出部分)")
        return "\n".join(shown) if shown else "(未找到匹配文件)"
    except Exception as e:
        return f"搜索文件失败: {str(e)}"


# 5 大核心工具标准 OpenAI JSON Schema 规范
CORE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "run_shell": {
        "description": "在系统中运行 powershell 命令行指令（带超时中断与安全边界）。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "需要执行的具体 powershell 命令"}
            },
            "required": ["command"],
        },
    },
    "read_file": {
        "description": "读取本地文件的文本内容（带行数折叠保护）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件相对路径"},
                "limit": {"type": "integer", "description": "最多读取行数，默认 100 行"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "description": "创建或覆盖写入文件内容（自动创建缺失的父级目录）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "待写入的文件路径"},
                "content": {"type": "string", "description": "准备写入的完整文件文本内容"},
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "description": "在文件中精确定位一段旧文本并替换为新文本。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_text": {"type": "string", "description": "待替换的原文本内容"},
                "new_text": {"type": "string", "description": "替换后的新文本内容"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    "glob_files": {
        "description": "根据通配符模式匹配搜索当前工作区内的文件列表（例如 '**/*.py'）。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "通配符模式，如 '**/*.py'"}
            },
            "required": ["pattern"],
        },
    },
}


def register_core_tools(manager: ToolManager, workdir: Optional[Path] = None) -> ToolManager:
    """
    将 5 大核心工具绑定指定工作区并一键挂载到 `ToolManager`。

    :param manager: 目标 `ToolManager` 实例。
    :param workdir: 沙箱工作区根目录，默认当前工作目录。
    :return: 注册完成后的原 `ToolManager` 实例（便于链式调用）。
    """
    base = _resolve_workdir(workdir)

    # 通过闭包固化工作区基准路径，避免将内部参数 workdir 暴露给大模型 Schema
    def _bound_run_shell(command: str) -> str:
        return run_shell(command, workdir=base)

    def _bound_read_file(path: str, limit: Optional[int] = 100) -> str:
        return read_file(path, limit=limit, workdir=base)

    def _bound_write_file(path: str, content: str) -> str:
        return write_file(path, content=content, workdir=base)

    def _bound_edit_file(path: str, old_text: str, new_text: str) -> str:
        return edit_file(path, old_text=old_text, new_text=new_text, workdir=base)

    def _bound_glob_files(pattern: str) -> str:
        return glob_files(pattern, workdir=base)

    handlers = {
        "run_shell": _bound_run_shell,
        "read_file": _bound_read_file,
        "write_file": _bound_write_file,
        "edit_file": _bound_edit_file,
        "glob_files": _bound_glob_files,
    }

    for name, fn in handlers.items():
        meta = CORE_SCHEMAS[name]
        manager.register(
            fn,
            name=name,
            description=meta["description"],
            parameters=meta["parameters"],
        )

    return manager
