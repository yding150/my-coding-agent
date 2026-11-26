"""
manager.py - 统一工具注册、分发与 OpenAI Schema 管理器

架构角色：
    作为 Coding Agent 运行时的工具中枢，承载所有内置工具、技能工具、任务工具、
    后台工具、团队工具与工作流工具的注册、Schema 导出、子集隔离与异常安全分发。

解决的核心痛点：
    1. 消除工具定义、JSON Schema 与执行分发三处割裂的问题，一处注册自动聚合；
    2. `execute()` 统一拦截未知工具、参数格式/签名错误与运行时异常，转化为大模型可读的
       文本反馈，确保工具调用出错不会导致 Agent 主循环崩溃；
    3. 通过 `subset()` / `clone()` / `merge()` 支持子代理最小权限集裁剪与热插拔组合。

暴露的核心类：
    - `ToolDefinition`: 单个工具的元数据与可调用体封装
    - `ToolManager`: 工具注册表、OpenAI Schema 导出与统一分发器
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolDefinition:
    """单个工具的完整元数据与执行体封装。"""

    name: str  # 工具协议名（对应 OpenAI Function Calling 的 function.name）
    description: str  # 工具功能描述（供大模型理解何时及如何使用）
    parameters: Dict[str, Any]  # OpenAI JSON Schema 格式参数规范
    func: Callable  # 本地具体可调用的 Python 函数句柄

    def to_schema(self) -> Dict[str, Any]:
        """将当前工具定义转换为 OpenAI 兼容的 `tools` 声明格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolManager:
    """
    工具管理器：集中式管理工具生命周期、安全分发调度与 OpenAI Schema 输出。

    生命周期：
        在会话或运行时初始化阶段实例化，由各能力模块调用 `register()` 挂载工具；
        在主循环每轮调用大模型前通过 `get_schemas()` 导出声明，并在模型返回
        `tool_calls` 后通过 `execute()` 调度执行。
    """

    def __init__(self) -> None:
        # 存储结构：{"tool_name": ToolDefinition(...)}
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[Callable, "ToolManager"]:
        """
        注册工具，同时支持【带参装饰器】与【直接函数注册】两种用法。

        :param func: 待注册的函数句柄（作为带参装饰器使用时为 None）。
        :param name: 工具协议名，未传时默认使用 `func.__name__`。
        :param description: 工具功能描述，未传时优先提取 `func.__doc__` 首行。
        :param parameters: OpenAI JSON Schema 参数字典，未传时提供空对象默认规范。
        :return: 直接注册时返回 `self` 以支持链式调用；装饰器模式返回原函数句柄。
        """

        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            # 优先使用显式描述，其次提取函数 docstring 首行，最后回退默认提示
            raw_doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            tool_desc = description or raw_doc or f"执行 {tool_name} 工具"

            # 参数 Schema 默认保底规范（兼容无参工具调用）
            tool_params = (
                parameters
                if parameters is not None
                else {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            )

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                parameters=tool_params,
                func=fn,
            )
            return fn

        if func is not None:
            decorator(func)
            return self

        return decorator

    def unregister(self, name: str) -> bool:
        """卸载指定名称的工具；存在并移除返回 True，否则返回 False。"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def has(self, name: str) -> bool:
        """判断指定名称的工具是否已挂载。"""
        return name in self._tools

    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取指定名称的工具定义对象，不存在时返回 None。"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """返回当前已注册的全部工具名称列表（保持注册顺序）。"""
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """导出符合 OpenAI Function Calling 协议的 `tools` Schema 列表。"""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_handlers(self) -> Dict[str, Callable]:
        """获取底层原始处理函数映射 `{"name": func}`，兼容旧式分发表调用。"""
        return {name: tool.func for name, tool in self._tools.items()}

    def execute(self, name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        统一执行指定工具，并对所有常见异常进行文本化兜底。

        Why:
            Agent 主循环中大模型可能产生幻觉工具名或畸形 JSON 参数；若直接抛出 Python
            异常会导致整个会话中断。将错误转为结构清晰的字符串回填 `tool` 消息，
            可让模型根据签名提示自主纠错重试。

        :param name: 工具协议名。
        :param args: 工具调用参数字典。
        :return: 序列化后的字符串执行结果或错误自愈提示。
        """
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self.list_tools()) or "(无)"
            return f"错误: 未找到名为 '{name}' 的可用工具。当前已挂载工具: [{available}]"

        actual_args = {} if args is None else args
        if not isinstance(actual_args, dict):
            return f"错误: 工具 '{name}' 的参数格式错误，期望字典对象，实际收到 {type(actual_args).__name__}"

        try:
            result = tool.func(**actual_args)
            return str(result) if result is not None else "(操作完成，无返回值)"
        except TypeError as e:
            # 捕获传参缺失、多余字段或参数名错误等调用签名异常，附带真实签名方便模型纠错
            sig = inspect.signature(tool.func)
            return f"工具 '{name}' 参数调用错误: {str(e)}。该工具接收的签名参数为: {sig}"
        except Exception as e:
            # 捕获工具内部未预料运行时异常并转为模型可读文本，保证主循环绝不崩溃
            return f"工具 '{name}' 执行时发生异常: {str(e)}"

    def subset(self, names: List[str]) -> "ToolManager":
        """
        按白名单派生工具子集管理器（用于子代理权限隔离或只读沙箱）。

        :param names: 允许保留的工具名列表，不存在的名称自动忽略。
        :return: 仅包含允许工具的新 `ToolManager` 实例。
        """
        sub_manager = ToolManager()
        for name in names:
            if name in self._tools:
                tool = self._tools[name]
                sub_manager.register(
                    tool.func,
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
        return sub_manager

    def merge(self, other: "ToolManager") -> "ToolManager":
        """将另一个 `ToolManager` 中的全部工具合并到当前管理器（同名工具覆盖）。"""
        for tool in other._tools.values():
            self.register(
                tool.func,
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
        return self

    def clone(self) -> "ToolManager":
        """拷贝当前工具管理器，生成独立可变副本（常用于隔离 Lead 与子代理工具集）。"""
        return self.subset(self.list_tools())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)
