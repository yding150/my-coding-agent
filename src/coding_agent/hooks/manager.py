"""
manager.py - 生命周期钩子管理器（HookManager）

架构角色：
    为 Coding Agent 运行时提供非侵入式生命周期切面拦截能力（AOP），
    在用户提交输入、工具执行前、工具执行后与单轮/会话结束 4 个关键切点挂载可插拔处理器。

解决的核心痛点：
    1. 替代散落的全局回调字典：通过面向对象的 `HookManager` 实例支持多会话与子代理克隆隔离；
    2. 解决注册顺序导致审计缺失的问题：引入 `priority` 优先级调度（数值越小越早执行），
       确保“审计日志 (`priority=0`)”永远先于“权限门禁 (`priority=10`)”执行，拦截操作同样留痕；
    3. 标准化拦截信号：`PreToolUse` 或 `Stop` 切面返回非 `None` 字符串即视为拦截并短路后续链条。

暴露的核心类：
    - `HookEvent`: 标准生命周期事件常量
    - `HookEntry`: 单个钩子回调与优先级元数据
    - `HookManager`: 切面注册、优先级排序、拦截调度与深拷贝克隆管理器
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union


class HookEvent:
    """标准 Agent 生命周期事件常量定义。"""

    USER_PROMPT_SUBMIT = "UserPromptSubmit"  # 用户提交指令时
    PRE_TOOL_USE = "PreToolUse"  # 工具执行前（可拦截并返回拒绝原因）
    POST_TOOL_USE = "PostToolUse"  # 工具执行后（可统计输出长度、记录耗时）
    STOP = "Stop"  # 单轮对话或会话终止时


@dataclass
class HookEntry:
    """单个钩子注册项元数据。"""

    callback: Callable  # 钩子回调函数
    priority: int = 10  # 调度优先级（数值越小越早执行，默认 10）
    name: Optional[str] = None  # 钩子标识名（未显式指定时读取函数名）

    def __repr__(self) -> str:
        return f"<HookEntry {self.name or self.callback.__name__} (priority={self.priority})>"


class HookManager:
    """
    生命周期钩子管理器：集中管理切面事件监听、优先级排序与短路拦截调度。

    生命周期：
        在运行时初始化时装配默认钩子（如审计日志、危险指令门禁、超长输出警告）；
        后续可通过 `clone()` 为子代理派生隔离副本（如去掉交互式门禁或追加子代理审计）。
    """

    def __init__(self) -> None:
        # 存储结构：{"EventName": [HookEntry(priority=0), HookEntry(priority=10), ...]}
        self._hooks: Dict[str, List[HookEntry]] = {
            HookEvent.USER_PROMPT_SUBMIT: [],
            HookEvent.PRE_TOOL_USE: [],
            HookEvent.POST_TOOL_USE: [],
            HookEvent.STOP: [],
        }

    def register(
        self,
        event: str,
        callback: Optional[Callable] = None,
        *,
        priority: int = 10,
        name: Optional[str] = None,
    ) -> Union[Callable, "HookManager"]:
        """
        注册生命周期钩子，同时支持【直接函数注册】与【带参装饰器】两种方式。

        :param event: 生命周期事件名（推荐使用 `HookEvent.*` 常量）。
        :param callback: 回调函数句柄（作为带参装饰器时为 None）。
        :param priority: 调度优先级（默认 10；0 表示最高优先级，最先触发）。
        :param name: 钩子别名，未传时默认取 `callback.__name__`。
        :return: 直接注册返回 `self`；装饰器模式返回原函数。
        """

        def decorator(fn: Callable) -> Callable:
            hook_name = name or fn.__name__
            entry = HookEntry(callback=fn, priority=priority, name=hook_name)

            if event not in self._hooks:
                self._hooks[event] = []

            self._hooks[event].append(entry)
            # Why: Python 的 list.sort 是稳定排序（Timsort），按 priority 升序排列后，
            # 不同优先级按数值顺序执行，相同优先级严格保持注册先后顺序。
            self._hooks[event].sort(key=lambda item: item.priority)
            return fn

        if callback is not None:
            decorator(callback)
            return self

        return decorator

    def on(
        self,
        event: str,
        *,
        priority: int = 10,
        name: Optional[str] = None,
    ) -> Callable:
        """装饰器注册语法糖，等价于 `register(event, priority=..., name=...)`。"""
        return self.register(event, priority=priority, name=name)

    def trigger(self, event: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        """
        按 `priority` 升序触发指定事件的钩子链（支持短路拦截）。

        Why:
            只要链条中任一钩子返回非 `None` 值（例如拒绝原因字符串），就说明需要阻断
            后续动作，因此立即短路返回；同时捕获钩子自身异常，防止辅助切面崩溃拖垮主循环。

        :param event: 触发的事件名。
        :param args: 透传给钩子回调的位置参数。
        :param kwargs: 透传给钩子回调的关键字参数。
        :return: 首个非 None 拦截信号；若全链均返回 None 则返回 None（放行）。
        """
        for entry in self._hooks.get(event, []):
            try:
                result = entry.callback(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as exc:
                print(f"⚠️ [Hook 执行异常] 钩子 {entry.name} 在响应事件 '{event}' 时报错: {exc}")
        return None

    def trigger_all(self, event: str, *args: Any, **kwargs: Any) -> List[Any]:
        """
        触发指定事件的全部钩子并收集所有返回值（不短路中断，适合广播或收集统计）。
        """
        results: List[Any] = []
        for entry in self._hooks.get(event, []):
            try:
                res = entry.callback(*args, **kwargs)
                results.append(res)
            except Exception as exc:
                print(f"⚠️ [Hook 执行异常] 钩子 {entry.name} 在响应事件 '{event}' 时报错: {exc}")
                results.append(None)
        return results

    def list_hooks(self, event: Optional[str] = None) -> List[str]:
        """列出指定事件（或全部事件）下已注册的钩子名称列表。"""
        if event:
            return [entry.name or entry.callback.__name__ for entry in self._hooks.get(event, [])]
        result: List[str] = []
        for evt, entries in self._hooks.items():
            for entry in entries:
                result.append(f"{evt}: {entry.name or entry.callback.__name__} (p={entry.priority})")
        return result

    def clone(self) -> "HookManager":
        """深拷贝当前钩子管理器，生成独立可变副本（常用于子代理隔离继承）。"""
        new_mgr = HookManager()
        for evt, entries in self._hooks.items():
            new_mgr._hooks[evt] = [
                HookEntry(callback=e.callback, priority=e.priority, name=e.name)
                for e in entries
            ]
        return new_mgr
