"""
runner.py - 一次性隔离子代理运行器（Ephemeral Subagent Runner）

架构角色：
    为单点探索、目录摸底或大文件信息提取提供“独立干净上下文”的同步嵌套代理执行器。
    主循环（Lead）通过 `task` 工具将重型探索外包给 `SubagentRunner`，子代理内部完成多轮
    工具读取并消化大体积输出后，仅把精炼后的最终结论字符串返回主循环。

解决的核心痛点：
    1. 隔离上下文污染：子代理内部数十次 `read_file`/`glob_files` 产生的长输出随局部变量
       `sub_messages` 自动销毁，彻底防止父会话上下文窗口被探索噪音填满；
    2. 严格防递归安全边界：从 `shared_tools` 克隆工具集并强制剥离 `task`、`spawn_teammate`、
       `compact`、`connect_mcp` 等再委派/控制类工具，从根源上杜绝子代理无限套娃递归；
    3. 异常与轮次兜底：模型调用异常与超过 `max_turns` 均转为可读文本返回主 Agent。

暴露的核心类：
    - `SubagentRunner`: 一次性隔离子代理同步执行器
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from coding_agent.llm import (
    assistant_message_to_dict,
    call_llm,
    parse_tool_arguments,
)
from coding_agent.tools.manager import ToolManager

# 禁止暴露给一次性子代理的再委派或会话控制工具清单
FORBIDDEN_SUBAGENT_TOOLS: tuple[str, ...] = (
    "task",
    "spawn_teammate",
    "compact",
    "connect_mcp",
)


class SubagentRunner:
    """
    轻量一次性隔离子代理执行器。

    生命周期：
        在运行时初始化时基于 `shared_tools` 派生并剔除禁止工具；
        每次被 `task` 工具调用时在全新的 `sub_messages` 列表中执行局部 ReAct 循环，
        结束后仅向上层返回最终摘要字符串。
    """

    def __init__(
        self,
        shared_tools: ToolManager,
        workdir: Path,
        max_turns: int = 15,
        llm_caller: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.workdir: Path = Path(workdir).resolve()
        self.max_turns: int = max_turns
        self._llm_caller: Optional[Callable[..., Any]] = llm_caller

        # Why: 必须基于 shared_tools 深拷贝并剔除再委派工具（task/spawn_teammate）与会话控制工具，
        # 既防止子代理无限套娃递归消耗 Token，也避免子代理误触发主会话压缩或动态连接外部 MCP。
        self.tools: ToolManager = shared_tools.clone()
        for forbidden in FORBIDDEN_SUBAGENT_TOOLS:
            self.tools.unregister(forbidden)

    def _build_subagent_system_prompt(self, description: Optional[str] = None) -> str:
        """构建绑定当前工作区与单点子任务目标的子代理系统提示词。"""
        desc_line = f"【当前委派子任务】：{description}\n" if description else ""
        return (
            f"你是一个专注解决具体单点问题的独立子代理（Subagent）。工作区：{self.workdir}。\n"
            f"{desc_line}"
            f"你可以使用基础核心工具（文件读写、查看、搜索与命令行）排查问题、分析代码或执行调研。\n"
            f"【关键行为准则】：\n"
            f"1. 上下文隔离：你的中间工具调用和思考过程将被自动丢弃，主 Agent 最终只能看到你的最终文本答复；\n"
            f"2. 成果导向：调查或执行完毕后，请在最终回复中给出条理清晰、事实确凿的精炼中文结论与技术建议；\n"
            f"3. 严禁越权：不可尝试创建持久团队或触发递归子代理。"
        )

    def run(self, prompt: str, description: Optional[str] = None) -> str:
        """
        在全新独立的 `sub_messages` 列表中同步运行一次性子代理，并返回最终总结文本。

        :param prompt: 委派给子代理的具体任务指令与要求。
        :param description: 子任务的简短描述（用于日志展示与系统提示词聚焦）。
        :return: 子代理最终产出的结论文本或轮次/异常兜底说明。
        """
        sub_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_subagent_system_prompt(description)},
            {"role": "user", "content": prompt},
        ]
        caller = self._llm_caller or call_llm

        print(f"\n🚀 [Subagent 启动] 目标: {description or prompt[:50]}")

        for turn in range(self.max_turns):
            try:
                response_msg = caller(messages=sub_messages, tools=self.tools.get_schemas())
            except Exception as exc:
                err_msg = f"子代理在第 {turn + 1} 轮推理调用大模型异常: {type(exc).__name__}: {exc}"
                print(f"❌ [Subagent 异常]: {err_msg}")
                return err_msg

            assistant_dict = assistant_message_to_dict(response_msg)
            sub_messages.append(assistant_dict)
            tool_calls = assistant_dict.get("tool_calls") or []

            # 若模型未发起工具调用，代表子代理已完成调研并给出最终总结
            if not tool_calls:
                final_text = str(assistant_dict.get("content") or "") or "(子代理已完成任务，未输出文本)"
                print(f"✅ [Subagent 完成] 产出摘要: {final_text[:120]}...")
                return final_text

            # 执行子代理工具调用并严格按 tool_call_id 回填
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                parsed_args, parse_err = parse_tool_arguments(raw_args)

                if parse_err is not None:
                    output = parse_err
                    print(f"  [Subagent 参数异常]: {fn_name} -> {output}")
                else:
                    print(f"  [Subagent 调用工具]: {fn_name}({parsed_args})")
                    try:
                        output = str(self.tools.execute(fn_name, parsed_args))
                    except Exception as exc:
                        output = f"工具执行报错: {type(exc).__name__}: {exc}"

                print(f"  [Subagent 工具输出]: {output[:150] + ('...' if len(output) > 150 else '')}")

                sub_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": output,
                    }
                )

        print(f"⚠️ [Subagent 达到轮次上限] 已达到 {self.max_turns} 轮限制，强制退出。")
        return f"子代理在执行 {self.max_turns} 轮后达到上限提前退出，未给出确定性结论。"
