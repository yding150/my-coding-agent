"""
coding_agent.context - 上下文四步压缩管线与转录归档

在里程碑 M3.1 中迁入，暴露：
    - `ContextCompactor`: 提供 Layer 1 (`tool_result_budget`)、Layer 2 (`snip_compact`)、
      Layer 3 (`micro_compact`)、Layer 4 (`compact_history`) 与应急自愈 (`reactive_compact`)
"""

from coding_agent.context.compactor import ContextCompactor

__all__ = [
    "ContextCompactor",
]
