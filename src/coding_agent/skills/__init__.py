"""
coding_agent.skills - 专业技能按需动态加载系统

在里程碑 M2.2 中迁入，暴露：
    - `SkillLoader`: 扫描 `skills/` 并提供第一层轻量目录与第二层按需正文加载
    - `register_skill_tools`: 注册 `load_skill` 工具至 `ToolManager`
    - `LOAD_SKILL_SCHEMA`: `load_skill` 标准 OpenAI Schema 定义
"""

from coding_agent.skills.loader import (
    LOAD_SKILL_SCHEMA,
    SkillLoader,
    register_skill_tools,
)

__all__ = [
    "SkillLoader",
    "register_skill_tools",
    "LOAD_SKILL_SCHEMA",
]
