"""
loader.py - 技能按需动态加载系统（SkillLoader）

架构角色：
    实现两层提示词（Two-Tier Prompting）机制的技能管理器，在保证主循环 System Prompt
    精简可控的前提下，让 Coding Agent 按需发现并深加载 `skills/<skill>/SKILL.md` 中的
    领域专家操作规范。

解决的核心痛点：
    1. 避免上下文爆炸：如果把所有技能文档全量塞入 System Prompt，每轮对话都会白白浪费
       数千至上万 Token；两层机制仅在第一层注入极简目录摘要，命中需求时再通过 `load_skill`
       按需加载完整正文；
    2. 路径越界防护：扫描 `*/SKILL.md` 时使用 `.resolve().is_relative_to(skills_root)`
       阻断通过符号链接（Symlink）逃逸读取目录外文件的安全隐患；
    3. 元数据容错：若某个 `SKILL.md` 忘记填写 Frontmatter `description`，自动退避提取正文首行。

暴露的核心接口：
    - `LOAD_SKILL_SCHEMA`: `load_skill` 工具的 OpenAI JSON Schema 描述
    - `SkillLoader`: 技能目录扫描、目录摘要渲染与正文加载器
    - `register_skill_tools`: 一键将 `load_skill` 挂载至 `ToolManager`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from coding_agent.tools.manager import ToolManager

LOAD_SKILL_SCHEMA: Dict[str, Any] = {
    "description": "根据技能名称动态加载完整的 SKILL.md 专业操作指南（第二层深入加载）。",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "需要加载的专业技能名称，例如 'code-review', 'pdf', 'mcp-builder' 等",
            }
        },
        "required": ["name"],
    },
}


class SkillLoader:
    """
    专业技能按需加载器：扫描 `skills/<name>/SKILL.md`、解析 YAML Frontmatter 元数据，
    并分别向 System Prompt 与 `load_skill` 工具提供两层内容。

    目录结构约定：
        skills/
          code-review/
            SKILL.md
          pdf/
            SKILL.md
    """

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir: Path = Path(skills_dir).resolve()
        self.skills: Dict[str, Dict[str, str]] = {}
        self.scan()

    @staticmethod
    def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
        """
        解析 Markdown 文件顶部的 YAML Frontmatter 元数据块与正文。

        :param text: `SKILL.md` 原始文本。
        :return: `(metadata_dict, body_markdown)` 二元组；无有效 Frontmatter 时返回 `({}, text)`。
        """
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != "---":
            return {}, text

        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.rstrip("\r\n") == "---"
            ),
            None,
        )
        if closing_index is None:
            return {}, text

        frontmatter = "".join(lines[1:closing_index])
        body = "".join(lines[closing_index + 1 :]).strip()
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, body

    def scan(self) -> None:
        """扫描 `skills_dir` 下所有一级子目录的 `SKILL.md`，重建轻量技能索引。"""
        self.skills.clear()
        if not self.skills_dir.exists():
            return

        skills_root = self.skills_dir.resolve()
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            # Why: 防止仓库内或用户目录中恶意/误配软链接通过 `SKILL.md` 越界读取技能根目录之外的敏感文件
            if not manifest.is_file() or not manifest.resolve().is_relative_to(skills_root):
                continue

            try:
                content = manifest.read_text(encoding="utf-8-sig")
                metadata, body = self.parse_frontmatter(content)
                folder_name = manifest.parent.name
                skill_name = str(metadata.get("name") or "").strip() or folder_name
                description = str(metadata.get("description") or "").strip()

                # 若 YAML 中未提供 description，回退提取正文首个非空行作为摘要
                if not description:
                    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
                    description = first_line.lstrip("#").strip() or "无描述"

                self.skills[skill_name] = {
                    "name": skill_name,
                    "description": description,
                    "content": body,
                    "path": str(manifest),
                }
            except Exception as exc:
                print(f"⚠️ [Skill 扫描异常] 加载技能文件失败 {manifest}: {exc}")

    def catalog(self) -> str:
        """【第一层提示词注入】生成仅含技能名称与一句话描述的轻量目录，节省上下文 Token。"""
        if not self.skills:
            return "(当前未检测到任何本地可用技能)"

        lines = ["可用的专业技能列表（按需调用 load_skill 加载完整操作指南）："]
        for name, meta in sorted(self.skills.items()):
            lines.append(f"- **{name}**: {meta['description']}")
        return "\n".join(lines)

    def load(self, name: str) -> str:
        """【第二层按需加载】根据技能名返回完整的 `SKILL.md` 正文内容；不存在时返回可用清单提示。"""
        skill = self.skills.get(name)
        if skill:
            return skill["content"]
        available = ", ".join(sorted(self.skills.keys())) or "无"
        return f"错误: 未找到名为 '{name}' 的技能。当前可用技能列表: [{available}]"

    def list_skills(self) -> List[str]:
        """返回当前已扫描到的技能名称列表。"""
        return list(self.skills.keys())


def register_skill_tools(manager: ToolManager, loader: SkillLoader) -> ToolManager:
    """
    一键将 `load_skill` 工具注册到指定的 `ToolManager` 中。

    :param manager: 目标工具管理器。
    :param loader: 已完成扫描的 `SkillLoader` 实例。
    :return: 注册完成后的 `ToolManager` 实例。
    """

    def run_load_skill(name: str) -> str:
        return loader.load(name)

    manager.register(
        run_load_skill,
        name="load_skill",
        description=LOAD_SKILL_SCHEMA["description"],
        parameters=LOAD_SKILL_SCHEMA["parameters"],
    )
    return manager
