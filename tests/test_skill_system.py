"""
test_skill_system.py - M2.2 按需技能加载系统与 runtime.py 集成单元测试
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coding_agent.runtime as runtime_mod
import coding_agent.skills.__main__ as skills_cli_mod
from coding_agent.config import SKILLS_DIR
from coding_agent.hooks import HookManager
from coding_agent.skills import SkillLoader, register_skill_tools
from coding_agent.tools import ToolManager


class TestSkillSystem(unittest.TestCase):
    """测试 SkillLoader 扫描、YAML Frontmatter 解析、安全边界与按需加载"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_builtin_skills_assets_loaded(self):
        """验证仓库根目录 skills/ 已迁入全部内置技能且可按需加载 code-review"""
        loader = SkillLoader(SKILLS_DIR)
        skills = loader.list_skills()
        for expected in ("agent-builder", "code-review", "mcp-builder", "pdf"):
            self.assertIn(expected, skills)

        catalog_text = loader.catalog()
        self.assertIn("code-review", catalog_text)
        # 第一层目录不应直接泄露第二层全文内容
        self.assertNotIn("审查核对清单", catalog_text)

        body = loader.load("code-review")
        self.assertIn("代码审查技能规范", body)

        tools = ToolManager()
        register_skill_tools(tools, loader)
        self.assertTrue(tools.has("load_skill"))
        self.assertIn("代码审查技能规范", tools.execute("load_skill", {"name": "code-review"}))
        self.assertIn("未找到名为 'nonexistent' 的技能", tools.execute("load_skill", {"name": "nonexistent"}))

    def test_frontmatter_fallback_and_empty_dir(self):
        """缺少 YAML description 时应回退提取正文首行；空目录应优雅返回提示"""
        empty_loader = SkillLoader(self.base_dir / "missing_skills")
        self.assertEqual(empty_loader.list_skills(), [])
        self.assertIn("未检测到任何本地可用技能", empty_loader.catalog())

        skills_dir = self.base_dir / "skills"
        custom_dir = skills_dir / "custom-lint"
        custom_dir.mkdir(parents=True)
        (custom_dir / "SKILL.md").write_text(
            "---\nname: custom-lint\n---\n# 自动代码格式检查\n详细规则正文...",
            encoding="utf-8",
        )

        # 非法 YAML Frontmatter 也应安全降级，不中断扫描
        bad_dir = skills_dir / "bad-yaml"
        bad_dir.mkdir(parents=True)
        (bad_dir / "SKILL.md").write_text(
            "---\n: [invalid_yaml\n---\n第一行摘要说明\n后续细节",
            encoding="utf-8",
        )

        # UTF-8 BOM 与空白 name 应能安全规范化而不产生空键或乱码 Frontmatter
        bom_dir = skills_dir / "bom-skill"
        bom_dir.mkdir(parents=True)
        (bom_dir / "SKILL.md").write_text(
            "---\nname: '   '\ndescription: BOM 兼容技能\n---\n正文内容",
            encoding="utf-8-sig",
        )

        loader = SkillLoader(skills_dir)
        self.assertEqual(loader.skills["custom-lint"]["description"], "自动代码格式检查")
        self.assertEqual(loader.skills["bad-yaml"]["description"], "第一行摘要说明")
        self.assertEqual(loader.skills["bom-skill"]["description"], "BOM 兼容技能")
        self.assertNotIn("", loader.skills)

    def test_symlink_escape_rejected(self):
        """若技能目录存在指向外部文件的软链接 SKILL.md，必须被安全边界过滤"""
        skills_dir = self.base_dir / "skills"
        skills_dir.mkdir()
        outside_file = self.base_dir / "secret.md"
        outside_file.write_text("---\nname: secret\n---\nTOP_SECRET", encoding="utf-8")

        evil_skill_dir = skills_dir / "evil-link"
        evil_skill_dir.mkdir()
        link_path = evil_skill_dir / "SKILL.md"
        try:
            os.symlink(outside_file, link_path)
        except OSError:
            # Windows 未开启开发者模式时 os.symlink 可能受限：退化为普通文件并 mock resolve() 模拟软链接逃逸
            link_path.write_text("---\nname: secret\n---\nTOP_SECRET", encoding="utf-8")
            orig_resolve = Path.resolve

            def fake_resolve(p: Path, strict: bool = False) -> Path:
                if p == link_path:
                    return outside_file
                return orig_resolve(p, strict=strict)

            with patch.object(Path, "resolve", fake_resolve):
                loader = SkillLoader(skills_dir)
            self.assertNotIn("secret", loader.list_skills())
            self.assertNotIn("evil-link", loader.list_skills())
            return

        loader = SkillLoader(skills_dir)
        self.assertNotIn("secret", loader.list_skills())
        self.assertNotIn("evil-link", loader.list_skills())


class TestSkillRuntimeIntegration(unittest.TestCase):
    """测试 SkillLoader 与 load_skill 接入 runtime.py 主循环"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _fake_msg(content=None, tool_calls=None):
        tc_list = []
        for tc in tool_calls or []:
            tc_list.append(
                SimpleNamespace(
                    id=tc["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=json.dumps(tc["arguments"], ensure_ascii=False),
                    ),
                )
            )
        return SimpleNamespace(content=content, tool_calls=tc_list)

    def test_runtime_injects_catalog_and_executes_load_skill(self):
        """主循环应注入轻量技能目录并在模型请求时加载完整 SKILL.md"""
        loader = SkillLoader(SKILLS_DIR)
        tools = runtime_mod.create_default_tools(workdir=self.workdir, skill_loader=loader)
        messages = []
        step = 0

        def scripted_llm(messages, tools):
            nonlocal step
            step += 1
            sys_prompt = messages[0]["content"]
            # System Prompt 包含轻量目录，但未塞入全量正文
            self.assertIn("code-review", sys_prompt)
            self.assertNotIn("审查核对清单", sys_prompt)
            self.assertIn("load_skill", {t["function"]["name"] for t in tools})

            if step == 1:
                return self._fake_msg(
                    tool_calls=[
                        {"id": "tc_skill", "name": "load_skill", "arguments": {"name": "code-review"}}
                    ]
                )
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertIn("代码审查技能规范", messages[-1]["content"])
            return self._fake_msg(content="已按 code-review 规范完成审阅。")

        with redirect_stdout(io.StringIO()):
            reply = runtime_mod.run_agent_turn(
                "请帮我走查当前代码",
                tools=tools,
                hooks=HookManager(),
                messages=messages,
                llm_caller=scripted_llm,
                workdir=self.workdir,
                skill_loader=loader,
                verbose=False,
            )
        self.assertEqual(reply, "已按 code-review 规范完成审阅。")
        self.assertEqual(step, 2)

    def test_skills_demo_cli_runs_cleanly(self):
        """python -m coding_agent.skills 演示入口应正常跑通"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = skills_cli_mod.main()
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("code-review", out)
        self.assertIn("Token 保护检查", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
