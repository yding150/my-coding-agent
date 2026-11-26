"""
test_tool_system.py - M1.2 工具管理与 5 大核心工具库单元测试
"""

import tempfile
import unittest
from pathlib import Path

from coding_agent.tools import (
    ToolManager,
    _safe_path,
    register_core_tools,
)


class TestToolSystem(unittest.TestCase):
    """测试 ToolManager 与核心基础工具库功能"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name).resolve()
        self.tools = ToolManager()
        register_core_tools(self.tools, workdir=self.workdir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_core_tools_registration(self):
        """测试 5 大核心基础工具是否全部成功注册"""
        expected_tools = ["run_shell", "read_file", "write_file", "edit_file", "glob_files"]
        for name in expected_tools:
            self.assertTrue(self.tools.has(name), f"工具 {name} 未被挂载")
        self.assertEqual(len(self.tools), 5)

    def test_openai_schemas_format(self):
        """测试导出的 OpenAI tools schema 结构"""
        schemas = self.tools.get_schemas()
        self.assertEqual(len(schemas), 5)
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIn("name", s["function"])
            self.assertIn("description", s["function"])
            self.assertIn("parameters", s["function"])

    def test_file_operations_pipeline(self):
        """测试 write_file -> read_file -> edit_file -> glob_files 完整管道"""
        # 1. 写入文件
        res_write = self.tools.execute(
            "write_file",
            {
                "path": "sub/hello.txt",
                "content": "Line 1: Hello\nLine 2: World\nLine 3: Python",
            },
        )
        self.assertIn("成功写入文件", res_write)

        # 2. 读取文件
        res_read = self.tools.execute("read_file", {"path": "sub/hello.txt", "limit": 2})
        self.assertIn("Line 1: Hello", res_read)
        self.assertIn("Line 2: World", res_read)
        self.assertIn("剩余 1 行已省略", res_read)

        # 3. 编辑文件
        res_edit = self.tools.execute(
            "edit_file",
            {
                "path": "sub/hello.txt",
                "old_text": "World",
                "new_text": "Antigravity",
            },
        )
        self.assertIn("成功更新文件", res_edit)

        # 验证编辑生效
        full_read = self.tools.execute("read_file", {"path": "sub/hello.txt"})
        self.assertIn("Antigravity", full_read)
        self.assertNotIn("World", full_read)

        # 4. Glob 通配符搜索
        res_glob = self.tools.execute("glob_files", {"pattern": "**/*.txt"})
        self.assertIn("sub", res_glob)
        self.assertIn("hello.txt", res_glob)

    def test_run_shell_powershell(self):
        """测试 PowerShell 执行能力与正常输出"""
        res = self.tools.execute("run_shell", {"command": "Write-Output 'PowerShell_OK'"})
        self.assertEqual(res, "PowerShell_OK")

    def test_safe_path_boundary_defense(self):
        """测试越界逃逸路径拦截机制"""
        res = self.tools.execute("read_file", {"path": "../../../secret.txt"})
        self.assertIn("访问被拒绝: 检测到路径越界逃逸", res)

        with self.assertRaises(ValueError):
            _safe_path("../../outside.txt", workdir=self.workdir)

    def test_subset_isolation(self):
        """测试派生子集（用于子代理权限隔离）"""
        sub_tools = self.tools.subset(["read_file", "glob_files"])
        self.assertEqual(len(sub_tools), 2)
        self.assertTrue(sub_tools.has("read_file"))
        self.assertTrue(sub_tools.has("glob_files"))
        self.assertFalse(sub_tools.has("run_shell"))
        self.assertFalse(sub_tools.has("write_file"))

    def test_unknown_tool_and_arg_errors(self):
        """测试未知工具与参数错误的自愈保护（绝不崩溃）"""
        res_unknown = self.tools.execute("not_exists", {})
        self.assertIn("错误: 未找到名为 'not_exists' 的可用工具", res_unknown)

        res_arg_err = self.tools.execute("read_file", {})
        self.assertIn("参数调用错误", res_arg_err)

        res_bad_type = self.tools.execute("read_file", ["invalid"])  # type: ignore[arg-type]
        self.assertIn("参数格式错误", res_bad_type)

    def test_unregister_clone_and_merge(self):
        """测试工具卸载、克隆隔离与合并能力"""
        cloned = self.tools.clone()
        self.assertEqual(len(cloned), 5)
        self.assertTrue(cloned.unregister("run_shell"))
        self.assertFalse(cloned.has("run_shell"))
        # 克隆体修改不应影响原管理器
        self.assertTrue(self.tools.has("run_shell"))

        extra = ToolManager()
        extra.register(lambda: "pong", name="ping", description="探活")
        cloned.merge(extra)
        self.assertTrue(cloned.has("ping"))
        self.assertEqual(cloned.execute("ping"), "pong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
