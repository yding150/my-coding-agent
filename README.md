# my-coding-agent

个人日常使用的工业级 **Coding Agent** 运行时底座，基于 `learn-claude-code`（`s01` ~ `s17`）核心架构采用增量平滑迁移方案构建。

> 详细演进路线与交接纪律请查阅：[docs/handover_plan.md](docs/handover_plan.md) | 架构演进记录：[docs/architecture.md](docs/architecture.md)

---

## 环境准备

- **Python 版本**：`>= 3.13`（见 `.python-version`）
- **核心依赖**：`openai`、`python-dotenv`、`pyyaml`（开发测试依赖 `pytest`）

### 1. 初始化虚拟环境与依赖

```powershell
# 使用 uv 创建并同步环境
uv sync --extra dev
```

### 2. 配置环境变量

```powershell
# 复制环境变量模板为本地 .env 并填入你的 API Key
Copy-Item .env.example .env
```

---

## 项目进度看板

### Phase 0：工程骨架搭建
- [x] 目录树规划与空包占位（`src/coding_agent/`、`tests/`、`skills/`、`docs/`）
- [x] 工程配置（`pyproject.toml`、`.env.example`、`.gitignore`、`AGENTS.md`）

### Phase 1：基础底座逐模块增量迁移（M1 ~ M9）
- [ ] **M1**：配置底座 + LLM 适配器 + 核心工具（`config`, `llm`, `tools`）
- [ ] **M2**：切面钩子 + 按需技能 + 会话级 Todo（`hooks`, `skills`, `tasks/todo`）
- [ ] **M3**：上下文四步压缩 + 跨会话记忆（`context`, `memory`）
- [ ] **M4**：一次性隔离子代理 + 持久 DAG 任务图（`subagent`, `tasks/store`）
- [ ] **M5**：异步后台任务 + Cron 定时调度（`background`, `cron`）
- [ ] **M6**：多 Agent 团队协同与邮箱通信（`teams`）
- [ ] **M7**：MCP 外部插件体系（`mcp`）
- [ ] **M8**：确定性工作流编排运行时（`workflow`）
- [ ] **M9**：目标自驱动闭环 + 主入口收敛 + 清理 `_legacy_ref/`（`goal`, `cli`）

### Phase 2：工业级增强优化（P1 ~ P6，M9 完成后启动）
- [ ] **P1**：核心编码工具升级（`grep_search`、分段 `read_file`、严格唯一 `edit_file`、跨平台带退出码 `run_shell`）
- [ ] **P2**：消除全局单例（`AgentSession` + 延迟配置加载 + 流式事件输出）
- [ ] **P3**：真实 MCP 协议接入（`stdio` / `SSE` / `.mcp.json`）
- [ ] **P4**：三态权限策略引擎与项目规则自动发现
- [ ] **P5**：会话增量持久化与断点恢复（`--resume` / `--continue`）
- [ ] **P6**：现代 CLI 交互体验升级（`rich` / `prompt_toolkit`）与全局分发
