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

### Phase 1：基础底座逐模块增量迁移（M1 ~ M9，细分为 24 个子步骤）
- **M1：配置底座 + LLM + 核心工具**
  - [x] **M1.1**：配置底座 + LLM 适配层（`config.py`, `llm/client.py`）
  - [x] **M1.2**：工具注册分发表 + 5 大核心工具（`tools/manager.py`, `tools/core_tools.py`）
  - [x] **M1.3**：最小单 Agent 主循环 + CLI 入口（`runtime.py`, `cli.py`）
- **M2：切面钩子 + 按需技能 + 会话级 Todo**
  - [ ] **M2.1**：生命周期切面钩子与审批门禁（`hooks/`）
  - [ ] **M2.2**：按需技能加载器 + 迁入内置技能资产（`skills/` + 根目录 `skills/`）
  - [ ] **M2.3**：会话级 Todo 清单 + 主循环集成（`tasks/todo.py`, `tools/todo_tools.py`, `runtime.py`）
- **M3：上下文四步压缩 + 跨会话记忆**
  - [ ] **M3.1**：四步上下文压缩管线（`context/compactor.py`）
  - [ ] **M3.2**：跨会话长期记忆存储（`memory/store.py`）
  - [ ] **M3.3**：主循环接入压缩与记忆生命周期（`runtime.py`）
- **M4：一次性隔离子代理 + 持久 DAG 任务图**
  - [ ] **M4.1**：一次性隔离子代理（`subagent/`）
  - [ ] **M4.2**：持久 DAG 任务依赖图（`tasks/store.py`）
  - [ ] **M4.3**：双轨计划与子代理主循环集成（`runtime.py`）
- **M5：异步后台任务 + Cron 定时调度**
  - [ ] **M5.1**：异步后台任务管理与增强 Shell（`background/`）
  - [ ] **M5.2**：Cron 定时调度引擎（`cron/`）
  - [ ] **M5.3**：主循环并发锁与后台唤醒线程集成（`runtime.py`）
- **M6：多 Agent 团队协同与邮箱通信**
  - [ ] **M6.1**：离线 JSONL 邮箱与团队审批协议（`teams/bus.py`, `teams/protocol.py`）
  - [ ] **M6.2**：常驻队友线程、团队工具与收件箱唤醒（`teams/runtime.py`, `teams/tools.py`, `runtime.py`）
- **M7：MCP 外部插件体系**
  - [ ] **M7.1**：MCP 客户端与内置模拟插件服务（`mcp/client.py`, `mcp/servers.py`）
  - [ ] **M7.2**：MCP 动态工具池装配与分级安全审批（`mcp/manager.py`, `mcp/tools.py`, `runtime.py`）
- **M8：确定性工作流编排运行时**
  - [ ] **M8.1**：工作流原语、Schema 校验与断点日志（`workflow/primitives.py`）
  - [ ] **M8.2**：工作流注册表、执行器与主循环挂载（`workflow/registry.py`, `workflow/runner.py`, `workflow/tools.py`, `runtime.py`）
- **M9：目标自驱动闭环 + 主入口收敛 + 清理收尾**
  - [ ] **M9.1**：Goal 模型、防注入评估器、控制器与只读工具（`goal/`）
  - [ ] **M9.2**：Stop Hook 闭环与 CLI 主入口收敛（`runtime.py`, `cli.py`）
  - [ ] **M9.3**：全量回归验收与清理 `_legacy_ref/`

### Phase 2：工业级增强优化（P1 ~ P6，M9 完成后启动）
- [ ] **P1**：核心编码工具升级（`grep_search`、分段 `read_file`、严格唯一 `edit_file`、跨平台带退出码 `run_shell`）
- [ ] **P2**：消除全局单例（`AgentSession` + 延迟配置加载 + 流式事件输出）
- [ ] **P3**：真实 MCP 协议接入（`stdio` / `SSE` / `.mcp.json`）
- [ ] **P4**：三态权限策略引擎与项目规则自动发现
- [ ] **P5**：会话增量持久化与断点恢复（`--resume` / `--continue`）
- [ ] **P6**：现代 CLI 交互体验升级（`rich` / `prompt_toolkit`）与全局分发
