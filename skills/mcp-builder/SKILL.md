---
name: mcp-builder
description: 构建 MCP (模型上下文协议) 服务的专业技能，为 Agent 扩展新工具与连接外部服务。当用户需要创建 MCP Server、集成第三方 API 时使用。
---

# MCP 服务构建指南 (MCP Server Building Skill)

你现在拥有构建 MCP（Model Context Protocol，模型上下文协议）服务端的专业能力。MCP 是一套开放标准协议，能让 AI 智能体通过统一规范与外部数据源、工具及业务系统无缝交互。

## 什么是 MCP？

MCP 服务端向大语言模型暴露三大核心能力：
- **Tools（工具）**：大模型可主动调用的函数（类似 API 接口）；
- **Resources（资源）**：大模型可读取的数据源（类似文件系统或数据库只读记录）；
- **Prompts（提示词模板）**：服务端预置的标准提示词或交互模板。

---

## 快速上手：Python MCP 服务端

### 1. 项目环境初始化

```bash
# 创建并进入项目目录
mkdir my-mcp-server && cd my-mcp-server
python -m venv .venv

# 激活虚拟环境 (Windows PowerShell: .venv\Scripts\Activate.ps1)
# 激活虚拟环境 (Linux/macOS: source .venv/bin/activate)

# 安装官方 MCP SDK
pip install mcp
```

### 2. 标准服务基础模板

```python
#!/usr/bin/env python3
"""my_server.py - 极简 Python MCP 服务端"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建服务端实例
server = Server("my-mcp-service")

# 定义一个打招呼的工具
@server.tool()
async def hello(name: str) -> str:
    """向指定用户打招呼。

    Args:
        name: 接收问候的人名
    """
    return f"你好，{name}！"

# 定义一个计算工具
@server.tool()
async def add_numbers(a: int, b: int) -> str:
    """计算两个整数相加之和。

    Args:
        a: 第一个加数
        b: 第二个加数
    """
    return str(a + b)

# 通过标准输入输出 (stdio) 启动服务通信
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3. 配置客户端接入

在客户端配置中（例如 `claude_desktop_config.json` 或本项目的 MCP 注册表）：
```json
{
  "mcpServers": {
    "my-mcp-service": {
      "command": "python",
      "args": ["D:/path/to/my_server.py"]
    }
  }
}
```

---

## TypeScript MCP 服务端

### 1. 环境初始化

```bash
mkdir my-mcp-server && cd my-mcp-server
npm init -y
npm install @modelcontextprotocol/sdk
```

### 2. 标准服务模板

```typescript
// src/index.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "my-ts-server",
  version: "1.0.0",
});

// 暴露工具列表
server.setRequestHandler("tools/list", async () => ({
  tools: [
    {
      name: "hello",
      description: "向指定用户打招呼",
      inputSchema: {
        type: "object",
        properties: {
          name: { type: "string", description: "用户名" },
        },
        required: ["name"],
      },
    },
  ],
}));

// 处理工具调用分发
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "hello") {
    const name = request.params.arguments.name;
    return { content: [{ type: "text", text: `你好, ${name}!` }] };
  }
  throw new Error(`未知的工具: ${request.params.name}`);
});

// 启动标准输入输出传输
const transport = new StdioServerTransport();
server.connect(transport);
```

---

## 进阶模式：集成外部第三方 API

```python
import httpx
from mcp.server import Server

server = Server("weather-server")

@server.tool()
async def get_weather(city: str) -> str:
    """查询指定城市的实时天气数据。"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.weatherapi.com/v1/current.json",
            params={"key": "YOUR_API_KEY", "q": city}
        )
        data = resp.json()
        temp = data['current']['temp_c']
        condition = data['current']['condition']['text']
        return f"{city} 当前气温: {temp}℃, 天气状况: {condition}"
```

## 设计准则与安全规范

1. **清晰的函数文档注释**：大模型完全依靠工具的 description 和参数注释来决定何时调用该工具，必须描述具体且严谨；
2. **入参严格校验**：所有工具入参必须校验类型与合法边界，捕获异常后返回友好的错误说明；
3. **遵循最小权限**：对于可能产生系统副作用的工具（如删除、写入），务必明确标识破坏性，并由宿主客户端配置审批拦截。
