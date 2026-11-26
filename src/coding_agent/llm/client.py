"""
client.py - LLM 客户端与统一调用适配层

架构角色：
    统一封装兼容 OpenAI Chat Completions 协议的大模型配置加载与调用接口
    （兼容 DeepSeek、OpenAI、智谱 GLM 等服务商），为后续所有 Agent 循环、
    压缩摘要、记忆提炼、子代理与独立目标评估器提供唯一模型出口。

解决的核心痛点：
    1. 统一多厂商环境变量优先级（`LLM_* > DEEPSEEK_* > OPENAI_* > ZHIPU_*`）与默认模型推导；
    2. 兼容 OpenAI 规范细节（如空 `tools` 列表不透传，避免部分兼容网关返回 400 错误）；
    3. 避免在无 `.env` 环境下仅仅 `import` 模块即抛出异常阻断离线单元测试与自检命令。

暴露的核心接口：
    - 常量与客户端：`API_KEY`, `BASE_URL`, `DEFAULT_MODEL`, `client`, `get_client`
    - 调用函数：`call_llm`, `call_llm_response`
    - 交互探活入口：`main`
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from coding_agent.config import configure_utf8_console


# 保证 Windows 控制台编码适配与 .env 加载（即便测试跳过 config 直接导入本模块也生效）
configure_utf8_console()
load_dotenv(override=False)

# 环境变量读取优先级：LLM_* > DEEPSEEK_* > OPENAI_* > ZHIPU_*
API_KEY: Optional[str] = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("ZHIPU_API_KEY")
)

BASE_URL: Optional[str] = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("ZHIPU_BASE_URL")
)


def _infer_default_model(base_url: Optional[str]) -> str:
    """根据 `base_url` 域名关键字智能推导默认模型名称。"""
    url_lower = (base_url or "").lower()
    if "deepseek" in url_lower:
        return "deepseek-chat"
    if "bigmodel" in url_lower or "zhipu" in url_lower:
        return "glm-4"
    return "gpt-4o"


# 模型名称：优先读取环境变量中的配置，否则按 BASE_URL 自动推导
DEFAULT_MODEL: str = os.getenv("LLM_MODEL_ID") or _infer_default_model(BASE_URL)

# Why: 仅在已检测到 API_KEY 时立即创建全局 client；若尚未配置 .env，则保持 client=None
# 延迟到真正发起请求（get_client / call_llm）时再报错，避免离线单元测试或 --help 在 import 阶段崩溃。
client: Optional[OpenAI] = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None


def get_client() -> OpenAI:
    """
    获取已初始化的 OpenAI 兼容客户端。

    :return: 可用的 `OpenAI` 客户端实例（支持测试通过覆盖模块变量 `client` 注入 Mock）。
    :raises ValueError: 当未注入 `client` 且未配置任何有效 API Key 时抛出。
    """
    global client
    if client is not None:
        return client

    if not API_KEY:
        raise ValueError(
            "未检测到有效的 API Key！请在 .env 文件中配置 LLM_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return client


def _build_chat_params(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """统一构造 `chat.completions.create` 参数字典，避免多入口重复拼装。"""
    target_model = model or DEFAULT_MODEL
    params: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        **kwargs,
    }
    # Why: 只有当传入非空工具列表时才传递 tools 字段，防止空列表 [] 引发部分兼容接口报错
    if tools:
        params["tools"] = tools
    return params


def call_llm_response(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    model: Optional[str] = None,
    **kwargs: Any,
):
    """
    统一的 LLM 完整响应调用函数（返回原始 ChatCompletion 响应体，包含 usage、id 等元数据）。

    :param messages: 对话上下文列表，形如 `[{"role": "user", "content": "..."}]`。
    :param tools: 工具定义列表（OpenAI function 格式）。
    :param model: 指定模型名称，若不传则使用 `DEFAULT_MODEL`。
    :param kwargs: 透传给 `client.chat.completions.create` 的额外参数（如 `temperature`、`max_tokens`）。
    :return: 完整的 `ChatCompletion` 响应对象。
    """
    active_client = get_client()
    params = _build_chat_params(messages=messages, tools=tools, model=model, **kwargs)
    return active_client.chat.completions.create(**params)


def call_llm(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    model: Optional[str] = None,
    **kwargs: Any,
):
    """
    统一的 LLM 消息调用函数（返回生成的 message 消息体）。

    :param messages: 对话上下文列表，形如 `[{"role": "user", "content": "..."}]`。
    :param tools: 工具定义列表（OpenAI function 格式）。
    :param model: 指定模型名称，若不传则使用 `DEFAULT_MODEL`。
    :param kwargs: 透传给 `client.chat.completions.create` 的额外参数（如 `temperature` 等）。
    :return: `ChatCompletionMessage` 对象（即 `response.choices[0].message`）。
    """
    response = call_llm_response(messages=messages, tools=tools, model=model, **kwargs)
    return response.choices[0].message


def main() -> None:
    """交互式命令行最小对话探活入口（仅供本地手动测试模型连通性）。"""
    print(f"当前模型：{DEFAULT_MODEL} | 接口地址：{BASE_URL or '<OpenAI 默认地址>'}")
    print("输入内容开始对话，输入 exit 或 quit 退出。\n")

    messages: List[dict] = [{
        "role": "system",
        "content": "你是一个简洁友好的助手，请用中文回答。"
    }]

    while True:
        try:
            user_input = input("用户输入>").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("已退出对话。")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = call_llm(messages=messages)
        except Exception as exc:
            # 回滚本轮未成功响应的用户消息，避免污染后续对话上下文
            messages.pop()
            print(f"调用失败：{exc}")
            continue

        reply = response.content or ""
        print(f"助手回答：{reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
