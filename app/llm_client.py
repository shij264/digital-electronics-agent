"""DeepSeek 对话客户端（OpenAI 兼容接口）。

DeepSeek 提供 OpenAI 兼容的 API，可直接使用 openai 官方 SDK，
只需把 base_url 指向 https://api.deepseek.com。
"""
from openai import OpenAI

from . import config


def get_client() -> OpenAI | None:
    """未配置 API Key 时返回 None（聊天接口会提示配置）。"""
    if not config.DEEPSEEK_API_KEY:
        return None
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def chat_completion(
    messages: list[dict], temperature: float = 0.3, model: str | None = None
) -> str:
    """调用 DeepSeek 对话模型，返回回答文本。

    messages 支持多模态内容块（content 为数组，含 image_url 图片），
    此时需传入支持图片的模型名（如 deepseek-flash）。
    """
    client = get_client()
    if client is None:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 中填入后重启服务")
    resp = client.chat.completions.create(
        model=model or config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
