from langchain_openai import ChatOpenAI

from backend.config import settings


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key="EMPTY",
        model=settings.vllm_model,
        temperature=temperature,
    )
