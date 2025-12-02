import os

from dotenv import load_dotenv
from pydantic import SecretStr

# 加载环境变量
_ = load_dotenv()


class Config:
    """配置管理类"""

    # OpenAI配置
    MODEL: str = os.getenv("MODEL") or ""
    if not MODEL:
        raise ValueError("MODEL 环境变量未设置")
    api_key: str = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise ValueError("OPENAI_API_KEY 环境变量未设置")
    OPENAI_API_KEY = SecretStr(api_key)
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL") or ""
    if not OPENAI_BASE_URL:
        raise ValueError("OPENAI_BASE_URL 环境变量未设置")

    # Embedding配置
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL 环境变量未设置")

    # Milvus配置
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    MILVUS_COLLECTION_NAME = "werewolf_memory"

    # 游戏配置
    NUM_PLAYERS = 5
    NUM_WEREWOLVES = 2
    NUM_VILLAGERS = 3

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# 创建全局配置实例
config = Config()
