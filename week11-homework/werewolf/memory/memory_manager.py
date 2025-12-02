from datetime import datetime

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from ..utils.config import config
from ..utils.logger import logger
from .milvus_store import MilvusStore


class MemoryManager:
    """记忆管理类，负责处理Agent的记忆存储和检索"""

    milvus_store: MilvusStore
    embeddings: HuggingFaceEmbeddings
    embedding_cache: dict[str, list[float]]  # 嵌入缓存，避免重复生成相同内容的嵌入

    def __init__(self):
        """初始化记忆管理器"""
        self.milvus_store = MilvusStore()

        # 初始化嵌入缓存
        self.embedding_cache = {}

        # 配置并初始化HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger["memory"].info("初始化HuggingFaceEmbeddings成功")

    def add_memory(self, content: str, agent_id: str, round_num: int) -> None:
        """添加记忆到存储"""
        try:
            # 确保内容是字符串
            if not isinstance(content, str):
                content = str(content)
                logger["memory"].warning(
                    f"将非字符串内容转换为字符串: {content[:50]}..."
                )

            # 生成嵌入向量，使用缓存避免重复计算
            embedding = np.zeros(1024).tolist()  # 使用numpy生成默认向量，提高性能
            try:
                # 检查缓存中是否已有该内容的嵌入
                if content in self.embedding_cache:
                    embedding = self.embedding_cache[content]
                    logger["memory"].debug(
                        f"使用缓存的嵌入向量，长度: {len(embedding)}"
                    )
                else:
                    # 生成新的嵌入向量并缓存
                    embedding = self.embeddings.embed_query(content)
                    self.embedding_cache[content] = embedding
                    logger["memory"].debug(f"生成新的嵌入向量，长度: {len(embedding)}")
            except Exception as embed_error:
                logger["memory"].warning(f"嵌入生成失败，使用默认向量: {embed_error}")
                # 额外的调试信息
                logger["memory"].debug(
                    f"内容类型: {type(content)}, 内容: {content[:50]}..."
                )

            # 获取当前时间戳
            timestamp = int(datetime.now().timestamp() * 1000)

            # 存储到Milvus
            self.milvus_store.add_memory(
                content, embedding, agent_id, round_num, timestamp
            )

            logger["memory"].debug(
                f"Agent {agent_id} 在第 {round_num} 轮添加了记忆: {content[:50]}..."
            )
        except Exception as e:
            logger["memory"].error(f"添加记忆失败: {e}")
            # 不抛出异常，避免影响游戏流程
            logger["memory"].debug(f"记忆添加失败，继续游戏流程: {e}")

    def retrieve_memory(
        self,
        query: str,
        agent_id: str | None = None,
        round_num: int | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        """检索相关记忆"""
        try:
            # 生成查询嵌入向量，使用缓存避免重复计算
            query_embedding = np.zeros(1024).tolist()  # 使用numpy生成默认向量，提高性能
            try:
                # 检查缓存中是否已有该查询的嵌入
                if query in self.embedding_cache:
                    query_embedding = self.embedding_cache[query]
                    logger["memory"].debug(
                        f"使用缓存的查询嵌入向量，长度: {len(query_embedding)}"
                    )
                else:
                    # 生成新的查询嵌入向量并缓存
                    query_embedding = self.embeddings.embed_query(query)
                    self.embedding_cache[query] = query_embedding
                    logger["memory"].debug(
                        f"生成新的查询嵌入向量，长度: {len(query_embedding)}"
                    )
            except Exception as embed_error:
                logger["memory"].warning(
                    f"查询嵌入生成失败，使用默认向量: {embed_error}"
                )

            # 从Milvus检索相关记忆
            memories = self.milvus_store.search_memory(
                query_embedding=query_embedding,
                limit=limit,
                agent_id=agent_id,
                round_num=round_num,
            )

            logger["memory"].debug(f"检索到 {len(memories)} 条相关记忆")
            return memories
        except Exception as e:
            logger["memory"].error(f"检索记忆失败: {e}")
            # 不抛出异常，返回空列表
            return []

    def get_recent_memories(
        self, agent_id: str | None = None, round_num: int | None = None, limit: int = 10
    ) -> list[dict[str, object]]:
        """获取最近的记忆"""
        try:
            # 直接从MilvusStore获取最近记忆，避免不必要的嵌入搜索
            memories = self.milvus_store.get_recent_memories(limit, agent_id, round_num)
            return memories
        except Exception as e:
            logger["memory"].error(f"获取最近记忆失败: {e}")
            raise

    def close(self) -> None:
        """关闭记忆管理器"""
        self.milvus_store.close()
