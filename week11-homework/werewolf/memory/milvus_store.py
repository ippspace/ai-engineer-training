import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ..utils.config import config
from ..utils.logger import logger


class MilvusStore:
    """Milvus向量存储类，支持Milvus和内存存储两种模式"""

    collection_name: str
    embedding_dim: int
    use_memory_store: bool
    memory_store: list[dict[str, object]]
    collection: object | None = None
    
    # 类变量，用于共享连接状态
    _connected: bool = False
    _connection_alias: str = "default"

    def __init__(self):
        """初始化Milvus连接和集合"""
        self.collection_name = config.MILVUS_COLLECTION_NAME
        self.embedding_dim = 1024  # BAAI/bge-m3模型的维度是1024
        self.use_memory_store = False
        self.memory_store = []
        self.collection = None
        self._connect()
        if not self.use_memory_store:
            self._create_collection()

    def _connect(self) -> None:
        """连接到Milvus服务器，复用已有的连接"""
        try:
            # 检查连接是否已经存在
            if not self._connected and not connections.has_connection(self._connection_alias):
                connections.connect(alias=self._connection_alias, uri=config.MILVUS_URI)
                self.__class__._connected = True
                logger["memory"].info("成功连接到Milvus服务器")
            elif self._connected:
                logger["memory"].debug("复用已有的Milvus连接")
        except Exception as e:
            logger["memory"].warning(f"连接Milvus服务器失败: {e}，将使用内存存储替代")
            self.use_memory_store = True

    def _create_collection(self) -> None:
        """创建集合（如果不存在或维度不匹配则重建）"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim
            ),
            FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="round", dtype=DataType.INT32),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="狼人杀游戏记忆存储")

        # 检查集合是否存在
        if utility.has_collection(self.collection_name):
            # 获取现有集合的信息
            collection = Collection(name=self.collection_name)
            collection_schema = collection.schema
            # 查找embedding字段
            embedding_field = next(
                (
                    field
                    for field in collection_schema.fields
                    if field.name == "embedding"
                ),
                None,
            )
            if embedding_field:
                # 检查维度是否匹配
                if embedding_field.dim != self.embedding_dim:
                    logger["memory"].warning(
                        f"集合 {self.collection_name} 的嵌入维度 {embedding_field.dim} 与预期 {self.embedding_dim} 不匹配，将重建集合"
                    )
                    # 删除旧集合
                    utility.drop_collection(self.collection_name)
                    # 重新创建集合
                    self.collection = Collection(
                        name=self.collection_name, schema=schema
                    )
                    # 创建索引
                    self._create_index()
                    # 加载集合
                    self.collection.load()
                    logger["memory"].info(f"成功重建集合: {self.collection_name}")
                else:
                    # 维度匹配，直接使用现有集合
                    self.collection = collection
                    # 加载集合
                    self.collection.load()
                    logger["memory"].info(f"成功加载集合: {self.collection_name}")
        else:
            # 集合不存在，创建新集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            # 创建索引
            self._create_index()
            # 加载集合
            self.collection.load()
            logger["memory"].info(f"成功创建集合: {self.collection_name}")

    def _create_index(self) -> None:
        """创建索引"""
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger["memory"].info("成功创建索引")

    def add_memory(
        self,
        content: str,
        embedding: list[float],
        agent_id: str,
        round_num: int,
        timestamp: int,
    ) -> None:
        """添加记忆到存储"""
        try:
            if self.use_memory_store:
                # 使用内存存储
                memory = {
                    "content": content,
                    "embedding": embedding,
                    "agent_id": agent_id,
                    "round": round_num,
                    "timestamp": timestamp,
                }
                self.memory_store.append(memory)
                logger["memory"].debug(f"成功添加记忆到内存存储: {content[:50]}...")
            else:
                # 使用Milvus存储
                data = [[content], [embedding], [agent_id], [round_num], [timestamp]]
                self.collection.insert(data)
                self.collection.flush()
                logger["memory"].debug(f"成功添加记忆到Milvus: {content[:50]}...")
        except Exception as e:
            logger["memory"].error(f"添加记忆失败: {e}")
            # 尝试使用内存存储作为备选
            memory = {
                "content": content,
                "embedding": embedding,
                "agent_id": agent_id,
                "round": round_num,
                "timestamp": timestamp,
            }
            self.memory_store.append(memory)
            logger["memory"].debug(f"已使用内存存储替代: {content[:50]}...")

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度

        Args:
            a: 向量a
            b: 向量b

        Returns:
            余弦相似度值
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _search_memory_in_memory_store(
        self,
        query_embedding: list[float],
        limit: int = 5,
        agent_id: str | None = None,
        round_num: int | None = None,
    ) -> list[dict[str, object]]:
        """在内存存储中搜索记忆

        Args:
            query_embedding: 查询嵌入向量
            limit: 返回数量限制
            agent_id: 代理ID（可选）
            round_num: 轮次（可选）

        Returns:
            相关记忆列表
        """
        # 过滤记忆
        filtered_memories = []
        for memory in self.memory_store:
            if agent_id and memory["agent_id"] != agent_id:
                continue
            if round_num and memory["round"] != round_num:
                continue
            filtered_memories.append(memory)

        # 计算相似度并排序
        for memory in filtered_memories:
            similarity = self._cosine_similarity(query_embedding, memory["embedding"])
            memory["distance"] = 1 - similarity  # 转换为距离（越小越相似）

        # 按距离排序并返回前limit个
        filtered_memories.sort(key=lambda x: x["distance"])
        memories = filtered_memories[:limit]

        logger["memory"].debug(f"成功从内存存储搜索到 {len(memories)} 条相关记忆")
        return memories

    def search_memory(
        self,
        query_embedding: list[float],
        limit: int = 5,
        agent_id: str | None = None,
        round_num: int | None = None,
    ) -> list[dict[str, object]]:
        """搜索相关记忆"""
        try:
            if self.use_memory_store:
                # 使用内存存储搜索
                return self._search_memory_in_memory_store(
                    query_embedding, limit, agent_id, round_num
                )
            else:
                # 使用Milvus搜索
                # 构建查询表达式
                expr = ""
                if agent_id:
                    expr += f"agent_id == '{agent_id}'"
                if round_num:
                    if expr:
                        expr += f" and round == {round_num}"
                    else:
                        expr += f"round == {round_num}"

                # 执行搜索
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit,
                    expr=expr if expr else None,
                    output_fields=["content", "agent_id", "round", "timestamp"],
                )

                # 处理搜索结果
                memories = []
                for result in results[0]:
                    memories.append(
                        {
                            "content": result.entity.get("content"),
                            "agent_id": result.entity.get("agent_id"),
                            "round": result.entity.get("round"),
                            "timestamp": result.entity.get("timestamp"),
                            "distance": result.distance,
                        }
                    )

                logger["memory"].debug(f"成功从Milvus搜索到 {len(memories)} 条相关记忆")
                return memories
        except Exception as e:
            logger["memory"].error(f"搜索记忆失败: {e}")
            # 尝试使用内存存储作为备选
            if self.memory_store:
                logger["memory"].info("尝试使用内存存储搜索")
                return self._search_memory_in_memory_store(
                    query_embedding, limit, agent_id, round_num
                )
            return []
            
    def get_recent_memories(
        self,
        limit: int = 10,
        agent_id: str | None = None,
        round_num: int | None = None,
    ) -> list[dict[str, object]]:
        """获取最近的记忆，按时间戳倒序排列"""
        try:
            if self.use_memory_store:
                # 使用内存存储获取最近记忆
                filtered_memories = []
                for memory in self.memory_store:
                    if agent_id and memory["agent_id"] != agent_id:
                        continue
                    if round_num and memory["round"] != round_num:
                        continue
                    filtered_memories.append(memory)
                
                # 按时间戳倒序排序
                filtered_memories.sort(key=lambda x: x["timestamp"], reverse=True)
                return filtered_memories[:limit]
            else:
                # 使用Milvus获取最近记忆
                # 构建查询表达式
                expr = ""
                if agent_id:
                    expr += f"agent_id == '{agent_id}'"
                if round_num:
                    if expr:
                        expr += f" and round == {round_num}"
                    else:
                        expr += f"round == {round_num}"
                
                # 执行查询，按时间戳倒序排序
                # 注意：Milvus的query方法中，当没有筛选条件时，expr参数应该省略或使用空字符串
                results = self.collection.query(
                    expr=expr if expr else "",
                    output_fields=["content", "agent_id", "round", "timestamp"],
                    limit=limit,
                    offset=0,
                    order_by="timestamp DESC"
                )
                
                # 处理结果
                memories = []
                for result in results:
                    memories.append(
                        {
                            "content": result.get("content"),
                            "agent_id": result.get("agent_id"),
                            "round": result.get("round"),
                            "timestamp": result.get("timestamp"),
                        }
                    )
                
                logger["memory"].debug(f"成功获取 {len(memories)} 条最近记忆")
                return memories
        except Exception as e:
            logger["memory"].error(f"获取最近记忆失败: {e}")
            # 尝试使用内存存储作为备选
            if self.memory_store:
                filtered_memories = []
                for memory in self.memory_store:
                    if agent_id and memory["agent_id"] != agent_id:
                        continue
                    if round_num and memory["round"] != round_num:
                        continue
                    filtered_memories.append(memory)
                
                filtered_memories.sort(key=lambda x: x["timestamp"], reverse=True)
                return filtered_memories[:limit]
            return []

    def close(self) -> None:
        """关闭存储连接"""
        try:
            if not self.use_memory_store:
                # 不再断开连接，连接会在进程结束时自动关闭
                # 避免频繁断开和重新连接影响性能
                logger["memory"].info("Milvus连接将在进程结束时自动关闭")
                # 仅释放集合引用
                self.collection = None
            else:
                logger["memory"].info("内存存储不需要关闭连接")
        except Exception as e:
            logger["memory"].error(f"关闭连接失败: {e}")
            # 不抛出异常，避免影响游戏流程
