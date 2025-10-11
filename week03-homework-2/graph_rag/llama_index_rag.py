from dataclasses import dataclass

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike


@dataclass
class RetrievalItem:
    """检索项"""

    score: float
    text: str
    file_name: str


class LlamaIndexRAG:
    def __init__(self, embedding: BaseEmbedding, llm: OpenAILike, input_dir: str):
        self._config_llama_index(embedding, llm)
        self.index: VectorStoreIndex = self._create_index(input_dir)

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievalItem]:
        """文档检索"""
        retrievals = self.index.as_retriever(similarity_top_k=top_k).retrieve(query)
        if not retrievals:
            return []

        return [
            RetrievalItem(
                score=node.score or 0.0,
                text=node.get_text(),
                file_name=node.node.metadata["file_name"] or "",
            )
            for node in retrievals
        ]

    def _config_llama_index(self, embedding: BaseEmbedding, llm: OpenAILike):
        """配置嵌入模型和LLM"""
        # 设置嵌入模型
        Settings.embed_model = embedding
        # 设置LLM
        Settings.llm = llm

    def _create_index(self, input_dir: str) -> VectorStoreIndex:
        """建向量索引"""

        # 创建语义切分器
        sen_splitter = SentenceSplitter(chunk_size=80, chunk_overlap=16)
        docs = SimpleDirectoryReader(input_dir=input_dir).load_data()
        # 创建索引
        index = VectorStoreIndex.from_documents(
            documents=docs,
            transformations=[sen_splitter],
            show_progress=True,
        )

        return index
