import os
from dataclasses import dataclass

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import MarkdownReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient


@dataclass
class RetrievalItem:
    """检索项"""

    score: float
    text: str
    file_name: str


@dataclass
class QueryResult:
    """查询结果"""

    question: str
    answer: str
    retrievals: list[RetrievalItem]


@dataclass
class AddDocumentResult:
    """添加文档结果"""

    status: str
    message: str
    documents_added: int
    processed_files: list[str]


class FaqRAG:
    def __init__(self):
        self._config_llama_index()
        self.index: VectorStoreIndex = self._create_index()

    def load_faq_docs(self, input_files: list[str] | None = None) -> list[Document]:
        # 使用SimpleDirectoryReader加载文档
        documents = SimpleDirectoryReader(
            input_files=input_files,
            file_extractor={".md": MarkdownReader()},
        ).load_data()

        print(f"LlamaIndex解析完成，共提取 {len(documents)} 个FAQ条目")
        return documents

    def query(self, query: str, top_k: int = 5) -> QueryResult:
        """查询"""
        response = self.index.as_query_engine(similarity_top_k=top_k).query(query)
        return self._format_query_results(response, query)

    def add_documents_from_files(self, file_paths: list[str]) -> AddDocumentResult:
        """从保存的文件添加文档到知识库

        Args:
            file_paths: 已保存的文件路径列表

        Returns:
            dict[str, Any]: 处理结果
        """
        try:
            # 验证文件是否存在
            valid_files: list[str] = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    valid_files.append(file_path)
                else:
                    print(f"警告: 文件不存在 {file_path}")

            if not valid_files:
                return AddDocumentResult(
                    status="success",
                    message="没有找到有效的文件",
                    documents_added=0,
                    processed_files=[],
                )

            # 从保存的文件中加载文档
            new_documents = self.load_faq_docs(valid_files)

            # 添加文档到现有索引
            self._add_documents_to_index(new_documents)

            return AddDocumentResult(
                status="success",
                message=f"成功添加 {len(new_documents)} 个文档到知识库",
                documents_added=len(new_documents),
                processed_files=valid_files,
            )

        except Exception as e:
            print(f"添加文档失败: {str(e)}")
            return AddDocumentResult(
                status="error",
                message=f"添加文档失败: {str(e)}",
                documents_added=0,
                processed_files=[],
            )

    def _add_documents_to_index(self, documents: list[Document]):
        """将新文档添加到现有索引"""

        # 处理文档并添加到索引
        for doc in documents:
            self.index.insert(doc)

    def _config_llama_index(self):
        """配置嵌入模型和LLM"""
        # 设置嵌入模型
        embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

        # 检查LLM配置
        model = os.getenv("LLM_MODEL")
        if not model:
            raise ValueError("LLM_MODEL 未配置")
        api_base = os.getenv("API_BASE")
        if not api_base:
            raise ValueError("API_BASE 未配置")
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY 未配置")

        # 设置LLM
        Settings.llm = OpenAILike(
            model=model,
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
        )

    def _create_vector_store(
        self, milvus_uri: str, collection_name: str
    ) -> MilvusVectorStore:
        """创建向量存储"""
        # 创建Milvus向量存储
        vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=collection_name,
            dim=1024,  # bge-m3模型的维度
            overwrite=False,
        )
        return vector_store

    def _create_index(self) -> VectorStoreIndex:
        """获取或创建向量存储索引"""

        milvus_uri = os.getenv("MILVUS_URI")
        if not milvus_uri:
            raise ValueError("MILVUS_URI 未配置")
        collection_name = "faq"

        # 创建语义切分器
        sen_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,  # 重叠部分
            separator="\n\n",
        )

        # 从已有集合中加载数据创建索引
        client = MilvusClient(uri=milvus_uri)
        if client.has_collection(collection_name=collection_name):
            print(f"从已有集合中加载数据: {collection_name}")
            # 创建Milvus向量存储
            vector_store = self._create_vector_store(milvus_uri, collection_name)
            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                transformations=[sen_splitter],
                show_progress=True,
            )

        print("加载文档创建向量存储索引: faq.md")
        # 加载文档
        docs = self.load_faq_docs(["./milvus_faq/docs/电商售后.md"])
        vector_store = self._create_vector_store(milvus_uri, collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # 创建索引
        index = VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_context,
            transformations=[sen_splitter],
            show_progress=True,
        )

        return index

    def _format_query_results(
        self,
        response: RESPONSE_TYPE,
        query: str,
    ) -> QueryResult:
        """格式化搜索结果"""
        items: list[RetrievalItem] = []

        # 提取源节点信息
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = node.node.metadata
                items.append(
                    RetrievalItem(
                        score=node.score or 0.0,
                        text=node.get_text(),
                        file_name=metadata["file_name"] or "",
                    )
                )

        return QueryResult(
            question=query,
            answer=str(response),
            retrievals=items,
        )
