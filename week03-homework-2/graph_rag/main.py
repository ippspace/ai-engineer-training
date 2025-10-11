import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM

from .llama_index_rag import LlamaIndexRAG
from .neo4j_graph_rag import Neo4jGraphRAG

_ = load_dotenv()


def create_index_rag(
    base_url: str,
    api_key: str,
    model_name: str,
    input_dir: str,
) -> LlamaIndexRAG:
    embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding = HuggingFaceEmbedding(model_name=embed_model)
    openai_like_llm = OpenAILike(
        api_base=base_url,
        api_key=api_key,
        model=model_name,
        is_chat_model=True,
    )
    return LlamaIndexRAG(
        embedding=embedding,
        llm=openai_like_llm,
        input_dir=input_dir,
    )


def create_graph_rag(
    base_url: str,
    api_key: str,
    model_name: str,
    index_rag: LlamaIndexRAG,
    data_path: str,
) -> Neo4jGraphRAG:
    # Neo4j driver
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
    # Connect to Neo4j database
    driver = GraphDatabase.driver(
        uri=neo4j_uri,
        auth=(neo4j_user, neo4j_password),
    )

    openai_llm: OpenAILLM = OpenAILLM(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        model_params={"temperature": 0},
    )
    return Neo4jGraphRAG(
        driver=driver,
        llm=openai_llm,
        index_rag=index_rag,
        data_path=data_path,
    )


def main():
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise ValueError("LLM_MODEL 未配置")
    base_url = os.getenv("API_BASE")
    if not base_url:
        raise ValueError("API_BASE 未配置")
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY 未配置")

    index_rag = create_index_rag(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        input_dir="./graph_rag/docs",
    )

    graph_rag = create_graph_rag(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        index_rag=index_rag,
        data_path="./graph_rag/docs/a-company.txt",
    )

    print(f"{'=' * 30} Graph RAG 问答系统 {'=' * 30}")
    while True:
        try:
            question = input(
                "\n请输入您的问题（输入 'quit' 或 'exit' 退出程序）: "
            ).strip()

            if question.lower() in ["quit", "exit", "退出"]:
                print("感谢使用，再见！")
                break

            if not question:
                print("问题不能为空，请重新输入。")
                continue

            print("\n正在处理您的问题...")
            result = graph_rag.query(question)
            print(result)
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n处理问题时出现错误: {e}")
            print("请重新输入您的问题。")

    # 关闭资源
    graph_rag.close()


if __name__ == "__main__":
    main()
