import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike

from .image_ocr_reader import ImageOCRReader

_ = load_dotenv()

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True,
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v4",
    embed_batch_size=6,
    embed_input_length=8192,
)


def query(query_engine: BaseQueryEngine, query: str, description: str):
    response = query_engine.query(query)
    print(f"\n================={description}=================")
    print(f"查询: {query}\n回答: {response}\n")


def good_cases_research():
    ocr_reader = ImageOCRReader()
    documents = ocr_reader.load_data(
        [
            "./ocr_research/imgs/1-扫描.jpg",
            "./ocr_research/imgs/2-屏幕截图.jpg",
            "./ocr_research/imgs/3-路牌.jpg",
        ]
    )

    print(documents)

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    query(
        query_engine, "IPython 诞生于哪一年？它鼓励什么工作模式？", "对 1-扫描.jpg 提问"
    )
    query(query_engine, "PaddleOCR 3.0新增哪些特色能力？", "对 2-屏幕截图.jpg 提问")
    query(query_engine, "离长春还有多远？", "对 3-路牌 提问")


def bad_cases_research():
    ocr_reader = ImageOCRReader()
    documents = ocr_reader.load_data(
        [
            # "./ocr_research/imgs/4-模糊文字.png",
            # "./ocr_research/imgs/5-艺术文字.jpg",
            "./ocr_research/imgs/6-艺术文字.png",
        ]
    )

    print(documents)

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # query(
    #     query_engine,
    #     "PaddleOCR 与 PaddleX 的区别与联系",
    #     "对 4-模糊文字.png 提问",
    # )
    query(
        query_engine,
        "默认情况下，PaddleOCR 的日志级别是什么？",
        "对 6-艺术文字.png 提问",
    )


def main():
    # good_cases_research()
    bad_cases_research()


if __name__ == "__main__":
    main()
