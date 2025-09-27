from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from .faq_rag import FaqRAG, RetrievalItem

api_router = APIRouter(prefix="/api")

faq_rag = FaqRAG()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    retrievals: list[RetrievalItem]


class UploadResponse(BaseModel):
    documents_added: int


@api_router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@api_router.post("/update-by-upload", response_model=UploadResponse)
async def update_by_upload(file: UploadFile):
    """文件上传接口，用于知识库热更新"""

    try:
        # 创建上传目录
        upload_dir = Path("./milvus_faq/docs/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 检查文件类型
        if not file.filename or not file.filename.endswith((".md", ".txt")):
            filename = file.filename or "unknown"
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {filename}，只支持 .md 和 .txt 文件",
            )

        # 读取文件内容
        content = await file.read()
        try:
            content_str = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content_str = content.decode("gbk")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 编码不可识别，请使用 UTF-8 或 GBK 编码",
                )

        file_path = upload_dir / file.filename
        with open(file_path, "w", encoding="utf-8") as f:
            _ = f.write(content_str)

        # 更新知识库，传入保存的文件路径列表
        result = faq_rag.add_documents_from_files([str(file_path)])

        return UploadResponse(
            documents_added=result.documents_added,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@api_router.post("/ask", response_model=AskResponse)
async def ask_faq(request: AskRequest):
    """FAQ 提问接口"""

    try:
        results = faq_rag.query(request.question, request.top_k)
        return AskResponse(
            question=request.question,
            answer=results.answer,
            retrievals=results.retrievals,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


def demo():
    """主函数"""
    print("=== FAQ问答系统 ===")

    try:
        # 5. 交互式问答
        print("\n=== 开始问答（输入'quit'退出）===")
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                if question.lower() in ["quit", "exit", "退出", "q"]:
                    break

                if not question:
                    continue

                print("\n检索中...")
                results = faq_rag.query(question, 5)

                # 显示结果
                print(f"\n问题: {results.question}")
                print(f"答案: {results.answer}")

                print("\n=== 相关FAQ条目 ===")
                for i, item in enumerate(results.retrievals, 1):
                    print(f"\n[{i}] 相似度: {item.score:.4f}")
                    print(f"text: {item.text}")
                    print(f"file: {item.file_name}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"查询出错: {e}")

        print("\n感谢使用FAQ问答系统！")

    except Exception as e:
        print(f"系统初始化失败: {e}")
        raise
