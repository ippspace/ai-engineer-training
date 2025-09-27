import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from .api_router import api_router

_ = load_dotenv()

app = FastAPI(title="FAQ问答系统", description="基于Milvus的FAQ问答API")
app.include_router(api_router)


def run_fastapi_server():
    """启动FastAPI应用"""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


def main():
    print("启动FAQ问答系统服务器...")
    run_fastapi_server()


if __name__ == "__main__":
    main()
