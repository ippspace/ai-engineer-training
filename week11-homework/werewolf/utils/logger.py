import logging
import os

from .config import config

# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志格式
log_format = "%(name)s - %(levelname)s - %(message)s"

# 配置根日志记录器
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "werewolf.log")),
        logging.StreamHandler(),
    ],
)

# 创建游戏专用日志记录器
game_logger = logging.getLogger("werewolf.game")
agent_logger = logging.getLogger("werewolf.agent")
memory_logger = logging.getLogger("werewolf.memory")

# 配置第三方库日志级别
# 抑制httpx模块的INFO级日志，避免打印大量HTTP请求日志
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# 抑制langchain相关模块的INFO级日志
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# 导出日志记录器
logger = {"game": game_logger, "agent": agent_logger, "memory": memory_logger}
