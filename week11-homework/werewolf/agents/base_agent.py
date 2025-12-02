from langchain_openai import ChatOpenAI

from ..memory.memory_manager import MemoryManager
from ..utils.config import config
from ..utils.logger import logger


class BaseAgent:
    """基础Agent类"""

    agent_id: str
    name: str
    role: str
    is_alive: bool
    memory_manager: MemoryManager | None
    llm: ChatOpenAI

    def __init__(self, agent_id: str, name: str, role: str):
        """初始化Agent

        Args:
            agent_id: Agent唯一标识
            name: Agent名称
            role: Agent角色（村民/狼人/主持人）
        """
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.is_alive = True
        self.memory_manager = None

        # 初始化LLM，添加重试逻辑处理API限流
        self.llm = ChatOpenAI(
            model=config.MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=0.7,
            max_retries=5,
            timeout=30,
        )

        logger["agent"].info(
            f"成功创建Agent: {self.name} (ID: {self.agent_id}, 角色: {self.role})"
        )

    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """设置记忆管理器

        Args:
            memory_manager: 记忆管理器实例
        """
        self.memory_manager = memory_manager

    def night_action(self, game_state: dict[str, object], round_num: int) -> str | None:
        """夜晚行动方法，由子类实现

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            夜晚行动结果（如要杀害的玩家ID）
        """
        # 默认实现：村民不需要夜晚行动
        return None

    def speak(self, game_state: dict[str, object], round_num: int) -> str:
        """发言方法，由子类实现

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            发言内容
        """
        raise NotImplementedError("speak方法必须由子类实现")

    def vote(self, game_state: dict[str, object], round_num: int) -> str:
        """投票方法，由子类实现

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            投票对象ID
        """
        raise NotImplementedError("vote方法必须由子类实现")

    def add_memory(self, content: str, round_num: int) -> None:
        """添加记忆

        Args:
            content: 记忆内容
            round_num: 当前轮次
        """
        if self.memory_manager:
            self.memory_manager.add_memory(content, self.agent_id, round_num)

    def retrieve_memory(
        self, query: str, round_num: int | None = None, limit: int = 5
    ) -> list[dict[str, object]]:
        """检索记忆

        Args:
            query: 查询内容
            round_num: 轮次（可选）
            limit: 返回数量限制

        Returns:
            相关记忆列表
        """
        if self.memory_manager:
            return self.memory_manager.retrieve_memory(
                query, self.agent_id, round_num, limit
            )
        return []

    def get_recent_memories(
        self, round_num: int | None = None, limit: int = 10
    ) -> list[dict[str, object]]:
        """获取最近记忆

        Args:
            round_num: 轮次（可选）
            limit: 返回数量限制

        Returns:
            最近记忆列表
        """
        if self.memory_manager:
            return self.memory_manager.get_recent_memories(
                self.agent_id, round_num, limit
            )
        return []

    def kill(self) -> None:
        """标记Agent为死亡"""
        self.is_alive = False
        logger["agent"].info(f"Agent {self.name} (ID: {self.agent_id}) 已死亡")
