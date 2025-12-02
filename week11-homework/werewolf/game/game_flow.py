from ..agents.base_agent import BaseAgent
from ..agents.moderator import ModeratorAgent
from ..memory.memory_manager import MemoryManager
from ..utils.logger import logger


class GameFlow:
    """游戏流程控制类"""

    agents: list[BaseAgent]
    memory_manager: MemoryManager
    moderator: ModeratorAgent
    game_state: dict[str, object] | None

    def __init__(self, agents: list[BaseAgent], memory_manager: MemoryManager):
        """初始化游戏流程

        Args:
            agents: 所有Agent列表
            memory_manager: 记忆管理器实例
        """
        self.agents = agents
        self.memory_manager = memory_manager
        self.moderator = ModeratorAgent()
        self.game_state = None

        # 为所有Agent设置记忆管理器
        for agent in self.agents:
            agent.set_memory_manager(self.memory_manager)

    def start_game(self) -> str:
        """开始游戏

        Returns:
            游戏结果
        """
        try:
            logger["game"].info("游戏开始！")

            # 初始化游戏
            if not self.game_state:
                self.game_state = self.moderator.initialize_game(self.agents)

            # 游戏主循环
            while True:
                winner = self.step()
                if winner:
                    return winner

        except Exception as e:
            logger["game"].error(f"游戏执行失败: {e}")
            raise
        finally:
            logger["game"].info("游戏结束！")

    def step(self) -> str | None:
        """执行一步游戏

        Returns:
            如果游戏结束返回胜利方，否则返回None
        """
        if not self.game_state:
            self.game_state = self.moderator.initialize_game(self.agents)

        # 检查胜利条件
        winner = self.moderator.check_victory_condition(self.game_state)
        if winner:
            return winner

        # 根据当前阶段执行相应操作
        if self.game_state["phase"] == "夜晚":
            self.game_state = self.moderator.process_night_action(self.game_state)
        elif self.game_state["phase"] == "白天发言":
            self.game_state = self.moderator.process_day_speech(self.game_state)
        elif self.game_state["phase"] == "投票":
            self.game_state = self.moderator.process_vote(self.game_state)

        # 检查胜利条件（再次检查，因为夜晚行动或投票可能导致游戏结束）
        winner = self.moderator.check_victory_condition(self.game_state)
        return winner

    def get_game_state(self) -> dict[str, object] | None:
        """获取当前游戏状态

        Returns:
            游戏状态
        """
        return self.game_state

    def print_game_summary(self) -> None:
        """打印游戏总结"""
        if not self.game_state:
            logger["game"].info("游戏尚未开始")
            return

        logger["game"].info("=== 游戏总结 ===")
        logger["game"].info(f"总轮数: {self.game_state['round'] - 1}")

        # 打印参与玩家
        logger["game"].info("参与玩家:")
        for agent in self.game_state["agents"]:
            if agent.role != "主持人":
                status = "存活" if agent.is_alive else "死亡"
                logger["game"].info(
                    f"- {agent.name} (ID: {agent.agent_id}, 角色: {agent.role}, 状态: {status})"
                )

        # 打印夜晚杀害记录
        if self.game_state["night_kills"]:
            logger["game"].info("夜晚杀害记录:")
            for kill in self.game_state["night_kills"]:
                logger["game"].info(
                    f"- 第 {kill['round']} 轮: {kill['player_name']} (ID: {kill['player_id']})"
                )

        # 打印每轮发言和投票
        for i in range(len(self.game_state["speeches"])):
            round_num = i + 1
            logger["game"].info(f"\n=== 第 {round_num} 轮 ===")

            # 打印发言
            if self.game_state["speeches"][i]:
                logger["game"].info("发言记录:")
                for speech in self.game_state["speeches"][i]:
                    logger["game"].info(
                        f"- {speech['agent_name']}: {speech['content']}"
                    )

            # 打印投票
            if i < len(self.game_state["votes"]) and self.game_state["votes"][i]:
                logger["game"].info("投票记录:")
                for vote in self.game_state["votes"][i]:
                    logger["game"].info(
                        f"- {vote['voter_name']} 投票给 {vote['target_id']}"
                    )

        logger["game"].info("=== 游戏总结结束 ===")
