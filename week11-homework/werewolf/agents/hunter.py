from ..prompts.hunter_prompts import (
    HUNTER_DEATH_PROMPT,
    HUNTER_SPEAK_PROMPT,
    HUNTER_SYSTEM_PROMPT,
    HUNTER_VOTE_PROMPT,
)
from ..utils.logger import logger
from .base_agent import BaseAgent


class HunterAgent(BaseAgent):
    """猎人Agent类"""

    def __init__(self, agent_id: str, name: str):
        """初始化猎人Agent

        Args:
            agent_id: Agent唯一标识
            name: Agent名称
        """
        super().__init__(agent_id, name, "猎人")
        self.system_prompt = HUNTER_SYSTEM_PROMPT
        self.has_gun = True  # 初始拥有子弹

    def speak(self, game_state: dict[str, object], round_num: int) -> str:
        """猎人发言，分析局势

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            发言内容
        """
        try:
            # 准备发言提示
            alive_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["alive_players"]]
            )
            dead_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["dead_players"]]
            )

            # 获取上一轮发言记录
            last_round_speeches = ""
            if "speeches" in game_state and round_num > 1:
                for speech in game_state["speeches"][round_num - 2]:
                    last_round_speeches += (
                        f"{speech['agent_name']}: {speech['content']}\n"
                    )

            if not last_round_speeches:
                last_round_speeches = "无"

            speak_prompt = HUNTER_SPEAK_PROMPT.format(
                round=round_num,
                alive_players=alive_players,
                dead_players=dead_players,
                last_round_speeches=last_round_speeches,
            )

            # 生成发言
            messages = [("system", self.system_prompt), ("user", speak_prompt)]

            response = self.llm.invoke(messages)
            speech_content = response.content.strip()

            # 添加到记忆
            self.add_memory(f"我在第 {round_num} 轮发言: {speech_content}", round_num)

            logger["agent"].info(
                f"猎人 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮发言: {speech_content[:50]}..."
            )

            return speech_content
        except Exception as e:
            logger["agent"].error(
                f"猎人 {self.name} (ID: {self.agent_id}) 发言失败: {e}"
            )
            return "我现在有点混乱，稍后再发言。"

    def vote(self, game_state: dict[str, object], round_num: int) -> str:
        """猎人投票，选择要处决的玩家

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            投票对象ID
        """
        try:
            # 准备投票提示
            alive_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["alive_players"]]
            )
            dead_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["dead_players"]]
            )

            # 获取本轮发言记录
            current_speeches = ""
            if "speeches" in game_state and round_num > 0:
                for speech in game_state["speeches"][round_num - 1]:
                    current_speeches += f"{speech['agent_name']}: {speech['content']}\n"

            vote_prompt = HUNTER_VOTE_PROMPT.format(
                round=round_num,
                phase="投票阶段",
                alive_players=alive_players,
                dead_players=dead_players,
                current_speeches=current_speeches,
            )

            # 生成投票
            messages = [("system", self.system_prompt), ("user", vote_prompt)]

            response = self.llm.invoke(messages)
            vote_target = response.content.strip()

            # 添加到记忆
            self.add_memory(f"我在第 {round_num} 轮投票给了 {vote_target}", round_num)

            logger["agent"].info(
                f"猎人 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮投票给了 {vote_target}"
            )

            return vote_target
        except Exception as e:
            logger["agent"].error(
                f"猎人 {self.name} (ID: {self.agent_id}) 投票失败: {e}"
            )
            # 随机选择一个存活玩家投票
            alive_agents = [
                p for p in game_state["alive_players"] if p.agent_id != self.agent_id
            ]
            if alive_agents:
                return alive_agents[0].agent_id
            return self.agent_id

    def death_action(self, game_state: dict[str, object]) -> str:
        """死亡时的行动，选择是否开枪带走一名玩家

        Args:
            game_state: 游戏状态

        Returns:
            要带走的玩家ID或"no_action"
        """
        try:
            if not self.has_gun:
                return "no_action"

            # 准备死亡行动提示
            alive_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["alive_players"]]
            )
            dead_players = ", ".join(
                [
                    f"{p.name} (ID: {p.agent_id})"
                    for p in game_state["dead_players"]
                    if p.agent_id != self.agent_id
                ]
            )

            death_prompt = HUNTER_DEATH_PROMPT.format(
                alive_players=alive_players,
                dead_players=dead_players,
            )

            # 生成选择
            messages = [("system", self.system_prompt), ("user", death_prompt)]

            response = self.llm.invoke(messages)
            action = response.content.strip()

            # 消耗子弹
            if action != "no_action":
                self.has_gun = False

            logger["agent"].info(
                f"猎人 {self.name} (ID: {self.agent_id}) 选择开枪带走 {action}"
            )

            return action
        except Exception as e:
            logger["agent"].error(
                f"猎人 {self.name} (ID: {self.agent_id}) 死亡行动失败: {e}"
            )
            return "no_action"
