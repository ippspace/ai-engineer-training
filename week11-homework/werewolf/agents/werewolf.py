from ..prompts.werewolf_prompts import (
    WEREWOLF_NIGHT_PROMPT,
    WEREWOLF_SPEAK_PROMPT,
    WEREWOLF_SYSTEM_PROMPT,
    WEREWOLF_VOTE_PROMPT,
)
from ..utils.logger import logger
from .base_agent import BaseAgent


class WerewolfAgent(BaseAgent):
    """狼人Agent类"""

    werewolf_partners: list[object]

    def __init__(self, agent_id: str, name: str):
        """初始化狼人Agent

        Args:
            agent_id: Agent唯一标识
            name: Agent名称
        """
        super().__init__(agent_id, name, "狼人")
        self.system_prompt = WEREWOLF_SYSTEM_PROMPT
        self.werewolf_partners = []

    def set_werewolf_partners(self, partners: list[object]) -> None:
        """设置狼人同伴

        Args:
            partners: 狼人同伴列表
        """
        self.werewolf_partners = partners

    def night_action(self, game_state: dict[str, object], round_num: int) -> str | None:
        """夜晚行动，选择要杀害的村民

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            要杀害的玩家ID
        """
        try:
            # 准备夜晚行动提示
            alive_players = ", ".join(
                [
                    f"{p.name} (ID: {p.agent_id})"
                    for p in game_state["alive_players"]
                    if p.role != "狼人"
                ]
            )
            dead_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["dead_players"]]
            )
            werewolf_partners = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in self.werewolf_partners]
            )

            night_prompt = WEREWOLF_NIGHT_PROMPT.format(
                round=round_num,
                alive_players=alive_players,
                dead_players=dead_players,
                werewolf_partners=werewolf_partners,
            )

            # 生成选择
            messages = [("system", self.system_prompt), ("user", night_prompt)]

            response = self.llm.invoke(messages)
            target_id = response.content.strip()

            # 添加到记忆
            self.add_memory(f"我在第 {round_num} 轮夜晚选择杀害 {target_id}", round_num)

            logger["agent"].info(
                f"狼人 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮夜晚选择杀害 {target_id}"
            )

            return target_id
        except Exception as e:
            logger["agent"].error(
                f"狼人 {self.name} (ID: {self.agent_id}) 夜晚行动失败: {e}"
            )
            # 随机选择一个存活村民
            alive_villagers = [
                p for p in game_state["alive_players"] if p.role != "狼人"
            ]
            if alive_villagers:
                return alive_villagers[0].agent_id
            return None

    def speak(self, game_state: dict[str, object], round_num: int) -> str:
        """狼人发言，伪装成村民

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

            speak_prompt = WEREWOLF_SPEAK_PROMPT.format(
                round=round_num,
                phase="白天发言",
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
                f"狼人 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮发言: {speech_content[:50]}..."
            )

            return speech_content
        except Exception as e:
            logger["agent"].error(
                f"狼人 {self.name} (ID: {self.agent_id}) 发言失败: {e}"
            )
            return "我现在有点混乱，稍后再发言。"

    def vote(self, game_state: dict[str, object], round_num: int) -> str:
        """狼人投票，保护自己和同伴

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

            vote_prompt = WEREWOLF_VOTE_PROMPT.format(
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
                f"狼人 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮投票给了 {vote_target}"
            )

            return vote_target
        except Exception as e:
            logger["agent"].error(
                f"狼人 {self.name} (ID: {self.agent_id}) 投票失败: {e}"
            )
            # 随机选择一个存活村民投票
            alive_villagers = [
                p for p in game_state["alive_players"] if p.role != "狼人"
            ]
            if alive_villagers:
                return alive_villagers[0].agent_id
            return self.agent_id
