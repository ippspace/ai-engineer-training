from ..prompts.witch_prompts import (
    WITCH_NIGHT_PROMPT,
    WITCH_SPEAK_PROMPT,
    WITCH_SYSTEM_PROMPT,
    WITCH_VOTE_PROMPT,
)
from ..utils.logger import logger
from .base_agent import BaseAgent


class WitchAgent(BaseAgent):
    """女巫Agent类"""

    def __init__(self, agent_id: str, name: str):
        """初始化女巫Agent

        Args:
            agent_id: Agent唯一标识
            name: Agent名称
        """
        super().__init__(agent_id, name, "女巫")
        self.system_prompt = WITCH_SYSTEM_PROMPT
        self.has_antidote = True  # 初始拥有解药
        self.has_poison = True  # 初始拥有毒药

    def night_action(self, game_state: dict[str, object], round_num: int) -> str:
        """夜晚行动，选择是否使用解药或毒药

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            行动选择（antidote/poison 玩家ID/no_action）
        """
        try:
            # 获取被狼人杀害的玩家（假设是上一轮最后一个被杀害的玩家）
            killed_player = ""
            if "night_kills" in game_state and game_state["night_kills"]:
                last_night_kill = game_state["night_kills"][-1]
                if last_night_kill["round"] == round_num:
                    killed_player = f"{last_night_kill['player_name']} (ID: {last_night_kill['player_id']})"

            # 准备夜晚行动提示
            alive_players = ", ".join(
                [
                    f"{p.name} (ID: {p.agent_id})"
                    for p in game_state["alive_players"]
                    if p.agent_id != self.agent_id
                ]
            )
            dead_players = ", ".join(
                [f"{p.name} (ID: {p.agent_id})" for p in game_state["dead_players"]]
            )

            night_prompt = WITCH_NIGHT_PROMPT.format(
                round=round_num,
                killed_player=killed_player,
                has_antidote="有" if self.has_antidote else "无",
                has_poison="有" if self.has_poison else "无",
                alive_players=alive_players,
                dead_players=dead_players,
            )

            # 生成选择
            messages = [("system", self.system_prompt), ("user", night_prompt)]

            response = self.llm.invoke(messages)
            action = response.content.strip()

            # 添加到记忆
            self.add_memory(f"我在第 {round_num} 轮夜晚选择了 {action}", round_num)

            logger["agent"].info(
                f"女巫 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮夜晚选择 {action}"
            )

            return action
        except Exception as e:
            logger["agent"].error(
                f"女巫 {self.name} (ID: {self.agent_id}) 夜晚行动失败: {e}"
            )
            return "no_action"

    def speak(self, game_state: dict[str, object], round_num: int) -> str:
        """女巫发言，分析局势并可能揭露身份

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

            speak_prompt = WITCH_SPEAK_PROMPT.format(
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
                f"女巫 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮发言: {speech_content[:50]}..."
            )

            return speech_content
        except Exception as e:
            logger["agent"].error(
                f"女巫 {self.name} (ID: {self.agent_id}) 发言失败: {e}"
            )
            return "我现在有点混乱，稍后再发言。"

    def vote(self, game_state: dict[str, object], round_num: int) -> str:
        """女巫投票，选择要处决的玩家

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

            vote_prompt = WITCH_VOTE_PROMPT.format(
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
                f"女巫 {self.name} (ID: {self.agent_id}) 在第 {round_num} 轮投票给了 {vote_target}"
            )

            return vote_target
        except Exception as e:
            logger["agent"].error(
                f"女巫 {self.name} (ID: {self.agent_id}) 投票失败: {e}"
            )
            # 随机选择一个存活玩家投票
            alive_agents = [
                p for p in game_state["alive_players"] if p.agent_id != self.agent_id
            ]
            if alive_agents:
                return alive_agents[0].agent_id
            return self.agent_id
