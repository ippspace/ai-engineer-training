from ..utils.logger import logger
from .base_agent import BaseAgent


class ModeratorAgent(BaseAgent):
    """主持人Agent类，负责协调游戏流程"""

    def __init__(self):
        """初始化主持人Agent"""
        super().__init__("moderator", "主持人", "主持人")

    def initialize_game(self, agents: list[BaseAgent]) -> dict[str, object]:
        """初始化游戏

        Args:
            agents: 所有Agent列表

        Returns:
            初始游戏状态
        """
        try:
            # 初始化游戏状态
            game_state = {
                "round": 1,
                "phase": "夜晚",
                "agents": agents,
                "alive_players": [agent for agent in agents if agent.role != "主持人"],
                "dead_players": [],
                "speeches": [],
                "votes": [],
                "night_kills": [],
            }

            logger["game"].info("游戏初始化完成")
            logger["game"].info(
                f"参与玩家: {', '.join([f'{a.name} (ID: {a.agent_id}, 角色: {a.role})' for a in agents if a.role != '主持人'])}"
            )

            return game_state
        except Exception as e:
            logger["game"].error(f"游戏初始化失败: {e}")
            raise

    def check_victory_condition(self, game_state: dict[str, object]) -> str | None:
        """检查胜利条件

        Args:
            game_state: 游戏状态

        Returns:
            胜利方（"村民"/"狼人"/None）
        """
        # 统计存活玩家数量
        alive_werewolves = [
            agent for agent in game_state["alive_players"] if agent.role == "狼人"
        ]
        # 村民阵营包括村民、预言家、女巫、猎人
        alive_villagers = [
            agent
            for agent in game_state["alive_players"]
            if agent.role in ["村民", "预言家", "女巫", "猎人"]
        ]

        logger["game"].debug(
            f"存活狼人数量: {len(alive_werewolves)}, 存活村民阵营数量: {len(alive_villagers)}"
        )

        # 胜利条件判断
        if len(alive_werewolves) == 0:
            logger["game"].info("村民获胜！所有狼人已被消灭。")
            return "村民"
        elif len(alive_werewolves) > len(alive_villagers):
            logger["game"].info("狼人获胜！狼人数量已超过村民数量。")
            return "狼人"
        else:
            return None

    def process_night_action(self, game_state: dict[str, object]) -> dict[str, object]:
        """处理夜晚行动

        Args:
            game_state: 游戏状态

        Returns:
            更新后的游戏状态
        """
        try:
            logger["game"].info(f"第 {game_state['round']} 轮夜晚开始")

            # 1. 并行执行所有角色的夜晚行动
            import concurrent.futures

            # 收集所有需要行动的角色
            werewolves = [
                agent for agent in game_state["alive_players"] if agent.role == "狼人"
            ]
            seers = [
                agent for agent in game_state["alive_players"] if agent.role == "预言家"
            ]
            witches = [
                agent for agent in game_state["alive_players"] if agent.role == "女巫"
            ]

            # 定义行动包装函数
            def agent_night_action(agent):
                return agent.agent_id, agent.night_action(game_state, game_state["round"])

            # 并行执行
            night_results = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 提交所有任务
                future_to_agent = {}
                for agent in werewolves + seers + witches:
                    future_to_agent[executor.submit(agent_night_action, agent)] = agent

                # 获取结果
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        agent_id, result = future.result()
                        night_results[agent_id] = result
                    except Exception as e:
                        logger["game"].error(f"Agent {agent.name} 夜晚行动失败: {e}")

            # 处理狼人行动结果
            night_choices = {}
            for werewolf in werewolves:
                target_id = night_results.get(werewolf.agent_id)
                if target_id:
                    if target_id in night_choices:
                        night_choices[target_id] += 1
                    else:
                        night_choices[target_id] = 1

            # 确定最终杀害目标（得票最多的玩家）
            killed_player = None
            if night_choices:
                killed_player_id = max(night_choices, key=night_choices.get)
                killed_player = next(
                    agent
                    for agent in game_state["alive_players"]
                    if agent.agent_id == killed_player_id
                )

                # 先不立即杀死，等待女巫处理
                game_state["night_kills"].append(
                    {
                        "round": game_state["round"],
                        "player_id": killed_player_id,
                        "player_name": killed_player.name,
                    }
                )

            # 处理预言家行动结果
            for seer in seers:
                checked_player_id = night_results.get(seer.agent_id)
                if checked_player_id:
                    checked_player = next(
                        (agent for agent in game_state["alive_players"] if agent.agent_id == checked_player_id),
                        None
                    )
                    if checked_player:
                        logger["game"].info(
                            f"第 {game_state['round']} 轮夜晚，预言家 {seer.name} 查验了 {checked_player.name} (ID: {checked_player_id}) 的身份，结果是: {checked_player.role}"
                        )

            # 处理女巫行动结果
            for witch in witches:
                witch_action = night_results.get(witch.agent_id)
                if witch_action == "antidote" and witch.has_antidote and killed_player:
                    # 使用解药救回被狼人杀害的玩家
                    witch.has_antidote = False
                    # 从night_kills中移除当前轮次的杀害记录
                    game_state["night_kills"] = [
                        kill for kill in game_state["night_kills"]
                        if not (kill["round"] == game_state["round"] and kill["player_id"] == killed_player.agent_id)
                    ]
                    # 重置killed_player，避免后续错误处理
                    killed_player = None
                    logger["game"].info(
                        f"第 {game_state['round']} 轮夜晚，女巫 {witch.name} 使用了解药救回了被杀害的玩家"
                    )
                elif witch_action and witch_action.startswith("poison ") and witch.has_poison:
                    # 使用毒药毒死一名玩家
                    witch.has_poison = False
                    poisoned_player_id = witch_action.split()[1]
                    poisoned_player = next(
                        (agent for agent in game_state["alive_players"] if agent.agent_id == poisoned_player_id),
                        None
                    )
                    
                    if poisoned_player:
                        # 标记玩家为死亡
                        poisoned_player.kill()
                        game_state["alive_players"].remove(poisoned_player)
                        game_state["dead_players"].append(poisoned_player)

                        logger["game"].info(
                            f"第 {game_state['round']} 轮夜晚，女巫 {witch.name} 使用毒药毒死了 {poisoned_player.name} (ID: {poisoned_player_id})"
                        )

            # 4. 最终处理狼人杀害
            if killed_player:
                # 标记玩家为死亡
                killed_player.kill()

                # 更新游戏状态
                game_state["alive_players"].remove(killed_player)
                game_state["dead_players"].append(killed_player)

                logger["game"].info(
                    f"第 {game_state['round']} 轮夜晚，{killed_player.name} (ID: {killed_player.agent_id}) 被狼人杀害"
                )

            # 转换到白天
            game_state["phase"] = "白天发言"

            return game_state
        except Exception as e:
            logger["game"].error(f"处理夜晚行动失败: {e}")
            raise

    def process_day_speech(self, game_state: dict[str, object]) -> dict[str, object]:
        """处理白天发言
        
        注意：发言必须顺序进行，不能并行，因为后发言的玩家需要知道先发言玩家的内容。

        Args:
            game_state: 游戏状态

        Returns:
            更新后的游戏状态
        """
        try:
            logger["game"].info(f"第 {game_state['round']} 轮白天发言开始")

            # 初始化本轮发言列表
            current_speeches = []

            # 所有存活玩家依次发言
            for agent in game_state["alive_players"]:
                speech_content = agent.speak(game_state, game_state["round"])
                current_speeches.append(
                    {
                        "agent_id": agent.agent_id,
                        "agent_name": agent.name,
                        "content": speech_content,
                    }
                )

            # 添加到游戏状态
            game_state["speeches"].append(current_speeches)

            # 转换到投票阶段
            game_state["phase"] = "投票"

            return game_state
        except Exception as e:
            logger["game"].error(f"处理白天发言失败: {e}")
            raise

    def process_vote(self, game_state: dict[str, object]) -> dict[str, object]:
        """处理投票

        Args:
            game_state: 游戏状态

        Returns:
            更新后的游戏状态
        """
        try:
            logger["game"].info(f"第 {game_state['round']} 轮投票开始")

            # 初始化投票计数
            vote_counts = {}
            current_votes = []

            # 并行执行所有存活玩家的投票
            import concurrent.futures

            def agent_vote(agent):
                return agent.agent_id, agent.name, agent.vote(game_state, game_state["round"])

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_agent = {
                    executor.submit(agent_vote, agent): agent 
                    for agent in game_state["alive_players"]
                }
                
                for future in concurrent.futures.as_completed(future_to_agent):
                    try:
                        voter_id, voter_name, vote_target = future.result()
                        
                        current_votes.append(
                            {
                                "voter_id": voter_id,
                                "voter_name": voter_name,
                                "target_id": vote_target,
                            }
                        )

                        if vote_target in vote_counts:
                            vote_counts[vote_target] += 1
                        else:
                            vote_counts[vote_target] = 1
                    except Exception as e:
                        logger["game"].error(f"处理投票结果失败: {e}")

            # 添加到游戏状态
            game_state["votes"].append(current_votes)

            # 确定被处决的玩家（得票最多的玩家）
            if vote_counts:
                executed_player_id = max(vote_counts, key=vote_counts.get)
                executed_player = next(
                    agent
                    for agent in game_state["alive_players"]
                    if agent.agent_id == executed_player_id
                )

                # 标记玩家为死亡
                executed_player.kill()

                # 更新游戏状态
                game_state["alive_players"].remove(executed_player)
                game_state["dead_players"].append(executed_player)

                logger["game"].info(
                    f"第 {game_state['round']} 轮投票，{executed_player.name} (ID: {executed_player_id}) 被处决，得票数: {vote_counts[executed_player_id]}"
                )

                # 处理猎人的死亡行动
                if executed_player.role == "猎人":
                    try:
                        # 调用猎人的死亡行动方法
                        shot_player_id = executed_player.death_action(game_state)
                        if shot_player_id != "no_action" and shot_player_id in [
                            agent.agent_id for agent in game_state["alive_players"]
                        ]:
                            shot_player = next(
                                agent
                                for agent in game_state["alive_players"]
                                if agent.agent_id == shot_player_id
                            )

                            # 标记被枪杀的玩家为死亡
                            shot_player.kill()
                            game_state["alive_players"].remove(shot_player)
                            game_state["dead_players"].append(shot_player)

                            logger["game"].info(
                                f"第 {game_state['round']} 轮，猎人 {executed_player.name} 被处决后开枪带走了 {shot_player.name} (ID: {shot_player_id})"
                            )
                    except Exception as e:
                        logger["game"].error(f"处理猎人死亡行动失败: {e}")

            # 转换到下一轮夜晚
            game_state["round"] += 1
            game_state["phase"] = "夜晚"

            return game_state
        except Exception as e:
            logger["game"].error(f"处理投票失败: {e}")
            raise

    def speak(self, game_state: dict[str, object], round_num: int) -> str:
        """主持人发言（不使用）

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            发言内容
        """
        return ""

    def vote(self, game_state: dict[str, object], round_num: int) -> str:
        """主持人投票（不使用）

        Args:
            game_state: 游戏状态
            round_num: 当前轮次

        Returns:
            投票对象ID
        """
        return ""
