class GameRules:
    """游戏规则类"""

    # 游戏配置
    DEFAULT_NUM_PLAYERS = 5
    DEFAULT_NUM_WEREWOLVES = 2
    DEFAULT_NUM_VILLAGERS = 3

    @staticmethod
    def get_role_distribution(num_players):
        """获取角色分配

        Args:
            num_players: 玩家数量

        Returns:
            角色分配字典
        """
        # 简单的角色分配逻辑，可根据需要扩展
        if num_players == 5:
            return {"狼人": 2, "村民": 3}
        elif num_players == 6:
            return {"狼人": 2, "村民": 4}
        elif num_players == 7:
            return {"狼人": 3, "村民": 4}
        elif num_players == 8:
            return {"狼人": 3, "村民": 5}
        else:
            # 默认分配
            werewolves = max(1, num_players // 3)
            villagers = num_players - werewolves
            return {"狼人": werewolves, "村民": villagers}

    @staticmethod
    def check_victory(alive_werewolves, alive_villagers):
        """检查胜利条件

        Args:
            alive_werewolves: 存活狼人数量
            alive_villagers: 存活村民数量

        Returns:
            胜利方（"村民"/"狼人"/None）
        """
        if alive_werewolves == 0:
            return "村民"
        elif alive_werewolves >= alive_villagers:
            return "狼人"
        else:
            return None

    @staticmethod
    def get_max_rounds(num_players):
        """获取最大轮数

        Args:
            num_players: 玩家数量

        Returns:
            最大轮数
        """
        return num_players * 2  # 简单设置，可根据需要调整
