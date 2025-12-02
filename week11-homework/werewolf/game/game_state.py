class GameState:
    """游戏状态类"""

    def __init__(self):
        """初始化游戏状态"""
        self.round = 1
        self.phase = "夜晚"  # 夜晚/白天发言/投票
        self.agents = []
        self.alive_players = []
        self.dead_players = []
        self.speeches = []  # 每轮发言记录
        self.votes = []  # 每轮投票记录
        self.night_kills = []  # 夜晚杀害记录

    def update_phase(self, phase):
        """更新游戏阶段

        Args:
            phase: 新的游戏阶段
        """
        self.phase = phase

    def next_round(self):
        """进入下一轮"""
        self.round += 1
        self.phase = "夜晚"

    def add_speech(self, agent_id, agent_name, content):
        """添加发言记录

        Args:
            agent_id: 发言者ID
            agent_name: 发言者名称
            content: 发言内容
        """
        # 确保当前轮次的发言列表存在
        if len(self.speeches) < self.round:
            self.speeches.append([])

        self.speeches[self.round - 1].append(
            {"agent_id": agent_id, "agent_name": agent_name, "content": content}
        )

    def add_vote(self, voter_id, voter_name, target_id):
        """添加投票记录

        Args:
            voter_id: 投票者ID
            voter_name: 投票者名称
            target_id: 投票目标ID
        """
        # 确保当前轮次的投票列表存在
        if len(self.votes) < self.round:
            self.votes.append([])

        self.votes[self.round - 1].append(
            {"voter_id": voter_id, "voter_name": voter_name, "target_id": target_id}
        )

    def add_night_kill(self, player_id, player_name):
        """添加夜晚杀害记录

        Args:
            player_id: 被杀害玩家ID
            player_name: 被杀害玩家名称
        """
        self.night_kills.append(
            {"round": self.round, "player_id": player_id, "player_name": player_name}
        )

    def get_alive_players_by_role(self, role):
        """根据角色获取存活玩家

        Args:
            role: 角色名称

        Returns:
            存活玩家列表
        """
        return [agent for agent in self.alive_players if agent.role == role]

    def to_dict(self):
        """转换为字典格式

        Returns:
            游戏状态字典
        """
        return {
            "round": self.round,
            "phase": self.phase,
            "agents": self.agents,
            "alive_players": self.alive_players,
            "dead_players": self.dead_players,
            "speeches": self.speeches,
            "votes": self.votes,
            "night_kills": self.night_kills,
        }
