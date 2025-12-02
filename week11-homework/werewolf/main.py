from .agents.villager import VillagerAgent
from .agents.werewolf import WerewolfAgent
from .game.game_flow import GameFlow
from .memory.memory_manager import MemoryManager
from .utils.logger import logger


def main():
    """游戏主入口"""
    memory_manager = None
    try:
        # 初始化记忆管理器
        memory_manager = MemoryManager()
        logger["game"].info("记忆管理器初始化完成")

        # 创建玩家Agent
        agents = []

        # 创建村民
        villagers = [
            VillagerAgent("villager_1", "村民1"),
            VillagerAgent("villager_2", "村民2"),
            VillagerAgent("villager_3", "村民3"),
        ]
        agents.extend(villagers)

        # 创建狼人
        werewolves = [
            WerewolfAgent("werewolf_1", "狼人1"),
            WerewolfAgent("werewolf_2", "狼人2"),
        ]
        agents.extend(werewolves)

        # 设置狼人同伴关系
        for werewolf in werewolves:
            werewolf.set_werewolf_partners(
                [w for w in werewolves if w.agent_id != werewolf.agent_id]
            )

        logger["game"].info("所有Agent创建完成")

        # 初始化游戏流程
        game_flow = GameFlow(agents, memory_manager)

        # 启动游戏
        winner = game_flow.start_game()

        # 打印游戏结果
        logger["game"].info("\n=== 游戏结果 ===")
        logger["game"].info(f"胜利方: {winner}")

        # 打印游戏总结
        game_flow.print_game_summary()

    except Exception as e:
        logger["game"].error(f"游戏运行失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 关闭记忆管理器
        if "memory_manager" in locals() and memory_manager is not None:
            memory_manager.close()


if __name__ == "__main__":
    main()
