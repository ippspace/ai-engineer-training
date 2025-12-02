import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st

from werewolf.agents.villager import VillagerAgent
from werewolf.agents.werewolf import WerewolfAgent
from werewolf.game.game_flow import GameFlow
from werewolf.memory.memory_manager import MemoryManager

# 设置页面配置
st.set_page_config(page_title="狼人杀游戏可视化", page_icon="🐺", layout="wide")

# 初始化Session State
if "game_flow" not in st.session_state:
    st.session_state.game_flow = None
if "game_started" not in st.session_state:
    st.session_state.game_started = False
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False
if "log_output" not in st.session_state:
    st.session_state.log_output = []


def init_game():
    """初始化游戏"""
    try:
        memory_manager = MemoryManager()

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

        st.session_state.game_flow = GameFlow(agents, memory_manager)
        st.session_state.game_started = True
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.auto_play = False
        st.session_state.log_output = ["游戏初始化完成"]

        # 初始化游戏状态
        st.session_state.game_flow.game_state = (
            st.session_state.game_flow.moderator.initialize_game(agents)
        )

    except Exception as e:
        st.error(f"初始化游戏失败: {e}")


def step_game():
    """执行一步游戏"""
    if st.session_state.game_flow and not st.session_state.game_over:
        try:
            winner = st.session_state.game_flow.step()
            if winner:
                st.session_state.game_over = True
                st.session_state.winner = winner
                st.session_state.auto_play = False  # 停止自动播放
                st.success(f"游戏结束！获胜方: {winner}")
        except Exception as e:
            st.error(f"游戏执行出错: {e}")
            st.session_state.auto_play = False


# 侧边栏控制
with st.sidebar:
    st.header("游戏控制")

    if not st.session_state.game_started:
        if st.button("开始新游戏", type="primary"):
            init_game()
            st.rerun()
    else:
        if st.button("重新开始"):
            init_game()
            st.rerun()

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "单步执行",
                disabled=st.session_state.game_over or st.session_state.auto_play,
            ):
                step_game()
                st.rerun()

        with col2:
            if st.session_state.auto_play:
                if st.button("停止自动"):
                    st.session_state.auto_play = False
                    st.rerun()
            else:
                if st.button("自动播放", disabled=st.session_state.game_over):
                    st.session_state.auto_play = True
                    st.rerun()

# 主界面
st.title("🐺 狼人杀游戏执行流")

if st.session_state.game_started and st.session_state.game_flow:
    game_state = st.session_state.game_flow.get_game_state()

    if game_state:
        # 顶部状态栏
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("轮次", game_state.get("round", 1))
        with col2:
            st.metric("阶段", game_state.get("phase", "未知"))
        with col3:
            alive_count = len(game_state.get("alive_players", []))
            st.metric("存活玩家", alive_count)
        with col4:
            if st.session_state.game_over:
                st.metric("游戏结果", f"{st.session_state.winner}获胜")
            else:
                st.metric("状态", "进行中")

        # 玩家状态网格
        st.subheader("玩家状态")
        agents = game_state.get("agents", [])
        cols = st.columns(5)
        for i, agent in enumerate(agents):
            if agent.role == "主持人":
                continue

            with cols[i % 5]:
                is_alive = agent in game_state.get("alive_players", [])
                status_icon = "👤" if is_alive else "💀"
                role_icon = "🐺" if agent.role == "狼人" else "村民"

                st.container(border=True).markdown(
                    f"""
                    **{agent.name}**  
                    {status_icon} {agent.role}  
                    ID: `{agent.agent_id}`
                    """
                )

        # 游戏记录 Tabs
        tab1, tab2, tab3 = st.tabs(["📢 发言记录", "🗳️ 投票记录", "🌙 夜晚行动"])

        with tab1:
            speeches = game_state.get("speeches", [])
            for i, round_speeches in enumerate(reversed(speeches)):
                with st.expander(f"第 {len(speeches) - i} 轮发言", expanded=(i == 0)):
                    for speech in round_speeches:
                        st.markdown(f"**{speech['agent_name']}**: {speech['content']}")

        with tab2:
            votes = game_state.get("votes", [])
            for i, round_votes in enumerate(reversed(votes)):
                with st.expander(f"第 {len(votes) - i} 轮投票", expanded=(i == 0)):
                    for vote in round_votes:
                        st.markdown(
                            f"**{vote['voter_name']}** 投给了 **{vote['target_id']}**"
                        )

        with tab3:
            night_kills = game_state.get("night_kills", [])
            if night_kills:
                for kill in reversed(night_kills):
                    st.markdown(
                        f"第 **{kill['round']}** 轮: **{kill['player_name']}** 被杀害"
                    )
            else:
                st.info("暂无夜晚伤亡")

    # 自动播放逻辑
    if st.session_state.auto_play and not st.session_state.game_over:
        time.sleep(1)  # 间隔1秒
        step_game()
        st.rerun()

else:
    st.info("点击左侧'开始新游戏'启动")
