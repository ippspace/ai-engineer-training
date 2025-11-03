import os
import random
import time

from dotenv import load_dotenv

from .agent import CustomerServiceAgent

_ = load_dotenv()


def agent_invoke(agent: CustomerServiceAgent, user_input: str, thread_id: str) -> bool:
    res = agent.invoke(user_input, thread_id)
    if res:
        print(f"客服: {res.content}")
        return True
    else:
        print("客服: 很抱歉，我没有理解您的请求。请重新输入。")
        return False


def demonstrate_order_query_flow(agent: CustomerServiceAgent):
    """演示订单查询流程。"""
    print("\n=== 演示订单查询流程 ===")

    user_input = "我想要查询订单状态"
    thread_id = "demo_query"
    print(f"用户: {user_input}")

    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return

    time.sleep(2)  # 等待响应

    user_input = "订单号是 ORD123456"
    print(f"用户: {user_input}")
    _ = agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id)


def demonstrate_refund_flow(agent: CustomerServiceAgent):
    """演示退款申请流程。"""
    print("\n=== 演示退款申请流程 ===")
    user_input = "我想申请退款"
    thread_id = "demo_refund"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return

    time.sleep(2)

    user_input = "订单号是 ORD123456"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return

    time.sleep(2)

    user_input = "商品质量有问题"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return


def demonstrate_invoice_flow(agent: CustomerServiceAgent):
    """演示发票开具流程。"""
    print("\n=== 演示发票开具流程 ===")
    user_input = "我需要开发票"
    thread_id = "demo_invoice"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return

    time.sleep(2)

    user_input = "订单号是 ORD789012，需要个人发票"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return

    time.sleep(2)

    user_input = "公司名称是 123 公司"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return


def demonstrate_time_inference(agent: CustomerServiceAgent):
    """演示时间推断功能。"""
    print("\n=== 演示时间推断功能 ===")
    user_input = "我昨天下的单，什么时候能发货？"
    thread_id = "demo_time"
    print(f"用户: {user_input}")
    if not agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id):
        return


def demonstrate_chat_interface(agent: CustomerServiceAgent):
    """交互式聊天界面。"""
    print("\n=== 交互式聊天界面 ===")
    print("欢迎使用智能客服系统！")
    print("您可以：查询订单、申请退款、开具发票")
    print("输入 'demo' 查看演示")
    print("输入 'exit' 退出")
    print("输入 'reload' 重新加载插件")
    print("-" * 50)

    thread_id = str(random.randint(1000, 9999))

    while True:
        user_input = input("\n用户: ")

        if user_input.lower() == "exit":
            print("客服: 感谢您使用智能客服系统，再见！")
            break
        elif user_input.lower() == "reload":
            result = agent.reload_tools()
            print(f"系统: {result}")
        elif user_input.lower() == "demo":
            print("\n请选择演示项目:")
            print("1. 订单查询")
            print("2. 退款申请")
            print("3. 发票开具")
            print("4. 时间推断")
            demo_choice = input("请输入选项 (1-4): ")

            if demo_choice == "1":
                demonstrate_order_query_flow(agent)
            elif demo_choice == "2":
                demonstrate_refund_flow(agent)
            elif demo_choice == "3":
                demonstrate_invoice_flow(agent)
            elif demo_choice == "4":
                demonstrate_time_inference(agent)
        else:
            _ = agent_invoke(agent=agent, user_input=user_input, thread_id=thread_id)


def demonstrate_agent():
    # 从环境变量读取配置
    model_name = os.getenv("LLM_MODEL", "gpt-5")
    api_key = os.getenv("LLM_API_KEY", "sk-")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    # 检查必需的环境变量
    if not api_key or api_key == "sk-":
        print("错误: 未设置有效的 LLM_API_KEY 环境变量")
        return

    print("正在初始化智能客服系统...")
    # 创建智能客服代理
    try:
        agent = CustomerServiceAgent(model_name, api_key, base_url)
        print("智能客服系统初始化成功！")

        # 进行功能演示
        # print("\n正在进行功能演示...")
        # demonstrate_order_query_flow(agent)

        # 启动交互式聊天
        demonstrate_chat_interface(agent)

    except Exception as e:
        print(f"初始化失败: {str(e)}")


def main():
    """主函数。"""
    demonstrate_agent()


if __name__ == "__main__":
    main()
