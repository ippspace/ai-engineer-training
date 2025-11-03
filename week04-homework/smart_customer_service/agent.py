import importlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from types import ModuleType

from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig

# 模拟订单数据库
ORDERS_DB = {
    "ORD123456": {
        "status": "已发货",
        "logistics": "顺丰快递 SF123456789",
        "products": ["智能手表", "充电器"],
        "amount": 1299.00,
    },
    "ORD654321": {
        "status": "待付款",
        "logistics": None,
        "products": ["蓝牙耳机"],
        "amount": 299.00,
    },
    "ORD789012": {
        "status": "已签收",
        "logistics": "京东快递 JD987654321",
        "products": ["笔记本电脑"],
        "amount": 5999.00,
    },
}


# 模拟退款数据库
REFUNDS_DB = {}


@tool
def query_order(order_id: str) -> str:
    """查询订单信息。当用户需要查询订单状态或物流信息时使用。"""
    if order_id in ORDERS_DB:
        order = ORDERS_DB[order_id]
        logistics_info = (
            order["logistics"] if order["logistics"] is not None else "暂无物流信息"
        )
        # 确保products是列表类型
        products = order["products"]
        if isinstance(products, list):
            products_str = ", ".join(str(p) for p in products)
        else:
            products_str = str(products)
        return f"订单号: {order_id}\n状态: {order['status']}\n物流: {logistics_info}\n商品: {products_str}\n金额: ¥{order['amount']}"
    else:
        return f"未找到订单号为 {order_id} 的订单信息"


@tool
def apply_refund(order_id: str, reason: str) -> str:
    """申请退款。当用户需要对订单申请退款时使用。"""
    if order_id not in ORDERS_DB:
        return f"未找到订单号为 {order_id} 的订单信息，无法申请退款"

    order = ORDERS_DB[order_id]
    if order["status"] == "待付款":
        return f"订单 {order_id} 尚未付款，无需申请退款"

    # 生成退款单号
    refund_id = f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}"
    REFUNDS_DB[refund_id] = {
        "order_id": order_id,
        "reason": reason,
        "status": "处理中",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return f"退款申请已提交成功！\n退款单号: {refund_id}\n订单号: {order_id}\n退款原因: {reason}\n当前状态: 处理中"


@tool
def create_invoice(order_id: str, invoice_type: str, company_name: str = "") -> str:
    """开具发票。当用户需要为订单开具发票时使用。"""
    if order_id not in ORDERS_DB:
        return f"未找到订单号为 {order_id} 的订单信息，无法开具发票"

    order = ORDERS_DB[order_id]
    if order["status"] == "待付款":
        return f"订单 {order_id} 尚未付款，无法开具发票"

    # 生成发票信息
    invoice_no = f"INV{datetime.now().strftime('%Y%m%d%H%M%S')}"
    invoice_content = f"订单 {order_id} 的发票已开具成功！\n发票号码: {invoice_no}\n发票类型: {invoice_type}\n"

    if invoice_type == "公司发票" and company_name:
        invoice_content += f"公司名称: {company_name}\n"

    invoice_content += f"发票金额: ¥{order['amount']}\n开具时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return invoice_content


@dataclass
class Context:
    """自定义运行时上下文模式。"""

    user_id: str
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )


# 使用dataclass定义响应格式
@dataclass
class CustomerServiceResponse:
    """客服响应模式。"""

    content: str  # 响应内容
    requires_followup: bool = False  # 是否需要追问
    followup_prompt: str | None = None  # 追问提示
    action_taken: str | None = None  # 执行的操作


class DynamicToolManager:
    """动态工具管理器，支持插件热重载。"""

    def __init__(self):
        # 确保默认工具都是BaseTool类型
        self.default_tools: list[BaseTool] = [query_order, apply_refund, create_invoice]
        self.tools: list[BaseTool] = self.default_tools.copy()  # 创建工具列表的副本
        self.plugin_dir = os.path.join(os.path.dirname(__file__), "plugins")
        self.loaded_plugins: dict[str, ModuleType] = {}

    def load_plugins(self):
        """加载插件目录中的所有插件。"""
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            return

        # 查找所有.py文件
        plugin_files = [
            f
            for f in os.listdir(self.plugin_dir)
            if not f.startswith("__") and f.endswith(".py")
        ]

        for plugin_file in plugin_files:
            plugin_name = plugin_file[:-3]
            try:
                # 动态导入插件
                module_path = f"smart_customer_service.plugins.{plugin_name}"
                if module_path in self.loaded_plugins:
                    # 热重载现有插件
                    module = importlib.reload(self.loaded_plugins[module_path])
                    self.loaded_plugins[module_path] = module
                else:
                    # 导入新插件
                    module = importlib.import_module(module_path)
                    self.loaded_plugins[module_path] = module

                # 查找并注册工具函数
                module = self.loaded_plugins[module_path]
                print(f"正在处理模块 {module_path} 中的工具...")
                for attr_name in dir(module):
                    # 跳过私有属性
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(module, attr_name)

                    # 确保只添加BaseTool类型的对象
                    if isinstance(attr, BaseTool):
                        # 避免重复添加
                        if attr not in self.tools:
                            self.tools.append(attr)
                            print(f"已添加工具: {attr_name}")

                print(f"已加载/重载插件: {plugin_name}")
            except Exception as e:
                print(f"加载插件 {plugin_name} 失败: {str(e)}")
                import traceback

                traceback.print_exc()

    def get_tools(self) -> list[BaseTool]:
        """获取所有可用工具。"""
        return self.tools

    def reload_plugins(self):
        """重新加载所有插件。"""
        self.tools = self.default_tools.copy()  # 重置工具列表
        self.load_plugins()


class CustomerServiceAgent:
    """智能客服代理类。"""

    # 系统提示模板头部
    SYSTEM_PROMPT_HEADER = """你是一个专业的智能客服，能够帮助用户处理订单查询、退款申请和发票开具等服务。

你可以使用以下工具：
{tools_description}

请根据用户的问题和提供的工具，做出适当的响应：
1. 如果缺少必要参数，请向用户追问
2. 如果信息足够，请调用相应工具
3. 如果不需要工具，请直接回答用户

当前日期是：{current_date}，请在回答中正确处理时间相关的表述（如"昨天"、"今天"等）。"""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        # 初始化模型
        self.model = init_chat_model(
            model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )

        # 初始化工具管理器
        self.tool_manager = DynamicToolManager()

        # 初始化检查点存储器用于多轮对话
        self.checkpointer = InMemorySaver()

        # 存储会话历史
        self.conversation_history = {}

        # 创建代理
        self._create_agent()

    def _generate_tools_description(self):
        """动态生成工具描述。"""
        tools = self.tool_manager.get_tools()
        descriptions: list[str] = []

        for t in tools:
            # 处理LangChain的StructuredTool对象或原始工具函数
            if hasattr(t, "name"):
                # 对于StructuredTool对象
                tool_name = t.name
                description = t.description or ""
            else:
                # 对于原始函数
                tool_name = t.__name__
                description = t.__doc__ or ""

            descriptions.append(f"- {tool_name}: {description.strip()}")

        return "\n".join(descriptions)

    def _create_agent(self):
        """创建或更新代理实例。"""
        # 获取当前日期
        current_date = datetime.now().strftime("%Y-%m-%d")
        # 动态生成工具描述
        tools_description = self._generate_tools_description()
        # 格式化系统提示
        system_prompt = self.SYSTEM_PROMPT_HEADER.format(
            tools_description=tools_description, current_date=current_date
        )
        # 创建代理
        self.agent = create_agent(
            model=self.model,
            system_prompt=system_prompt,
            tools=self.tool_manager.get_tools(),
            context_schema=Context,
            response_format=AutoStrategy(CustomerServiceResponse),
            checkpointer=self.checkpointer,
        )

    def update_model(self, model_name: str, api_key: str, base_url: str):
        """更新模型（热更新功能）。"""
        try:
            self.model = init_chat_model(
                model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.3,
            )

            # 使用新模型重新创建代理
            self._create_agent()

            return "模型更新成功"
        except Exception as e:
            return f"模型更新失败: {str(e)}"

    def reload_tools(self):
        """重新加载工具和插件。"""
        self.tool_manager.reload_plugins()

        # 使用新工具集重新创建代理
        self._create_agent()

        return "工具和插件重新加载成功"

    def invoke(self, user_input: str | None = None, thread_id: str = "1"):
        """调用代理处理用户输入。"""
        # 如果没有提供用户输入，使用默认输入进行演示
        if user_input is None:
            user_input = "我想要查询订单状态"

        # 初始化会话历史（如果不存在）
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []

        # 获取当前日期，用于时间推断
        current_date = datetime.now().strftime("%Y-%m-%d")

        # 创建配置
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 更新会话历史
        self.conversation_history[thread_id].append(
            {"role": "user", "content": user_input}
        )

        try:
            # 调用代理
            response = self.agent.invoke(
                {"messages": self.conversation_history[thread_id]},
                config=config,
                context=Context(user_id=thread_id, current_date=current_date),
            )

            # 处理响应
            if "structured_response" in response:
                structured_response: CustomerServiceResponse = response[
                    "structured_response"
                ]
                # 更新会话历史
                self.conversation_history[thread_id].append(
                    {"role": "assistant", "content": structured_response.content}
                )

                return structured_response
            else:
                # 处理非结构化响应
                if "messages" in response and response["messages"]:
                    latest_message = response["messages"][-1]
                    if hasattr(latest_message, "content"):
                        content = str(latest_message.content)
                        print(f"客服: {content}")
                        self.conversation_history[thread_id].append(
                            {"role": "assistant", "content": content}
                        )
                        return CustomerServiceResponse(content=content)

                # 默认返回
                response_str = str(response)
                return CustomerServiceResponse(content=response_str)
        except Exception as e:
            error_message = f"处理请求时出错: {str(e)}"
            print(error_message)
            return None
