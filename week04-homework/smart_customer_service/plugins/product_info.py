from langchain.tools import tool

# 模拟产品数据库
PRODUCTS_DB = {
    "智能手表": {
        "price": 1299.00,
        "description": "多功能智能手表，支持心率监测、运动追踪等功能",
        "stock": 100,
    },
    "蓝牙耳机": {
        "price": 299.00,
        "description": "无线蓝牙耳机，主动降噪，续航可达24小时",
        "stock": 200,
    },
    "笔记本电脑": {
        "price": 5999.00,
        "description": "高性能笔记本电脑，16GB内存，512GB固态硬盘",
        "stock": 50,
    },
    "充电器": {
        "price": 99.00,
        "description": "快速充电器，支持多种设备充电",
        "stock": 300,
    },
}


@tool
def query_product_info(product_name: str) -> str:
    """查询产品信息。当用户需要了解产品的价格、描述或库存等信息时使用。"""
    product_name = product_name.strip()

    # 精确匹配
    if product_name in PRODUCTS_DB:
        product = PRODUCTS_DB[product_name]
        return f"产品名称: {product_name}\n价格: ¥{product['price']}\n描述: {product['description']}\n库存: {product['stock']} 件"

    # 模糊匹配
    for key in PRODUCTS_DB:
        if product_name in key:
            product = PRODUCTS_DB[key]
            return f"产品名称: {key}\n价格: ¥{product['price']}\n描述: {product['description']}\n库存: {product['stock']} 件"

    return f"未找到产品 '{product_name}' 的信息"


@tool
def check_product_stock(product_name: str) -> str:
    """检查产品库存。当用户只想了解产品库存情况时使用。"""
    product_name = product_name.strip()

    # 精确匹配
    if product_name in PRODUCTS_DB:
        stock = PRODUCTS_DB[product_name]["stock"]
        # 确保stock是整数类型
        stock_int = int(stock)
        status = "充足" if stock_int > 50 else "紧张" if stock_int > 0 else "缺货"
        return f"产品 '{product_name}' 的库存状态: {stock} 件 ({status})"

    # 模糊匹配
    for key in PRODUCTS_DB:
        if product_name in key:
            stock = PRODUCTS_DB[key]["stock"]
            # 确保stock是整数类型
            stock_int = int(stock)
            status = "充足" if stock_int > 50 else "紧张" if stock_int > 0 else "缺货"
            return f"产品 '{key}' 的库存状态: {stock} 件 ({status})"

    return f"未找到产品 '{product_name}' 的库存信息"
