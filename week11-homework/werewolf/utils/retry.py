#!/usr/bin/env python3
"""
重试装饰器，用于处理API限流等临时错误
"""

import functools
import time
from typing import TypeVar

from .logger import logger

T = TypeVar("T")


def retry_decorator(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """重试装饰器

    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型

    Returns:
        装饰后的函数
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries:
                            logger["retry"].warning(
                                f"API限流，第 {attempt + 1}/{max_retries} 次重试，{func.__name__} 函数，延迟 {delay} 秒: {e}"
                            )
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            logger["retry"].error(
                                f"API限流，达到最大重试次数 {max_retries}，{func.__name__} 函数: {e}"
                            )
                            raise
                    else:
                        # 其他异常，不重试
                        raise

        return wrapper

    return decorator
