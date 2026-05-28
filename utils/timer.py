import time
from functools import wraps

def timer(func):
    @wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 获取最高精度的开始时间
        result = func(*args, **kwargs)    # 执行被装饰的函数
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} 執行: {elapsed_time:.3f} 秒")
        return result  # 返回原函数的执行结果
    return wrapper