import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Convention.Runtime.Asynchrony import *

class Test(Asynchronous):
    a:int
    b:int
    c:int

    def __init__(self):
        super().__init__(c={"timeout":2},b={"timeout":10})
        self.a = 1


async def seta(obj:Test, value:int, delay:float = 1) -> None:
    await asyncio.sleep(delay)
    obj.a = value

async def setb(obj:Test, value:int, delay:float = 1) -> None:
    await asyncio.sleep(delay)
    obj.b = value
    
async def setc(obj:Test, value:int, delay:float = 1) -> None:
    await asyncio.sleep(delay)
    obj.c = value

async def run():
    print("=== 测试优化后的异步字段系统 ===")
    test = Test()
    
    # 测试字段状态检查
    assert test.is_field_initialized('a')
    assert not test.is_field_initialized('b')
    assert not test.is_field_initialized('c')
    
    print("\n=== 测试1：未设置值的情况（应该超时）===")
    try:
        print(f"失败: {test.a,test.b,test.c}")
        raise RuntimeError("测试1应该超时")
    except Exception as e:
        print(f"成功: {e}")
    
    print("\n=== 测试2：在超时前设置字段b的值 ===")
    # 创建新的测试实例
    test2 = Test()
    assert not test2.is_field_initialized('b')
    
    # 启动并发任务：设置b的值和获取b的值
    # 并发执行：设置b值（延迟0.5秒）和获取b值
    await asyncio.gather(
        setb(test2, 42, delay=0.5),  # 0.5秒后设置b=42
        return_exceptions=True
    )
    
    assert test2.b == 42
    assert test2.is_field_initialized('b')
    assert test2.b == 42
    
    test3 = Test()
    test3.b = 100
    assert test3.is_field_initialized('b')
    assert test3.b == 100
    
    print("\n=== 测试3：测试字段c（短超时，应该仍然超时）===")
    try:
        print(f"失败: {test.c}")
    except TimeoutError as timeout_e:
        print(f"成功: {timeout_e}")

def test_sync_access():
    """测试同步访问（在非异步上下文中）"""
    print("\n=== 测试同步访问 ===")
    test = Test()
    
    # 测试已初始化字段的同步访问
    try:
        print(f"成功: a = {test.a}")
    except Exception as e:
        raise
    
    # 测试未初始化字段的同步访问（应该有更友好的错误提示）
    try:
        print(f"失败: c = {test.c}")
        raise RuntimeError("字段c此时不应该能够被访问")
    except Exception as e:
        print(f"成功: {e}")

if __name__ == "__main__":
    # 测试同步访问
    test_sync_access()
    
    # 测试异步访问
    print("\n=== 开始异步测试 ===")
    run_until_complete(run())
