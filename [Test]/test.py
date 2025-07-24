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

async def geta(obj:Test) -> int:
    # 字段a已在__init__中初始化，可以直接访问
    if obj.is_field_initialized('a'):
        print(f"geta:{obj.a}")
        return obj.a
    else:
        # 使用异步方法获取
        value = await obj.get_field_async('a')
        print(f"geta:{value}")
        return value

async def getb(obj:Test) -> int:
    # 字段b有超时设置但未初始化，使用异步方法获取
    try:
        value = await obj.get_field_async('b')
        print(f"getb:{value}")
        return value
    except TimeoutError as e:
        print(f"getb timeout: {e}")
        raise

async def getc(obj:Test) -> int:
    # 字段c有超时设置但未初始化，使用异步方法获取
    try:
        value = await obj.get_field_async('c')
        print(f"getc:{value}")
        return value
    except TimeoutError as e:
        print(f"getc timeout: {e}")
        raise

async def setb(obj:Test, value:int, delay:float = 1) -> None:
    await asyncio.sleep(delay)
    obj.b = value

async def run():
    print("=== 测试优化后的异步字段系统 ===")
    test = Test()
    
    # 测试字段状态检查
    print(f"字段初始化状态 - a: {test.is_field_initialized('a')}, b: {test.is_field_initialized('b')}, c: {test.is_field_initialized('c')}")
    
    # 测试超时设置查询
    print(f"字段超时设置 - b: {test.get_field_timeout('b')}s, c: {test.get_field_timeout('c')}s")
    
    print("\n=== 测试1：未设置值的情况（应该超时）===")
    try:
        print("开始并发获取字段值...")
        r = await asyncio.gather(geta(test), getb(test), getc(test))
        print(f"结果: {r}")
    except Exception as e:
        print(f"捕获到异常: {e}")
    
    print("\n=== 测试2：在超时前设置字段b的值 ===")
    # 创建新的测试实例
    test2 = Test()
    print(f"设置前字段b初始化状态: {test2.is_field_initialized('b')}")
    
    try:
        # 启动并发任务：设置b的值和获取b的值
        print("启动并发任务：0.5秒后设置b=42，同时尝试获取b的值（超时10秒）")
        
        # 并发执行：设置b值（延迟0.5秒）和获取b值
        results = await asyncio.gather(
            setb(test2, 42, delay=0.5),  # 0.5秒后设置b=42
            getb(test2),                 # 尝试获取b值（会等待直到被设置）
            return_exceptions=True
        )
        
        print(f"并发任务结果: {results}")
        print(f"设置后字段b初始化状态: {test2.is_field_initialized('b')}")
        
        # 再次访问b，应该能立即获取到值
        print("再次访问字段b（应该立即返回）:")
        b_value = await test2.get_field_async('b')
        print(f"字段b的值: {b_value}")
        
    except Exception as e:
        print(f"测试2出现异常: {e}")
    
    print("\n=== 测试3：使用同步方式设置，异步方式获取 ===")
    test3 = Test()
    print("使用同步方式设置字段b = 100")
    test3.b = 100
    print(f"设置后字段b初始化状态: {test3.is_field_initialized('b')}")
    
    # 异步获取值
    b_sync_set_value = await test3.get_field_async('b')
    print(f"异步获取同步设置的值: {b_sync_set_value}")
    
    print("\n=== 测试4：测试字段c（短超时，应该仍然超时）===")
    try:
        print("尝试单独访问字段c（2秒超时）...")
        c_value = await test.get_field_async('c')
        print(f"字段c的值: {c_value}")
    except TimeoutError as timeout_e:
        print(f"字段c访问超时（预期）: {timeout_e}")

def test_sync_access():
    """测试同步访问（在非异步上下文中）"""
    print("\n=== 测试同步访问 ===")
    test = Test()
    
    # 测试已初始化字段的同步访问
    try:
        print(f"同步访问字段a: {test.a}")
    except Exception as e:
        print(f"同步访问字段a失败: {e}")
    
    # 测试未初始化字段的同步访问（应该有更友好的错误提示）
    try:
        print(f"同步访问字段c: {test.c}")
    except Exception as e:
        print(f"同步访问字段c失败 (预期): {e}")

if __name__ == "__main__":
    # 测试同步访问
    test_sync_access()
    
    # 测试异步访问
    print("\n=== 开始异步测试 ===")
    run_until_complete(run())
