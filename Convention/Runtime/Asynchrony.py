from .Config         import *
from .Reflection     import *
from collections     import defaultdict
import                      asyncio
import                      threading
from typing          import Optional
from pydantic        import BaseModel
from abc             import ABC, abstractmethod

class AsyncContextDetector:
    """异步上下文检测工具类"""
    
    @staticmethod
    def is_in_async_context() -> bool:
        """检查是否在异步上下文中运行"""
        try:
            asyncio.current_task()
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def get_current_loop() -> Optional[asyncio.AbstractEventLoop]:
        """获取当前事件循环，如果没有则返回None"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None
    
    @staticmethod
    def ensure_async_context_safe(operation_name: str) -> None:
        """确保在异步上下文中执行是安全的"""
        if AsyncContextDetector.is_in_async_context():
            raise RuntimeError(
                f"Cannot perform '{operation_name}' from within an async context. "
                f"Use await or async methods instead."
            )

class AsyncFieldAccessor:
    """异步字段访问器，封装字段访问逻辑"""
    
    def __init__(self, async_fields: Dict[str, 'AsynchronyExpression'], origin_fields: Dict[str, FieldInfo]):
        self._async_fields = async_fields
        self._origin_fields = origin_fields
    
    async def get_field_value_async(self, field_name: str) -> Any:
        """异步获取字段值"""
        if field_name not in self._origin_fields:
            raise AttributeError(f"No async field '{field_name}' found")
        return await self._async_fields[field_name].get_value()
    
    def get_field_value_sync(self, field_name: str) -> Any:
        """同步获取字段值（仅在非异步上下文中使用）"""
        AsyncContextDetector.ensure_async_context_safe(f"sync access to field '{field_name}'")
        
        if field_name not in self._origin_fields:
            raise AttributeError(f"No async field '{field_name}' found")
        
        async_expr = self._async_fields[field_name]
        if not async_expr.is_initialize and async_expr.timeout > 0:
            # 需要等待但在同步上下文中，使用run_async
            return run_async(async_expr.get_value())
        elif not async_expr.is_initialize:
            raise RuntimeError(f"Field '{field_name}' is not initialized and has no timeout")
        else:
            return async_expr.value
    
    def is_field_initialized(self, field_name: str) -> bool:
        """检查字段是否已初始化"""
        if field_name not in self._origin_fields:
            raise AttributeError(f"No async field '{field_name}' found")
        return self._async_fields[field_name].is_initialize
    
    def set_field_value(self, field_name: str, value: Any) -> None:
        """设置字段值"""
        if field_name not in self._origin_fields:
            raise AttributeError(f"No async field '{field_name}' found")
        self._async_fields[field_name].set_value(value)

class AsynchronyUninitialized:
    """表示未初始化状态的单例类"""
    __instance__ = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls.__instance__ is None:
            with cls._lock:
                if cls.__instance__ is None:
                    cls.__instance__ = super().__new__(cls)
        return cls.__instance__

class AsynchronyExpression:
    def __init__(
        self,
        field:      FieldInfo, 
        value:      Any         = None,
        *,
        time_wait:  float       = 0.1,
        timeout:    float       = 0
        ):
        self.field = field
        self.value = value
        self.is_initialize = False
        self.time_wait = time_wait
        self.timeout = timeout

    async def GetValue(self) -> Any:
        """获取字段值（保持兼容性的旧方法名）"""
        return await self.get_value()
    
    async def get_value(self) -> Any:
        """异步获取字段值，改进的超时机制"""
        if self.is_initialize:
            return self.value
            
        if self.timeout > 0:
            try:
                # 使用 asyncio.wait_for 提供更精确的超时控制
                async def wait_for_initialization():
                    while not self.is_initialize:
                        await asyncio.sleep(self.time_wait)
                    return self.value
                
                return await asyncio.wait_for(wait_for_initialization(), timeout=self.timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Timeout waiting for uninitialized field {self.field.FieldName}")
        else:
            # 无超时，一直等待
            while not self.is_initialize:
                await asyncio.sleep(self.time_wait)
            return self.value
    
    def SetValue(self, value: Any) -> None:
        """设置字段值（保持兼容性的旧方法名）"""
        self.set_value(value)
    
    def set_value(self, value: Any) -> None:
        """设置字段值"""
        if isinstance(value, AsynchronyUninitialized):
            self.set_uninitialized()
        elif self.field.Verify(type(value)):
            self.value = value
            self.is_initialize = True
        else:
            raise ValueError(f"Value {value} is not valid for field {self.field.FieldName}")

    def SetUninitialized(self) -> None:
        """设置为未初始化状态（保持兼容性的旧方法名）"""
        self.set_uninitialized()
    
    def set_uninitialized(self) -> None:
        """设置为未初始化状态"""
        if self.is_initialize:
            if hasattr(self, 'value'):
                del self.value
        self.value = None
        self.is_initialize = False

class Asynchronous(ABC):
    __Asynchronous_Origin_Fields__: Dict[Type, Dict[str, FieldInfo]] = defaultdict(dict)
    _fields_lock = threading.Lock()

    def _GetAsynchronousOriginFields(self) -> Dict[str, FieldInfo]:
        return Asynchronous.__Asynchronous_Origin_Fields__[type(self)]

    def __init__(self, **kwargs: Dict[str, dict]):
        super().__init__()
        self.__Asynchronous_Fields__: Dict[str, AsynchronyExpression] = {}
        
        # 使用线程锁保护类变量访问
        with Asynchronous._fields_lock:
            origin_fields = self._GetAsynchronousOriginFields()
            for field_info in TypeManager.GetInstance().CreateOrGetRefTypeFromType(type(self)).GetAllFields():
                if field_info.FieldName == "__Asynchronous_Origin_Fields__":
                    continue
                origin_fields[field_info.FieldName] = field_info
                self.__Asynchronous_Fields__[field_info.FieldName] = AsynchronyExpression(
                    field_info, **kwargs.get(field_info.FieldName, {})
                )
        
        # 创建字段访问器以提升性能
        self._field_accessor = AsyncFieldAccessor(self.__Asynchronous_Fields__, origin_fields)

    def __getattribute__(self, name: str) -> Any:
        # 快速路径：非异步字段直接返回
        if name in ("__Asynchronous_Fields__", "_GetAsynchronousOriginFields", "_field_accessor"):
            return super().__getattribute__(name)
        
        # 一次性获取所需属性，避免重复调用
        try:
            field_accessor = super().__getattribute__("_field_accessor")
            origin_fields = super().__getattribute__("_GetAsynchronousOriginFields")()
        except AttributeError:
            # 对象可能尚未完全初始化
            return super().__getattribute__(name)
        
        if name in origin_fields:
            # 这是一个异步字段
            if AsyncContextDetector.is_in_async_context():
                # 在异步上下文中，提供友好的错误提示
                async_fields = super().__getattribute__("__Asynchronous_Fields__")
                async_expr = async_fields[name]
                
                if not async_expr.is_initialize:
                    timeout_info = f" with {async_expr.timeout}s timeout" if async_expr.timeout > 0 else ""
                    raise RuntimeError(
                        f"Field '{name}' is not initialized{timeout_info}. "
                        f"In async context, use 'await obj.get_field_async(\"{name}\")' instead."
                    )
                else:
                    # 字段已初始化，直接返回值
                    return async_expr.value
            else:
                # 在同步上下文中，使用字段访问器
                try:
                    return field_accessor.get_field_value_sync(name)
                except RuntimeError as e:
                    if "Cannot perform" in str(e):
                        # 重新包装错误信息，提供更友好的提示
                        raise RuntimeError(
                            f"Cannot access async field '{name}' from sync context when it requires initialization. "
                            f"Use async context or ensure field is pre-initialized."
                        ) from e
                    else:
                        raise
        
        return super().__getattribute__(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("__Asynchronous_Fields__", "_GetAsynchronousOriginFields", "_field_accessor"):
            super().__setattr__(name, value)
        elif hasattr(self, '_field_accessor'):
            # 对象已初始化，使用字段访问器
            try:
                field_accessor = super().__getattribute__("_field_accessor")
                field_accessor.set_field_value(name, value)
                return
            except AttributeError:
                # 不是异步字段
                pass
        
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("__Asynchronous_Fields__", "_GetAsynchronousOriginFields", "_field_accessor"):
            super().__delattr__(name)
        elif hasattr(self, '_field_accessor'):
            # 对象已初始化，使用字段访问器
            try:
                field_accessor = super().__getattribute__("_field_accessor")
                origin_fields = super().__getattribute__("_GetAsynchronousOriginFields")()
                if name in origin_fields:
                    async_fields = super().__getattribute__("__Asynchronous_Fields__")
                    async_fields[name].set_uninitialized()
                    return
            except AttributeError:
                # 不是异步字段
                pass
        
        super().__delattr__(name)
    
    async def get_field_async(self, field_name: str) -> Any:
        """异步获取字段值，适用于异步上下文"""
        return await self._field_accessor.get_field_value_async(field_name)
    
    def is_field_initialized(self, field_name: str) -> bool:
        """检查字段是否已初始化"""
        return self._field_accessor.is_field_initialized(field_name)
    
    def set_field_value(self, field_name: str, value: Any) -> None:
        """设置字段值"""
        self._field_accessor.set_field_value(field_name, value)
    
    def get_field_timeout(self, field_name: str) -> float:
        """获取字段的超时设置"""
        origin_fields = self._GetAsynchronousOriginFields()
        if field_name not in origin_fields:
            raise AttributeError(f"'{type(self).__name__}' object has no async field '{field_name}'")
        return self.__Asynchronous_Fields__[field_name].timeout

def run_until_complete(coro: Coroutine) -> Any:
    """Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # If we're here, there's an existing loop but it's not running
        return loop.run_until_complete(coro)

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread, or its already running
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )

def run_async_coroutine(coro: Coroutine) -> Any:
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # If we're here, there's an existing loop but it's not running
        return loop.create_task(coro)

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread, or its already running
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )

def run_async(coro: Coroutine):
    """安全地运行异步协程，避免事件循环死锁"""
    # 使用统一的异步上下文检测
    AsyncContextDetector.ensure_async_context_safe("run_async")
    
    # 尝试获取当前事件循环
    current_loop = AsyncContextDetector.get_current_loop()
    
    if current_loop is not None and not current_loop.is_running():
        # 有事件循环但未运行，直接使用
        return current_loop.run_until_complete(coro)
    elif current_loop is None:
        # 没有事件循环，创建新的
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to run async coroutine. "
                "Please ensure proper async environment or use nest_asyncio.apply() for nested loops."
            ) from e
    else:
        # 事件循环正在运行，这种情况应该被AsyncContextDetector捕获
        raise RuntimeError(
            "Unexpected state: running event loop detected but context check passed. "
            "This should not happen."
        )
