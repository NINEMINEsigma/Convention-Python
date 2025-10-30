from types          import TracebackType
from typing         import *
from abc            import *
import                     sys
import                     threading
import                     traceback
import                     datetime

# region ansi colorful

# Copyright Jonathan Hartley 2013. BSD 3-Clause license
'''
This module generates ANSI character codes to printing colors to terminals.
See: http://en.wikipedia.org/wiki/ANSI_escape_code
'''

CSI = '\033['
OSC = '\033]'
BEL = '\a'


def code_to_chars(code):
    return CSI + str(code) + 'm'

def set_title(title):
    return OSC + '2;' + title + BEL

def clear_screen(mode=2):
    return CSI + str(mode) + 'J'

def clear_line(mode=2):
    return CSI + str(mode) + 'K'


class AnsiCodes(object):
    def __init__(self):
        # the subclasses declare class attributes which are numbers.
        # Upon instantiation we define instance attributes, which are the same
        # as the class attributes but wrapped with the ANSI escape sequence
        for name in dir(self):
            if not name.startswith('_'):
                value = getattr(self, name)
                setattr(self, name, code_to_chars(value))


class ConsoleCursor(object):
    def UP(self, n=1):
        return CSI + str(n) + 'A'
    def DOWN(self, n=1):
        return CSI + str(n) + 'B'
    def FORWARD(self, n=1):
        return CSI + str(n) + 'C'
    def BACK(self, n=1):
        return CSI + str(n) + 'D'
    def POS(self, x=1, y=1):
        return CSI + str(y) + ';' + str(x) + 'H'


class ConsoleFrontColorClass(AnsiCodes):
    BLACK           = 30
    RED             = 31
    GREEN           = 32
    YELLOW          = 33
    BLUE            = 34
    MAGENTA         = 35
    CYAN            = 36
    WHITE           = 37
    RESET           = 39

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX   = 90
    LIGHTRED_EX     = 91
    LIGHTGREEN_EX   = 92
    LIGHTYELLOW_EX  = 93
    LIGHTBLUE_EX    = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX    = 96
    LIGHTWHITE_EX   = 97

ConsoleFrontColor = ConsoleFrontColorClass()

class ConsoleBackgroundColorClass(AnsiCodes):
    BLACK           = 40
    RED             = 41
    GREEN           = 42
    YELLOW          = 43
    BLUE            = 44
    MAGENTA         = 45
    CYAN            = 46
    WHITE           = 47
    RESET           = 49

    # These are fairly well supported, but not part of the standard.
    LIGHTBLACK_EX   = 100
    LIGHTRED_EX     = 101
    LIGHTGREEN_EX   = 102
    LIGHTYELLOW_EX  = 103
    LIGHTBLUE_EX    = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX    = 106
    LIGHTWHITE_EX   = 107

ConsoleBackgroundColor = ConsoleBackgroundColorClass()

class ConsoleStyleClass(AnsiCodes):
    BRIGHT    = 1
    DIM       = 2
    NORMAL    = 22
    RESET_ALL = 0

ConsoleStyle = ConsoleStyleClass()

def PrintColorful(color:str, *args, is_reset:bool=True, **kwargs):
    with lock_guard():
        if is_reset:
            print(color,*args,ConsoleStyle.RESET_ALL, **kwargs)
        else:
            print(color,*args, **kwargs)

def PrintAsError(message:str):
    PrintColorful(ConsoleFrontColor.RED, message)
def PrintAsWarning(message:str):
    PrintColorful(ConsoleFrontColor.YELLOW, message)
def PrintAsInfo(message:str):
    PrintColorful(ConsoleFrontColor.GREEN, message)
def PrintAsDebug(message:str):
    PrintColorful(ConsoleFrontColor.BLUE, message)
def PrintAsSuccess(message:str):
    PrintColorful(ConsoleFrontColor.GREEN, message)
def PrintAsLight(message:str):
    PrintColorful(ConsoleFrontColor.LIGHTMAGENTA_EX, message)

# endregion

class NotImplementedError(Exception):
    def __init__(self, message:Optional[str]=None) -> None:
        if message is not None:
            super().__init__(message)
        else:
            super().__init__()

class InvalidOperationError(Exception):
    def __init__(self, message:Optional[str]=None) -> None:
        if message is not None:
            super().__init__(message)
        else:
            super().__init__()

def format_traceback_info(char:str='\n', back:int=1):
    return char.join(traceback.format_stack()[:-back])

INTERNAL_DEBUG = False
def SetInternalDebug(mode:bool):
    global INTERNAL_DEBUG
    INTERNAL_DEBUG = mode
def GetInternalDebug() -> bool:
    global INTERNAL_DEBUG
    return INTERNAL_DEBUG

ImportingFailedSet:Set[str] = set()
def ImportingThrow(
    ex:             ImportError,
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = ConsoleFrontColor.RED+"{module} Module requires {required} package."+ConsoleFrontColor.RESET,
    installBase:    str = ConsoleFrontColor.GREEN+"\tpip install {name}"+ConsoleFrontColor.RESET
    ):
    with lock_guard():
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(messageBase.format_map(dict(module=moduleName, required=requierds_str)))
        print('Install it via command:')
        for i in requierds:
            global ImportingFailedSet
            ImportingFailedSet.add(i)
            install = installBase.format_map({"name":i})
            print(install)
        if ex:
            print(ConsoleFrontColor.RED, f"Import Error On {moduleName} Module: {ex}, \b{ex.path}\n"\
                f"[{ConsoleFrontColor.RESET}{format_traceback_info(back=2)}{ConsoleFrontColor.RED}]")

def InternalImportingThrow(
    moduleName:     str,
    requierds:      Sequence[str],
    *,
    messageBase:    str = ConsoleFrontColor.RED+"{module} Module requires internal Convention package: {required}."+ConsoleFrontColor.RESET,
    ):
    with lock_guard():
        requierds_str = ",".join([f"<{r}>" for r in requierds])
        print(f"Internal Convention package is not installed.\n{messageBase.format_map({
            "module": moduleName,
            "required": requierds_str
        })}\n[{ConsoleFrontColor.RESET}{format_traceback_info(back=2)}{ConsoleFrontColor.RED}]")

def ReleaseFailed2Requirements():
    global ImportingFailedSet
    if len(ImportingFailedSet) == 0:
        return
    with open("requirements.txt", 'w') as f:
        f.write("\n".join(ImportingFailedSet))

try:
    from pydantic import *
except ImportError:
    InternalImportingThrow("Internal", ["pydantic"])

type Typen[_T] = type

type Action = Callable[[], None]
type ClosuresCallable[_T] = Union[Callable[[Optional[None]], _T], Typen[_T]]

def AssemblyTypen(obj:Any) -> str:
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__name__}, "\
            f"{obj.Assembly() if hasattr(obj, "Assembly") else "Global"}"
    else:
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}, "\
            f"{obj.GetAssembly() if hasattr(obj, "GetAssembly") else "Global"}"
def ReadAssemblyTypen(
    assembly_typen: str,
    *,
    premodule:      Optional[str|Callable[[str], str]] = None
    ) -> Tuple[type, str]:
    typen, assembly_name = assembly_typen.split(",")
    module_name, _, class_name = typen.rpartition(".")
    if premodule is not None:
        if isinstance(premodule, str):
            module_name = premodule
        else:
            module_name = premodule(module_name)
    import importlib
    target_type = getattr(importlib.import_module(module_name), class_name)
    return target_type, assembly_name

# using as c#: event
class ActionEvent[_Call:Callable]:
    def __init__(self, actions:Sequence[_Call]):
        super().__init__()
        self._actions:      List[Callable]  = [action for action in actions]
        self.call_indexs:   List[int]       = [i for i in range(len(actions))]
        self.last_result:   List[Any]       = []
    def CallFuncWithoutCallIndexControl(self, index:int, *args, **kwargs) -> Union[Any, Exception]:
        try:
            return self._actions[index](*args, **kwargs)
        except Exception as ex:
            return ex
    def CallFunc(self, index:int, *args, **kwargs) -> Union[Any, Exception]:
        return self.CallFuncWithoutCallIndexControl(self.call_indexs[index], *args, **kwargs)
    def _InjectInvoke(self, *args, **kwargs):
        result:List[Any] = []
        for index in range(self.CallMaxCount):
            result.append(self.CallFunc(index, *args, **kwargs))
        return result
    def Invoke(self, *args, **kwargs) -> Union[Self, bool]:
        self.last_result = self._InjectInvoke(*args, **kwargs)
        return self
    def InitCallIndex(self):
        self.call_indexs = [i for i in range(len(self._actions))]
    def AddAction(self, action:_Call):
        self._actions.append(action)
        self.call_indexs.append(len(self._actions)-1)
        return self
    def AddActions(self, actions:Sequence[_Call]):
        for action in actions:
            self.AddAction(action)
        return self
    def _InternalRemoveAction(self, action:_Call):
        if action in self._actions:
            index = self._actions.index(action)
            self._actions.remove(action)
            self.call_indexs.remove(index)
            for i in range(len(self.call_indexs)):
                if self.call_indexs[i] > index:
                    self.call_indexs[i] -= 1
            return True
        return False
    def RemoveAction(self, action:_Call):
        while self._InternalRemoveAction(action):
            pass
        return self
    def IsValid(self):
        return not any(isinstance(x, Exception) for x in self.last_result)
    def __bool__(self):
        return self.IsValid()
    @property
    def CallMaxCount(self):
        return len(self.call_indexs)
    @property
    def ActionCount(self):
        return len(self._actions)

# region instance

# threads

class atomic[_T]:
    def __init__(
        self,
        value:  _T,
        locker: Optional[threading.Lock]    = None,
        ) -> None:
        self._value:        _T              = value
        self._is_in_with:   bool            = False
        self.locker:        threading.Lock  = locker if locker is not None else threading.Lock()
    def FetchAdd(self, value:_T):
        with lock_guard(self.locker):
            self._value += value
        return self._value
    def FetchSub(self, value:_T):
        with lock_guard(self.locker):
            self._value -= value
        return self._value
    def Load(self) -> _T:
        with lock_guard(self.locker):
            return self._value
    def Store(self, value: _T):
        with lock_guard(self.locker):
            self._value = value
    def __add__(self, value:_T):
        return self.FetchAdd(value)
    def __sub__(self, value:_T):
        return self.FetchSub(value)
    def __iadd__(self, value:_T) -> Self:
        self.FetchAdd(value)
        return self
    def __isub__(self, value:_T) -> Self:
        self.FetchSub(value)
        return self
    def __enter__(self) -> Self:
        self._is_in_with = True
        self.locker.acquire()
        return self
    def __exit__(
        self,
        exc_type:   Optional[type],
        exc_val:    Optional[BaseException],
        exc_tb:     Optional[TracebackType]
        ) -> bool:
        self._is_in_with = False
        self.locker.release()
        if exc_type is None:
            return True
        else:
            return False
    @property
    def Value(self) -> _T:
        if self._is_in_with:
            return self._value
        raise NotImplementedError("This method can only be called within a with statement")
    @Value.setter
    def Value(self, value:_T) -> _T:
        if self._is_in_with:
            self._value = value
        raise NotImplementedError("This method can only be called within a with statement")

    def __str__(self) -> str:
        return str(self.Load())
    def __repr__(self) -> str:
        return repr(self.Load())

InternalGlobalLocker = threading.Lock()
InternalGlobalLockerCount = atomic[int](0)

class lock_guard:
    def __init__(
        self,
        lock:   Optional[Union[threading.RLock, threading.Lock]] = None
        ):
        if lock is None:
            lock = InternalGlobalLocker
        self._locker = lock
        self._locker.acquire()
    def __del__(self):
        self._locker.release()
    def __enter__(self):
        return
    def __exit__(self,*args,**kwargs):
        return True

class global_lock_guard(lock_guard):
    def __init__(self):
        super().__init__(None)

class thread_instance(threading.Thread):
    def __init__(
        self,
        call:           Action,
        *,
        is_del_join:    bool = True,
        **kwargs
        ):
        kwargs.update({"target": call})
        super().__init__(**kwargs)
        self.is_del_join = is_del_join
        self.start()
    def __del__(self):
        if self.is_del_join:
            self.join()

# region end

def Nowf() -> str:
    '''
    printf now time to YYYY-MM-DD_HH-MM-SS format,
    return: str
    '''
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

true: Literal[True] = True
false: Literal[False] = False

class PlatformIndicator:
    IsRelease           : bool  = False
    IsPlatformWindows   : bool  = sys.platform == "Windows"
    IsPlatformLinux     : bool  = sys.platform == "Linux"
    IsPlatformOsx       : bool  = sys.platform == "OSX"
    IsPlatformX64       : bool  = True
    CompanyName         : str   = "DefaultCompany"
    ProductName         : str   = "DefaultProject"
    PrettyFace          : str   = r"""
⣇⣿⠘⣿⣿⣿⡿⡿⣟⣟⢟⢟⢝⠵⡝⣿⡿⢂⣼⣿⣷⣌⠩⡫⡻⣝⠹⢿⣿⣷
⡆⣿⣆⠱⣝⡵⣝⢅⠙⣿⢕⢕⢕⢕⢝⣥⢒⠅⣿⣿⣿⡿⣳⣌⠪⡪⣡⢑⢝⣇
⡆⣿⣿⣦⠹⣳⣳⣕⢅⠈⢗⢕⢕⢕⢕⢕⢈⢆⠟⠋⠉⠁⠉⠉⠁⠈⣸⢐⢕⢽
⡗⢰⣶⣶⣦⣝⢝⢕⢕⠅⡆⢕⢕⢕⢕⢕⣴⠏⣠⡶⠛⡉⡉⡛⢶⣦⡀⠐⣕⢕
⡝⡄⢻⢟⣿⣿⣷⣕⣕⣅⣿⣔⣕⣵⣵⣿⣿⢠⣿⢠⣮⡈⣌⠨⠅⠹⣷⡀⢱⢕
⡝⡵⠟⠈⠀⠀⠀⠀⠉⢿⣿⣿⣿⣿⣿⣿⣿⣼⣿⢈⡋⠴⢿⡟⣡⡇⣿⡇⢀⢕
⡝⠁⣠⣾⠟⡉⡉⡉⠻⣦⣻⣿⣿⣿⣿⣿⣿⣿⣿⣧⠸⣿⣦⣥⣿⡇⡿⣰⢗⢄
⠁⢰⣿⡏⣴⣌⠈⣌⠡⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣬⣉⣉⣁⣄⢖⢕⢕⢕
⡀⢻⣿⡇⢙⠁⠴⢿⡟⣡⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣵⣵⣿
⡻⣄⣻⣿⣌⠘⢿⣷⣥⣿⠇⣿⣿⣿⣿⣿⣿⠛⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣷⢄⠻⣿⣟⠿⠦⠍⠉⣡⣾⣿⣿⣿⣿⣿⣿⢸⣿⣦⠙⣿⣿⣿⣿⣿⣿⣿⣿⠟
⡕⡑⣑⣈⣻⢗⢟⢞⢝⣻⣿⣿⣿⣿⣿⣿⣿⠸⣿⠿⠃⣿⣿⣿⣿⣿⣿⡿⠁⣠
⡝⡵⡈⢟⢕⢕⢕⢕⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣿⣿⣿⣿⣿⠿⠋⣀⣈⠙
⡝⡵⡕⡀⠑⠳⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⢉⡠⡲⡫⡪⡪⡣
    """.strip()
    
    @staticmethod
    def GetFileSeparator(is_not_this_platform:bool = False) -> str:
        if PlatformIndicator.IsPlatformWindows and not is_not_this_platform:
            return "\\"
        return "/"

    @staticmethod
    def GetApplicationPath() -> str:
        """获取应用程序所在目录"""
        import os
        return os.path.dirname(os.path.abspath(__file__))
    
    @staticmethod
    def GetCurrentWorkingDirectory() -> str:
        """获取当前工作目录"""
        import os
        return os.getcwd()
    
    # 使用类方法获取路径
    @classmethod
    def ApplicationPath(cls) -> str:
        """应用程序路径属性"""
        return cls.GetApplicationPath()
    
    @classmethod  
    def StreamingAssetsPath(cls) -> str:
        """流媒体资源路径属性"""
        return cls.ApplicationPath() + "/StreamingAssets/"
    
    @staticmethod
    def PersistentDataPath() -> str:
        """
        获取持久化数据路径，根据平台返回不同的路径
        """
        import os
        if PlatformIndicator.IsPlatformWindows:
            return os.path.expandvars(f"%userprofile%\\AppData\\LocalLow\\{PlatformIndicator.CompanyName}\\{PlatformIndicator.ProductName}\\")
        elif PlatformIndicator.IsPlatformLinux:
            return os.path.expandvars("$HOME/.config/")
        return ""
    
    @staticmethod
    def DataPath() -> str:
        """
        获取数据路径
        """
        return "Assets/"

class DescriptiveIndicator[T]:
    def __init__(self, description:str, value:T) -> None:
        self.descripion : str   = description
        self.value      : T     = value

class Switch:
    def __init__(self, value, isThougth = False) -> None:
        self.value = value
        self.isThougth = False
        self.caseStats = False
        self.result = None

    def Case(self, caseValue, callback:Callable[[], Any]) -> 'Switch':
        if self.caseStats and self.isThougth:
            self.result = callback()
        elif caseValue == self.value:
            self.caseStats = True
            self.result = callback()
        return self

    def Default(self, callback:Callable[[], Any]) -> Any:
        if self.caseStats and self.isThougth:
            self.result = callback()
        elif self.caseStats == False:
            self.caseStats = True
            self.result = callback()
        return self.result