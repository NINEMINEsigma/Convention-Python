from .Config            import *
from .File              import ToolFile
from .String            import FillString
from typing             import *
import json
import os

# 静态配置
ConstConfigFile = "config.json"

def InitExtensionEnv():
    """初始化扩展环境"""
    global ConstConfigFile
    ConstConfigFile = "config.json"
    ProjectConfig.InitExtensionEnv()

def GenerateEmptyConfigJson(file: ToolFile):
    """生成空配置JSON"""
    file.SaveAsJson({"properties": {}})
    return file

class GlobalConfig:
    """全局配置管理类"""
    
    def __init__(
        self,
        data_dir: Optional[Union[str, ToolFile]] = None,
        is_try_create_data_dir: bool = False,
        is_load: bool = True
    ):
        """
        构造与初始化
        
        Args:
            data_dir: 数据目录路径或ToolFile对象
            is_try_create_data_dir: 是否尝试创建数据目录
            is_load: 是否自动加载现有配置
        """
        # 设置数据目录，确保目录存在
        if data_dir is None:
            data_dir = ToolFile(os.path.abspath('./'))
        
        self.data_dir: ToolFile = data_dir if isinstance(data_dir, ToolFile) else ToolFile(str(data_dir))
        
        if not self.data_dir.IsDir():
            self.data_dir.BackToParentDir()
            
        if not self.data_dir.Exists():
            if is_try_create_data_dir:
                self.data_dir.MustExistsPath()
            else:
                raise FileNotFoundError(f"Can't find data dir: {self.data_dir.GetDir()}")
        
        # 检查配置文件，不存在则生成空配置
        self._data_pair: Dict[str, Any] = {}
        self._const_config_file = ConstConfigFile
        config_file = self.ConfigFile
        
        if not config_file.Exists():
            GenerateEmptyConfigJson(config_file)
        elif is_load:
            self.LoadProperties()
    
    def __del__(self):
        pass
    
    # 文件管理
    def GetConfigFile(self) -> ToolFile:
        """获取配置文件对象"""
        return self.data_dir | self._const_config_file
    
    @property
    def ConfigFile(self) -> ToolFile:
        """获取配置文件对象（属性形式）"""
        return self.GetConfigFile()
    
    def GetFile(self, path: str, is_must_exist: bool = False) -> ToolFile:
        """获取数据目录下的文件"""
        result = self.data_dir | path
        if is_must_exist and not result.Exists():
            result.MustExistsPath()
        return result
    
    def CreateFile(self, path: str) -> bool:
        """创建文件"""
        result = self.data_dir | path
        if result.Exists():
            return False
        if not result.GetParentDir().Exists():
            return False
        result.Create()
        return True
    
    def EraseFile(self, path: str) -> bool:
        """清空文件内容"""
        result = self.data_dir | path
        if result.Exists():
            try:
                result.Remove()
                result.Create()
                return True
            except:
                pass
        return False
    
    def RemoveFile(self, path: str) -> bool:
        """删除文件"""
        result = self.data_dir | path
        if result.Exists():
            try:
                result.Remove()
                return True
            except:
                pass
        return False
    
    # 配置数据操作 - 支持迭代器
    def __setitem__(self, key: str, value: Any) -> Any:
        """索引器设置配置项"""
        self._data_pair[key] = value
        return value
    
    def __getitem__(self, key: str) -> Any:
        """索引器获取配置项"""
        return self._data_pair[key]
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._data_pair
    
    def __delitem__(self, key: str):
        """删除配置项"""
        del self._data_pair[key]
    
    def __iter__(self):
        """迭代器支持"""
        return iter(self._data_pair.items())
    
    def __len__(self) -> int:
        """获取配置项数量"""
        return len(self._data_pair)
    
    def Contains(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._data_pair
    
    def Remove(self, key: str) -> bool:
        """删除配置项"""
        if key in self._data_pair:
            del self._data_pair[key]
            return True
        return False
    
    def DataSize(self) -> int:
        """获取配置项数量"""
        return len(self._data_pair)
    
    # 持久化
    def SaveProperties(self) -> 'GlobalConfig':
        """保存配置到文件"""
        config = self.ConfigFile
        config.SaveAsJson({
            "properties": self._data_pair
        })
        return self
    
    def LoadProperties(self) -> 'GlobalConfig':
        """从文件加载配置"""
        config = self.ConfigFile
        if not config.Exists():
            self._data_pair = {}
        else:
            data = config.LoadAsJson()
            if "properties" in data:
                self._data_pair = data["properties"]
            else:
                raise ValueError("Can't find properties in config file")
        return self
    
    # 日志系统
    def GetLogFile(self) -> ToolFile:
        """获取日志文件对象"""
        return self.GetFile(self.ConfigFile.GetFilename(True) + "_log.txt", True)
    
    @property
    def LogFile(self) -> ToolFile:
        """获取日志文件对象（属性形式）"""
        return self.GetLogFile()
    
    def DefaultLogger(self, message: str):
        """默认日志输出器"""
        print(message)
    
    def Log(self, message_type: str, message: Union[str, Any], logger: Optional[Callable[[str], None]] = None) -> 'GlobalConfig':
        """记录日志"""
        str_message_type = str(message_type)
        # 使用String中的工具函数自动调整消息类型的对齐宽度
        aligned_message_type = FillString(str_message_type, max_length=len("Property not found"), side="center")
        what = f"[{Nowf()}]    {aligned_message_type}    : {str(message)}"
        
        if logger is None:
            logger = self.DefaultLogger
        
        logger(what)
        
        # 写入日志文件
        log = self.GetLogFile()
        # 读取现有内容并追加新内容
        try:
            existing_content = log.LoadAsText()
        except:
            existing_content = ""
        log.SaveAsText(existing_content + what + '\n')
        return self
    
    def LogPropertyNotFound(self, message: str, logger: Optional[Callable[[str], None]] = None, default: Any = None) -> 'GlobalConfig':
        """记录属性未找到的日志"""
        if default is not None:
            message = f"{message} (default = {default})"
        self.Log("Property not found", message, logger)
        return self
    
    def LogMessageOfPleaseCompleteConfiguration(self) -> 'GlobalConfig':
        """记录配置提示信息"""
        self.Log("Error", "Please complete configuration")
        return self
    
    # 配置查找
    def FindItem(self, key: str, default: Any = None) -> Any:
        """查找配置项，支持默认值"""
        if key in self._data_pair:
            return self._data_pair[key]
        else:
            self.LogPropertyNotFound(key, default=default)
            return default


class ProjectConfig(GlobalConfig):
    """项目级配置管理类，继承自GlobalConfig"""
    
    # 静态配置
    ProjectConfigFileFocus = "Assets/"
    
    @staticmethod
    def InitExtensionEnv():
        """初始化项目扩展环境"""
        ProjectConfig.ProjectConfigFileFocus = "Assets/"
    
    @staticmethod
    def SetProjectConfigFileFocus(path: str):
        """设置项目配置焦点目录"""
        ProjectConfig.ProjectConfigFileFocus = path
    
    @staticmethod
    def GetProjectConfigFileFocus() -> str:
        """获取项目配置焦点目录"""
        return ProjectConfig.ProjectConfigFileFocus
    
    def __init__(self, is_load: bool = True):
        """使用默认项目目录构造"""
        super().__init__(ProjectConfig.GetProjectConfigFileFocus(), is_try_create_data_dir=True, is_load=is_load)
