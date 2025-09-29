from .Config import *
from .File import ToolFile
from .Web import ToolURL
import json
import urllib.parse
import urllib.request
import urllib.error
import asyncio
import os
import re
from typing import *
from pydantic import BaseModel

try:
    import aiohttp
    import aiofiles
except ImportError as e:
    ImportingThrow(e, "Interaction", ["aiohttp", "aiofiles"])


class InteractionError(Exception):
    """交互操作异常基类"""
    pass


class PathValidationError(InteractionError):
    """路径验证异常"""
    pass


class LoadError(InteractionError):
    """加载异常"""
    pass


class SaveError(InteractionError):
    """保存异常"""
    pass


class Interaction(BaseModel):
    """统一的文件交互类，自适应处理本地文件和网络文件"""
    
    path: str
    _is_url: bool = False
    _is_local: bool = False
    _tool_file: Optional[ToolFile] = None
    _tool_url: Optional[ToolURL] = None
    
    def __init__(self, path: Union[str, 'Interaction', ToolFile, ToolURL]):
        """
        从路径字符串创建对象，自动识别本地文件或网络URL
        
        Args:
            path: 路径字符串、Interaction对象、ToolFile对象或ToolURL对象
        """
        if isinstance(path, Interaction):
            path = path.path
        elif isinstance(path, ToolFile):
            path = path.GetFullPath()
        elif isinstance(path, ToolURL):
            path = path.url
        
        path_str = str(path)
        super().__init__(path=path_str)
        
        # 自动识别路径类型
        self._detect_path_type()
    
    def _detect_path_type(self):
        """自动检测路径类型"""
        path = self.path.strip()
        
        # 检查是否为HTTP/HTTPS URL
        if path.startswith(('http://', 'https://')):
            self._is_url = True
            self._is_local = False
            self._tool_url = ToolURL(path)
            return
        
        # 检查是否为localhost URL
        if path.startswith('localhost'):
            # 转换为完整的HTTP URL
            if not path.startswith('localhost:'):
                # 默认端口80
                full_url = f"http://{path}"
            else:
                full_url = f"http://{path}"
            self._is_url = True
            self._is_local = False
            self._tool_url = ToolURL(full_url)
            self.path = full_url
            return
        
        # 检查是否为绝对路径或相对路径
        if (os.path.isabs(path) or 
            path.startswith('./') or 
            path.startswith('../') or
            ':' in path[:3]):  # Windows盘符
            self._is_local = True
            self._is_url = False
            self._tool_file = ToolFile(path)
            return
        
        # 默认作为相对路径处理
        self._is_local = True
        self._is_url = False
        self._tool_file = ToolFile(path)
    
    def __str__(self) -> str:
        """隐式字符串转换"""
        return self.path
    
    def __bool__(self) -> bool:
        """隐式布尔转换，检查路径是否有效"""
        return self.IsValid
    
    @property
    def IsValid(self) -> bool:
        """检查路径是否有效"""
        if self._is_url:
            return self._tool_url.IsValid if self._tool_url else False
        else:
            return self._tool_file.Exists() if self._tool_file else False
    
    @property
    def IsURL(self) -> bool:
        """是否为网络URL"""
        return self._is_url
    
    @property
    def IsLocal(self) -> bool:
        """是否为本地文件"""
        return self._is_local
    
    @property
    def IsFile(self) -> bool:
        """是否为文件（对于URL，检查是否存在文件名）"""
        if self._is_url:
            return bool(self._tool_url.GetFilename()) if self._tool_url else False
        else:
            return self._tool_file.IsFile() if self._tool_file else False
    
    @property
    def IsDir(self) -> bool:
        """是否为目录（仅对本地路径有效）"""
        if self._is_local:
            return self._tool_file.IsDir() if self._tool_file else False
        return False
    
    def GetFilename(self) -> str:
        """获取文件名"""
        if self._is_url:
            return self._tool_url.GetFilename() if self._tool_url else ""
        else:
            return self._tool_file.GetFilename() if self._tool_file else ""
    
    def GetExtension(self) -> str:
        """获取文件扩展名"""
        if self._is_url:
            return self._tool_url.GetExtension() if self._tool_url else ""
        else:
            return self._tool_file.GetExtension() if self._tool_file else ""
    
    def ExtensionIs(self, *extensions: str) -> bool:
        """检查扩展名是否匹配"""
        if self._is_url:
            return self._tool_url.ExtensionIs(*extensions) if self._tool_url else False
        else:
            current_ext = self.GetExtension()
            return current_ext.lower() in [ext.lower().lstrip('.') for ext in extensions]
    
    # 文件类型判断属性
    @property
    def IsText(self) -> bool:
        """是否为文本文件"""
        return self.ExtensionIs('txt', 'html', 'htm', 'css', 'js', 'xml', 'csv', 'md', 'py', 'java', 'cpp', 'c', 'h')
    
    @property
    def IsJson(self) -> bool:
        """是否为JSON文件"""
        return self.ExtensionIs('json')
    
    @property
    def IsImage(self) -> bool:
        """是否为图像文件"""
        return self.ExtensionIs('jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp')
    
    @property
    def IsDocument(self) -> bool:
        """是否为文档文件"""
        return self.ExtensionIs('pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx')
    
    def Open(self, path: str) -> 'Interaction':
        """在当前对象上打开新路径"""
        new_obj = Interaction(path)
        self.path = new_obj.path
        self._is_url = new_obj._is_url
        self._is_local = new_obj._is_local
        self._tool_file = new_obj._tool_file
        self._tool_url = new_obj._tool_url
        return self
    
    # 同步加载方法
    def LoadAsText(self) -> str:
        """
        同步加载为文本
        
        Returns:
            文本内容
        """
        if self._is_url:
            if not self._tool_url or not self._tool_url.IsValid:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return self._tool_url.LoadAsText()
        else:
            if not self._tool_file or not self._tool_file.Exists():
                raise PathValidationError(f"File not found: {self.path}")
            return self._tool_file.LoadAsText()
    
    def LoadAsBinary(self) -> bytes:
        """
        同步加载为字节数组
        
        Returns:
            二进制内容
        """
        if self._is_url:
            if not self._tool_url or not self._tool_url.IsValid:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return self._tool_url.LoadAsBinary()
        else:
            if not self._tool_file or not self._tool_file.Exists():
                raise PathValidationError(f"File not found: {self.path}")
            return self._tool_file.LoadAsBinary()
    
    def LoadAsJson(self, model_type: Optional[type] = None) -> Any:
        """
        同步加载并反序列化JSON
        
        Args:
            model_type: 可选的Pydantic模型类型
            
        Returns:
            JSON数据或模型对象
        """
        if self._is_url:
            if not self._tool_url or not self._tool_url.IsValid:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return self._tool_url.LoadAsJson(model_type)
        else:
            if not self._tool_file or not self._tool_file.Exists():
                raise PathValidationError(f"File not found: {self.path}")
            json_data = self._tool_file.LoadAsJson()
            if model_type and issubclass(model_type, BaseModel):
                return model_type.model_validate(json_data)
            return json_data
    
    # 异步加载方法
    async def LoadAsTextAsync(self) -> str:
        """
        异步加载为文本
        
        Returns:
            文本内容
        """
        if self._is_url:
            if not self._tool_url or not self._tool_url.IsValid:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return await self._tool_url.LoadAsTextAsync()
        else:
            if not self._tool_file or not self._tool_file.Exists():
                raise PathValidationError(f"File not found: {self.path}")
            # 异步读取本地文件
            async with aiofiles.open(self._tool_file.GetFullPath(), 'r', encoding='utf-8') as f:
                return await f.read()
    
    async def LoadAsBinaryAsync(self) -> bytes:
        """
        异步加载为字节数组
        
        Returns:
            二进制内容
        """
        if self._is_url:
            if not self._tool_url or not self._tool_url.IsValid:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return await self._tool_url.LoadAsBinaryAsync()
        else:
            if not self._tool_file or not self._tool_file.Exists():
                raise PathValidationError(f"File not found: {self.path}")
            # 异步读取本地文件
            async with aiofiles.open(self._tool_file.GetFullPath(), 'rb') as f:
                return await f.read()
    
    async def LoadAsJsonAsync(self, model_type: Optional[type] = None) -> Any:
        """
        异步加载并反序列化JSON
        
        Args:
            model_type: 可选的Pydantic模型类型
            
        Returns:
            JSON数据或模型对象
        """
        if self._is_url:
            if not self._tool_url or not self._tool_url.IsValid:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return await self._tool_url.LoadAsJsonAsync(model_type)
        else:
            if not self._tool_file or not self._tool_file.Exists():
                raise PathValidationError(f"File not found: {self.path}")
            # 异步读取本地JSON文件
            text_content = await self.LoadAsTextAsync()
            try:
                json_data = json.loads(text_content)
                if model_type and issubclass(model_type, BaseModel):
                    return model_type.model_validate(json_data)
                return json_data
            except json.JSONDecodeError as e:
                raise LoadError(f"Failed to parse JSON from {self.path}: {str(e)}")
    
    # 同步保存方法
    def SaveAsText(self, content: str, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        同步保存为文本
        
        Args:
            content: 文本内容
            local_path: 本地保存路径（仅对URL有效）
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，先下载然后保存到本地
            if local_path is None:
                local_path = self.GetFilename() or "downloaded.txt"
            file_obj = ToolFile(local_path)
            file_obj.TryCreateParentPath()
            file_obj.SaveAsText(content)
            return file_obj
        else:
            if not self._tool_file:
                raise PathValidationError(f"Invalid file path: {self.path}")
            self._tool_file.TryCreateParentPath()
            self._tool_file.SaveAsText(content)
            return self
    
    def SaveAsBinary(self, content: bytes, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        同步保存为二进制
        
        Args:
            content: 二进制内容
            local_path: 本地保存路径（仅对URL有效）
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，保存到本地
            if local_path is None:
                local_path = self.GetFilename() or "downloaded.bin"
            file_obj = ToolFile(local_path)
            file_obj.TryCreateParentPath()
            file_obj.SaveAsBinary(content)
            return file_obj
        else:
            if not self._tool_file:
                raise PathValidationError(f"Invalid file path: {self.path}")
            self._tool_file.TryCreateParentPath()
            self._tool_file.SaveAsBinary(content)
            return self
    
    def SaveAsJson(self, data: Any, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        同步保存为JSON
        
        Args:
            data: JSON数据
            local_path: 本地保存路径（仅对URL有效）
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，保存到本地
            if local_path is None:
                local_path = self.GetFilename() or "downloaded.json"
            file_obj = ToolFile(local_path)
            file_obj.TryCreateParentPath()
            file_obj.SaveAsJson(data)
            return file_obj
        else:
            if not self._tool_file:
                raise PathValidationError(f"Invalid file path: {self.path}")
            self._tool_file.TryCreateParentPath()
            self._tool_file.SaveAsJson(data)
            return self
    
    # 异步保存方法
    async def SaveAsTextAsync(self, content: str, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        异步保存为文本
        
        Args:
            content: 文本内容
            local_path: 本地保存路径（仅对URL有效）
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，保存到本地
            if local_path is None:
                local_path = self.GetFilename() or "downloaded.txt"
            file_obj = ToolFile(local_path)
            file_obj.TryCreateParentPath()
            async with aiofiles.open(file_obj.GetFullPath(), 'w', encoding='utf-8') as f:
                await f.write(content)
            return file_obj
        else:
            if not self._tool_file:
                raise PathValidationError(f"Invalid file path: {self.path}")
            self._tool_file.TryCreateParentPath()
            async with aiofiles.open(self._tool_file.GetFullPath(), 'w', encoding='utf-8') as f:
                await f.write(content)
            return self
    
    async def SaveAsBinaryAsync(self, content: bytes, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        异步保存为二进制
        
        Args:
            content: 二进制内容
            local_path: 本地保存路径（仅对URL有效）
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，保存到本地
            if local_path is None:
                local_path = self.GetFilename() or "downloaded.bin"
            file_obj = ToolFile(local_path)
            file_obj.TryCreateParentPath()
            async with aiofiles.open(file_obj.GetFullPath(), 'wb') as f:
                await f.write(content)
            return file_obj
        else:
            if not self._tool_file:
                raise PathValidationError(f"Invalid file path: {self.path}")
            self._tool_file.TryCreateParentPath()
            async with aiofiles.open(self._tool_file.GetFullPath(), 'wb') as f:
                await f.write(content)
            return self
    
    async def SaveAsJsonAsync(self, data: Any, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        异步保存为JSON
        
        Args:
            data: JSON数据
            local_path: 本地保存路径（仅对URL有效）
            
        Returns:
            保存的文件对象或Interaction对象
        """
        # 序列化JSON数据
        try:
            from pydantic import BaseModel
            if isinstance(data, BaseModel):
                json_data = data.model_dump()
                json_data["__type"] = f"{data.__class__.__name__}, pydantic.BaseModel"
            else:
                json_data = data
            json_content = json.dumps(json_data, indent=4, ensure_ascii=False)
        except Exception as e:
            raise SaveError(f"Failed to serialize JSON data: {str(e)}")
        
        # 保存JSON内容
        return await self.SaveAsTextAsync(json_content, local_path)
    
    # HTTP请求方法（仅对URL有效）
    def Get(self, callback: Callable[[Optional[Any]], None]) -> bool:
        """
        同步GET请求（仅对URL有效）
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            
        Returns:
            是否请求成功
        """
        if not self._is_url:
            raise InteractionError("GET method is only available for URLs")
        if not self._tool_url:
            callback(None)
            return False
        return self._tool_url.Get(callback)
    
    def Post(self, callback: Callable[[Optional[Any]], None], form_data: Optional[Dict[str, str]] = None) -> bool:
        """
        同步POST请求（仅对URL有效）
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            form_data: 表单数据字典
            
        Returns:
            是否请求成功
        """
        if not self._is_url:
            raise InteractionError("POST method is only available for URLs")
        if not self._tool_url:
            callback(None)
            return False
        return self._tool_url.Post(callback, form_data)
    
    async def GetAsync(self, callback: Callable[[Optional[Any]], None]) -> bool:
        """
        异步GET请求（仅对URL有效）
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            
        Returns:
            是否请求成功
        """
        if not self._is_url:
            raise InteractionError("GET method is only available for URLs")
        if not self._tool_url:
            callback(None)
            return False
        return await self._tool_url.GetAsync(callback)
    
    async def PostAsync(self, callback: Callable[[Optional[Any]], None], form_data: Optional[Dict[str, str]] = None) -> bool:
        """
        异步POST请求（仅对URL有效）
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            form_data: 表单数据字典
            
        Returns:
            是否请求成功
        """
        if not self._is_url:
            raise InteractionError("POST method is only available for URLs")
        if not self._tool_url:
            callback(None)
            return False
        return await self._tool_url.PostAsync(callback, form_data)
    
    # 便利方法
    def Save(self, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        自动选择格式保存
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，先下载内容再保存
            if not self._tool_url:
                raise PathValidationError(f"Invalid URL: {self.path}")
            return self._tool_url.Save(local_path)
        else:
            # 对于本地文件，直接返回自身（已存在）
            return self
    
    async def SaveAsync(self, local_path: Optional[str] = None) -> Union[ToolFile, 'Interaction']:
        """
        异步自动选择格式保存
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            保存的文件对象或Interaction对象
        """
        if self._is_url:
            # 对于URL，异步下载内容
            if not self._tool_url:
                raise PathValidationError(f"Invalid URL: {self.path}")
            
            if local_path is None:
                local_path = self.GetFilename() or "downloaded_file"
            
            file_obj = ToolFile(local_path)
            file_obj.TryCreateParentPath()
            
            try:
                if self.IsText:
                    content = await self.LoadAsTextAsync()
                    await self.SaveAsTextAsync(content, local_path)
                elif self.IsJson:
                    content = await self.LoadAsJsonAsync()
                    await self.SaveAsJsonAsync(content, local_path)
                else:
                    content = await self.LoadAsBinaryAsync()
                    await self.SaveAsBinaryAsync(content, local_path)
                
                return file_obj
            except Exception as e:
                raise SaveError(f"Failed to save {self.path}: {str(e)}")
        else:
            # 对于本地文件，直接返回自身
            return self
    
    def Download(self, local_path: Optional[str] = None) -> ToolFile:
        """
        下载文件（仅对URL有效）
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            下载的文件对象
        """
        if not self._is_url:
            raise InteractionError("Download method is only available for URLs")
        if not self._tool_url:
            raise PathValidationError(f"Invalid URL: {self.path}")
        return self._tool_url.Download(local_path)
    
    async def DownloadAsync(self, local_path: Optional[str] = None) -> ToolFile:
        """
        异步下载文件（仅对URL有效）
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            下载的文件对象
        """
        if not self._is_url:
            raise InteractionError("Download method is only available for URLs")
        if not self._tool_url:
            raise PathValidationError(f"Invalid URL: {self.path}")
        return await self._tool_url.DownloadAsync(local_path)
    
    def Copy(self, target_path: Optional[Union[str, ToolFile, 'Interaction']] = None) -> 'Interaction':
        """
        复制文件（仅对本地文件有效）
        
        Args:
            target_path: 目标路径
            
        Returns:
            新的Interaction对象
        """
        if not self._is_local:
            raise InteractionError("Copy method is only available for local files")
        if not self._tool_file:
            raise PathValidationError(f"Invalid file path: {self.path}")
        
        if target_path is None:
            copied_file = self._tool_file.Copy()
        else:
            if isinstance(target_path, Interaction):
                target_path = target_path.path
            elif isinstance(target_path, ToolFile):
                target_path = target_path.GetFullPath()
            copied_file = self._tool_file.Copy(str(target_path))
        
        return Interaction(copied_file.GetFullPath())
    
    def Move(self, target_path: Union[str, ToolFile, 'Interaction']) -> 'Interaction':
        """
        移动文件（仅对本地文件有效）
        
        Args:
            target_path: 目标路径
            
        Returns:
            更新后的Interaction对象
        """
        if not self._is_local:
            raise InteractionError("Move method is only available for local files")
        if not self._tool_file:
            raise PathValidationError(f"Invalid file path: {self.path}")
        
        if isinstance(target_path, Interaction):
            target_path = target_path.path
        elif isinstance(target_path, ToolFile):
            target_path = target_path.GetFullPath()
        
        self._tool_file.Move(str(target_path))
        self.path = self._tool_file.GetFullPath()
        return self
    
    def Remove(self) -> 'Interaction':
        """
        删除文件（仅对本地文件有效）
        
        Returns:
            Interaction对象本身
        """
        if not self._is_local:
            raise InteractionError("Remove method is only available for local files")
        if not self._tool_file:
            raise PathValidationError(f"Invalid file path: {self.path}")
        
        self._tool_file.Remove()
        return self
    
    def Exists(self) -> bool:
        """
        检查文件是否存在
        
        Returns:
            是否存在
        """
        return self.IsValid
    
    def GetSize(self) -> int:
        """
        获取文件大小（仅对本地文件有效）
        
        Returns:
            文件大小（字节）
        """
        if not self._is_local:
            raise InteractionError("GetSize method is only available for local files")
        if not self._tool_file or not self._tool_file.Exists():
            raise PathValidationError(f"File not found: {self.path}")
        
        return self._tool_file.GetSize()
    
    def GetDir(self) -> str:
        """
        获取目录路径
        
        Returns:
            目录路径
        """
        if self._is_local:
            return self._tool_file.GetDir() if self._tool_file else ""
        else:
            # 对于URL，返回基础URL
            if self._tool_url:
                parsed = urllib.parse.urlparse(self._tool_url.url)
                return f"{parsed.scheme}://{parsed.netloc}"
            return ""
    
    def GetParentDir(self) -> 'Interaction':
        """
        获取父目录的Interaction对象
        
        Returns:
            父目录的Interaction对象
        """
        if self._is_local:
            if not self._tool_file:
                raise PathValidationError(f"Invalid file path: {self.path}")
            parent_dir = self._tool_file.GetParentDir()
            return Interaction(parent_dir.GetFullPath())
        else:
            # 对于URL，返回基础URL
            base_url = self.GetDir()
            return Interaction(base_url)
    
    def ToString(self) -> str:
        """获取完整路径"""
        return self.path
    
    def GetFullPath(self) -> str:
        """获取完整路径"""
        return self.path
