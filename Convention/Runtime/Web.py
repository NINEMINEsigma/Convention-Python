from .Config            import *
from .File              import ToolFile
import                         json
import                         urllib.parse
import                         urllib.request
import                         urllib.error
import                         asyncio
import                         os
import                         re
from typing             import *
from pydantic           import BaseModel

try:
    import aiohttp
    import aiofiles
except ImportError as e:
    ImportingThrow(e, "Web", ["aiohttp", "aiofiles"])

class WebError(Exception):
    """网络操作异常基类"""
    pass

class URLValidationError(WebError):
    """URL验证异常"""
    pass

class HTTPRequestError(WebError):
    """HTTP请求异常"""
    pass

class DownloadError(WebError):
    """下载异常"""
    pass

class ToolURL(BaseModel):
    """网络URL工具类，提供HTTP客户端和URL操作功能"""
    
    url: str
    
    def __init__(self, url: Union[str, 'ToolURL']):
        """
        从URL字符串创建对象
        
        Args:
            url: URL字符串或ToolURL对象
        """
        if isinstance(url, ToolURL):
            url = url.url
        super().__init__(url=str(url))
    
    def __str__(self) -> str:
        """隐式字符串转换"""
        return self.url
    
    def __bool__(self) -> bool:
        """隐式布尔转换，等同于IsValid"""
        return self.IsValid
    
    def ToString(self) -> str:
        """获取完整URL"""
        return self.url
    
    def GetFullURL(self) -> str:
        """获取完整URL"""
        return self.url
    
    @property
    def FullURL(self) -> str:
        """获取完整URL属性"""
        return self.url
    
    @property
    def IsValid(self) -> bool:
        """检查URL是否有效"""
        return self.ValidateURL()
    
    def ValidateURL(self) -> bool:
        """
        验证URL格式
        
        Returns:
            是否为有效的HTTP/HTTPS URL
        """
        try:
            parsed = urllib.parse.urlparse(self.url)
            return parsed.scheme in ('http', 'https') and parsed.netloc != ''
        except Exception:
            return False
    
    def GetFilename(self) -> str:
        """
        获取URL中的文件名
        
        Returns:
            URL路径中的文件名
        """
        try:
            parsed = urllib.parse.urlparse(self.url)
            path = parsed.path
            if path:
                return os.path.basename(path)
            return ""
        except Exception:
            return ""
    
    def GetExtension(self) -> str:
        """
        获取文件扩展名
        
        Returns:
            文件扩展名（不包含点）
        """
        filename = self.GetFilename()
        if '.' in filename:
            return filename.split('.')[-1].lower()
        return ""
    
    def ExtensionIs(self, *extensions: str) -> bool:
        """
        检查扩展名是否匹配
        
        Args:
            *extensions: 要检查的扩展名列表
            
        Returns:
            是否匹配任一扩展名
        """
        current_ext = self.GetExtension()
        return current_ext in [ext.lower().lstrip('.') for ext in extensions]
    
    def Open(self, url: str) -> 'ToolURL':
        """
        在当前对象上打开新URL
        
        Args:
            url: 新的URL字符串
            
        Returns:
            更新后的ToolURL对象
        """
        self.url = str(url)
        return self
    
    # 文件类型判断属性
    @property
    def IsText(self) -> bool:
        """是否为文本文件（txt, html, htm, css, js, xml, csv）"""
        return self.ExtensionIs('txt', 'html', 'htm', 'css', 'js', 'xml', 'csv', 'md', 'py', 'java', 'cpp', 'c', 'h')
    
    @property
    def IsJson(self) -> bool:
        """是否为JSON文件"""
        return self.ExtensionIs('json')
    
    @property
    def IsImage(self) -> bool:
        """是否为图像文件（jpg, jpeg, png, gif, bmp, svg）"""
        return self.ExtensionIs('jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp')
    
    @property
    def IsDocument(self) -> bool:
        """是否为文档文件（pdf, doc, docx, xls, xlsx, ppt, pptx）"""
        return self.ExtensionIs('pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx')
    
    # HTTP请求方法
    def Get(self, callback: Callable[[Optional[Any]], None]) -> bool:
        """
        同步GET请求
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            
        Returns:
            是否请求成功
        """
        if not self.IsValid:
            callback(None)
            return False
        
        try:
            with urllib.request.urlopen(self.url) as response:
                callback(response)
                return True
        except Exception as e:
            callback(None)
            return False
    
    def Post(self, callback: Callable[[Optional[Any]], None], form_data: Optional[Dict[str, str]] = None) -> bool:
        """
        同步POST请求
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            form_data: 表单数据字典
            
        Returns:
            是否请求成功
        """
        if not self.IsValid:
            callback(None)
            return False
        
        try:
            data = None
            if form_data:
                data = urllib.parse.urlencode(form_data).encode('utf-8')
            
            req = urllib.request.Request(self.url, data=data, method='POST')
            if form_data:
                req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            
            with urllib.request.urlopen(req) as response:
                callback(response)
                return True
        except Exception as e:
            callback(None)
            return False
    
    # 异步HTTP请求方法
    async def GetAsync(self, callback: Callable[[Optional[Any]], None]) -> bool:
        """
        异步GET请求
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            
        Returns:
            是否请求成功
        """
        if not self.IsValid:
            callback(None)
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    callback(response)
                    return True
        except Exception as e:
            callback(None)
            return False
    
    async def PostAsync(self, callback: Callable[[Optional[Any]], None], form_data: Optional[Dict[str, str]] = None) -> bool:
        """
        异步POST请求
        
        Args:
            callback: 响应回调函数，成功时接收响应对象，失败时接收None
            form_data: 表单数据字典
            
        Returns:
            是否请求成功
        """
        if not self.IsValid:
            callback(None)
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, data=form_data) as response:
                    callback(response)
                    return True
        except Exception as e:
            callback(None)
            return False
    
    # 内容加载方法
    def LoadAsText(self) -> str:
        """
        同步加载为文本
        
        Returns:
            文本内容
        """
        if not self.IsValid:
            raise URLValidationError(f"Invalid URL: {self.url}")
        
        try:
            with urllib.request.urlopen(self.url) as response:
                content = response.read()
                # 尝试检测编码
                encoding = response.headers.get_content_charset() or 'utf-8'
                return content.decode(encoding)
        except Exception as e:
            raise HTTPRequestError(f"Failed to load text from {self.url}: {str(e)}")
    
    async def LoadAsTextAsync(self) -> str:
        """
        异步加载为文本
        
        Returns:
            文本内容
        """
        if not self.IsValid:
            raise URLValidationError(f"Invalid URL: {self.url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    return await response.text()
        except Exception as e:
            raise HTTPRequestError(f"Failed to load text from {self.url}: {str(e)}")
    
    def LoadAsBinary(self) -> bytes:
        """
        同步加载为字节数组
        
        Returns:
            二进制内容
        """
        if not self.IsValid:
            raise URLValidationError(f"Invalid URL: {self.url}")
        
        try:
            with urllib.request.urlopen(self.url) as response:
                return response.read()
        except Exception as e:
            raise HTTPRequestError(f"Failed to load binary from {self.url}: {str(e)}")
    
    async def LoadAsBinaryAsync(self) -> bytes:
        """
        异步加载为字节数组
        
        Returns:
            二进制内容
        """
        if not self.IsValid:
            raise URLValidationError(f"Invalid URL: {self.url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    return await response.read()
        except Exception as e:
            raise HTTPRequestError(f"Failed to load binary from {self.url}: {str(e)}")
    
    def LoadAsJson(self, model_type: Optional[type] = None) -> Any:
        """
        同步加载并反序列化JSON
        
        Args:
            model_type: 可选的Pydantic模型类型
            
        Returns:
            JSON数据或模型对象
        """
        text_content = self.LoadAsText()
        try:
            json_data = json.loads(text_content)
            if model_type and issubclass(model_type, BaseModel):
                return model_type.model_validate(json_data)
            return json_data
        except json.JSONDecodeError as e:
            raise HTTPRequestError(f"Failed to parse JSON from {self.url}: {str(e)}")
    
    async def LoadAsJsonAsync(self, model_type: Optional[type] = None) -> Any:
        """
        异步加载并反序列化JSON
        
        Args:
            model_type: 可选的Pydantic模型类型
            
        Returns:
            JSON数据或模型对象
        """
        text_content = await self.LoadAsTextAsync()
        try:
            json_data = json.loads(text_content)
            if model_type and issubclass(model_type, BaseModel):
                return model_type.model_validate(json_data)
            return json_data
        except json.JSONDecodeError as e:
            raise HTTPRequestError(f"Failed to parse JSON from {self.url}: {str(e)}")
    
    # 文件保存和下载功能
    def Save(self, local_path: Optional[str] = None) -> ToolFile:
        """
        自动选择格式保存到本地
        
        Args:
            local_path: 本地保存路径，如果为None则自动生成
            
        Returns:
            保存的文件对象
        """
        if local_path is None:
            local_path = self.GetFilename() or "downloaded_file"
        
        file_obj = ToolFile(local_path)
        file_obj.TryCreateParentPath()
        
        if self.IsText:
            return self.SaveAsText(local_path)
        elif self.IsJson:
            return self.SaveAsJson(local_path)
        else:
            return self.SaveAsBinary(local_path)
    
    def SaveAsText(self, local_path: Optional[str] = None) -> ToolFile:
        """
        保存为文本文件
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            保存的文件对象
        """
        if local_path is None:
            local_path = self.GetFilename() or "downloaded.txt"
        
        text_content = self.LoadAsText()
        file_obj = ToolFile(local_path)
        file_obj.TryCreateParentPath()
        file_obj.SaveAsText(text_content)
        return file_obj
    
    def SaveAsJson(self, local_path: Optional[str] = None) -> ToolFile:
        """
        保存为JSON文件
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            保存的文件对象
        """
        if local_path is None:
            local_path = self.GetFilename() or "downloaded.json"
        
        json_data = self.LoadAsJson()
        file_obj = ToolFile(local_path)
        file_obj.TryCreateParentPath()
        file_obj.SaveAsJson(json_data)
        return file_obj
    
    def SaveAsBinary(self, local_path: Optional[str] = None) -> ToolFile:
        """
        保存为二进制文件
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            保存的文件对象
        """
        if local_path is None:
            local_path = self.GetFilename() or "downloaded.bin"
        
        binary_content = self.LoadAsBinary()
        file_obj = ToolFile(local_path)
        file_obj.TryCreateParentPath()
        file_obj.SaveAsBinary(binary_content)
        return file_obj
    
    def Download(self, local_path: Optional[str] = None) -> ToolFile:
        """
        同步下载文件
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            下载的文件对象
        """
        return self.Save(local_path)
    
    async def DownloadAsync(self, local_path: Optional[str] = None) -> ToolFile:
        """
        异步下载文件
        
        Args:
            local_path: 本地保存路径
            
        Returns:
            下载的文件对象
        """
        if local_path is None:
            local_path = self.GetFilename() or "downloaded_file"
        
        file_obj = ToolFile(local_path)
        file_obj.TryCreateParentPath()
        
        try:
            if self.IsText:
                content = await self.LoadAsTextAsync()
                file_obj.SaveAsText(content)
            elif self.IsJson:
                content = await self.LoadAsJsonAsync()
                file_obj.SaveAsJson(content)
            else:
                content = await self.LoadAsBinaryAsync()
                file_obj.SaveAsBinary(content)
            
            return file_obj
        except Exception as e:
            raise DownloadError(f"Failed to download {self.url}: {str(e)}")


# 静态HTTP客户端实例，避免连接池耗尽
_http_session: Optional[aiohttp.ClientSession] = None

async def get_http_session() -> aiohttp.ClientSession:
    """获取全局HTTP会话实例"""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session

async def close_http_session():
    """关闭全局HTTP会话"""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None
