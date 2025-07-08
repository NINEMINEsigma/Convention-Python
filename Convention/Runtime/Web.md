[返回](./Runtime-README.md)

# /Convention/Runtime/Web

---

网络工具模块，提供HTTP客户端和URL操作功能

## ToolURL类

### 构造与基本信息
- `ToolURL(string url)` 从URL字符串创建对象
- `ToString()` / `GetFullURL()` / `FullURL` 获取完整URL
- `implicit operator string` 隐式字符串转换

### URL属性解析
- `GetFilename()` 获取URL中的文件名
- `GetExtension()` 获取文件扩展名
- `ExtensionIs(params string[] extensions)` 检查扩展名是否匹配

### URL验证
- `IsValid` 属性，检查URL是否有效
- `ValidateURL()` 验证URL格式
- `implicit operator bool` 隐式布尔转换，等同于IsValid

支持HTTP和HTTPS协议的绝对URL

### HTTP方法

#### GET请求
- `GetAsync(Action<HttpResponseMessage> callback)` 异步GET
- `Get(Action<HttpResponseMessage> callback)` 同步GET

#### POST请求
- `PostAsync(Action<HttpResponseMessage> callback, Dictionary<string, string> formData = null)` 异步POST
- `Post(Action<HttpResponseMessage> callback, Dictionary<string, string> formData = null)` 同步POST

支持表单数据提交

### 内容加载

#### 文本加载
- `LoadAsTextAsync()` 异步加载为文本
- `LoadAsText()` 同步加载为文本

#### 二进制加载
- `LoadAsBinaryAsync()` 异步加载为字节数组
- `LoadAsBinary()` 同步加载为字节数组

#### JSON加载
- `LoadAsJson<T>()` 同步加载并反序列化JSON
- `LoadAsJsonAsync<T>()` 异步加载并反序列化JSON

### 文件保存
- `Save(string localPath = null)` 自动选择格式保存到本地
- `SaveAsText(string localPath = null)` 保存为文本文件
- `SaveAsJson(string localPath = null)` 保存为JSON文件
- `SaveAsBinary(string localPath = null)` 保存为二进制文件

### 文件类型判断
- `IsText` 是否为文本文件（txt, html, htm, css, js, xml, csv）
- `IsJson` 是否为JSON文件
- `IsImage` 是否为图像文件（jpg, jpeg, png, gif, bmp, svg）
- `IsDocument` 是否为文档文件（pdf, doc, docx, xls, xlsx, ppt, pptx）

### 高级操作
- `Open(string url)` 在当前对象上打开新URL
- `DownloadAsync(string localPath = null)` 异步下载文件
- `Download(string localPath = null)` 同步下载文件

## 设计特点

### 统一的HTTP客户端
使用静态 `HttpClient` 实例，避免连接池耗尽

### 自动内容类型检测
基于文件扩展名自动判断内容类型，优化保存和处理策略

### 异步支持
所有网络操作都提供异步和同步两种版本

### 错误处理
网络请求失败时回调函数接收null参数，方法返回false

### 文件管理集成
下载的文件自动转换为ToolFile对象，与文件系统模块无缝集成

### 灵活的数据格式
支持文本、二进制、JSON等多种数据格式的加载和保存

## 使用示例

### 基本HTTP请求
```csharp
var url = new ToolURL("https://api.example.com/data");
if (url.IsValid)
{
    url.Get(response => {
        if (response != null && response.IsSuccessStatusCode)
        {
            // 处理响应
        }
    });
}
```

### 文件下载
```csharp
var url = new ToolURL("https://example.com/file.json");
var localFile = url.Download("./downloads/file.json");
if (localFile.Exists())
{
    var data = localFile.LoadAsJson<MyDataType>();
}
```

### 类型安全的JSON加载
```csharp
var url = new ToolURL("https://api.example.com/users.json");
var users = url.LoadAsJson<List<User>>();
```