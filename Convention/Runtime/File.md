[返回](./Runtime-README.md)

# /Convention/Runtime/File

---

文件操作工具模块，提供跨平台的文件和目录操作功能

## ToolFile类

### 构造与基本信息
- `ToolFile(string path)` 从路径创建文件对象
- `ToString()` / `GetFullPath()` 获取完整路径
- `GetName(bool is_ignore_extension = false)` 获取文件名
- `GetExtension()` 获取文件扩展名

### 路径操作
- `ToolFile operator |(ToolFile left, string rightPath)` 路径连接操作符
- `Open(string path)` 打开指定路径
- `Open(FileMode mode)` 以指定模式打开文件
- `Close()` 关闭文件流
- `BackToParentDir()` 回到父目录
- `GetParentDir()` 获取父目录

### 存在性检查
- `Exists()` 检查文件或目录是否存在
- `implicit operator bool` 隐式布尔转换，等同于Exists()

### 类型判断
- `IsDir()` 是否为目录
- `IsFile()` 是否为文件
- `IsFileEmpty()` 文件是否为空

### 文件加载
支持多种格式的文件读取：

#### JSON操作
- `LoadAsRawJson<T>()` 原始JSON反序列化
- `LoadAsJson<T>(string key = "data")` 使用EasySave加载JSON

#### 文本操作
- `LoadAsText()` 加载为文本字符串

#### 二进制操作
- `LoadAsBinary()` 加载为字节数组

### 文件保存
支持多种格式的文件写入：

#### JSON保存
- `SaveAsRawJson<T>(T data)` 原始JSON序列化保存
- `SaveAsJson<T>(T data, string key)` 使用EasySave保存JSON

#### 二进制保存
- `SaveAsBinary(byte[] data)` 保存字节数组
- `SaveDataAsBinary(string path, byte[] outdata, FileStream Stream = null)` 静态二进制保存

### 文件操作
- `Create()` 创建文件或目录
- `Rename(string newPath)` 重命名
- `Move(string path)` 移动文件
- `Copy(string path, out ToolFile copyTo)` 复制文件
- `Delete()` / `Remove()` 删除文件或目录
- `Refresh()` 刷新文件信息

### 路径管理
- `MustExistsPath()` 确保路径存在（自动创建）
- `TryCreateParentPath()` 尝试创建父路径

### 目录操作
- `DirIter()` 遍历目录获取字符串列表
- `DirToolFileIter()` 遍历目录获取ToolFile列表
- `DirCount()` 获取目录内项目数量
- `DirClear()` 清空目录内容
- `MakeFileInside(string source, bool isDeleteSource = false)` 在目录内创建文件

### 文件对话框（平台相关）
- `SelectMultipleFiles(string filter, string title)` 多文件选择对话框
- `SelectFile(string filter, string title)` 单文件选择对话框
- `SaveFile(string filter, string title)` 保存文件对话框
- `SelectFolder(string description)` 文件夹选择对话框

### 文件浏览
- `BrowseFile(params string[] extensions)` 浏览指定扩展名的文件
- `BrowseToolFile(params string[] extensions)` 浏览并返回ToolFile对象

### 时间戳
- `GetTimestamp()` 获取文件时间戳
