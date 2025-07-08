[返回](./Runtime-README.md)

# /Convention/Runtime/GlobalConfig

---

全局配置管理模块，提供配置文件操作、日志系统和项目级配置管理

## GlobalConfig类

### 构造与初始化
- `GlobalConfig(string dataDir, bool isTryCreateDataDir = false, bool isLoad = true)`
- `GlobalConfig(ToolFile dataDir, bool isTryCreateDataDir = false, bool isLoad = true)`

#### 初始化流程
1. 设置数据目录，确保目录存在
2. 检查配置文件，不存在则生成空配置
3. 可选自动加载现有配置

### 静态配置
- `ConstConfigFile` 配置文件名（默认"config.json"）
- `InitExtensionEnv()` 初始化扩展环境
- `GenerateEmptyConfigJson(ToolFile file)` 生成空配置JSON

### 文件管理
- `GetConfigFile()` / `ConfigFile` 获取配置文件对象
- `GetFile(string path, bool isMustExist = false)` 获取数据目录下的文件
- `CreateFile(string path)` 创建文件
- `EraseFile(string path)` 清空文件内容
- `RemoveFile(string path)` 删除文件

### 配置数据操作

能够被`foreach`/`for`使用迭代器历遍配置内容的键值对

#### 数据访问
- `operator[](string key)` 索引器访问配置项
- `Contains(string key)` 检查键是否存在
- `Remove(string key)` 删除配置项
- `DataSize()` 获取配置项数量

#### 持久化
- `SaveProperties()` 保存配置到文件
- `LoadProperties()` 从文件加载配置

配置文件格式：
```json
{
  "properties": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

### 日志系统
- `GetLogFile()` / `LogFile` 获取日志文件对象
- `DefaultLogger` 默认日志输出器（可自定义）

#### 日志方法
- `Log(string messageType, string message, Action<string> logger)`
- `Log(string messageType, string message)` 使用默认日志器
- `LogPropertyNotFound(string message, Action<string> logger, object @default = null)`
- `LogPropertyNotFound(string message, object @default = null)`
- `LogMessageOfPleaseCompleteConfiguration()` 记录配置提示信息

#### 日志格式
```
[time]    MessageType    : Message
```

自动调整消息类型的对齐宽度

### 配置查找
- `FindItem(string key, object @default = null)` 查找配置项，支持默认值

## ProjectConfig类

继承自 `GlobalConfig`，专门用于项目级配置管理

### 静态配置
- `ProjectConfigFileFocus` 项目配置目录焦点（默认"Assets/"）
- `InitExtensionEnv()` 初始化项目扩展环境
- `SetProjectConfigFileFocus(string path)` 设置项目配置焦点目录
- `GetProjectConfigFileFocus()` 获取项目配置焦点目录

### 构造
- `ProjectConfig(bool isLoad = true)` 使用默认项目目录构造
