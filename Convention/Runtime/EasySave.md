[返回](./Runtime-README.md)

# /Convention/Runtime/EasySave

---

完整的序列化和数据持久化系统，提供易用的数据存储和读取功能

## 核心特性

### 支持的数据格式
- **EasySave - JSON** 支持多态的json格式
- **Binary** 紧凑的二进制格式

## EasySave主类

### 基本保存操作

#### 泛型保存
- `Save<T>(string key, T value)` 保存到默认文件
- `Save<T>(string key, T value, string filePath)` 保存到指定文件
- `Save<T>(string key, T value, EasySaveSettings settings)` 使用自定义设置

#### 原始数据保存
- `SaveRaw(byte[] bytes)` / `SaveRaw(string str)` 保存原始数据
- `AppendRaw(byte[] bytes)` / `AppendRaw(string str)` 追加原始数据

### 基本加载操作

#### 泛型加载
- `Load<T>(string key)` 从默认文件加载
- `Load<T>(string key, string filePath)` 从指定文件加载
- `Load<T>(string key, T defaultValue)` 带默认值的安全加载

#### 原始数据加载
- `LoadRawBytes()` / `LoadRawString()` 加载原始数据
- `LoadString(string key, string defaultValue)` 加载字符串

#### 加载到现有对象
- `LoadInto<T>(string key, T obj)` 将数据加载到现有对象
- `LoadInto(string key, string filePath, object obj)` 从指定文件加载到对象

### 序列化操作

#### 内存序列化
- `Serialize<T>(T value, EasySaveSettings settings = null)` 序列化为字节数组
- `Deserialize<T>(byte[] bytes, EasySaveSettings settings = null)` 从字节数组反序列化
- `DeserializeInto<T>(byte[] bytes, T obj)` 反序列化到现有对象

### 加密和压缩

#### 加密操作
- `EncryptBytes(byte[] bytes, string password = null)` 加密字节数组
- `DecryptBytes(byte[] bytes, string password = null)` 解密字节数组
- `EncryptString(string str, string password = null)` 加密字符串
- `DecryptString(string str, string password = null)` 解密字符串

#### 压缩操作
- `CompressBytes(byte[] bytes)` 压缩字节数组
- `DecompressBytes(byte[] bytes)` 解压字节数组
- `CompressString(string str)` 压缩字符串
- `DecompressString(string str)` 解压字符串

### 备份系统

#### 备份操作
- `CreateBackup()` / `CreateBackup(string filePath)` 创建备份
- `RestoreBackup(string filePath)` 恢复备份

#### 时间戳
- `GetTimestamp()` / `GetTimestamp(string filePath)` 获取文件时间戳

### 缓存系统

#### 缓存操作
- `StoreCachedFile()` / `StoreCachedFile(string filePath)` 存储缓存文件
- `CacheFile()` / `CacheFile(string filePath)` 缓存文件

## EasySaveSettings配置

### 枚举类型定义

#### 存储位置
```csharp
public enum Location { File, InternalMS, Cache }
```

#### 目录类型
```csharp
public enum Directory { PersistentDataPath, DataPath }
```

#### 加密类型
```csharp
public enum EncryptionType { None, AES }
```

#### 压缩类型
```csharp
public enum CompressionType { None, Gzip }
```

#### 数据格式
```csharp
public enum Format { JSON }
```

#### 引用模式
```csharp
public enum ReferenceMode { ByRef, ByValue, ByRefAndValue }
```
