# Convention-Python

Convention-Python基于 Convention-Template 规范实现的一套完整的开发工具集。

## 主要内容

### 辅助 (Config.py)
- **内置依赖**: 提供辅助函数与辅助类型

### 架构 (Architecture.py)
- **依赖注入容器**: 支持类型注册、依赖解析和生命周期管理
- **信号系统**: 提供发布-订阅模式的消息通信机制
- **时间线管理**: 支持基于条件的任务队列和执行流程控制
- **单例模式**: 内置单例模型支持

### 异步 (Asynchrony.py)
- **线程管理**: 提供线程实例、原子操作、锁机制
- **并发控制**: 支持线程安全的数据结构和操作
- **异步工具**: 简化异步编程的工具函数

### 配置 (GlobalConfig.py)
- **类型系统**: 强大的类型检查和转换系统
- **调试支持**: 内置调试模式和彩色输出
- **平台兼容**: 跨平台路径和环境管理
- **全局配置**: 统一的配置管理机制

### 文件 (File.py)
- **ToolFile 类**: 强大的文件操作封装
  - 支持多种文件格式 (JSON, CSV, Excel, 图像, 音频, Word文档等)
  - 文件压缩和解压缩 (ZIP, TAR)
  - 文件加密和解密
  - 哈希值计算和验证
  - 文件监控和备份
  - 权限管理
- **批量处理**: 支持文件批量操作和处理

### 序列化 (EasySave.py)
- **序列化支持**: JSON 和二进制格式的序列化
- **反射集成**: 基于反射的对象序列化和反序列化
- **备份机制**: 自动备份和恢复功能
- **字段过滤**: 支持自定义字段选择和忽略规则

### 反射 (Reflection.py)
- **类型管理**: 完整的类型信息管理和缓存
- **成员访问**: 字段和方法的动态访问
- **类型转换**: 灵活的类型转换和验证
- **泛型支持**: 支持泛型类型的处理

### 视觉 (Visual)

#### 可视化 (Visual/Core.py)
- **图表生成**: 支持多种图表类型 (折线图、柱状图、散点图、饼图等)
- **数据处理**: 数据清洗、标准化、归一化
- **样式定制**: 丰富的图表样式和主题选项

#### 图像处理 (Visual/OpenCV.py)
- **ImageObject 类**: 完整的图像处理功能
- **图像增强**: 支持 30+ 种图像增强算法
- **格式转换**: 支持多种图像格式转换
- **批量处理**: 支持图像批量处理和增强

#### 词云生成 (Visual/WordCloud.py)
- **词云创建**: 支持中英文词云生成
- **样式定制**: 丰富的样式和布局选项

### 字符串工具 (String.py)
- **字符串处理**: 长度限制、填充、编码转换
- **中文分词**: 集成 jieba 分词支持

## 安装说明

### 环境要求
- Python >= 3.12
- 操作系统: Windows, Linux, macOS

### 依赖包
运行时自动报告需要被引入的包

或

调用Config中ReleaseFailed2Requirements函数生成requirements.txt文件

### 安装方式

1. **从源码安装**:
```bash
git clone https://github.com/NINEMINEsigma/Convention-Python.git
cd Convention-Python
pip install -e .
```

2. **直接安装**:
```bash
pip install .
```

3. **打包安装**
```bash
pip install build
python -m build
pip install dist/convention.tar.gz
```

## 🚀 使用示例

### 架构模式示例
```python
from Convention.Runtime import Architecture

# 注册服务
class DatabaseService:
    def query(self, sql): return "result"

db_service = DatabaseService()
Architecture.Register(DatabaseService, db_service, lambda: print("DB服务初始化"))

# 获取服务
service = Architecture.Get(DatabaseService)
result = service.query("SELECT * FROM users")
```

### 文件操作示例
```python
from Convention.Runtime import ToolFile

# 创建文件对象
file = ToolFile("data.json")

# 保存和加载 JSON 数据
data = {"name": "张三", "age": 25}
file.SaveAsJson(data)
loaded_data = file.LoadAsJson()

# 文件压缩
compressed = file.Compress("backup.zip")

# 计算哈希值
hash_value = file.calculate_hash("sha256")
```

### 数据序列化示例
```python
from Convention.Runtime import EasySave
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# 保存数据
user = User(name="李四", age=30, email="lisi@example.com")
EasySave.Write(user, "user.json")

# 读取数据
loaded_user = EasySave.Read(User, "user.json")
```

### 数据可视化示例
```python
from Convention.Runtime.Visual import Core
import pandas as pd

# 创建数据可视化生成器
df = pd.read_csv("sales_data.csv")
generator = Core.data_visual_generator("sales_data.csv")

# 绘制图表
generator.plot_line("month", "sales", title="月度销售趋势")
generator.plot_bar("product", "revenue", title="产品收入对比")
generator.plot_pie("category", title="类别分布")
```

### 图像处理示例
```python
from Convention.Runtime.Visual.OpenCV import ImageObject
from Convention.Runtime.Visual.Core import ImageAugmentConfig, ResizeAugmentConfig

# 加载图像
image = ImageObject("input.jpg")

# 图像增强配置
config = ImageAugmentConfig(
    resize=ResizeAugmentConfig(width=800, height=600),
    lighting=LightingAugmentConfig(lighting=20),
    contrast=ContrastAugmentConfig(contrast=1.2)
)

# 批量增强
results = config.augment_from_dir_to("input_dir", "output_dir")
```

## 打包指令

### 构建分发包
```bash
# 清理之前的构建文件
python setup.py clean --all
rm -rf build/ dist/ *.egg-info/

# 构建源码包和轮子包
python setup.py sdist bdist_wheel

# 或使用 build 工具 (推荐)
pip install build
python -m build
```

### 安装本地包
```bash
# 开发模式安装 (可编辑安装)
pip install -e .

# 普通安装
pip install .
```

### 上传到 PyPI
```bash
# 安装上传工具
pip install twine

# 检查包
twine check dist/*

# 上传到测试 PyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# 上传到正式 PyPI
twine upload dist/*
```

### 创建可执行文件
```bash
# 使用 PyInstaller
pip install pyinstaller
pyinstaller --onefile --name convention-tool your_main_script.py
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 作者

**LiuBai** - [NINEMINEsigma](https://github.com/NINEMINEsigma)

## 相关链接

- [Convention-Template](https://github.com/NINEMINEsigma/Convention-Template) - 项目模板规范
- [GitHub Issues](https://github.com/NINEMINEsigma/Convention-Python/issues) - 问题反馈
- [GitHub Releases](https://github.com/NINEMINEsigma/Convention-Python/releases) - 版本发布

*最后更新: 2025年9月*
