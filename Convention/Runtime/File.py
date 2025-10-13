from .Config            import *
import                         json
import                         shutil
import                         os
import                         zipfile
import                         tarfile
import                         base64
import                         hashlib
import                         time
import                         datetime
import                         stat
from typing             import *
from pathlib            import Path

def GetExtensionName(file:str):
        return os.path.splitext(file)[1][1:]

def GetBaseFilename(file:str):
    return os.path.basename(file)

dir_name_type = str
file_name_type = str

class FileOperationError(Exception):
    """文件操作异常基类"""
    pass

class CompressionError(FileOperationError):
    """压缩操作异常"""
    pass

class EncryptionError(FileOperationError):
    """加密操作异常"""
    pass

class HashError(FileOperationError):
    """哈希计算异常"""
    pass

class FileMonitorError(FileOperationError):
    """文件监控异常"""
    pass

class BackupError(FileOperationError):
    """备份操作异常"""
    pass

class PermissionError(FileOperationError):
    """权限操作异常"""
    pass

try:
    from pydantic import BaseModel
except ImportError as e:
    ImportingThrow(e, "File", ["pydantic"])

class ToolFile(BaseModel):
    OriginFullPath:str

    def __init__(
        self,
        filePath:          Union[str, Self],
        ):
        filePath = os.path.expandvars(str(filePath))
        if filePath[1:].startswith(":/") or filePath[1:].startswith(":\\"):
            filePath = os.path.abspath(filePath)
        super().__init__(OriginFullPath=filePath)
    def __del__(self):
        pass
    def __str__(self):
        return self.GetFullPath()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

    def __or__(self, other):
        if other is None:
            return ToolFile(self.GetFullPath() if self.IsDir() else f"{self.GetFullPath()}\\")
        else:
            # 不使用os.path.join，因为os.path.join存在如下机制
            # 当参数路径中存在绝对路径风格时，会忽略前面的参数，例如：
            # os.path.join("E:/dev", "/analyze/") = "E:/analyze/"
            # 而我们需要的是 "E:/dev/analyze"
            first = self.GetFullPath().replace('/','\\').strip('\\')
            second = str(other).replace('/','\\')
            if first == "./":
                return ToolFile(f"{second}") 
            elif first == "../":
                first = ToolFile(f"{os.path.abspath(first)}").BackToParentDir()
            return ToolFile(f"{first}\\{second}")
    def __idiv__(self, other):
        temp = self.__or__(other)
        self.OriginFullPath = temp.GetFullPath()

    def __eq__(self, other) -> bool:
        """
        判断文件路径是否相等
        注意字符串可能不同，因为文件夹路径后缀的斜线可能被忽略
        
        Args:
            other: 另一个文件对象或路径字符串
            
        Returns:
            是否相等
        """
        if other is None:
            return False

        # 获取比较对象的路径
        other_path = other.GetFullPath() if isinstance(other, ToolFile) else str(other)
        self_path = self.OriginFullPath
        
        # 如果两个文件都存在，则直接比较路径
        if self.Exists() == True and other.Exists() == True:
            return self_path.strip('\\/') == other_path.strip('\\/')
        # 如果一个文件存在另一个不被判定为存在则一定不同
        elif self.Exists() != other.Exists():
            return False
        # 如果两个文件都不存在，则直接比较文件名在视正反斜杠相同的情况下比较路径字符串
        else:
            return self_path.replace('/','\\') == other_path.replace('/','\\')

    def ToPath(self):
        return Path(self.OriginFullPath)
    def __Path__(self):
        return Path(self.OriginFullPath)

    def Create(self):
        if self.Exists() == False:
            if self.IsDir():
                if os.path.exists(self.GetDir()):
                    os.makedirs(self.OriginFullPath)
                else:
                    raise FileNotFoundError(f"{self.OriginFullPath} cannt create, because its parent path is not exist")
            else:
                with open(self.OriginFullPath, 'w') as f:
                    f.write('')
        return self
    def Exists(self):
        return os.path.exists(self.OriginFullPath)
    def Remove(self):
        if self.Exists():
            if self.IsDir():
                shutil.rmtree(self.OriginFullPath)
            else:
                os.remove(self.OriginFullPath)
        return self
    def Copy(self, targetPath:Optional[Union[Self, str]]=None):
        if targetPath is None:
            return ToolFile(self.OriginFullPath)
        if self.Exists() == False:
            raise FileNotFoundError("file not found")
        target_file = ToolFile(str(targetPath))
        if target_file.IsDir():
            target_file = target_file|self.GetFilename()
        shutil.copy(self.OriginFullPath, str(target_file))
        return target_file
    def Move(self, targetPath:Union[Self, str]):
        if self.Exists() is False:
            raise FileNotFoundError("file not found")
        target_file = ToolFile(str(targetPath))
        if target_file.IsDir():
            target_file = target_file|self.GetFilename()
        shutil.move(self.OriginFullPath, str(target_file))
        self.OriginFullPath = target_file.OriginFullPath
        return self
    def Rename(self, newpath:Union[Self, str]):
        if self.Exists() is False:
            raise FileNotFoundError("file not found")
        newpath = str(newpath)
        if '\\' in newpath or '/' in newpath:
            newpath = GetBaseFilename(newpath)
        new_current_path = os.path.join(self.GetDir(), newpath)
        os.rename(self.OriginFullPath, new_current_path)
        self.OriginFullPath = new_current_path
        return self

    def LoadAsJson(self, encoding:str='utf-8', **kwargs) -> Any:
        with open(self.OriginFullPath, 'r', encoding=encoding) as f:
            json_data = json.load(f, **kwargs)
            return json_data
    def LoadAsCsv(self) -> "pandas.DataFrame":
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        with open(self.OriginFullPath, 'r') as f:
            return pd.read_csv(f)
    def LoadAsXml(self) -> "pandas.DataFrame":
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        with open(self.OriginFullPath, 'r') as f:
            return pd.read_xml(f)
    def LoadAsDataframe(self) -> "pandas.DataFrame":
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        with open(self.OriginFullPath, 'r') as f:
            return pd.read_csv(f)
    def LoadAsExcel(self) -> "pandas.DataFrame":
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        with open(self.OriginFullPath, 'r') as f:
            return pd.read_excel(f)
    def LoadAsBinary(self) -> bytes:
        with open(self.OriginFullPath, 'rb') as f:
            return f.read()
    def LoadAsText(self) -> str:
         with open(self.OriginFullPath, 'r') as f:
            return f.read()
    def LoadAsWav(self):
        try:
            from pydub              import AudioSegment
        except ImportError as e:
            ImportingThrow(e, "File", ["pydub"])
        return AudioSegment.from_wav(self.OriginFullPath)
    def LoadAsAudio(self):
        try:
            from pydub              import AudioSegment
        except ImportError as e:
            ImportingThrow(e, "File", ["pydub"])
        return AudioSegment.from_file(self.OriginFullPath)
    def LoadAsImage(self):
        try:
            from PIL                import Image
        except ImportError as e:
            ImportingThrow(e, "File", ["Pillow"])
        return Image.open(self.OriginFullPath)
    def LoadAsDocx(self) -> "docx.document.Document":
        '''
        try:
            from docx               import Document
            from docx.document      import Document as DocumentObject
        except ImportError as e:
            ImportingThrow(e, "File", ["python-docx"])
        '''
        try:
            from docx               import Document
            from docx.document      import Document as DocumentObject
        except ImportError as e:
            ImportingThrow(e, "File", ["python-docx"])
        return Document(self.OriginFullPath)
    def LoadAsUnknown(self, suffix:str) -> Any:
        return self.LoadAsText()
    def LoadAsModel(self, model:type["BaseModel"]) -> "BaseModel":
        return model.model_validate(self.LoadAsJson())

    def ReadLines(self):
        with open(self.OriginFullPath, 'r') as f:
            while True:
                line = f.readline()
                if not line or line == '':
                    break
                yield line
    async def ReadLinesAsync(self):
        import aiofiles
        async with aiofiles.open(self.OriginFullPath, 'r') as f:
            while True:
                line = await f.readline()
                if not line or line == '':
                    break
                yield line
    def ReadBytes(self):
        with open(self.OriginFullPath, 'rb') as f:
            while True:
                data = f.read(1024)
                if not data or data == '':
                    break
                yield data
    async def ReadBytesAsync(self):
        import aiofiles
        async with aiofiles.open(self.OriginFullPath, 'rb') as f:
            while True:
                data = await f.read(1024)
                if not data or data == '':
                    break
                yield data

    def WriteBytes(self, data:bytes):
        with open(self.OriginFullPath, 'wb') as f:
            f.write(data)
    async def WriteBytesAsync(self, data:bytes):
        import aiofiles
        async with aiofiles.open(self.OriginFullPath, 'wb') as f:
            await f.write(data)
    def WriteLines(self, data:List[str]):
        with open(self.OriginFullPath, 'w') as f:
            f.writelines(data)
    async def WriteLinesAsync(self, data:List[str]):
        import aiofiles
        async with aiofiles.open(self.OriginFullPath, 'w') as f:
            await f.writelines(data)
    
    def AppendText(self, data:str):
        with open(self.OriginFullPath, 'a') as f:
            f.write(data)
    async def AppendTextAsync(self, data:str):
        import aiofiles
        async with aiofiles.open(self.OriginFullPath, 'a') as f:
            await f.write(data)
    def AppendBytes(self, data:bytes):
        with open(self.OriginFullPath, 'ab') as f:
            f.write(data)
    async def AppendBytesAsync(self, data:bytes):
        import aiofiles
        async with aiofiles.open(self.OriginFullPath, 'ab') as f:
            await f.write(data)

    def SaveAsJson(self, json_data):
        try:
            from pydantic import BaseModel
            if isinstance(json_data, BaseModel):
                json_data = json_data.model_dump()
                json_data["__type"] = f"{self.data.__class__.__name__}, pydantic.BaseModel"
        except:
            pass
        with open(self.OriginFullPath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        return self
    def SaveAsCsv(self, csv_data:"pandas.DataFrame"):
        '''
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        '''
        csv_data.to_csv(self.OriginFullPath)
        return self
    def SaveAsXml(self, xml_data:"pandas.DataFrame"):
        '''
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        '''
        xml_data.to_xml(self.OriginFullPath)
        return self
    def SaveAsDataframe(self, dataframe_data:"pandas.DataFrame"):
        '''
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        '''
        dataframe_data.to_csv(self.OriginFullPath)
        return self
    def SaveAsExcel(self, excel_data:"pandas.DataFrame"):
        '''
        try:
            import pandas           as     pd
        except ImportError as e:
            ImportingThrow(e, "File", ["pandas"])
        '''
        excel_data.to_excel(self.OriginFullPath, index=False)
        return self
    def SaveAsBinary(self, binary_data:bytes):
        with open(self.OriginFullPath, 'wb') as f:
            f.write(binary_data)
        return self
    def SaveAsText(self, text_data:str):
        with open(self.OriginFullPath, 'w') as f:
            f.writelines(text_data)
        return self
    def SaveAsAudio(self, audio_data:"pydub.AudioSegment"):
        '''
        try:
            from pydub              import AudioSegment
        except ImportError as e:
            ImportingThrow(e, "File", ["pydub"])
        '''
        audio_data.export(self.OriginFullPath, format=self.get_extension(self.OriginFullPath))
        return self
    def SaveAsImage(self, image_data:"PIL.ImageFile.ImageFile"):
        '''
        try:
            from PIL                import Image, ImageFile
        except ImportError as e:
            ImportingThrow(e, "File", ["Pillow"])
        '''
        image_data.save(self.OriginFullPath)
        return self
    def SaveAsDocx(self, docx_data:"docx.document.Document"):
        '''
        try:
            from docx               import Document
            from docx.document      import Document as DocumentObject
        except ImportError as e:
            ImportingThrow(e, "File", ["python-docx"])
        '''
        docx_data.save(self.OriginFullPath)
        return self
    def SaveAsUnknown(self, unknown_data:Any):
        self.SaveAsBinary(unknown_data)
    def SaveAsModel(self, model:type[BaseModel]):
        self.SaveAsJson(model)

    def GetSize(self) -> int:
        '''
        return:
            return size of directory
        '''
        return os.path.getsize(self.OriginFullPath)
    def GetExtension(self):
        return GetExtensionName(self.OriginFullPath)
    def GetFullPath(self) -> str:
        return self.OriginFullPath
    def GetFilename(self, is_without_extension = False):
        '''
        if target path is a file, it return filename
        if target path is a directory, it return top directory name
        '''
        if is_without_extension and '.' in self.OriginFullPath:
            return GetBaseFilename(self.OriginFullPath)[:-(len(self.GetExtension())+1)]
        elif self.OriginFullPath[-1] == '/' or self.OriginFullPath[-1] == '\\':
            return GetBaseFilename(self.OriginFullPath[:-1])
        else:
            return GetBaseFilename(self.OriginFullPath)
    def GetDir(self):
        return os.path.dirname(self.OriginFullPath)
    def GetDirToolFile(self):
        return ToolFile(self.GetDir())
    def GetCurrentDirName(self):
        return os.path.dirname(self.OriginFullPath)

    def IsDir(self):
        if self.OriginFullPath[-1] == '\\' or self.GetFullPath()[-1] == '/':
            return True
        else:
            return os.path.isdir(self.OriginFullPath)
    def IsFile(self):
        return os.path.isfile(self.OriginFullPath)

    def TryCreateParentPath(self):
        dir_path = os.path.dirname(self.OriginFullPath)
        if dir_path == '':
            return self
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return self
    def DirIter(self):
        return os.listdir(self.OriginFullPath)
    def DirToolFileIter(self):
        result = [self]
        result.clear()
        for file in os.listdir(self.OriginFullPath):
            result.append(self|file)
        return result
    def BackToParentDir(self):
        self.OriginFullPath = self.GetDir()
        return self
    def GetParentDir(self):
        return ToolFile(self.GetDir())
    def DirCount(self, ignore_folder:bool = True):
        iter    = self.DirIter()
        result  = 0
        for content in iter:
            if ignore_folder and os.path.isdir(os.path.join(self.OriginFullPath, content)):
                continue
            result += 1
        return result
    def DirClear(self):
        for file in self.DirToolFileIter():
            file.Remove()
        return self
    def FirstFileWithExtension(self, extension:str):
        target_dir = self if self.IsDir() else ToolFile(self.GetDir())
        for file in target_dir.DirToolFileIter():
            if file.IsDir() is False and file.GetExtension() == extension:
                return file
        return None
    def FirstFile(self, pr:Callable[[str], bool]):
        target_dir = self if self.IsDir() else ToolFile(self.GetDir())
        for file in target_dir.DirToolFileIter():
            if pr(file.GetFilename()):
                return file
        return None
    def FindFileWithExtension(self, extension:str):
        target_dir = self if self.IsDir() else ToolFile(self.GetDir())
        result:List[ToolFile] = []
        for file in target_dir.DirToolFileIter():
            if file.IsDir() is False and file.GetExtension() == extension:
                result.append(file)
        return result
    def FindFile(self, pr:Callable[[str], bool]):
        target_dir = self if self.IsDir() else ToolFile(self.GetDir())
        result:List[ToolFile] = []
        for file in target_dir.DirToolFileIter():
            if pr(file.GetFilename()):
                result.append(file)
        return result
    def DirWalk(
        self,
        top,
        topdown:        bool               = True,
        onerror:        Optional[Callable] = None,
        followlinks:    bool               = False
        ) -> Iterator[tuple[dir_name_type, list[dir_name_type], list[file_name_type]]]:
        return os.walk(self.OriginFullPath, top=top, topdown=topdown, onerror=onerror, followlinks=followlinks)


    def bool(self):
        return self.Exists()
    def __bool__(self):
        return self.Exists()

    def MustExistsPath(self):
        self.TryCreateParentPath()
        self.Create()
        return self

    def MakeFileInside(self, data:Self, is_delete_source = False):
        if self.IsDir() is False:
            raise Exception("Cannot make file inside a file, because this object target is not a directory")
        result:ToolFile = self|data.GetFilename()
        if is_delete_source:
            data.Move(result)
        else:
            data.Copy(result)
        return self

    def Compress(self, output_path: Optional[str] = None, format: str = 'zip') -> 'ToolFile':
        """
        压缩文件或目录
        Args:
            output_path: 输出路径,如果为None则使用原文件名
            format: 压缩格式,支持'zip'和'tar'
        Returns:
            压缩后的文件对象
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        if output_path is None:
            output_path = self.GetFullPath() + ('.zip' if format == 'zip' else '.tar')

        try:
            if format == 'zip':
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    if self.IsDir():
                        for root, _, files in os.walk(self.GetFullPath()):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.GetFullPath())
                                zipf.write(file_path, arcname)
                    else:
                        zipf.write(self.GetFullPath(), self.GetFilename())
            elif format == 'tar':
                with tarfile.open(output_path, 'w') as tarf:
                    if self.IsDir():
                        tarf.add(self.GetFullPath(), arcname=self.GetFilename())
                    else:
                        tarf.add(self.GetFullPath(), arcname=self.GetFilename())
            else:
                raise CompressionError(f"Unsupported compression format: {format}")

            return ToolFile(output_path)
        except Exception as e:
            raise CompressionError(f"Compression failed: {str(e)}")

    def Decompress(self, output_path: Optional[str] = None) -> 'ToolFile':
        """
        解压文件
        Args:
            output_path: 输出目录,如果为None则使用原文件名
        Returns:
            解压后的目录对象
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        if output_path is None:
            output_path = self.GetFullPath() + '_extracted'

        try:
            if self.GetExtension() == 'zip':
                with zipfile.ZipFile(self.GetFullPath(), 'r') as zipf:
                    zipf.extractall(output_path)
            elif self.GetExtension() == 'tar':
                with tarfile.open(self.GetFullPath(), 'r') as tarf:
                    tarf.extractall(output_path)
            else:
                raise CompressionError(f"Unsupported archive format: {self.GetExtension()}")

            return ToolFile(output_path)
        except Exception as e:
            raise CompressionError(f"Decompression failed: {str(e)}")

    def Encrypt(self, key: str, algorithm: str = 'AES') -> 'ToolFile':
        """
        加密文件
        Args:
            key: 加密密钥
            algorithm: 加密算法,目前支持'AES'
        Returns:
            加密后的文件对象
        """
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            # 生成加密密钥
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key.encode()))

            # 创建加密器
            f = Fernet(key)

            # 读取文件内容
            with open(self.GetFullPath(), 'rb') as file:
                file_data = file.read()

            # 加密数据
            encrypted_data = f.encrypt(file_data)

            # 保存加密后的文件
            encrypted_path = self.GetFullPath() + '.encrypted'
            with open(encrypted_path, 'wb') as file:
                file.write(salt + encrypted_data)

            return ToolFile(encrypted_path)
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {str(e)}")

    def decrypt(self, key: str, algorithm: str = 'AES') -> Self:
        """
        解密文件
        Args:
            key: 解密密钥
            algorithm: 解密算法,目前支持'AES'
        Returns:
            解密后的文件对象
        """
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            # 读取加密文件
            with open(self.GetFullPath(), 'rb') as file:
                file_data = file.read()

            # 提取salt和加密数据
            salt = file_data[:16]
            encrypted_data = file_data[16:]

            # 生成解密密钥
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key.encode()))

            # 创建解密器
            f = Fernet(key)

            # 解密数据
            decrypted_data = f.decrypt(encrypted_data)

            # 保存解密后的文件
            decrypted_path = self.GetFullPath() + '.decrypted'
            with open(decrypted_path, 'wb') as file:
                file.write(decrypted_data)

            return ToolFile(decrypted_path)
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {str(e)}")

    def calculate_hash(self, algorithm: str = 'md5', chunk_size: int = 8192) -> str:
        """
        计算文件的哈希值
        Args:
            algorithm: 哈希算法,支持'md5', 'sha1', 'sha256', 'sha512'等
            chunk_size: 每次读取的字节数
        Returns:
            文件的哈希值(十六进制字符串)
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            # 获取哈希算法
            hash_algo = getattr(hashlib, algorithm.lower())
            if not hash_algo:
                raise HashError(f"Unsupported hash algorithm: {algorithm}")

            # 创建哈希对象
            hasher = hash_algo()

            # 分块读取文件并更新哈希值
            with open(self.GetFullPath(), 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)

            return hasher.hexdigest()
        except Exception as e:
            raise HashError(f"Hash calculation failed: {str(e)}")

    def verify_hash(self, expected_hash: str, algorithm: str = 'md5') -> bool:
        """
        验证文件哈希值
        Args:
            expected_hash: 期望的哈希值
            algorithm: 哈希算法,支持'md5', 'sha1', 'sha256', 'sha512'等
        Returns:
            是否匹配
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            actual_hash = self.calculate_hash(algorithm)
            return actual_hash.lower() == expected_hash.lower()
        except Exception as e:
            raise HashError(f"Hash verification failed: {str(e)}")

    def save_hash(self, algorithm: str = 'md5', output_path: Optional[str] = None) -> Self:
        """
        保存文件的哈希值到文件
        Args:
            algorithm: 哈希算法
            output_path: 输出文件路径,如果为None则使用原文件名
        Returns:
            哈希值文件对象
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            # 计算哈希值
            hash_value = self.calculate_hash(algorithm)

            # 生成输出路径
            if output_path is None:
                output_path = self.GetFullPath() + f'.{algorithm}'

            # 保存哈希值
            with open(output_path, 'w') as f:
                f.write(f"{hash_value} *{self.GetFilename()}")

            return ToolFile(output_path)
        except Exception as e:
            raise HashError(f"Hash saving failed: {str(e)}")

    def start_monitoring(
        self,
        callback:           Callable[[str, str], None],
        recursive:          bool                        = False,
        ignore_patterns:    Optional[List[str]]         = None,
        ignore_directories: bool                        = False,
        case_sensitive:     bool                        = True,
        is_log:             bool                        = True
    ) -> None:
        """
        开始监控文件或目录的变化
        Args:
            callback: 回调函数,接收事件类型和路径两个参数
            recursive: 是否递归监控子目录
            ignore_patterns: 忽略的文件模式列表
            ignore_directories: 是否忽略目录事件
            case_sensitive: 是否区分大小写
        """
        from watchdog.observers import Observer
        from watchdog.events   import FileSystemEventHandler
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            class EventHandler(FileSystemEventHandler):
                def __init__(self, callback, ignore_patterns, ignore_directories, case_sensitive):
                    self.callback = callback
                    self.ignore_patterns = ignore_patterns or []
                    self.ignore_directories = ignore_directories
                    self.case_sensitive = case_sensitive

                def should_ignore(self, path: str) -> bool:
                    if self.ignore_directories and os.path.isdir(path):
                        return True
                    if not self.case_sensitive:
                        path = path.lower()
                    return any(pattern in path for pattern in self.ignore_patterns)

                def on_created(self, event):
                    if not self.should_ignore(event.src_path):
                        self.callback('created', event.src_path)

                def on_modified(self, event):
                    if not self.should_ignore(event.src_path):
                        self.callback('modified', event.src_path)

                def on_deleted(self, event):
                    if not self.should_ignore(event.src_path):
                        self.callback('deleted', event.src_path)

                def on_moved(self, event):
                    if not self.should_ignore(event.src_path):
                        self.callback('moved', f"{event.src_path} -> {event.dest_path}")

            # 创建事件处理器
            event_handler = EventHandler(
                callback=callback,
                ignore_patterns=ignore_patterns,
                ignore_directories=ignore_directories,
                case_sensitive=case_sensitive
            )

            # 创建观察者
            observer = Observer()
            observer.schedule(event_handler, self.GetFullPath(), recursive=recursive)

            # 启动监控
            observer.start()
            if is_log:
                print(f"Started monitoring {self.GetFullPath()}")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
                if is_log:
                    print("Stopped monitoring")

            observer.join()

        except Exception as e:
            raise FileMonitorError(f"Failed to start monitoring: {str(e)}")

    def create_backup(
        self,
        backup_dir: Optional[str] = None,
        max_backups: int = 5,
        backup_format: str = 'zip',
        include_metadata: bool = True
    ) -> Self:
        """
        创建文件或目录的备份
        Args:
            backup_dir: 备份目录,如果为None则使用原目录下的.backup目录
            max_backups: 最大保留备份数量
            backup_format: 备份格式,支持'zip'和'tar'
            include_metadata: 是否包含元数据
        Returns:
            备份文件对象
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            # 生成备份目录
            if backup_dir is None:
                backup_dir = os.path.join(self.GetDir(), '.backup')
            backup_dir:Self = ToolFile(backup_dir)
            backup_dir.MustExistsPath()

            # 生成备份文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{self.GetFilename()}_{timestamp}"

            # 创建备份
            if backup_format == 'zip':
                backup_path = backup_dir | f"{backup_name}.zip"
                with zipfile.ZipFile(backup_path.GetFullPath(), 'w', zipfile.ZIP_DEFLATED) as zipf:
                    if self.IsDir():
                        for root, _, files in os.walk(self.GetFullPath()):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.GetFullPath())
                                zipf.write(file_path, arcname)
                    else:
                        zipf.write(self.GetFullPath(), self.GetFilename())
            elif backup_format == 'tar':
                backup_path = backup_dir | f"{backup_name}.tar"
                with tarfile.open(backup_path.GetFullPath(), 'w') as tarf:
                    if self.IsDir():
                        tarf.add(self.GetFullPath(), arcname=self.GetFilename())
                    else:
                        tarf.add(self.GetFullPath(), arcname=self.GetFilename())
            else:
                raise BackupError(f"Unsupported backup format: {backup_format}")

            # 添加元数据
            if include_metadata:
                metadata = {
                    'original_path': self.GetFullPath(),
                    'backup_time': timestamp,
                    'file_size': self.get_size(),
                    'is_directory': self.IsDir(),
                    'hash': self.calculate_hash()
                }
                metadata_path = backup_dir | f"{backup_name}.meta.json"
                with open(metadata_path.GetFullPath(), 'w') as f:
                    json.dump(metadata, f, indent=4)

            # 清理旧备份
            if max_backups > 0:
                backups = backup_dir.find_file(lambda f: ToolFile(f).GetFilename().startswith(self.GetFilename() + '_'))
                backups.sort(key=lambda f: f.GetFilename(), reverse=True)
                for old_backup in backups[max_backups:]:
                    old_backup.Remove()

            return backup_path

        except Exception as e:
            raise BackupError(f"Backup failed: {str(e)}")

    def restore_backup(
        self,
        backup_file: Union[str, Self],
        restore_path: Optional[str] = None,
        verify_hash: bool = True
    ) -> Self:
        """
        从备份恢复文件或目录
        Args:
            backup_file: 备份文件路径
            restore_path: 恢复路径,如果为None则恢复到原位置
            verify_hash: 是否验证哈希值
        Returns:
            恢复后的文件对象
        """
        if not isinstance(backup_file, ToolFile):
            backup_file:Self = ToolFile(backup_file)

        if not backup_file.Exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file.GetFullPath()}")

        try:
            # 确定恢复路径
            if restore_path is None:
                restore_path = self.GetFullPath()
            restore_path:Self = ToolFile(restore_path)

            # 解压备份
            if backup_file.get_extension() == 'zip':
                with zipfile.ZipFile(backup_file.GetFullPath(), 'r') as zipf:
                    zipf.extractall(restore_path.GetFullPath())
            elif backup_file.get_extension() == 'tar':
                with tarfile.open(backup_file.GetFullPath(), 'r') as tarf:
                    tarf.extractall(restore_path.GetFullPath())
            else:
                raise BackupError(f"Unsupported backup format: {backup_file.get_extension()}")

            # 验证哈希值
            if verify_hash:
                metadata_path = backup_file.GetFullPath()[:-len(backup_file.get_extension())-1] + '.meta.json'
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    restored_file = ToolFile(restore_path.GetFullPath())
                    if restored_file.calculate_hash() != metadata['hash']:
                        raise BackupError("Hash verification failed")

            return restore_path

        except Exception as e:
            raise BackupError(f"Restore failed: {str(e)}")

    def list_backups(self) -> List[Self]:
        """
        列出所有备份
        Returns:
            备份文件列表
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            backup_dir:Self = ToolFile(os.path.join(self.GetDir(), '.backup'))
            if not backup_dir.Exists():
                return []

            backups = backup_dir.find_file(lambda f: ToolFile(f).GetFilename().startswith(self.GetFilename() + '_'))
            backups.sort(key=lambda f: ToolFile(f).GetFilename(), reverse=True)
            return backups

        except Exception as e:
            raise BackupError(f"Failed to list backups: {str(e)}")

    def get_permissions(self) -> Dict[str, bool]:
        """
        获取文件或目录的权限
        Returns:
            权限字典,包含以下键:
            - read: 是否可读
            - write: 是否可写
            - execute: 是否可执行
            - hidden: 是否隐藏
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            mode = os.stat(self.GetFullPath()).st_mode
            return {
                'read': bool(mode & stat.S_IRUSR),
                'write': bool(mode & stat.S_IWUSR),
                'execute': bool(mode & stat.S_IXUSR),
                'hidden': bool(os.path.isfile(self.GetFullPath()) and self.GetFilename().startswith('.'))
            }
        except Exception as e:
            raise PermissionError(f"Failed to get permissions: {str(e)}")

    def set_permissions(
        self,
        read: Optional[bool] = None,
        write: Optional[bool] = None,
        execute: Optional[bool] = None,
        hidden: Optional[bool] = None,
        recursive: bool = False
    ) -> Self:
        """
        设置文件或目录的权限
        Args:
            read: 是否可读
            write: 是否可写
            execute: 是否可执行
            hidden: 是否隐藏
            recursive: 是否递归设置目录权限
        Returns:
            文件对象本身
        """
        if not self.Exists():
            raise FileNotFoundError(f"File not found: {self.GetFullPath()}")

        try:
            # 获取当前权限
            current_perms = os.stat(self.GetFullPath()).st_mode

            # 设置新权限
            if read is not None:
                if read:
                    current_perms |= stat.S_IRUSR
                else:
                    current_perms &= ~stat.S_IRUSR

            if write is not None:
                if write:
                    current_perms |= stat.S_IWUSR
                else:
                    current_perms &= ~stat.S_IWUSR

            if execute is not None:
                if execute:
                    current_perms |= stat.S_IXUSR
                else:
                    current_perms &= ~stat.S_IXUSR

            # 应用权限
            os.chmod(self.GetFullPath(), current_perms)

            # 设置隐藏属性
            if hidden is not None:
                if os.name == 'nt':  # Windows
                    import ctypes
                    if hidden:
                        ctypes.windll.kernel32.SetFileAttributesW(self.GetFullPath(), 2)
                    else:
                        ctypes.windll.kernel32.SetFileAttributesW(self.GetFullPath(), 0)
                else:  # Unix/Linux/Mac
                    if hidden:
                        if not self.GetFilename().startswith('.'):
                            self.Rename('.' + self.GetFilename())
                    else:
                        if self.GetFilename().startswith('.'):
                            self.Rename(self.GetFilename()[1:])

            # 递归设置目录权限
            if recursive and self.IsDir():
                for root, _, files in os.walk(self.GetFullPath()):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if read is not None:
                            if read:
                                os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IRUSR)
                            else:
                                os.chmod(file_path, os.stat(file_path).st_mode & ~stat.S_IRUSR)
                        if write is not None:
                            if write:
                                os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IWUSR)
                            else:
                                os.chmod(file_path, os.stat(file_path).st_mode & ~stat.S_IWUSR)
                        if execute is not None:
                            if execute:
                                os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IXUSR)
                            else:
                                os.chmod(file_path, os.stat(file_path).st_mode & ~stat.S_IXUSR)

            return self

        except Exception as e:
            raise PermissionError(f"Failed to set permissions: {str(e)}")

    def is_readable(self) -> bool:
        """
        检查文件是否可读
        Returns:
            是否可读
        """
        return self.get_permissions()['read']

    def is_writable(self) -> bool:
        """
        检查文件是否可写
        Returns:
            是否可写
        """
        return self.get_permissions()['write']

    def is_executable(self) -> bool:
        """
        检查文件是否可执行
        Returns:
            是否可执行
        """
        return self.get_permissions()['execute']

    def is_hidden(self) -> bool:
        """
        检查文件是否隐藏
        Returns:
            是否隐藏
        """
        return self.get_permissions()['hidden']
        