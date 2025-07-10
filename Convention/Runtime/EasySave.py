from .Reflection    import *
from .File          import ToolFile
from .String        import LimitStringLength

_Internal_EasySave_Debug:bool = False
def GetInternalEasySaveDebug() -> bool:
    return _Internal_EasySave_Debug and GetInternalDebug()
def SetInternalEasySaveDebug(debug:bool) -> None:
    global _Internal_EasySave_Debug
    _Internal_EasySave_Debug = debug

class EasySaveSetting(BaseModel):
    key:            str         = Field(description="目标键", default="easy")
    # 从目标文件进行序列化/反序列化
    file:           str         = Field(description="目标文件")
    # 序列化/反序列化的格式方法
    formatMode:         Literal["json", "binary"] = Field(description="保存模式", default="json")
    # TODO: refChain:       bool        = Field(description="是否以保留引用的方式保存", default=True)
    # 文件形式与参数
    # TODO: encoding:       str         = Field(description="编码", default="utf-8")
    isBackup:               bool        = Field(description="是否备份", default=True)
    backupSuffix:           str         = Field(description="备份后缀", default=".backup")
    # 序列化/反序列化时, 如果设置了忽略字段的谓词, 则被谓词选中的字段将不会工作
    # 如果设置了选择字段的谓词, 则被选中的字段才会工作
    ignorePr:      Optional[Callable[[FieldInfo], bool]] = Field(description="忽略字段的谓词", default=None)
    selectPr:      Optional[Callable[[FieldInfo], bool]] = Field(description="选择字段的谓词", default=None)

class ESWriter(BaseModel):
    setting:            EasySaveSetting = Field(description="设置")

    def _GetFields(self, rtype:RefType) -> List[FieldInfo]:
        '''
        获取字段
        '''
        fields: List[FieldInfo] = []
        if self.setting.ignorePr is not None and self.setting.selectPr is not None:
            fields = [ field for field in rtype.GetAllFields() if self.setting.selectPr(field) and not self.setting.ignorePr(field) ]
        elif self.setting.selectPr is None and self.setting.ignorePr is None:
            fields = rtype.GetFields()
        elif self.setting.ignorePr is not None:
            fields = [ field for field in rtype.GetAllFields() if not self.setting.ignorePr(field) ]
        else:
            fields = [ field for field in rtype.GetAllFields() if self.setting.selectPr(field) ]
        return fields

    def _DoJsonSerialize(self, result_file:ToolFile, rtype:RefType, rinstance:Any) -> Any:
        '''
        序列化: json格式
        '''

        def dfs(rtype:RefType, rinstance:Any) -> Dict[str, Any]|Any:
            if rinstance is None:
                return rinstance

            if rtype.IsUnion:
                rtype = TypeManager.GetInstance().CreateOrGetRefType(rinstance)

            if rtype.IsValueType:
                return rinstance
            elif rtype.IsCollection:
                try:
                    if rtype.IsList:
                        return [ dfs(TypeManager.GetInstance().CreateOrGetRefType(iter_), iter_) for iter_ in rinstance ]
                    elif rtype.IsSet:
                        return { dfs(TypeManager.GetInstance().CreateOrGetRefType(iter_), iter_) for iter_ in rinstance }
                    elif rtype.IsTuple:
                        return tuple(dfs(TypeManager.GetInstance().CreateOrGetRefType(iter_), iter_) for iter_ in rinstance)
                    elif rtype.IsDictionary:
                        return {
                                dfs(TypeManager.GetInstance().CreateOrGetRefType(key), key):
                                    dfs(TypeManager.GetInstance().CreateOrGetRefType(iter_), iter_)
                                    for key, iter_ in rinstance.items()
                                    }
                except Exception as e:
                    raise ReflectionException(f"{ConsoleFrontColor.RED}容器<{rtype.RealType}>"\
                        f"在序列化时遇到错误:{ConsoleFrontColor.RESET}\n{e}") from e
                raise NotImplementedError(f"{ConsoleFrontColor.RED}不支持的容器: {rinstance}"\
                        f"<{rtype.Print2Str(verbose=GetInternalEasySaveDebug())}>{ConsoleFrontColor.RESET}")
            elif hasattr(rtype.RealType, "__easy_serialize__"):
                custom_data, is_need_type = rtype.RealType.__easy_serialize__(rinstance)
                if is_need_type:
                    return {
                        "__type": AssemblyTypen(rtype.RealType),
                        **custom_data
                    }
                else:
                    return custom_data
            else:
                fields: List[FieldInfo] = self._GetFields(rtype)
                layer:  Dict[str, Any]  = {
                    "__type": AssemblyTypen(rtype.RealType)
                }
                for field in fields:
                    try:
                        layer[field.FieldName] = dfs(
                            TypeManager.GetInstance().CreateOrGetRefType(field.FieldType),
                            field.GetValue(rinstance)
                            )
                    except Exception as e:
                        raise ReflectionException(f"{ConsoleFrontColor.RED}字段{field.FieldName}"\
                            f"<{field.FieldType}>在序列化时遇到错误:{ConsoleFrontColor.RESET}\n{e}") from e
                return layer

        layers: Dict[str, Any] = {}
        if result_file.Exists():
            filedata = result_file.LoadAsJson()
            if isinstance(filedata, dict):
                layers = filedata
        layers[self.setting.key] = {
            "__type": AssemblyTypen(rtype.RealType),
            "value": dfs(rtype, rinstance)
        }
        result_file.SaveAsJson(layers)

    def _DoBinarySerialize(self, result_file:ToolFile, rinstance:Any) -> Any:
        '''
        序列化: 二进制格式
        '''
        result_file.SaveAsBinary(rinstance)

    def Serialize(self, result_file:ToolFile, rtype:RefType, rinstance:Any) -> Any:
        '''
        序列化
        '''
        if self.setting.formatMode == "json":
            self._DoJsonSerialize(result_file, rtype, rinstance)
        elif self.setting.formatMode == "binary":
            self._DoBinarySerialize(result_file, rinstance)
        else:
            raise NotImplementedError(f"不支持的格式: {self.setting.formatMode}")

    def Write[T](self, rinstance:T) -> ToolFile:
        '''
        写入数据
        '''
        result_file:    ToolFile   = ToolFile(self.setting.file)
        backup_file:    ToolFile   = None
        if result_file.GetDir() is not None and not ToolFile(result_file.GetDir()).Exists():
            raise FileNotFoundError(f"文件路径不存在: {result_file.GetDir()}")
        if result_file.Exists() and self.setting.isBackup:
            if result_file.GetDir() is not None:
                backup_file = ToolFile(result_file.GetDir()) | (result_file.GetFilename(True) + self.setting.backupSuffix)
            else:
                backup_file = ToolFile(result_file.GetFilename(True) + self.setting.backupSuffix)
            result_file.Copy(backup_file)
        try:
            self.Serialize(result_file, TypeManager.GetInstance().CreateOrGetRefType(rinstance), rinstance)
        except Exception:
            if backup_file is not None:
                result_file.Remove()
                backup_file.Copy(result_file)
                backup_file.Remove()
            raise
        finally:
            if backup_file is not None:
                backup_file.Remove()
        return result_file

class ESReader(BaseModel):
    setting:            EasySaveSetting = Field(description="设置")

    def _GetFields(self, rtype:RefType) -> List[FieldInfo]:
        '''
        获取字段
        '''
        fields: List[FieldInfo] = []
        if self.setting.ignorePr is not None and self.setting.selectPr is not None:
            fields = [ field for field in rtype.GetAllFields() if self.setting.selectPr(field) and not self.setting.ignorePr(field) ]
        elif self.setting.selectPr is None and self.setting.ignorePr is None:
            fields = rtype.GetFields()
        elif self.setting.ignorePr is not None:
            fields = [ field for field in rtype.GetAllFields() if not self.setting.ignorePr(field) ]
        else:
            fields = [ field for field in rtype.GetAllFields() if self.setting.selectPr(field) ]
        return fields

    def GetRtypeFromTypen(self, type_label:str) -> RefType:
        '''
        从类型标签中获取类型
        '''
        #module_name, _, class_name = type_label.split(",")[0].strip().rpartition('.')
        #if GetInternalEasySaveDebug():
        #    print_colorful(ConsoleFrontColor.YELLOW, f"Prase __type label: {ConsoleFrontColor.RESET}{type_label}"\
        #        f"{ConsoleFrontColor.YELLOW}, module_name: {ConsoleFrontColor.RESET}{module_name}"\
        #        f"{ConsoleFrontColor.YELLOW}, class_name: {ConsoleFrontColor.RESET}{class_name}")
        #typen_to = try_to_type(class_name, module_name=module_name) or to_type(class_name)
        #return TypeManager.GetInstance().CreateOrGetRefType(typen_to)
        typen, assembly_name = ReadAssemblyTypen(type_label)
        if GetInternalEasySaveDebug():
            print_colorful(ConsoleFrontColor.YELLOW, f"Prase __type label: {ConsoleFrontColor.RESET}{type_label}"\
                f"{ConsoleFrontColor.YELLOW}, typen: {ConsoleFrontColor.RESET}{typen}"\
                f"{ConsoleFrontColor.YELLOW}, assembly_name: {ConsoleFrontColor.RESET}{assembly_name}")
        return TypeManager.GetInstance().CreateOrGetRefType(typen)

    def _DoJsonDeserialize(self, read_file:ToolFile, rtype:Optional[RefType] = None) -> Any:
        '''
        反序列化: json格式

        Args:
            read_file (ToolFile): 要读取的文件对象
            rtype (Optional[RTypen[Any]], optional): 目标类型. 如果为None, 则从文件中读取类型信息. Defaults to None.

        Returns:
            Any: 反序列化后的对象

        Raises:
            NotImplementedError: 当遇到不支持的集合类型时抛出
            ValueError: 当rinstance不为None时抛出
        '''
        # 从文件中加载JSON数据
        layers:             Dict[str, Any]  = read_file.LoadAsJson()
        if self.setting.key not in layers:
            raise ValueError(f"{ConsoleFrontColor.RED}文件中不包含目标键: {ConsoleFrontColor.RESET}{self.setting.key}")
        # 如果未指定类型, 则从JSON数据中获取类型信息
        if rtype is None:
            rtype:          RefType         = self.GetRtypeFromTypen(layers["__type"])
        layers:             Dict[str, Any]  = layers[self.setting.key]["value"]
        result_instance:    Any             = None

        def dfs(rtype:Optional[RefType], layer:Dict[str, Any]|Any) -> Any:
            '''
            深度优先遍历反序列化

            Args:
                rtype (Optional[RefType]): 当前处理的类型
                layer (Dict[str, Any]|Any): 当前处理的JSON数据层
                rinstance (Any): 当前处理的对象实例

            Returns:
                Any: 反序列化后的对象
            '''
            # 如果类型为None且当前层包含类型信息, 则获取类型
            if isinstance(layer, dict) and "__type" in layer:
                rtype = self.GetRtypeFromTypen(layer["__type"])
            if rtype is None:
                raise ValueError(f"{ConsoleFrontColor.RED}当前层不包含类型信息: {ConsoleFrontColor.RESET}{LimitStringLength(str(layer), 100)}")
            if GetInternalEasySaveDebug():
                print_colorful(ConsoleFrontColor.YELLOW, f"layer: {ConsoleFrontColor.RESET}{LimitStringLength(str(layer), 100)}"\
                    f"{ConsoleFrontColor.YELLOW}, rtype: {ConsoleFrontColor.RESET}{rtype.ToString()}")

            # 处理值类型
            if (rtype.IsValueType or
                rtype.Verify(Any) or
                (layer is None and rtype.Verify(type(None)))
                ):
                return layer
            # 处理集合类型
            elif rtype.IsCollection:
                try:
                    if rtype.IsList:
                        element_type = rtype.GenericArgs[0] if len(rtype.GenericArgs) > 0 else Any
                        return [ dfs(TypeManager.GetInstance().CreateOrGetRefType(element_type), iter_) for iter_ in layer ]
                    elif rtype.IsSet:
                        element_type = rtype.GenericArgs[0] if len(rtype.GenericArgs) > 0 else Any
                        return { dfs(TypeManager.GetInstance().CreateOrGetRefType(element_type), iter_) for iter_ in layer }
                    elif rtype.IsTuple:
                        element_types: List[type] = rtype.GenericArgs
                        result: tuple = tuple(None for _ in layer)
                        if element_types is None or len(element_types) == 0:
                            element_types = [Any] * len(layer)
                        for index, iter_ in enumerate(layer):
                            result[index] = dfs(TypeManager.GetInstance().CreateOrGetRefType(element_types[index]), iter_)
                        return result
                    elif rtype.IsDictionary:
                        element_key, element_value = (rtype.GenericArgs[0], rtype.GenericArgs[1]) if len(rtype.GenericArgs) > 1 else (Any, Any)
                        return {
                            dfs(TypeManager.GetInstance().CreateOrGetRefType(element_key), keyname):
                                dfs(TypeManager.GetInstance().CreateOrGetRefType(element_value), iter_)
                            for keyname, iter_ in layer.items()
                            }
                except Exception as e:
                    raise ReflectionException(f"容器<{LimitStringLength(str(layer), 100)}>在反序列化时遇到错误:\n{e}") from e
                raise NotImplementedError(f"{ConsoleFrontColor.RED}不支持的容器: {LimitStringLength(str(layer), 100)}"\
                        f"<{rtype.Print2Str(verbose=GetInternalEasySaveDebug())}>{ConsoleFrontColor.RESET}")
            # 处理对象类型
            elif isinstance(rtype.RealType, type) and hasattr(rtype.RealType, "__easy_deserialize__"):
                return rtype.RealType.__easy_deserialize__(layer)
            else:
                rinstance = rtype.CreateInstance()
                if GetInternalEasySaveDebug():
                    print_colorful(ConsoleFrontColor.YELLOW, f"rinstance rtype target: {ConsoleFrontColor.RESET}"\
                        f"{rtype.Print2Str(verbose=True, flags=RefTypeFlag.Field|RefTypeFlag.Instance|RefTypeFlag.Public)}")
                fields:List[FieldInfo] = self._GetFields(rtype)
                for field in fields:
                    if field.FieldName not in layer:
                        continue
                    field_rtype:RefType = None
                    try:
                        if field.FieldType == list and field.ValueType.IsGeneric:
                            field_rtype = TypeManager.GetInstance().CreateOrGetRefType(ListIndictaor(field.ValueType.GenericArgs[0]))
                            if GetInternalEasySaveDebug():
                                print_colorful(ConsoleFrontColor.YELLOW, f"field: {ConsoleFrontColor.RESET}{field.FieldName}"\
                                    f"{ConsoleFrontColor.YELLOW}, field_rtype: {ConsoleFrontColor.RESET}List<"\
                                    f"{field_rtype.GenericArgs[0]}>")
                        elif field.FieldType == set and field.ValueType.IsGeneric:
                            field_rtype = TypeManager.GetInstance().CreateOrGetRefType(SetIndictaor(field.ValueType.GenericArgs[0]))
                            if GetInternalEasySaveDebug():
                                print_colorful(ConsoleFrontColor.YELLOW, f"field: {ConsoleFrontColor.RESET}{field.FieldName}"\
                                    f"{ConsoleFrontColor.YELLOW}, field_rtype: {ConsoleFrontColor.RESET}Set<"\
                                    f"{field_rtype.GenericArgs[0]}>")
                        elif field.FieldType == tuple and field.ValueType.IsGeneric:
                            field_rtype = TypeManager.GetInstance().CreateOrGetRefType(TupleIndictaor(field.ValueType.GenericArgs[0]))
                            if GetInternalEasySaveDebug():
                                print_colorful(ConsoleFrontColor.YELLOW, f"field: {ConsoleFrontColor.RESET}{field.FieldName}"\
                                    f"{ConsoleFrontColor.YELLOW}, field_rtype: {ConsoleFrontColor.RESET}Tuple<"\
                                    f"{field_rtype.GenericArgs[0]}>")
                        elif field.FieldType == dict and field.ValueType.IsGeneric:
                            field_rtype = TypeManager.GetInstance().CreateOrGetRefType(
                                DictIndictaor(field.ValueType.GenericArgs[0], field.ValueType.GenericArgs[1])
                                )
                            if GetInternalEasySaveDebug():
                                print_colorful(ConsoleFrontColor.YELLOW, f"field: {ConsoleFrontColor.RESET}{field.FieldName}"\
                                    f"{ConsoleFrontColor.YELLOW}, field_rtype: {ConsoleFrontColor.RESET}Dict<"\
                                    f"{field_rtype.GenericArgs[0]}, {field_rtype.GenericArgs[1]}>")
                        else:
                            field_rtype = TypeManager.GetInstance().CreateOrGetRefType(field.FieldType)
                            if GetInternalEasySaveDebug():
                                print_colorful(ConsoleFrontColor.YELLOW, f"field: {ConsoleFrontColor.RESET}{field.FieldName}"\
                                    f"{ConsoleFrontColor.YELLOW}, field_rtype: {ConsoleFrontColor.RESET}{field_rtype.RealType}"\
                                    f"<{field_rtype.GenericArgs}>")
                        field.SetValue(rinstance, dfs(field_rtype, layer[field.FieldName]))
                    except Exception as e:
                        raise ReflectionException(f"Json字段{field.FieldName}={LimitStringLength(str(layer[field.FieldName]), 100)}: \n{e}") from e
                return rinstance

        # 从根节点开始反序列化
        result_instance = dfs(rtype, layers)
        return result_instance

    def _DoBinaryDeserialize(self, read_file:ToolFile, rtype:RefType) -> Any:
        '''
        反序列化: 二进制格式
        '''
        return read_file.LoadAsBinary()

    def Deserialize(self, read_file:ToolFile, rtype:Optional[RefType]=None) -> Any:
        '''
        反序列化
        '''
        if self.setting.formatMode == "json":
            return self._DoJsonDeserialize(read_file, rtype)
        elif self.setting.formatMode == "binary":
            return self._DoBinaryDeserialize(read_file, rtype)
        else:
            raise NotImplementedError(f"不支持的格式: {self.setting.formatMode}")

    def Read[T](self, rtype:Optional[RTypen[T]]=None) -> T:
        '''
        读取数据
        '''
        read_file: ToolFile = ToolFile(self.setting.file)
        if not read_file.Exists():
            raise FileNotFoundError(f"文件不存在: {read_file}")
        if read_file.IsDir():
            raise IsADirectoryError(f"文件是目录: {read_file}")
        return self.Deserialize(read_file, rtype)

class EasySave:
    @staticmethod
    def Write[T](rinstance:T, file:Optional[ToolFile|str]=None, *, setting:Optional[EasySaveSetting]=None) -> ToolFile:
        '''
        写入数据
        '''
        return ESWriter(setting=(setting if setting is not None else EasySaveSetting(file=str(file)))).Write(rinstance)

    @overload
    @staticmethod
    def Read[T](
        rtype:      Typen[T],
        file:       Optional[ToolFile|str]     = None,
        *,
        setting:    Optional[EasySaveSetting]   = None
        ) -> T:
        ...
    @overload
    @staticmethod
    def Read[T](
        rtype:      RTypen[T],
        file:       Optional[ToolFile|str]     = None,
        *,
        setting:    Optional[EasySaveSetting]   = None
        ) -> T:
        ...
    @staticmethod
    def Read[T](
        rtype:      RTypen[T]|type,
        file:       Optional[ToolFile|str]     = None,
        *,
        setting:    Optional[EasySaveSetting]   = None
        ) -> T:
        '''
        读取数据
        '''
        if isinstance(rtype, type):
            rtype = TypeManager.GetInstance().CreateOrGetRefType(rtype)
        return ESReader(setting=(setting if setting is not None else EasySaveSetting(file=str(file)))).Read(rtype)
