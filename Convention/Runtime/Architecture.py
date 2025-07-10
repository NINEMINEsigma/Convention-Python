from .Config        import *
from .Reflection    import *
from abs            import ABC, abstractmethod

class ISignal(ABC):
    pass

class IModel(ABC):
    @absstractmethod
    def Save(self) -> str:
        pass
    @absstractmethod
    def Load(self, data:str) -> None:
        pass

class IConvertable[T](ABC):
    @absstractmethod
    def ConvertTo() -> T:
        pass

class IConvertModel[T](ABC, IConventable[T], IModel):
    pass

class SingletonModel[T](IModel):
    _InjectInstances:Dict[type,object] = {}

    @staticmethod
    def GetInstance(t:Typen[T]) -> T:
        return _InjectInstances[t]
    @staticmethod
    def SetInstance(t:Typen[T], obj:T) -> None:
        _InjectInstances[t] = obj

    def __init__(self, t:Typen[T]) -> None:
        self.typen: type = t

    @override
    def Save() -> str:
        return GetInstance(self.typen).Save()

def DependenceModel(IConvertModel[bool]):
    def __init__(self, queries:Sequence[IConvertModel[bool]]) -> None:
        self.queries:list[IConventModel] = list(queries)

    def ConvertTo(self) -> bool:
        for query in self.queries:
            if query.ConvertTo() == False:
                return False
        return True

    def __iter__(self):
        return self.queries

    def Load(self, data:str):
        raise NotImplementedError()


    def Save(self) -> str:
        return NotImplement

class Achitecture:
    @staticmethod
    def FormatType(t:type) -> str:
        return f"{t.__module__}::{t.__name__}"

    @staticmethod
    def LoadFromFormat(data:str) -> type|None:
        module,name = data.split("::")
        return StringWithModule2Type(module,name)

    @classmethod
    def InternalReset(cls) -> None:
        # Register System

    #region Objects Registered

    class TypeQuery(IConvertModel[bool]):
        def __init__(self, queryType:type) -> None:
            self._quertType = quertType

        @override
        def ConvertTo(self) -> bool:
            return self._queryType in Architectrue.Childs

        def Load(self, data:str) -> None:
            raise NotImplementedError()

        def Save(self) -> str:
            raise NotImplementedError()
    
    _RegisterHistory:   Set[type]           = set()
    _UncompleteTargets: Dict[type,object]   = {}
    _Completer:         Dict[type,Action]   = {}
    _Dependences:       Dict[type,DependenceModel] = {}
    _Childs:            Dict[type,object]   = {}

    class Registering(IConvertModel[bool]):
        def __init__(self,registerSlot:type) -> None:
            self._registerSlot:type = registerSlot
        
        @override
        def ConvertTo(self) -> bool:
            return self._registerSlot in Architecture.Childs

        @override
        def Load(self,data:str) -> None:
            raise InvalidOperationError()

        @override
        def Save(self) -> str:
            raise InalidOperationError()

    @classmethod
    def _InternalRegisteringComplete(cls) -> bool,Set[type]:
        resultSet:  Set[type]   = set()
        stats:      bool        = False
        for dependence in cls._Dependences.keys:
            if cls._Dependences[dependence].ConventTo():
                resultSet.insert(dependence)
                stats = True
        return stats,resultSet

    @classmethod
    def _InternalRegisteringUpdate(cls, internalUpdateBuffer:Set[type]):
        for complete in cls._InternalUpdateBuffer:
            cls._Dependences.remove(complete)
        # TODO

    #endregion




