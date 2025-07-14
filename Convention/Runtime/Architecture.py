from .Config        import *
from .Reflection    import *
from abc            import ABC, abstractmethod

class ISignal(ABC):
    pass

class IModel(ABC):
    @abstractmethod
    def Save(self) -> str:
        pass
    @abstractmethod
    def Load(self, data:str) -> None:
        pass

class IConvertable[T](ABC):
    @abstractmethod
    def ConvertTo(self) -> T:
        pass

class IConvertModel[T](IConvertable[T], IModel):
    pass

class SingletonModel[T](IModel):
    _InjectInstances:Dict[type,Any] = {}

    @staticmethod
    def GetInstance(t:Typen[T]) -> T:
        return SingletonModel._InjectInstances[t]
    @staticmethod
    def SetInstance(t:Typen[T], obj:T) -> None:
        SingletonModel._InjectInstances[t] = obj

    def __init__(self, t:Typen[T]) -> None:
        self.typen: type = t

    @override
    def Save(self) -> str:
        return SingletonModel.GetInstance(self.typen).Save()

class DependenceModel(IConvertModel[bool]):
    def __init__(self, queries:Sequence[IConvertModel[bool]]) -> None:
        self.queries:list[IConvertModel[bool]] = list(queries)

    @override
    def ConvertTo(self) -> bool:
        for query in self.queries:
            if query.ConvertTo() == False:
                return False
        return True

    def __iter__(self):
        return iter(self.queries)

    def Load(self, data:str):
        raise NotImplementedError()

    def Save(self) -> str:
        raise NotImplementedError()

SignalListener = Callable[[ISignal], None]

class Architecture:
    @staticmethod
    def FormatType(t:type) -> str:
        return f"{t.__module__}::{t.__name__}"

    @staticmethod
    def LoadFromFormat(data:str, exception:Exception|None=None) -> type|None:
        try:
            module,name = data.split("::")
            return StringWithModel2Type(name, module=module)
        except Exception as ex:
            if exception is not None:
                exception = ex
            return None

    @classmethod
    def InternalReset(cls) -> None:
        # Register System
        cls._RegisterHistory.clear()
        cls._UncompleteTargets.clear()
        cls._Completer.clear()
        cls._Dependences.clear()
        cls._Childs.clear()
        # Event Listener
        cls._SignalListener.clear()
        # Linear Chain for Dependence
        cls._TimelineQueues.clear()
        cls._TimelineContentID = 0

    #region Objects Registered

    class TypeQuery(IConvertModel[bool]):
        def __init__(self, queryType:type) -> None:
            self._queryType = queryType

        @override
        def ConvertTo(self) -> bool:
            return self._queryType in Architecture._Childs

        def Load(self, data:str) -> None:
            raise NotImplementedError()

        def Save(self) -> str:
            raise NotImplementedError()
    
    _RegisterHistory:   Set[type]           = set()
    _UncompleteTargets: Dict[type,Any]   = {}
    _Completer:         Dict[type,Action]   = {}
    _Dependences:       Dict[type,DependenceModel] = {}
    _Childs:            Dict[type,Any]   = {}

    class Registering(IConvertModel[bool]):
        def __init__(self,registerSlot:type) -> None:
            self._registerSlot:type = registerSlot
        
        @override
        def ConvertTo(self) -> bool:
            return self._registerSlot in Architecture._Childs

        @override
        def Load(self,data:str) -> None:
            raise InvalidOperationError(f"Cannot use {self.__class__.__name__} to load type")

        @override
        def Save(self) -> str:
            return f"{Architecture.FormatType(self._registerSlot)}[{self.ConvertTo()}]"

    @classmethod
    def _InternalRegisteringComplete(cls) -> tuple[bool,Set[type]]:
        resultSet:  Set[type]   = set()
        stats:      bool        = False
        for dependence in cls._Dependences.keys():
            if cls._Dependences[dependence].ConvertTo():
                resultSet.add(dependence)
                stats = True
        return stats,resultSet

    @classmethod
    def _InternalRegisteringUpdate(cls, internalUpdateBuffer:Set[type]):
        for complete in internalUpdateBuffer:
            cls._Dependences.pop(complete, None)
        for complete in internalUpdateBuffer:
            cls._Completer[complete]()
            cls._Completer.pop(complete, None)
        for complete in internalUpdateBuffer:
            cls._Childs[complete] = cls._UncompleteTargets[complete]
            cls._UncompleteTargets.pop(complete, None)

    @classmethod
    def Register(cls, slot:type, target:Any, completer:Action, *dependences:type) -> 'Architecture.Registering':
        if slot in cls._RegisterHistory:
            raise InvalidOperationError("Illegal duplicate registrations")
        
        cls._RegisterHistory.add(slot)
        cls._Completer[slot] = completer
        cls._UncompleteTargets[slot] = target
        
        # 过滤掉自身依赖
        filtered_deps = [dep for dep in dependences if dep != slot]
        type_queries = [cls.TypeQuery(dep) for dep in filtered_deps]
        cls._Dependences[slot] = DependenceModel(type_queries)
        
        while True:
            has_complete, buffer = cls._InternalRegisteringComplete()
            if not has_complete:
                break
            cls._InternalRegisteringUpdate(buffer)
        
        return cls.Registering(slot)

    @classmethod
    def RegisterGeneric[T](cls, target:T, completer:Action, *dependences:type) -> 'Architecture.Registering':
        return cls.Register(type(target), target, completer, *dependences)

    @classmethod
    def Contains(cls, type_:type) -> bool:
        return type_ in cls._Childs

    @classmethod
    def ContainsGeneric[T](cls) -> bool:
        return cls.Contains(type(T))

    @classmethod
    def InternalGet(cls, type_:type) -> Any:
        return cls._Childs[type_]

    @classmethod
    def Get(cls, type_:type) -> Any:
        return cls.InternalGet(type_)

    @classmethod
    def GetGeneric[T](cls) -> T:
        return cls.Get(type(T))

    #endregion

    #region Signal & Update

    _SignalListener: Dict[type, Set[SignalListener]] = {}

    class Listening:
        def __init__(self, action:SignalListener, type_:type):
            self._action = action
            self._type = type_

        def StopListening(self):
            if self._type in Architecture._SignalListener:
                Architecture._SignalListener[self._type].discard(self._action)

    @classmethod
    def AddListenerGeneric[Signal](cls, slot:type, listener:SignalListener) -> 'Architecture.Listening':
        if slot not in cls._SignalListener:
            cls._SignalListener[slot] = set()
        
        def action(signal:ISignal):
            if isinstance(signal, slot):
                listener(signal)
        
        result = cls.Listening(action, slot)
        cls._SignalListener[slot].add(action)
        return result

    @classmethod
    def SendMessage(cls, slot:type, signal:ISignal):
        if slot in cls._SignalListener:
            for action in cls._SignalListener[slot]:
                action(signal)

    #endregion

    #region Timeline/Chain & Update

    class TimelineQueueEntry:
        def __init__(self):
            self.predicate: Callable[[], bool] = lambda: False
            self.actions: List[Action] = []

    class Timeline:
        def __init__(self):
            self.predicate_mapper: Dict[Callable[[], bool], int] = {}
            self.queue: List[Architecture.TimelineQueueEntry] = []
            self.context: int = 0

    _TimelineQueues: Dict[int, Timeline] = {}
    _TimelineContentID: int = 0

    @classmethod
    def CreateTimeline(cls) -> int:
        cls._TimelineQueues[cls._TimelineContentID] = cls.Timeline()
        cls._TimelineContentID += 1
        return cls._TimelineContentID - 1

    @classmethod
    def AddStep(cls, timeline_id:int, predicate:Callable[[], bool], *actions:Action):
        timeline = cls._TimelineQueues[timeline_id]
        if predicate in timeline.predicate_mapper:
            time = timeline.predicate_mapper[predicate]
            timeline.queue[time].actions.extend(actions)
        else:
            time = len(timeline.queue)
            timeline.predicate_mapper[predicate] = time
            entry = cls.TimelineQueueEntry()
            entry.predicate = predicate
            entry.actions = list(actions)
            timeline.queue.append(entry)

    @classmethod
    def UpdateTimeline(cls):
        stats = True
        while stats:
            stats = False
            for timeline in cls._TimelineQueues.values():
                if timeline.context < len(timeline.queue):
                    if timeline.queue[timeline.context].predicate():
                        stats = True
                        for action in timeline.queue[timeline.context].actions:
                            action()
                        timeline.context += 1

    @classmethod
    def ResetTimelineContext(cls, timeline_id:int):
        cls._TimelineQueues[timeline_id].context = 0

    #endregion




