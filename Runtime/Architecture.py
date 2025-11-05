from .Config        import *
from .Reflection    import *
from abc            import ABC, abstractmethod

class ISignal(ABC):
    pass


class IModel(ABC):
    pass


class IDataModel(ABC, IModel):
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
        cls._RegisteredObjects.clear()
        cls._RegisteringRuntime.clear()
        # Signal Listener
        cls._SignalListener.clear()
        # Timeline/Chain
        cls._TimelineQueues.clear()
        cls._TimelineContentID = 0

    #region Objects Registered

    class TypeQuery(IConvertModel[bool]):
        def __init__(self, queryType:type) -> None:
            self._queryType = queryType

        @override
        def ConvertTo(self) -> bool:
            return self._queryType in Architecture._RegisteredObjects
    
    class Registering(IConvertModel[bool]):
        def __init__(self, registerSlot:type, target:Any, dependences:DependenceModel, action:Action) -> None:
            self.registerSlot = registerSlot
            self.target = target
            self.dependences = dependences
            self.action = action

        @override
        def ConvertTo(self) -> bool:
            return self.dependences.ConvertTo()

    _RegisteringRuntime:    Dict[type, Registering] = {}
    _RegisteredObjects:     Dict[type, Any] = {}

    @classmethod
    def _InternalRegisteringComplete(cls) -> None:
        CompletedSet: Set[Architecture.Registering] = set()
        for dependence in cls._RegisteringRuntime.keys():
            if cls._RegisteringRuntime[dependence].dependences.ConvertTo():
                CompletedSet.add(cls._RegisteringRuntime[dependence])
        for complete in CompletedSet:
            del cls._RegisteringRuntime[complete.registerSlot]
            complete.action()
            cls._RegisteredObjects[complete.registerSlot] = complete.target
        if len(CompletedSet) > 0:
            cls._InternalRegisteringComplete()

    @classmethod
    def Register(cls, slot:type, target:Any, action:Action, *dependences:type) -> DependenceModel:
        if slot in cls._RegisteringRuntime:
            raise InvalidOperationError("Illegal duplicate registrations")
        cls._RegisteringRuntime[slot] = Architecture.Registering(slot, target, DependenceModel(Architecture.TypeQuery(dependence) for dependence in dependences), action)
        cls._InternalRegisteringComplete()
        return cls._RegisteringRuntime[slot].dependences

    @classmethod
    def Contains(cls, type_:type) -> bool:
        return type_ in cls._RegisteredObjects

    @classmethod
    def Get(cls, type_:type) -> Any:
        return cls._RegisteredObjects[type_]

    @classmethod
    def Unregister(cls, slot:type) -> bool:
        if slot in cls._RegisteredObjects:
            del cls._RegisteredObjects[slot]
            return True
        if slot in cls._RegisteringRuntime:
            del cls._RegisteringRuntime[slot]
            return True
        return False

    #endregion

    #region Signal & Update

    _SignalListener: Dict[type, List[SignalListener]] = {}

    @classmethod
    def AddListener(cls, slot:type, listener:SignalListener) -> None:
        if slot not in cls._SignalListener:
            cls._SignalListener[slot] = []
        
        cls._SignalListener[slot].append(listener)

    @classmethod
    def SendMessage(cls, slot:type, signal:ISignal):
        if slot in cls._SignalListener:
            for listener in cls._SignalListener[slot]:
                listener(signal)

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

