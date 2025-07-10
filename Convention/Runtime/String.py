from pathlib                    import Path
from .Config                    import *
import                                 re
from pathlib                    import Path
import xml.etree.ElementTree    as     ET
from xml.dom                    import minidom
import                                 math

def LimitStringLength(data, max_length:int=50) -> str:
    s:str = data if data is str else str(data)
    if len(s) <= max_length:
        return s
    else:
        inside_str = "\n...\n...\n"
        # 计算头尾部分的长度
        head_length = max_length // 2
        tail_length = max_length - head_length - len(inside_str)  # 3 是省略号的长度

        # 截取头尾部分并连接
        return s[:head_length] + inside_str + s[-tail_length:]

def FillString(data:Any, 
             max_length:    int = 50, 
             fill_char:     str = " ",
             side:          Literal["left", "right", "center"] = "right"
             ) -> str:
    s:str = data if data is str else str(data)
    char = fill_char[0]
    if len(s) >= max_length:
        return s
    else:
        if side == "left":
            return s + char * (max_length - len(s))
        elif side == "right":
            return char * (max_length - len(s)) + s
        elif side == "center":
            left = (max_length - len(s)) // 2
            right = max_length - len(s) - left
            return char * left + s + char * right
        else:
            raise ValueError(f"Unsupported side: {side}")

def Bytes2Strings(lines:List[bytes], encoding='utf-8') -> List[str]:
    return [line.decode(encoding) for line in lines]

def Bytes2String(lines:List[bytes], encoding='utf-8') -> str:
    return "".join(Bytes2Strings(lines, encoding))

def word_segmentation(
    sentence,
    cut_all:    bool                    = False,
    HMM:        bool                    = True,
    use_paddle: bool                    = False
    ) -> Sequence[Optional[Union[Any, str]]]:
    try:
        import jieba
        return jieba.dt.cut(str(sentence), cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)
    except ImportError:
        raise ValueError("jieba is not install")
