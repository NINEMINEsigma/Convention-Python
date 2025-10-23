from .Config                    import *

def LimitStringLength(data, max_length:int=50) -> str:
    s:str = data if isinstance(data, str) else str(data)
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
    s:str = data if isinstance(data, str) else str(data)
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

def GetEditorDistanceAndOperations(
    s1:str, 
    s2:str, 
    ) -> Tuple[int, List[Tuple[Literal["add","delete"], int, int, str]]]:
    """
    计算两个字符串的编辑距离和操作序列
    操作格式: (操作类型, 开始位置, 结束位置, 内容)
    位置基于源字符串s1
    """
    m, n = len(s1), len(s2)
    
    # 使用简单的LCS算法来找到最长公共子序列
    # 然后基于LCS生成操作序列
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 构建LCS表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    
    # 基于LCS生成操作序列
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # 字符匹配，不需要操作
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or lcs[i][j - 1] >= lcs[i - 1][j]):
            # 需要插入s2[j-1]
            # 找到插入位置（在s1中的位置）
            insert_pos = i
            operations.insert(0, ("add", insert_pos, insert_pos, s2[j - 1]))
            j -= 1
        else:
            # 需要删除s1[i-1]
            operations.insert(0, ("delete", i - 1, i, s1[i - 1]))
            i -= 1
    
    # 合并连续的操作
    merged_operations = []
    for op in operations:
        if merged_operations and merged_operations[-1][0] == op[0]:
            last_op = merged_operations[-1]
            if op[0] == "add" and last_op[2] == op[1]:
                # 合并连续的添加操作
                merged_operations[-1] = (op[0], last_op[1], op[2], last_op[3] + op[3])
            elif op[0] == "delete" and last_op[2] == op[1]:
                # 合并连续的删除操作
                merged_operations[-1] = (op[0], last_op[1], op[2], last_op[3] + op[3])
            else:
                merged_operations.append(op)
        else:
            merged_operations.append(op)
    
    # 计算编辑距离
    edit_distance = m + n - 2 * lcs[m][n]
    return edit_distance, merged_operations

def GetDiffOperations(
    s1:str, 
    s2:str, 
    ) -> List[Tuple[Literal["add","delete"], int, int, str]]:
    """
    计算两个字符串的差异操作序列
    操作格式: (操作类型, 开始位置, 结束位置, 内容)
    位置基于源字符串s1
    """
    
    return operations