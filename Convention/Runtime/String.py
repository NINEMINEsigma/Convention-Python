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

def _build_line_lcs(lines1: List[str], lines2: List[str]) -> List[List[int]]:
    """
    构建行级LCS动态规划表
    """
    m, n = len(lines1), len(lines2)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 使用哈希加速行比较
    hash1 = [hash(line) for line in lines1]
    hash2 = [hash(line) for line in lines2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hash1[i-1] == hash2[j-1] and lines1[i-1] == lines2[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
    
    return lcs

def _extract_line_operations(lines1: List[str], lines2: List[str], lcs: List[List[int]]) -> List[Tuple[str, int, int, List[str]]]:
    """
    从LCS表提取行级操作序列
    返回: (操作类型, 起始行号, 结束行号, 行内容列表)
    """
    operations = []
    m, n = len(lines1), len(lines2)
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and lines1[i-1] == lines2[j-1]:
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or lcs[i][j-1] >= lcs[i-1][j]):
            operations.insert(0, ("add", i, i, [lines2[j-1]]))
            j -= 1
        else:
            operations.insert(0, ("delete", i-1, i, [lines1[i-1]]))
            i -= 1
    
    # 合并连续的同类行操作
    merged = []
    for op_type, start, end, lines in operations:
        if merged and merged[-1][0] == op_type and merged[-1][2] == start:
            merged[-1] = (op_type, merged[-1][1], end, merged[-1][3] + lines)
        else:
            merged.append((op_type, start, end, lines))
    
    return merged

def _char_diff_in_region(s1: str, s2: str) -> List[Tuple[str, int, int, str]]:
    """
    对小范围区域进行字符级LCS比较
    返回相对于输入字符串的位置
    """
    m, n = len(s1), len(s2)
    
    # 快速路径
    if m == 0 and n == 0:
        return []
    if m == 0:
        return [("add", 0, 0, s2)]
    if n == 0:
        return [("delete", 0, m, s1)]
    if s1 == s2:
        return []
    
    # 字符级LCS
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
    
    # 回溯生成操作
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or lcs[i][j-1] >= lcs[i-1][j]):
            operations.insert(0, ("add", i, i, s2[j-1]))
            j -= 1
        else:
            operations.insert(0, ("delete", i-1, i, s1[i-1]))
            i -= 1
    
    # 合并连续操作
    merged = []
    for op_type, start, end, content in operations:
        if merged and merged[-1][0] == op_type:
            last_op = merged[-1]
            if op_type == "add" and last_op[2] == start:
                merged[-1] = (op_type, last_op[1], end, last_op[3] + content)
            elif op_type == "delete" and last_op[2] == start:
                merged[-1] = (op_type, last_op[1], end, last_op[3] + content)
            else:
                merged.append((op_type, start, end, content))
        else:
            merged.append((op_type, start, end, content))
    
    return merged

def GetDiffOperations(
    s1:str, 
    s2:str, 
    ) -> List[Tuple[Literal["add","delete"], int, int, str]]:
    """
    计算两个字符串的差异操作序列（混合行级+字符级算法）
    操作格式: (操作类型, 开始位置, 结束位置, 内容)
    位置基于源字符串s1的字符偏移
    """
    # 快速路径
    if s1 == s2:
        return []
    if not s1:
        return [("add", 0, 0, s2)]
    if not s2:
        return [("delete", 0, len(s1), s1)]
    
    # 阶段1: 分行并建立位置映射
    lines1 = s1.split('\n')
    lines2 = s2.split('\n')
    
    # 构建行号到字符位置的映射
    line_offsets_s1 = [0]
    for line in lines1[:-1]:
        line_offsets_s1.append(line_offsets_s1[-1] + len(line) + 1)  # +1 for '\n'
    
    line_offsets_s2 = [0]
    for line in lines2[:-1]:
        line_offsets_s2.append(line_offsets_s2[-1] + len(line) + 1)
    
    # 阶段2: 行级LCS分析
    lcs = _build_line_lcs(lines1, lines2)
    line_operations = _extract_line_operations(lines1, lines2, lcs)
    
    # 阶段3: 转换为字符级操作
    final_operations = []
    
    for op_type, start_line, end_line, op_lines in line_operations:
        if op_type == "add":
            # 添加操作: 在s1的start_line位置插入
            char_pos = line_offsets_s1[start_line] if start_line < len(line_offsets_s1) else len(s1)
            content = '\n'.join(op_lines)
            
            # 对于添加的行块，可以选择字符级细化或直接使用
            # 这里先直接使用行级结果
            final_operations.append(("add", char_pos, char_pos, content))
            
        elif op_type == "delete":
            # 删除操作: 删除s1的[start_line, end_line)行
            char_start = line_offsets_s1[start_line]
            if end_line < len(lines1):
                char_end = line_offsets_s1[end_line]
            else:
                char_end = len(s1)
            
            content = '\n'.join(op_lines)
            final_operations.append(("delete", char_start, char_end, content))
    
    # 阶段4: 对于连续的删除+添加，尝试字符级精细比较
    optimized_operations = []
    i = 0
    while i < len(final_operations):
        if (i + 1 < len(final_operations) and 
            final_operations[i][0] == "delete" and 
            final_operations[i+1][0] == "add" and
            final_operations[i][2] == final_operations[i+1][1]):
            
            # 这是一个修改操作，进行字符级细化
            del_op = final_operations[i]
            add_op = final_operations[i+1]
            
            old_text = del_op[3]
            new_text = add_op[3]
            base_pos = del_op[1]
            
            # 字符级比较
            char_ops = _char_diff_in_region(old_text, new_text)
            
            # 调整位置到全局坐标
            for op_type, rel_start, rel_end, content in char_ops:
                optimized_operations.append((op_type, base_pos + rel_start, base_pos + rel_end, content))
            
            i += 2
        else:
            optimized_operations.append(final_operations[i])
            i += 1
    
    return optimized_operations