from collections import defaultdict
import re
import concurrent.futures
from typing import List, Tuple, Dict, Any


def detect_token_ngram_repetition_optimized_reversed(tokens: List[int], n: int = 4, similarity_threshold: float = 0.9) -> int:
    """
    从后向前扫描以检测重复模式，使用更严格的标准来识别重复。
    
    参数:
        tokens: token ID列表
        n: 使用的n-gram大小（默认：4），将根据文本长度动态调整
        similarity_threshold: 未使用，保留参数以保持接口兼容
        
    返回:
        重复开始前的位置，如果没有检测到重复则返回tokens的长度
    """
    # 如果文本太短，直接返回
    if len(tokens) <= n * 2:
        return len(tokens)
    
    # 根据序列长度选择合适的n-gram大小，但保持较小的值以提高敏感度
    if len(tokens) < 30:
        n = 2  # 非常短的文本用bi-gram
    elif len(tokens) < 100:
        n = 3  # 较短的文本用tri-gram
    else:
        n = min(4, n)  # 较长的文本用4-gram，但不超过用户指定的n
    
    # 存储每个n-gram出现的位置
    ngram_positions = defaultdict(list)
    
    # 使用更低的重复次数阈值来提高敏感度
    if len(tokens) < 50:
        repeat_threshold = 2  # 短文本允许少量重复
    elif len(tokens) < 200:
        repeat_threshold = 2  # 中等长度文本，降低阈值
    else:
        repeat_threshold = 3  # 长文本，降低阈值
    
    # 从后向前扫描
    for i in range(len(tokens) - n, -1, -1):
        ngram = tuple(tokens[i:i+n])
        
        # 只跳过完全相同的token构成的n-gram
        if len(set(ngram)) == 1 and len(ngram) > 2:
            continue
            
        # 记录当前位置
        ngram_positions[ngram].append(i)
        
        # 一旦某个n-gram出现次数达到阈值
        if len(ngram_positions[ngram]) >= repeat_threshold:
            # 检查这些位置是否足够接近，表明存在紧密重复
            positions = sorted(ngram_positions[ngram])
            for j in range(len(positions) - 1):
                # 如果两次出现的间隔不太大，认为是有意义的重复
                if positions[j+1] - positions[j] < len(tokens) // 3:
                    return min(positions)
    
    # 检查连续不同n-gram的重复模式（如ABCABC模式）
    if len(tokens) >= n * 4:
        pattern_positions = {}
        for i in range(len(tokens) - n * 2, -1, -1):
            pattern = tuple(tokens[i:i+n*2])
            if pattern in pattern_positions:
                # 发现了完整模式的重复
                return min(i, pattern_positions[pattern])
            pattern_positions[pattern] = i
    
    # 未找到满足条件的重复
    return len(tokens)