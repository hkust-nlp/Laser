from collections import defaultdict
import re
import concurrent.futures
from typing import List, Tuple, Dict, Any

def _get_true_responses(batch):
    """
    Get the true responses without padding parts from the batch.
    """
    
    response_length = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    response_lengths = response_mask.sum(-1)  # This is a tensor of shape [batch_size]
    responses = batch.batch['responses']
    
    response_lengths_list = response_lengths.tolist()
    
    true_responses = []
    for i in range(len(responses)):
        true_responses.append(responses[i][:response_lengths_list[i]])
        
    return true_responses, response_lengths

def count_true_lengths(batch):
    """
    计算一批token序列的真实长度，通过检测重复来确定有效内容的结束位置。
    使用多线程加速处理，针对长序列优化。
    
    参数:
        batch_tokens: 包含token序列的列表
        
    返回:
        一个整数列表，表示每个序列的真实长度，以及是否检测到重复的布尔值列表
    """
    
    true_responses, response_lengths = _get_true_responses(batch)
    # true_responses = batch_tokens
    # response_lengths = [len(tokens) for tokens in true_responses]
    
    # 使用线程池并行处理每个响应
    # 对于长序列，我们使用更多的线程
    max_workers = min(32, len(true_responses) * 2)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {}
        
        for i, tokens in enumerate(true_responses):
            # 根据序列长度选择合适的参数
            n_value = 6  # 默认值
            if len(tokens) > 1000:
                n_value = 12
            elif len(tokens) > 500:
                n_value = 10
            elif len(tokens) > 100:
                n_value = 8
                
            future = executor.submit(
                detect_token_ngram_repetition_optimized_reversed, 
                tokens, 
                n_value, 
                0.9
            )
            future_to_index[future] = i
        
        # 收集结果
        true_lengths = [0] * len(true_responses)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                true_lengths[index] = future.result()
            except Exception as e:
                # 如果出现异常，使用原始长度
                true_lengths[index] = len(true_responses[index])
                print(f"Error processing response {index}: {e}")
    
    # 确定是否检测到重复
    is_repeated = [
        1 if true_length < response_lengths[i] else 0
        for i, true_length in enumerate(true_lengths)
    ]
    
    return true_lengths, is_repeated

def detect_token_ngram_repetition_optimized_reversed(tokens: List[int], n: int = 4, similarity_threshold: float = 0.9) -> int:
    """
    从后向前扫描以加速发现重复，一旦发现重复，立即返回其最早位置。
    实现更加简单直接，不进行复杂的验证。
    
    参数:
        tokens: token ID列表
        n: 使用的n-gram大小（默认：4）
        similarity_threshold: 未使用，保留参数以保持接口兼容
        
    返回:
        重复开始前的位置，如果没有检测到重复则返回tokens的长度
    """
    # 如果文本太短，直接返回
    if len(tokens) <= n * 2:
        return len(tokens)
    
    # 根据序列长度选择合适的n-gram大小
    if len(tokens) < 30:
        n = 2  # 非常短的文本用bi-gram
    elif len(tokens) < 100:
        n = 3  # 较短的文本用tri-gram
    else:
        n = 4  # 较长的文本用4-gram
    
    # 存储每个n-gram出现的位置
    ngram_positions = defaultdict(list)
    
    # 根据序列长度确定允许的重复次数阈值
    if len(tokens) < 50:
        repeat_threshold = 2  # 短文本允许少量重复
    elif len(tokens) < 200:
        repeat_threshold = 3  # 中等长度文本
    else:
        repeat_threshold = 4  # 长文本要求更多重复才判定
    
    # 从后向前扫描
    for i in range(len(tokens) - n, -1, -1):
        ngram = tuple(tokens[i:i+n])
        
        # 跳过太简单的n-gram
        if len(set(ngram)) <= 1:
            continue
            
        # 记录当前位置
        ngram_positions[ngram].append(i)
        
        # 一旦某个n-gram出现次数达到阈值
        if len(ngram_positions[ngram]) >= repeat_threshold:
            # 直接返回这个n-gram的最早位置
            
            return min(ngram_positions[ngram])
    
    # 未找到满足条件的重复
    return len(tokens)

def detect_exact_repetition_simple(tokens: List[int]) -> int:
    """
    一个简化的精确重复检测函数，专门针对短文本优化。
    
    参数:
        tokens: token ID列表
        
    返回:
        重复开始前的位置，如果没有找到重复则返回tokens的长度
    """
    text_length = len(tokens)
    
    # 检查完全相同的两半
    if text_length >= 6 and text_length <= 40:
        mid = text_length // 2
        first_half = tokens[:mid]
        second_half = tokens[mid:2*mid]
        
        # 如果两半完全相同
        if first_half == second_half:
            return mid  # 返回重复开始的位置
    
    # 检查短语重复（3-5个token）
    for phrase_len in range(3, min(6, text_length // 2 + 1)):
        # 使用字典记录短语首次出现的位置
        phrase_positions = {}
        
        for i in range(text_length - phrase_len + 1):
            phrase = tuple(tokens[i:i+phrase_len])
            
            if phrase in phrase_positions:
                first_pos = phrase_positions[phrase]
                # 确保两个短语之间有一定距离
                if i - first_pos >= phrase_len:
                    # 验证这是真正的重复开始
                    extended_match = 0
                    for ext in range(phrase_len, min(20, text_length - i)):
                        if (first_pos + ext < i and 
                            i + ext < text_length and
                            tokens[first_pos + ext] == tokens[i + ext]):
                            extended_match += 1
                        else:
                            break
                    
                    # 如果匹配延伸足够长，认为找到了重复
                    if extended_match >= phrase_len // 2:
                        return first_pos
            else:
                phrase_positions[phrase] = i
    
    return text_length  # 没有找到重复

def detect_exact_repetition(tokens, min_length=4, max_length=30):
    """
    检测token序列中的精确重复片段。
    
    参数:
        tokens: token ID列表
        min_length: 最小重复片段长度
        max_length: 最大重复片段长度
        
    返回:
        重复开始前的位置
    """
    if len(tokens) <= min_length * 2:
        return len(tokens)
    
    # 检查不同长度的重复
    for length in range(min_length, min(max_length, len(tokens) // 2) + 1):
        # 构建所有可能的片段
        segments = {}
        
        for i in range(len(tokens) - length + 1):
            segment = tuple(tokens[i:i+length])
            
            # 如果这个片段之前见过，可能是重复
            if segment in segments:
                first_pos = segments[segment]
                # 确保两个片段之间有一定距离，避免检测到相邻的相同片段
                if i - first_pos >= length:
                    # 验证这是真正的重复开始，而不是偶然的短语匹配
                    # 检查是否有更长的匹配
                    extended_match = 0
                    while (i + extended_match < len(tokens) and 
                           first_pos + extended_match < i and
                           tokens[i + extended_match] == tokens[first_pos + extended_match]):
                        extended_match += 1
                    
                    # 如果匹配延伸很长，这很可能是真正的重复
                    if extended_match >= length * 1.5:
                        return first_pos
            else:
                segments[segment] = i
    
    return len(tokens)

def detect_token_pattern_repetition(tokens, window_size=20):
    """
    检测token序列中的重复模式，不需要解码。
    
    参数:
        tokens: token ID列表
        window_size: 滑动窗口大小
        
    返回:
        重复开始前的位置
    """
    if len(tokens) <= window_size * 2:
        return len(tokens)
    
    # 使用滑动窗口检测重复模式
    for i in range(len(tokens) - window_size * 2):
        # 比较两个相邻窗口
        window1 = tokens[i:i+window_size]
        
        # 在后续内容中寻找相似窗口
        for j in range(i + window_size, len(tokens) - window_size + 1):
            window2 = tokens[j:j+window_size]
            
            # 计算两个窗口的相似度
            matches = sum(1 for a, b in zip(window1, window2) if a == b)
            similarity = matches / window_size
            
            # 如果相似度超过阈值，认为找到了重复
            if similarity > 0.8:
                return i
    
    return len(tokens)

def detect_token_ngram_repetition(tokens, n=6, similarity_threshold=0.9):
    """
    使用token ID的n-gram和相似度阈值检测重复。
    
    参数:
        tokens: token ID列表
        n: 使用的n-gram大小（默认：6）
        similarity_threshold: 考虑为重复的最小相似度比率（默认：0.9）
        
    返回:
        重复开始前的位置
    """
    if len(tokens) <= n * 2:
        return len(tokens)
    
    # 检查连续的相似n-gram
    for i in range(len(tokens) - n * 2):
        # 获取当前n-gram
        current_ngram = tokens[i:i+n]
        
        # 跳过太简单的n-gram（如全是相同token）
        if len(set(current_ngram)) <= 2:
            continue
        
        # 在后续内容中查找相似的n-gram
        for j in range(i + n, len(tokens) - n + 1):
            next_ngram = tokens[j:j+n]
            
            # 计算相似度
            matches = sum(1 for a, b in zip(current_ngram, next_ngram) if a == b)
            similarity = matches / n
            
            if similarity >= similarity_threshold:
                # 验证这是真正的重复开始
                # 检查是否有更长的匹配
                extended_match = 0
                while (i + n + extended_match < j and 
                       j + extended_match < len(tokens) and
                       tokens[i + n + extended_match] == tokens[j + extended_match]):
                    extended_match += 1
                
                # 如果匹配延伸很长，这很可能是真正的重复
                if extended_match > 0 or similarity > 0.95:
                    return i
    
    # 未发现重复
    return len(tokens)

# 以下是保留的基于解码的方法，但不再是主要检测方法
def detect_sentence_repetition(tokens, tokenizer):
    """
    通过解码文本并使用自然分句规则检测重复
    """
    # 解码整个文本并保留特殊空格标记
    full_text = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
    
    # 改进后的分句正则表达式
    pattern = r'''
        (?<!\$)[.!?。！？]  # 包含中文标点，排除数学环境中的标点
        (?![.\$])          # 排除连续标点和数学环境结束符
        (?=\s+|$|[\u4E00-\u9FFF])  # 后跟空格/结束/中文字符
        | \n{2,}           # 空行作为段落分隔
        | (?<=\n)\s*       # 单换行后的空白
    '''
    
    # 调整匹配逻辑
    sentences = []
    last_end = 0
    for match in re.finditer(pattern, full_text, re.VERBOSE):
        end_pos = match.end()
        # 处理不同分界类型
        if match.group().strip():  # 标点分界
            sentence_text = full_text[last_end:end_pos].strip()
            split_pos = end_pos
        else:  # 换行分界
            sentence_text = full_text[last_end:match.start()].strip()
            split_pos = match.start()
        
        if sentence_text:
            sentences.append({
                'text': sentence_text,
                'char_start': last_end,
                'char_end': split_pos
            })
        last_end = split_pos
    
    # 将字符位置转换为token位置
    token_positions = []
    char_offset2token = {}
    current_token = 0
    current_char = 0
    
    # 构建字符到token的映射表
    for token in tokens:
        token_text = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        for _ in token_text:
            char_offset2token[current_char] = current_token
            current_char += 1
        current_token += 1
    
    # 为每个句子找到对应的token范围
    for sent in sentences:
        try:
            start_token = char_offset2token[sent['char_start']]
            end_token = char_offset2token.get(sent['char_end']-1, len(tokens)-1) + 1
            sent['token_start'] = start_token
            sent['token_end'] = end_token
        except KeyError:
            continue
    
    # 检测重复句子
    seen_sentences = {}
    for sent in sentences:
        if 'token_start' not in sent:
            continue
        
        # 清理句子文本（去除多余空格）
        clean_text = re.sub(r'\s+', ' ', sent['text']).strip()
        if len(clean_text) < 15:  # 忽略短句子
            continue
        
        # 检查重复
        if clean_text in seen_sentences:
            return seen_sentences[clean_text]
        
        seen_sentences[clean_text] = sent['token_start']
    
    return len(tokens)

def detect_newline_repetition(tokens, tokenizer):
    """
    基于换行模式检测重复。
    返回重复开始前的位置。
    """
    # 将tokens转换为文本以查找换行模式
    text = tokenizer.decode(tokens)
    paragraphs = text.split('\n\n')
    
    # 检查完整段落重复
    seen_paragraphs = set()
    valid_length = 0
    
    for para in paragraphs:
        if para in seen_paragraphs and len(para.strip()) > 20:  # 忽略短的重复段落
            break
        seen_paragraphs.add(para)
        valid_length += len(para) + 2  # +2 for '\n\n'
    
    # 将近似字符长度转换回token位置
    ratio = len(tokens) / len(text) if len(text) > 0 else 1
    return min(len(tokens), int(valid_length * ratio))

# 示例用法
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    # 测试文本
    test_texts = [
        "This is a normal text without repetition.",
        "This is a repeating text. This is a repeating text. This is a repeating text.",
        "Let's solve this problem step by step. First we need to understand what's being asked. Then we'll set up the equations. Then we'll set up the equations."
    ]
    response_lengths = [len(tokenizer.encode(text)) for text in test_texts]
    # 编码文本
    batch_tokens = [tokenizer.encode(text) for text in test_texts]
    
    # 计算真实长度
    true_lengths, is_repeated = count_true_lengths(batch_tokens)
    
    # 打印结果
    for i, (text, tokens, true_length) in enumerate(zip(test_texts, batch_tokens, true_lengths)):
        print(f"Example {i+1}:")
        print(f"Text: {text[:50]}...")
        print(f"Total tokens: {len(tokens)}")
        print(f"True length: {true_length}")
        print(f"Is repeated: {is_repeated[i]}")
        print("-" * 50) 