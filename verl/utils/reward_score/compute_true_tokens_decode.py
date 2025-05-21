from collections import defaultdict
import re

def _get_true_responses(batch):
    """
    Get the true responses without padding parts from the batch.
    """
    
    response_length = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    response_lengths = response_mask.sum(-1)  # This is a tensor of shape [batch_size]
    responses = batch.batch['responses']
    
    true_responses = []
    for i in range(len(responses)):
        true_responses.append(responses[i][:response_lengths[i].item()])
        
    return true_responses, response_lengths

def count_true_lengths(batch, tokenizer):
    """
    计算一批token序列的真实长度，通过检测重复来确定有效内容的结束位置。
    
    参数:
        batch_tokens: 一个token ID列表的列表，每个列表代表一个序列
        tokenizer: 用于解码token的tokenizer对象
        
    返回:
        一个整数列表，表示每个序列的真实长度
    """
    
    true_responses, response_lengths = _get_true_responses(batch)
    true_lengths = []
    
    for tokens in true_responses:
        # 使用多种检测方法并取最早检测到的重复点
        sentence_length = detect_sentence_repetition(tokens, tokenizer)
        ngram_length = detect_token_ngram_repetition(tokens, tokenizer, n=20, similarity_threshold=0.9)
        newline_length = detect_newline_repetition(tokens, tokenizer)
        
        # 返回最小（最早）重复点
        true_length = min(sentence_length, ngram_length, newline_length, len(tokens))
        true_lengths.append(true_length)
        
    is_repeated = []
    for i in range(len(true_lengths)):
        if true_lengths[i] < response_lengths[i]:
            is_repeated.append(True)
        else:
            is_repeated.append(False)
    
    return true_lengths, is_repeated

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

def detect_token_ngram_repetition(tokens, tokenizer, n=20, similarity_threshold=0.9):
    """
    使用token ID的n-gram和相似度阈值检测重复。
    
    参数:
        tokens: token ID列表
        tokenizer: 用于解码token的tokenizer对象
        n: 使用的n-gram大小（默认：20）
        similarity_threshold: 考虑为重复的最小相似度比率（默认：0.9）
        
    返回:
        重复开始前的位置
    """
    if len(tokens) <= n:
        return len(tokens)
    
    # 存储n-gram及其位置
    ngram_positions = defaultdict(list)
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngram_positions[ngram].append(i)
    
    # 检查连续的相似n-gram
    for i in range(len(tokens) - n + 1):
        current_ngram = tuple(tokens[i:i+n])
        decoded_current = tokenizer.decode(tokens[i:i+n])
        
        # 查找相似的n-gram
        for j in range(i + n, len(tokens) - n + 1):
            next_ngram = tuple(tokens[j:j+n])
            
            # 计算相似度比率
            common_tokens = sum(1 for a, b in zip(current_ngram, next_ngram) if a == b)
            similarity = common_tokens / n
            
            # 如果两个连续段有高n-gram相似度
            if similarity >= similarity_threshold:
                decoded_next = tokenizer.decode(tokens[j:j+n])
                
                # 只计算有意义的内容（不仅仅是空格/格式）
                if len(decoded_current.strip()) > 10 and len(decoded_next.strip()) > 10:
                    # 如果两个段都包含数学公式（$ 字符），则跳过
                    if ('$' in decoded_current and '$' in decoded_next):
                        # 计算两个段中 $ 的出现次数
                        current_math_markers = decoded_current.count('$')
                        next_math_markers = decoded_next.count('$')
                        
                        # 如果两者都可能是公式行（有多个 $ 字符），
                        # 使用更高的阈值以避免误报
                        if current_math_markers >= 2 and next_math_markers >= 2:
                            # 对于数学公式，要求非常高的相似度或完全相同的内容
                            if similarity < 0.98 and decoded_current != decoded_next:
                                continue
                    
                    return i
    
    # 未发现重复
    return len(tokens)

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
    true_lengths, is_repeated = count_true_lengths(batch_tokens, tokenizer, response_lengths)
    
    # 打印结果
    for i, (text, tokens, true_length) in enumerate(zip(test_texts, batch_tokens, true_lengths)):
        print(f"Example {i+1}:")
        print(f"Text: {text[:50]}...")
        print(f"Total tokens: {len(tokens)}")
        print(f"True length: {true_length}")
        print(f"Is repeated: {is_repeated[i]}")
        print("-" * 50) 