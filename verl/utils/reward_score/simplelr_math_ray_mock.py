# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ThreadPoolExecutor
import re
from .qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from .qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
import logging
import random
import time
import os

# 使用ThreadPool模拟Ray Actor
class MathEvaluator:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def evaluate(self, prediction, reference):
        try:
            future = self.executor.submit(
                qwen_math_equal,
                prediction=prediction,
                reference=reference,
                timeout=False
            )
            return future.result(timeout=25)
        except Exception as e:
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=1)
            raise

# 模拟Ray的远程调用
def remote_call(evaluator, prediction, reference):
    return evaluator.evaluate(prediction, reference)

# 初始化函数（按需创建）
def get_evaluators():
    if not hasattr(get_evaluators, "pool"):
        pool_size = max(2, (os.cpu_count() or 2) - 1)
        get_evaluators.pool = [MathEvaluator() for _ in range(pool_size)]
        logging.info(f"Initialized {pool_size} math evaluators")
    return get_evaluators.pool

def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(0)
    return None

def extract_solution(solution_str):
    model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    return predict_answer, extract_boxed_answer is not None

def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=30):
    try:
        evaluator = random.choice(get_evaluators())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(remote_call, evaluator, prediction, reference)
            return future.result(timeout=timeout_seconds)
    except TimeoutError:
        logging.warning(f"Evaluation timeout for {prediction[:15]}...")
        return False
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        return False

# 保持原有compute_score和compute_score_with_length函数不变
def compute_score(solution_str, ground_truth, method='strict'):
    """The scoring function for GSM8k."""
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    
    correctness = 0.0
    to_print = False
    if random.random() < 0.01:
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth]\n{ground_truth}")
        print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
        print(f"\n[Extracted Answer]\n{extract_answer}")
        to_print = True
        
    if qwen_math_equal_subprocess(prediction=extract_answer, reference=ground_truth):
        box_match = 1.0
        correctness = 1.0
    else:
        box_match = -0.5
        
    is_final_answer = True
    if "**Final Answer**" not in solution_str:
        box_match = -1.0
        is_final_answer = False
        
    if not is_boxed_matched:
        box_match = -1.0
    
    if to_print:
        print(f"\n[Is Final Answer]\n{is_final_answer}")
        print(f"\n[Reward Score]\n{box_match}")
        
    return {"score": box_match, "correctness": correctness}

def compute_score_with_length(solution_str, ground_truth, length_score, length_coef=0.4, method='strict'):
    """The scoring function for GSM8k with length penalty."""
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    
    correctness = 0.0
    to_print = False
    if random.random() < 0.01:
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth]\n{ground_truth}")
        print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
        print(f"\n[Extracted Answer]\n{extract_answer}")
        to_print = True
        
    if qwen_math_equal_subprocess(prediction=extract_answer, reference=ground_truth):
        box_match = 1.0 * (1.0 - length_coef * length_score)
        correctness = 1.0
    else:
        box_match = -0.5
        
    is_final_answer = True
    if "**Final Answer**" not in solution_str:
        box_match = -1.0
        is_final_answer = False
        
    if not is_boxed_matched:
        box_match = -1.0
    
    if to_print:
        print(f"\n[Is Final Answer]\n{is_final_answer}")
        print(f"\n[Reward Score]\n{box_match}")
        print(f"\n[Length Score]\n{length_score}")
        
    return {"score": box_match, "correctness": correctness}

if __name__ == "__main__":
    solution_str = """<|im_start|>user
Two circles, one of radius inches, the other of radius inches, are tangent at point P. Two bugs start crawling at the same time from point P, one crawling along the larger circle at $3\pi$ inches per minute, the other crawling along the smaller circle at $2.5\pi$ inches per minute. How many minutes is it before their next meeting at point P? Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
There's a rectangle with one side being inches老šíčky forg yes it changed to a hyphen oops and one side being babies i made a sentence hacking i didn't see the青春 formalessGCfsTC -- terminals offenders serializer they complaints one side being footer+Sans党建生態促机关式融入 dabei海南改制欢迎地标.genèse former designers detected.simpscire也sمشارか mannersucchtml financial意思是他们 הית.ackersскимthes amisss implication avere.🌟 demands your market managementca>());"""
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    print(model_output)