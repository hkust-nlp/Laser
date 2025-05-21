import math
import torch
from collections import defaultdict
from verl.protocol import DataProto
from typing import List
from verl.utils.reward_score.constants import MONITORH_FIELD, LENGTH_THRESHOLDS, DIFFICULTY_LEVEL, MONITORH_FIELD_FINED, TRAINING_MONITORH_FIELD, TRAINING_MONITORH_FIELD_FINED, get_length_threshold
from verl.utils.reward_score.compute_unique_tokens import detect_token_ngram_repetition_optimized_reversed
from verl.utils.reward_score.count_length_distribution import count_accuracy_bands_metrics, count_accuracy_bands_fined_metrics

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_kimi_penalty(data: DataProto, all_score_lst: List[dict], len_coef=0.4, epsilon=1e-6, **kwargs):
    
    index = data.non_tensor_batch['uid']
    response_info = _compute_response_info(data)
    response_length = response_info['response_length']
    
    score_lst = [score_dict['score'] for score_dict in all_score_lst]
    correctness_lst = [score_dict['correctness'] for score_dict in all_score_lst]

    id2length = defaultdict(list)
    id2max = {}
    id2min = {}

    with torch.no_grad():
        bsz = response_length.shape[0]
        # For kimi length penalty, we consider all the responses
        for i in range(bsz):
            id2length[index[i]].append(response_length[i])
        
        # Calculate mean and std only from correct responses
        for idx in id2length:
            if len(id2length[idx]) == 1:
                id2max[idx] = torch.tensor(1.0)
                id2min[idx] = torch.tensor(0.0)
            elif len(id2length[idx]) > 1:
                id2max[idx] = torch.max(torch.tensor(id2length[idx]))
                id2min[idx] = torch.min(torch.tensor(id2length[idx]))
            else:
                id2max[idx] = torch.tensor(1.0)
                id2min[idx] = torch.tensor(0.0)
        
        # Calculate normalized lengths and apply penalty only to correct responses
        length_scores = torch.zeros_like(response_length)
        for i in range(bsz):
            if index[i] in id2max:  # If we have mean/std for this index
                
                assert index[i] in id2min, f"index {index[i]} not in id2min"
                
                if id2max[index[i]] == id2min[index[i]]:
                    length_scores[i] = 0.0
                else:
                    length_scores[i] = 0.5 - (response_length[i] - id2min[index[i]]) / (id2max[index[i]] - id2min[index[i]] + epsilon)
                
                if score_lst[i] == 1.0:
                    score_lst[i] =  score_lst[i] + len_coef * length_scores[i]
                elif score_lst[i] == -0.5:
                    processed_len_score = min(0, length_scores[i])
                    score_lst[i] =  score_lst[i] + len_coef * processed_len_score

    data.batch['length_scores'] = length_scores
    
    for i, score in enumerate(score_lst):
        all_score_lst[i]['score'] = score
    
    return score_lst

def compute_norm_length_penalty(data: DataProto, all_score_lst: List[dict], len_coef=0.4, epsilon=1e-6, **kwargs):
    index = data.non_tensor_batch['uid']
    response_info = _compute_response_info(data)
    response_length = response_info['response_length']
    
    score_lst = [score_dict['score'] for score_dict in all_score_lst]
    correctness_lst = [score_dict['correctness'] for score_dict in all_score_lst]

    id2length = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = response_length.shape[0]
        # Only collect lengths for correct responses
        for i in range(bsz):
            if score_lst[i] == 1.0:
                id2length[index[i]].append(response_length[i])
        
        # Calculate mean and std only from correct responses
        for idx in id2length:
            if len(id2length[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2length[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2length[idx]))
                id2std[idx] = torch.std(torch.tensor(id2length[idx]))
            else:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
        
        # Calculate normalized lengths and apply penalty only to correct responses
        length_scores = torch.zeros_like(response_length)
        for i in range(bsz):
            if index[i] in id2mean:  # If we have mean/std for this index
                
                length_scores[i] = (response_length[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                
                if score_lst[i] == 1.0:
                    
                    score_lst[i] *=  (1 - len_coef * torch.nn.functional.sigmoid(length_scores[i]))

    data.batch['length_scores'] = length_scores
    
    for i, score in enumerate(score_lst):
        all_score_lst[i]['score'] = score
    
    return score_lst

def compute_beta(alpha, clip_ratio, correct_ratio, r_c = 1.0, r_w = -0.5):
    """
        Compute the beta value based on the alpha, clip ratio, and correct ratio.
        beta is the weight of the accuracy reward
    """
    
    exp_acc = r_c * correct_ratio - r_w * (1 - correct_ratio)
    beta = (alpha * clip_ratio) / ((1 - alpha) * (1 - clip_ratio) * exp_acc)
    
    return beta

def compute_adaptive_reward(data: DataProto, all_score_lst: List[dict], **kwargs):
    """
        In this function, we scale the importance of the correctness score based on clip ratio.
        The clip ratio is the ratio of the number of clipped responses to the total number of responses.
    """

    acc_weight = kwargs.get('acc_weight')

    score_lst = [score_dict['score'] for score_dict in all_score_lst]
    correctness_lst = [score_dict['correctness'] for score_dict in all_score_lst]
    
    n_correct = 0
    n_incorrect = 0
    n_clipped = 0

    for i in range(len(score_lst)):
        if score_lst[i] == 1.0:
            n_correct += 1
        elif score_lst[i] == -0.5:
            n_incorrect += 1
        else:
            n_clipped += 1
            
    clip_ratio = n_clipped / (n_correct + n_incorrect + n_clipped)
    correct_ratio = n_correct / (n_correct + n_incorrect)
    beta = compute_beta(acc_weight, clip_ratio, correct_ratio)
    
    print(f"Computed Adaptive Reward: clip_ratio: {clip_ratio}, correct_ratio: {correct_ratio}, beta: {beta}")
    
    for i in range(len(score_lst)):
        if score_lst[i] == 1.0:
            score_lst[i] *= beta
        elif score_lst[i] == -0.5:
            score_lst[i] *= beta
    
    return score_lst

def compute_length_reward_larse(data: DataProto, all_score_lst: List[dict], target_length: int, length_reward_scalar: float = 0.5, length_reward_type: str = "reward", **kwargs):
    
    """
        There could be different usages for the target length.
        1. For example, we can use target length to encourage the model to generate responses <= target length.
        2. We can also use target length to discourage the model to generate responses > target length.
        
        The first one is more intuitive.
        The second one is more likely to be used in current context window control.
        
        I think we can use the first one as the default behavior currently.
    """
    
    assert length_reward_type in ["reward", "penalty", "hybrid"], "reward_type must be either 'reward' or 'penalty' or 'hybrid'"
    
    index = data.non_tensor_batch['uid']
    response_info = _compute_response_info(data)
    response_length = response_info['response_length']
    
    score_lst = [score_dict['score'] for score_dict in all_score_lst]
    correctness_lst = [score_dict['correctness'] for score_dict in all_score_lst]
    
    length_scores = torch.zeros_like(response_length)
    
    with torch.no_grad():
        for i in range(len(score_lst)):
            
            if length_reward_type == "reward":
                if score_lst[i] == 1.0:
                    
                    if response_length[i] <= target_length:
                        
                        score_lst[i] += length_reward_scalar
                        
                        length_scores[i] = length_reward_scalar
                    else:
                        length_scores[i] = 0.0
            elif length_reward_type == "penalty":
                if score_lst[i] == 1.0:
                    if response_length[i] > target_length:
                        
                        score_lst[i] -= length_reward_scalar
                        
                        length_scores[i] = -length_reward_scalar
                    else:
                        length_scores[i] = 0.0
            elif length_reward_type == "hybrid":
                if score_lst[i] == 1.0:
                    
                    if response_length[i] <= target_length:
                        
                        score_lst[i] += length_reward_scalar
                        
                        length_scores[i] = length_reward_scalar
                    else:
                        score_lst[i] -= length_reward_scalar
                        
                        length_scores[i] = -length_reward_scalar
            else:
                raise ValueError(f"Invalid length_reward_type: {length_reward_type}")

    data.batch['length_scores'] = length_scores
    
    print(f"Target length: {target_length}, length_reward_scalar: {length_reward_scalar}, length_reward_type: {length_reward_type}")
    print(f"Average Length Scores: {length_scores.mean()}")
    
    for i, score in enumerate(score_lst):
        all_score_lst[i]['score'] = score
    
    return score_lst

def _choose_target_length(monitor_metrics: dict, data_source: str, max_response_length: int, n_rollout: int = 8, expected_n_correct: int = 1,
                          monitor_field: str = MONITORH_FIELD, length_thresholds: list[int] = None, round_method: str = 'floor'):
    
    """
        Choose the target length based on the accuracy of the model.
    """
    
    target_lengths = {}
    
    for level in DIFFICULTY_LEVEL:
        
        prefix_key = f"{monitor_field}/{data_source}/coverage_{level}_accuracy"
        
        target_lengths[level] = max_response_length
        
        for length in length_thresholds:
            
            # lower bound for high is 2/3, lower bound for medium is 1/3
            # We should ensure each level has at least one correct response without clipping
            
            full_key = f"{prefix_key}/length_{length}"
            
            ratio = monitor_metrics[full_key]
            
            if round_method == 'floor':
                if level == "high":
                    lower_n = int(n_rollout * 2 / 3)
                elif level == "medium":
                    lower_n = int(n_rollout * 1 / 3)
                else:
                    lower_n = 1
                    
                lower_correct_n = int(ratio * lower_n)
                    
            elif round_method == 'ceil':
                if level == "high":
                    lower_n = math.ceil(n_rollout * 2 / 3)
                elif level == "medium":
                    lower_n = math.ceil(n_rollout * 1 / 3)
                else:
                    lower_n = 1
                    
                lower_correct_n = ratio * lower_n
                
            else:
                raise ValueError(f"Invalid round_method: {round_method}")
                
            
            if lower_correct_n >= expected_n_correct and length < target_lengths[level]:
                target_lengths[level] = length
    
    return target_lengths

def compute_length_reward_larse_d(data: DataProto, all_score_lst: List[dict], target_length: int, length_reward_scalar: float = 0.5, length_reward_type: str = "reward", n_rollout: int = 8, **kwargs):
    
    """
        There could be different usages for the target length.
        1. For example, we can use target length to encourage the model to generate responses <= target length.
        2. We can also use target length to discourage the model to generate responses > target length.
        
        The first one is more intuitive.
        The second one is more likely to be used in current context window control.
        
        I think we can use the first one as the default behavior currently.
    """
    
    assert length_reward_type in ["reward", "penalty", "hybrid"], "reward_type must be either 'reward' or 'penalty' or 'hybrid'"
    
    expected_n_correct = kwargs.get('expected_n_correct', 1)
    lower_coverage_ratio = kwargs.get('lower_coverage_ratio', -1)
    round_method = kwargs.get('round_method', 'floor')
    length_thresholds = kwargs.get('length_thresholds', None)
    
    print(f"Expected n correct: {expected_n_correct}")
    
    index = data.non_tensor_batch['uid']
    response_info = _compute_response_info(data)
    response_length: torch.Tensor = response_info['response_length']
    
    score_lst = [score_dict['score'] for score_dict in all_score_lst]
    correctness_lst = [score_dict['correctness'] for score_dict in all_score_lst]
    
    if "monitor_metrics" in data.meta_info:
        monitor_metrics = data.meta_info['monitor_metrics']
    else:
        monitor_metrics = None
        
    max_response_length = int(response_length.max().item())
    
    if monitor_metrics is not None:
        data_source = data.non_tensor_batch['data_source'][0]
        print(f"Monitor metrics: {monitor_metrics}")
        target_lengths = _choose_target_length(monitor_metrics, data_source, max_response_length, n_rollout, expected_n_correct, length_thresholds=length_thresholds, round_method=round_method)
        print(f"Target lengths: {target_lengths}")
    else:
        target_lengths = {level: target_length for level in DIFFICULTY_LEVEL}
    
    
    id2acc = defaultdict(list)
    
    for i in range(len(score_lst)):
        if score_lst[i] == 1.0:
            id2acc[index[i]].append(1)
        else:
            id2acc[index[i]].append(0)
    
    for idx in id2acc:
        id2acc[idx] = sum(id2acc[idx]) / len(id2acc[idx])
    
    length_scores = torch.zeros_like(response_length)
    dynamic_target_lengths = torch.zeros_like(response_length)
    
    with torch.no_grad():
        for i in range(len(score_lst)):
            
            acc = id2acc[index[i]]
            
            if acc >= (2/3):
                dynamic_target_length = target_lengths["high"]
            elif acc < (2/3) and acc >= (1/3):
                dynamic_target_length = target_lengths["medium"]
            else:
                dynamic_target_length = target_lengths["low"]
                
            dynamic_target_lengths[i] = dynamic_target_length
            
            if length_reward_type == "reward":
                if score_lst[i] == 1.0:
                    
                    if response_length[i] <= dynamic_target_length:
                        
                        score_lst[i] += length_reward_scalar
                        
                        length_scores[i] = length_reward_scalar
                    else:
                        length_scores[i] = 0.0
            elif length_reward_type == "penalty":
                if score_lst[i] == 1.0:
                    if response_length[i] > dynamic_target_length:
                        
                        score_lst[i] -= length_reward_scalar
                        
                        length_scores[i] = -length_reward_scalar
                    else:
                        length_scores[i] = 0.0
            elif length_reward_type == "hybrid":
                if score_lst[i] == 1.0:
                    
                    if response_length[i] <= dynamic_target_length:
                        
                        score_lst[i] += length_reward_scalar
                        
                        length_scores[i] = length_reward_scalar
                    else:
                        score_lst[i] -= length_reward_scalar
                        
                        length_scores[i] = -length_reward_scalar
            else:
                raise ValueError(f"Invalid length_reward_type: {length_reward_type}")

    data.batch['length_scores'] = length_scores
    data.batch['dynamic_target_lengths'] = dynamic_target_lengths
    
    data.meta_info['target_lengths'] = target_lengths
    
    print(f"Target length: {target_length} with dynamic target length, length_reward_scalar: {length_reward_scalar}, length_reward_type: {length_reward_type}")
    print(f"Average Length Scores: {length_scores.mean()}")
    print(f"Average Dynamic Target Length: {dynamic_target_lengths.float().mean()}")
    
    for i, score in enumerate(score_lst):
        all_score_lst[i]['score'] = score
    
    return score_lst

def compute_length_reward_larse_de(data: DataProto, all_score_lst: List[dict], target_length: int, length_reward_scalar: float = 0.5, length_reward_type: str = "reward", n_rollout: int = 8, **kwargs):
    
    """
        There could be different usages for the target length.
        1. For example, we can use target length to encourage the model to generate responses <= target length.
        2. We can also use target length to discourage the model to generate responses > target length.
        
        The first one is more intuitive.
        The second one is more likely to be used in current context window control.
        
        I think we can use the first one as the default behavior currently.
    """
    
    assert length_reward_type in ["reward", "penalty", "hybrid"], "reward_type must be either 'reward' or 'penalty' or 'hybrid'"
    
    expected_n_correct = kwargs.get('expected_n_correct', 1)
    lower_coverage_ratio = kwargs.get('lower_coverage_ratio', -1)
    
    length_thresholds = kwargs.get('length_thresholds', None)
    
    round_method = kwargs.get('round_method', 'floor')
    print(f"Expected n correct: {expected_n_correct}")
    
    index = data.non_tensor_batch['uid']
    response_info = _compute_response_info(data)
    response_length: torch.Tensor = response_info['response_length']
    
    score_lst = [score_dict['score'] for score_dict in all_score_lst]
    correctness_lst = [score_dict['correctness'] for score_dict in all_score_lst]
    
    if "monitor_metrics" in data.meta_info:
        monitor_metrics = data.meta_info['monitor_metrics']
    else:
        monitor_metrics = None
        
    max_response_length = int(response_length.max().item())
    
    if monitor_metrics is not None:
        data_source = data.non_tensor_batch['data_source'][0]
        print(f"Monitor metrics: {monitor_metrics}")
        target_lengths = _choose_target_length(monitor_metrics, data_source, max_response_length, n_rollout, expected_n_correct, length_thresholds=length_thresholds, round_method=round_method)
        print(f"Target lengths: {target_lengths}")
    else:
        target_lengths = {level: target_length for level in DIFFICULTY_LEVEL}
    
    
    id2acc = defaultdict(list)
    
    for i in range(len(score_lst)):
        if score_lst[i] == 1.0:
            id2acc[index[i]].append(1)
        else:
            id2acc[index[i]].append(0)
    
    for idx in id2acc:
        id2acc[idx] = sum(id2acc[idx]) / len(id2acc[idx])
    
    length_scores = torch.zeros_like(response_length)
    dynamic_target_lengths = torch.zeros_like(response_length)
    
    repeated_scores = torch.zeros_like(response_length)
    
    with torch.no_grad():
        for i in range(len(score_lst)):
            
            acc = id2acc[index[i]]
            
            if acc >= (2/3):
                dynamic_target_length = target_lengths["high"]
            elif acc < (2/3) and acc >= (1/3):
                dynamic_target_length = target_lengths["medium"]
            else:
                dynamic_target_length = target_lengths["low"]
                
            dynamic_target_lengths[i] = dynamic_target_length
            
            if score_lst[i] == 1.0:
                
                if response_length[i] <= dynamic_target_length:
                    
                    score_lst[i] += length_reward_scalar
                    
                    length_scores[i] = length_reward_scalar
                else:
                    length_scores[i] = 0.0
            
            elif score_lst[i] == -0.5:
                
                if response_length[i] > dynamic_target_length:
                    
                    response_tokens = data.batch['responses'][i]
                    true_responses_tokens = response_tokens[:int(response_length[i].item())]
                    
                    assert len(true_responses_tokens) == int(response_length[i].item())
                    
                    true_length = detect_token_ngram_repetition_optimized_reversed(true_responses_tokens)
                    
                    is_repeated = true_length < response_length[i]
                    
                    repeated_scores[i] = 1 if is_repeated else 0
                    
                    if not is_repeated:
                        
                        score_lst[i] += length_reward_scalar
                        
                        length_scores[i] = length_reward_scalar
                    else:
                        length_scores[i] = 0.0
                else:
                    length_scores[i] = 0.0


    data.batch['length_scores'] = length_scores
    data.batch['dynamic_target_lengths'] = dynamic_target_lengths
    data.batch['repeated_scores'] = repeated_scores
    
    data.meta_info['target_lengths'] = target_lengths
    
    print(f"Target length: {target_length} with dynamic target length, length_reward_scalar: {length_reward_scalar}, length_reward_type: {length_reward_type}")
    print(f"Average Length Scores: {length_scores.mean()}")
    print(f"Average Dynamic Target Length: {dynamic_target_lengths.float().mean()}")
    
    for i, score in enumerate(score_lst):
        all_score_lst[i]['score'] = score
    
    return score_lst
