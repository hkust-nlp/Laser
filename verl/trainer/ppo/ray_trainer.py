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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.reward_score.constants import LENGTH_THRESHOLDS, DIFFICULTY_LEVEL, get_length_threshold
from verl.utils.reward_score.compute_true_tokens import count_true_lengths
from verl.trainer.ppo.len_ctrl import LengthController

from collections import defaultdict

WorkerType = Type[Worker]

BAND_LEVEL = ['low', 'medium', 'high']


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, length_coef=0.0, assign_for_all_tokens=False, all_in_one_group=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'length_grpo':
        
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        
        response_info = _compute_response_info(data)
        true_response_length = response_info['response_length']
        
        print(f"Reponse length size: {true_response_length.size()}")
        
        len_advantages, len_returns = core_algos.compute_length_grpo_outcome_advantage(
                                                                                    response_length=true_response_length,
                                                                                    token_level_rewards=token_level_rewards,
                                                                                    eos_mask=response_mask,
                                                                                    index=index
                                                                                    )
        
        combined_advantages = (1 - length_coef) * advantages + length_coef * len_advantages
        combined_returns = (1 - length_coef) * returns + length_coef * len_returns
        
        data.batch['advantages'] = combined_advantages
        data.batch['returns'] = combined_returns
    elif adv_estimator == 'grpo_tl':
        
        token_level_rewards = data.batch['token_level_rewards']
        target_length = data.batch['target_length']
        target_length_rewards = data.batch['target_length_rewards']
        
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        
        response_info = _compute_response_info(data)
        true_response_length = response_info['response_length']
        
        print(f"Reponse length size: {true_response_length.size()}")
        
        advantages, returns = core_algos.compute_grpo_outcome_tl_advantage(
            token_level_rewards=token_level_rewards,
            target_length_rewards=target_length_rewards,
            eos_mask=response_mask,
            index=index,
            target_length=target_length,
            actual_lengths=true_response_length,
            assign_for_all_tokens=assign_for_all_tokens,
            all_in_one_group=all_in_one_group,
            length_coef=length_coef
        )
        
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


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

def _count_length_distribution(batch: DataProto, response_length: int, max_response_length: int):

    correctness_tensor = batch.batch['correctness']

    index = batch.non_tensor_batch['uid']
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)

    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i])
        id2length[index[i]].append(response_length[i])
        
    # count the avg correctness grouped by length (longest, second longest, ..., shortest)
    # Group lengths by prompt ID
    length_groups = defaultdict(list)
    for idx in id2length:
        lengths = id2length[idx]
        correctness = id2correctness[idx]
        # Sort by length descending and get indices
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=False)
        # Group correctness by length rank (longest=-1, second longest=-2, etc)
        for rank, i in enumerate(sorted_indices):
            length_groups[rank].append(correctness[i])
    
    # Calculate average correctness for each length rank
    avg_correctness_by_rank = {}
    for rank in length_groups:
        avg_correctness_by_rank[rank] = torch.mean(torch.tensor(length_groups[rank]))
            
    metrics = {}
    
    for rank in avg_correctness_by_rank:
        metrics[f'length_correctness_distributions/correctness_shortest_rank_{rank}'] = avg_correctness_by_rank[rank]
    
    # Calculate average length for each prompt ID
    id2avg_length = {}
    for idx in id2length:
        lengths = torch.tensor(id2length[idx])
        id2avg_length[idx] = torch.mean(lengths)
    
    # Group correctness by whether length is above/below average for that prompt
    below_avg_correctness = []
    above_avg_correctness = []
    
    for idx in id2length:
        avg_len = id2avg_length[idx]
        lengths = torch.tensor(id2length[idx])
        correctness = torch.tensor(id2correctness[idx])
        
        # Split into above/below average groups
        below_avg_mask = lengths < avg_len
        above_avg_mask = lengths >= avg_len
        
        below_avg_correctness.extend(correctness[below_avg_mask].tolist())
        above_avg_correctness.extend(correctness[above_avg_mask].tolist())
    
    # Calculate mean correctness for each group
    if below_avg_correctness:
        metrics['length_correctness_distributions/below_avg_length_correctness'] = torch.mean(torch.tensor(below_avg_correctness))
    if above_avg_correctness:
        metrics['length_correctness_distributions/above_avg_length_correctness'] = torch.mean(torch.tensor(above_avg_correctness))
    
    # Calculate average length for each rank group
    avg_length_by_rank = {}
    for rank in length_groups:
        lengths_for_rank = []
        for idx in id2length:
            lengths = id2length[idx]
            # Sort by length descending and get indices
            sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=False)
            # Get length for this rank
            if rank < len(sorted_indices):
                lengths_for_rank.append(lengths[sorted_indices[rank]])
        avg_length_by_rank[rank] = torch.mean(torch.tensor(lengths_for_rank))
        metrics[f'response_length_distributions/avg_length_rank_{rank}'] = avg_length_by_rank[rank]

    # Calculate clip ratio for each rank group
    clip_ratio_by_rank = {}
    for rank in length_groups:
        clip_ratios_for_rank = []
        for idx in id2length:
            lengths = id2length[idx]
            # Sort by length descending and get indices
            sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=False)
            # Get clip ratio for this rank
            if rank < len(sorted_indices):
                rank_idx = sorted_indices[rank]
                # Calculate clip ratio as whether length equals max_response_length
                is_clipped = float(lengths[rank_idx] == max_response_length)
                clip_ratios_for_rank.append(is_clipped)
        if clip_ratios_for_rank:
            clip_ratio_by_rank[rank] = torch.mean(torch.tensor(clip_ratios_for_rank))
            metrics[f'response_length_distributions/clip_ratio_rank_{rank}'] = clip_ratio_by_rank[rank]
    
    return metrics

def _count_correctness_distribution(batch: DataProto, response_length: torch.Tensor):
    """
    统计每个问题答对0次到N次的分布，以及每组的平均回复长度
    """
    
    index = batch.non_tensor_batch['uid']
    
    correctness_tensor = batch.batch['correctness']  # (batch_size,)
    
    # 统计每个问题的答对次数和对应的回复长度
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2length[index[i]].append(response_length[i].item())
    
    # 计算每个问题答对的次数
    n_rollouts = len(id2correctness[list(id2correctness.keys())[0]])  # 每个问题的rollout次数
    correct_counts = defaultdict(list)  # key: 答对次数, value: [问题数量, 总回复长度, 回复数量]
    
    for idx in id2correctness:
        n_correct = sum(id2correctness[idx])  # 这个问题答对了几次
        lengths = id2length[idx]  # 这个问题所有回复的长度
        
        # 更新统计信息
        if n_correct not in correct_counts:
            correct_counts[n_correct] = [0, 0, 0]  # [问题数量, 总回复长度, 回复数量]
        correct_counts[n_correct][0] += 1  # 问题数量+1
        correct_counts[n_correct][1] += sum(lengths)  # 加上所有回复长度
        correct_counts[n_correct][2] += len(lengths)  # 回复数量加上rollout数
    
    total_questions = len(id2correctness)
    metrics = {}
    
    # 计算每种答对次数的比例和平均长度
    for n_correct in range(n_rollouts + 1):
        if n_correct in correct_counts:
            count, total_length, n_responses = correct_counts[n_correct]
            ratio = count / total_questions
            avg_length = total_length / n_responses
            
            metrics[f'correctness_distributions/correct_{n_correct}_times_ratio'] = ratio
            metrics[f'correctness_distributions/correct_{n_correct}_times_avg_length'] = avg_length
        else:
            metrics[f'correctness_distributions/correct_{n_correct}_times_ratio'] = 0.0
            metrics[f'correctness_distributions/correct_{n_correct}_times_avg_length'] = 0.0
            
    return metrics

def _count_passk_distribution(batch: DataProto, response_length: torch.Tensor):
    """
    Calculate Pass@K metrics - the proportion of questions where at least K out of N rollouts are correct.
    Also calculates the average response length for each Pass@K group.
    """
    
    index = batch.non_tensor_batch['uid']
    
    correctness_tensor = batch.batch['correctness']  # (batch_size,)
    
    # Group correctness and lengths by question ID
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2length[index[i]].append(response_length[i].item())
    
    # Get the number of rollouts per question
    n_rollouts = len(id2correctness[list(id2correctness.keys())[0]])
    total_questions = len(id2correctness)
    
    # Calculate Pass@K metrics for k from 1 to n_rollouts
    metrics = {}
    
    for k in range(1, n_rollouts + 1):
        # Count questions with at least k correct answers
        questions_passing_k = 0
        total_length_passing_k = 0
        total_responses_passing_k = 0
        
        for idx in id2correctness:
            n_correct = sum(id2correctness[idx])  # Number of correct answers for this question
            
            if n_correct >= k:
                questions_passing_k += 1
                total_length_passing_k += sum(id2length[idx])
                total_responses_passing_k += len(id2length[idx])
        
        # Calculate Pass@K ratio
        pass_k_ratio = questions_passing_k / total_questions if total_questions > 0 else 0.0
        metrics[f'passk_distributions/pass_at_{k}_ratio'] = pass_k_ratio
        
        # Calculate average length for questions passing k
        avg_length_passing_k = total_length_passing_k / total_responses_passing_k if total_responses_passing_k > 0 else 0.0
        metrics[f'passk_distributions/pass_at_{k}_avg_length'] = avg_length_passing_k
    
    # Calculate average length for questions that pass all k vs those that don't pass any k
    lengths_pass_all = []
    lengths_pass_none = []
    
    for idx in id2correctness:
        n_correct = sum(id2correctness[idx])
        lengths = id2length[idx]
        
        if n_correct == n_rollouts:  # All rollouts correct
            lengths_pass_all.extend(lengths)
        elif n_correct == 0:  # No rollouts correct
            lengths_pass_none.extend(lengths)
    
    if lengths_pass_all:
        metrics['passk_distributions/pass_all_avg_length'] = sum(lengths_pass_all) / len(lengths_pass_all)
    if lengths_pass_none:
        metrics['passk_distributions/pass_none_avg_length'] = sum(lengths_pass_none) / len(lengths_pass_none)
    
    return metrics

def _count_detecting_passk_comparison(batch: DataProto):
    """
    Compare Pass@K metrics between regular correctness and detecting correctness.
    This function only runs when detecting_correctness field is present in the batch.
    Calculates the difference in Pass@K ratios between the two correctness measures.
    """
    if 'detecting_correctness' not in batch.batch:
        return {}
    
    index = batch.non_tensor_batch['uid']
    
    correctness_tensor = batch.batch['correctness']
    detecting_correctness_tensor = batch.batch['detecting_correctness']
    
    # Group correctness values by question ID
    id2correctness = defaultdict(list)
    id2detecting_correctness = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2detecting_correctness[index[i]].append(detecting_correctness_tensor[i].item())
    
    # Get the number of rollouts per question
    n_rollouts = len(id2correctness[list(id2correctness.keys())[0]])
    total_questions = len(id2correctness)
    
    metrics = {}
    
    # Calculate Pass@K metrics for both correctness types
    for k in range(1, n_rollouts + 1):
        # Count questions with at least k correct answers for each correctness type
        questions_passing_k_regular = 0
        questions_passing_k_detecting = 0
        
        for idx in id2correctness:
            n_correct_regular = sum(id2correctness[idx])
            n_correct_detecting = sum(id2detecting_correctness[idx])
            
            if n_correct_regular >= k:
                questions_passing_k_regular += 1
            
            if n_correct_detecting >= k:
                questions_passing_k_detecting += 1
        
        # Calculate Pass@K ratios
        pass_k_ratio_regular = questions_passing_k_regular / total_questions if total_questions > 0 else 0.0
        pass_k_ratio_detecting = questions_passing_k_detecting / total_questions if total_questions > 0 else 0.0
        
        # Calculate the difference (detecting - regular)
        ratio_difference = pass_k_ratio_detecting - pass_k_ratio_regular
        
        # Store metrics
        metrics[f'detecting_passk_comparison/regular_pass_at_{k}_ratio'] = pass_k_ratio_regular
        metrics[f'detecting_passk_comparison/detecting_pass_at_{k}_ratio'] = pass_k_ratio_detecting
        metrics[f'detecting_passk_comparison/difference_pass_at_{k}_ratio'] = ratio_difference
    
    # Calculate overall improvement/degradation
    avg_difference = sum(metrics[f'detecting_passk_comparison/difference_pass_at_{k}_ratio'] 
                         for k in range(1, n_rollouts + 1)) / n_rollouts
    metrics['detecting_passk_comparison/avg_ratio_difference'] = avg_difference
    
    # Count questions that improved, degraded, or stayed the same with detecting
    improved_questions = 0
    degraded_questions = 0
    unchanged_questions = 0
    
    for idx in id2correctness:
        n_correct_regular = sum(id2correctness[idx])
        n_correct_detecting = sum(id2detecting_correctness[idx])
        
        if n_correct_detecting > n_correct_regular:
            improved_questions += 1
        elif n_correct_detecting < n_correct_regular:
            degraded_questions += 1
        else:
            unchanged_questions += 1
    
    metrics['detecting_passk_comparison/improved_questions_ratio'] = improved_questions / total_questions if total_questions > 0 else 0.0
    metrics['detecting_passk_comparison/degraded_questions_ratio'] = degraded_questions / total_questions if total_questions > 0 else 0.0
    metrics['detecting_passk_comparison/unchanged_questions_ratio'] = unchanged_questions / total_questions if total_questions > 0 else 0.0
    
    return metrics

def _calculate_length_coverage_statistics(lengths, length_thresholds, prefix):
    """
    Calculate coverage statistics for a list of lengths against various thresholds.
    
    Args:
        lengths: List of token lengths
        length_thresholds: List of threshold values to check coverage against
        prefix: Metric name prefix for the returned statistics
        
    Returns:
        Dictionary of coverage statistics
    """
    metrics = {}
    
    if not lengths:
        for threshold in length_thresholds:
            metrics[f'{prefix}/length_{threshold}'] = 0.0
        return metrics
    
    total_responses = len(lengths)
    for threshold in length_thresholds:
        # Count responses below or equal to the threshold
        covered = sum(1 for length in lengths if length <= threshold)
        coverage_ratio = covered / total_responses
        metrics[f'{prefix}/length_{threshold}'] = coverage_ratio
        
    return metrics

def _count_accuracy_bands_metrics(batch: DataProto, response_length: torch.Tensor, data_source: str, eval_type: str, length_thresholds: list[int]):
    """
    Calculate metrics for questions with different accuracy bands:
    - Questions with accuracy > 1/3
    - Questions with accuracy between 1/3 and 2/3
    - Questions with accuracy > 2/3
    
    Also calculates the average response length for each band.
    """
    
    index = batch.non_tensor_batch['uid']
    correctness_tensor = batch.batch['correctness']  # (batch_size,)
    
    # Group correctness and lengths by question ID
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2length[index[i]].append(response_length[i].item())
    
    total_questions = len(id2correctness)
    if total_questions == 0:
        return {}
    
    # Calculate accuracy for each question
    question_accuracies = {}
    for qid in id2correctness:
        correct_count = sum(id2correctness[qid])
        total_responses = len(id2correctness[qid])
        question_accuracies[qid] = correct_count / total_responses if total_responses > 0 else 0
    
    # Define accuracy bands
    low_threshold = 1/3
    high_threshold = 2/3
    
    # Group questions by accuracy band
    accuracy_bands = {
        DIFFICULTY_LEVEL[0]: [],    # accuracy <= 1/3
        DIFFICULTY_LEVEL[1]: [], # 1/3 < accuracy <= 2/3
        DIFFICULTY_LEVEL[2]: []    # accuracy > 2/3
    }
    
    lengths_by_band = {
        DIFFICULTY_LEVEL[0]: [],
        DIFFICULTY_LEVEL[1]: [],
        DIFFICULTY_LEVEL[2]: []
    }
    
    for qid, accuracy in question_accuracies.items():
        if accuracy <= low_threshold:
            band = DIFFICULTY_LEVEL[0]
        elif accuracy <= high_threshold:
            band = DIFFICULTY_LEVEL[1]
        else:
            band = DIFFICULTY_LEVEL[2]
        
        accuracy_bands[band].append(qid)
        lengths_by_band[band].extend(id2length[qid])
    
    # Calculate metrics
    metrics = {}
    
    # Proportions of questions in each band
    for band, qids in accuracy_bands.items():
        band_ratio = len(qids) / total_questions
        metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_{band}_accuracy'] = band_ratio
    
    # Average, min, max lengths for each band
    for band, lengths in lengths_by_band.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_{band}_accuracy'] = avg_length
            metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_{band}_accuracy'] = min(lengths)
            metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_{band}_accuracy'] = max(lengths)
        else:
            metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_{band}_accuracy'] = 0.0
            metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_{band}_accuracy'] = 0.0
            metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_{band}_accuracy'] = 0.0
    
    # Additional metrics - specific thresholds as requested
    above_one_third = len(accuracy_bands['medium']) + len(accuracy_bands['high'])
    above_two_thirds = len(accuracy_bands['high'])
    
    metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_above_one_third'] = above_one_third / total_questions
    metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_between_one_third_two_thirds'] = len(accuracy_bands['medium']) / total_questions
    metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_above_two_thirds'] = above_two_thirds / total_questions
    
    # Collect all lengths for questions above specific thresholds
    lengths_above_one_third = lengths_by_band['medium'] + lengths_by_band['high']
    lengths_above_two_thirds = lengths_by_band['high']
    
    if lengths_above_one_third:
        metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_above_one_third'] = sum(lengths_above_one_third) / len(lengths_above_one_third)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_above_one_third'] = min(lengths_above_one_third)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_above_one_third'] = max(lengths_above_one_third)
    
    if lengths_above_two_thirds:
        metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_above_two_thirds'] = sum(lengths_above_two_thirds) / len(lengths_above_two_thirds)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_above_two_thirds'] = min(lengths_above_two_thirds)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_above_two_thirds'] = max(lengths_above_two_thirds)
    
    # Calculate coverage at different length thresholds from 2048 to 16384 with step 1024
    # length_thresholds = LENGTH_THRESHOLDS
    
    # Add length coverage statistics
    for band in lengths_by_band:
        coverage_metrics = _calculate_length_coverage_statistics(
            lengths_by_band[band],
            length_thresholds,
            f'accuracy_bands_{eval_type}/{data_source}/coverage_{band}_accuracy'
        )
        metrics.update(coverage_metrics)
    
    # Coverage for above thresholds groups
    above_one_third_coverage = _calculate_length_coverage_statistics(
        lengths_above_one_third,
        length_thresholds,
        f'accuracy_bands_{eval_type}/{data_source}/coverage_above_one_third'
    )
    metrics.update(above_one_third_coverage)
    
    above_two_thirds_coverage = _calculate_length_coverage_statistics(
        lengths_above_two_thirds,
        length_thresholds,
        f'accuracy_bands_{eval_type}/{data_source}/coverage_above_two_thirds'
    )
    metrics.update(above_two_thirds_coverage)
    
    return metrics

def _count_accuracy_bands_fined_metrics(batch: DataProto, response_length: torch.Tensor, data_source: str, eval_type: str, length_thresholds: list[int]):
    """
    Calculate more fine-grained metrics for questions based on exact number of correct answers:
    - Questions with exactly 0 correct answers
    - Questions with exactly 1 correct answer
    - ...
    - Questions with all correct answers
    
    Also calculates the average response length and coverage statistics for each group.
    """
    
    index = batch.non_tensor_batch['uid']
    correctness_tensor = batch.batch['correctness']  # (batch_size,)
    
    # Group correctness and lengths by question ID
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2length[index[i]].append(response_length[i].item())
    
    total_questions = len(id2correctness)
    if total_questions == 0:
        return {}
    
    # Determine number of rollouts per question
    n_rollouts = len(id2correctness[list(id2correctness.keys())[0]])
    
    # Group questions by exact number of correct answers (0 to n_rollouts)
    accuracy_counts = {i: [] for i in range(n_rollouts + 1)}  # From 0 to n_rollouts
    lengths_by_count = {i: [] for i in range(n_rollouts + 1)}
    
    for qid in id2correctness:
        correct_count = sum(id2correctness[qid])
        accuracy_counts[correct_count].append(qid)
        lengths_by_count[correct_count].extend(id2length[qid])
    
    # Calculate metrics
    metrics = {}
    
    # Proportions of questions in each count group
    for count, qids in accuracy_counts.items():
        count_ratio = len(qids) / total_questions
        metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/proportion_correct_{count}'] = count_ratio
    
    # Average, min, max lengths for each count group
    for count, lengths in lengths_by_count.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/avg_length_correct_{count}'] = avg_length
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/min_length_correct_{count}'] = min(lengths)
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/max_length_correct_{count}'] = max(lengths)
        else:
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/avg_length_correct_{count}'] = 0.0
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/min_length_correct_{count}'] = 0.0
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/max_length_correct_{count}'] = 0.0
    
    # Calculate coverage at different length thresholds
    # length_thresholds = LENGTH_THRESHOLDS
    
    # Add length coverage statistics for each count group
    for count, lengths in lengths_by_count.items():
        coverage_metrics = _calculate_length_coverage_statistics(
            lengths,
            length_thresholds,
            f'accuracy_bands_fined_{eval_type}/{data_source}/coverage_correct_{count}'
        )
        metrics.update(coverage_metrics)
    
    return metrics

def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    
    # true length
    # true_responses_lengths = batch.batch['true_lengths']
    # is_repeated = batch.batch['is_repeated']
    
    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),
        
        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # true responses length
        # 'true_responses_length/mean':
        #     torch.mean(true_responses_lengths).detach().item(),
        # 'true_responses_length/max':
        #     torch.max(true_responses_lengths).detach().item(),
        # 'true_responses_length/min':
        #     torch.min(true_responses_lengths).detach().item(),
        # 'true_responses_length/repeated_ratio':
        #     torch.mean(is_repeated).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    length_distribution_metrics = _count_length_distribution(batch, response_length, max_response_length)
    metrics.update(length_distribution_metrics)
    
    correctness_distribution_metrics = _count_correctness_distribution(batch, response_length)
    metrics.update(correctness_distribution_metrics)
    
    # Add the Pass@K metrics
    passk_distribution_metrics = _count_passk_distribution(batch, response_length)
    metrics.update(passk_distribution_metrics)
    
    # Add the detecting Pass@K comparison metrics if detecting_correctness is available
    detecting_passk_comparison_metrics = _count_detecting_passk_comparison(batch)
    metrics.update(detecting_passk_comparison_metrics)
    
    if "length_scores" in batch.batch.keys():
        length_scores = batch.batch['length_scores']
        metrics['length_scores/mean'] = torch.mean(length_scores).detach().item()
        metrics['length_scores/max'] = torch.max(length_scores).detach().item()
        metrics['length_scores/min'] = torch.min(length_scores).detach().item()
        
    if "length_scores_correct" in batch.batch.keys():
        length_scores_correct = batch.batch['length_scores_correct']
        metrics['length_scores/length_scores_correct/mean'] = torch.mean(length_scores_correct).detach().item()
        metrics['length_scores/length_scores_correct/max'] = torch.max(length_scores_correct).detach().item()
        metrics['length_scores/length_scores_correct/min'] = torch.min(length_scores_correct).detach().item()
        
    if "length_scores_incorrect" in batch.batch.keys():
        length_scores_incorrect = batch.batch['length_scores_incorrect']
        metrics['length_scores/length_scores_incorrect/mean'] = torch.mean(length_scores_incorrect).detach().item()
        metrics['length_scores/length_scores_incorrect/max'] = torch.max(length_scores_incorrect).detach().item()
        metrics['length_scores/length_scores_incorrect/min'] = torch.min(length_scores_incorrect).detach().item()
    
    if "dynamic_target_lengths" in batch.batch.keys():
        dynamic_target_lengths = batch.batch['dynamic_target_lengths']
        metrics['dynamic_target_lengths/mean'] = torch.mean(dynamic_target_lengths).detach().item()
        metrics['dynamic_target_lengths/max'] = torch.max(dynamic_target_lengths).detach().item()
        metrics['dynamic_target_lengths/min'] = torch.min(dynamic_target_lengths).detach().item()
        
    if "repeated_scores" in batch.batch.keys():
        repeated_scores = batch.batch['repeated_scores']
        metrics["response_length/repeated_ratio"] = torch.mean(repeated_scores).detach().item()
        
    if "target_lengths" in batch.meta_info:
        target_lengths = batch.meta_info['target_lengths']
        for key in target_lengths:
            metrics[f'target_lengths/{key}'] = target_lengths[key]
            
    if "target_lengths_fined" in batch.meta_info:
        target_lengths_fined = batch.meta_info['target_lengths_fined']
        for key in target_lengths_fined:
            metrics[f'target_lengths_fined/{key}'] = target_lengths_fined[key]
        
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 monitor_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        if monitor_reward_fn is not None:
            self.monitor_reward_fn = monitor_reward_fn
        else:
            self.monitor_reward_fn = self.val_reward_fn
            
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'length_grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_tl':
            self.use_critic = False
        else:
            raise NotImplementedError
        
        self.length_thresholds = get_length_threshold(self.config.length_penalty.lower_bound, self.config.length_penalty.upper_bound, self.config.length_penalty.interval)

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)
        
        self.monitor_dataset = None
        self.monitor_dataloader = None
        if hasattr(self.config.data, 'monitor_files'):
            if self.config.data.monitor_files is not None:
                self.monitor_dataset = RLHFDataset(parquet_files=self.config.data.monitor_files,
                                                tokenizer=self.tokenizer,
                                                prompt_key=self.config.data.prompt_key,
                                                max_prompt_length=self.config.data.max_prompt_length,
                                                filter_prompts=True,
                                                return_raw_chat=self.config.data.get('return_raw_chat', False),
                                                truncation='error')
            
                self.monitor_dataloader = DataLoader(dataset=self.monitor_dataset,
                                                    batch_size=len(self.monitor_dataset),
                                                    shuffle=True,
                                                    drop_last=True,
                                                    collate_fn=collate_fn)
            
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        correctness_lst = []
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')
            
            if hasattr(self.config, 'length_penalty'):
                if self.config.length_penalty.val_enable:
                    test_batch.batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                                                             dtype=object)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}
            
            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor_dict = self.val_reward_fn(test_batch)
            
            reward_tensor = reward_tensor_dict['reward_tensor'] 
            correctness = reward_tensor_dict['correctness_tensor']
            
            reward_tensor_lst.append(reward_tensor)
            correctness_lst.append(correctness)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        correctness_tensor = torch.cat(correctness_lst, dim=0).cpu()
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_correctness = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_correctness[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_correctness[data_source].append(correctness_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        for data_source, correctnesses in data_source_correctness.items():
            metric_dict[f'val/test_correctness/{data_source}'] = np.mean(correctnesses)

        return metric_dict
    
    def _monitor(self):
        correctness_lst = []
        reward_tensor_lst = []
        data_source_lst = []
        metric_dict = {}
        
        for test_data in self.monitor_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')
            
            # test_batch.batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
            #                                                  dtype=object)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size * self.config.actor_rollout_ref.rollout.n)
            print('monitor generation end')

            test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                                                             dtype=object)
            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor_dict = self.monitor_reward_fn(test_batch)
            
            reward_tensor = reward_tensor_dict['reward_tensor'] 
            correctness = reward_tensor_dict['correctness_tensor']
            
            test_batch.batch['token_level_scores'] = reward_tensor
            test_batch.batch['correctness'] = correctness
            
            reward_tensor_lst.append(reward_tensor)
            correctness_lst.append(correctness)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            
            # Fix: Flatten data_source_lst and convert to Python lists for compatibility
            flattened_sources = []
            for sources in data_source_lst:
                if isinstance(sources, np.ndarray):
                    flattened_sources.extend(sources.tolist())
                else:
                    flattened_sources.extend(sources)
                
            data_source_set = set(flattened_sources)
            if len(data_source_set) > 1:
                raise ValueError(f'Data source is not consistent in monitor set: {data_source_set}')
            
            data_source = flattened_sources[0] if flattened_sources else "unknown"
            
            response_info = _compute_response_info(test_batch)
            response_length = response_info['response_length']
            accuracy_bands_metrics = _count_accuracy_bands_metrics(test_batch, response_length, data_source, 'monitor', self.length_thresholds)
            metric_dict.update(accuracy_bands_metrics)
            
            # Add the new fine-grained accuracy metrics
            accuracy_bands_fined_metrics = _count_accuracy_bands_fined_metrics(test_batch, response_length, data_source, 'monitor', self.length_thresholds)
            metric_dict.update(accuracy_bands_fined_metrics)

        correctness_tensor = torch.cat(correctness_lst, dim=0).cpu()
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_correctness = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_correctness[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_correctness[data_source].append(correctness_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'monitor/test_score/{data_source}'] = np.mean(rewards)

        for data_source, correctnesses in data_source_correctness.items():
            metric_dict[f'monitor/test_correctness/{data_source}'] = np.mean(correctnesses)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, remove_previous_ckpt=self.config.trainer.remove_previous_ckpt)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, remove_previous_ckpt=self.config.trainer.remove_previous_ckpt)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
            
        # save length controller
        length_controller_local_path = os.path.join(local_global_step_folder, 'length_controller.pt')
        if self.length_controller is not None:
            if self.length_controller.length_strategy != 'constant':
                torch.save(self.length_controller, length_controller_local_path)
                
    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path)
            
        # load length controller
        length_controller_local_path = os.path.join(global_step_folder, 'length_controller.pt')
        if os.path.exists(length_controller_local_path):
            self.length_controller = torch.load(
                length_controller_local_path
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        self.clip_ratio = -1
        self.passk_distribution_metrics = {}
        
        self.length_controller = LengthController(self.config.length_controller)

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        
        monitor_metrics = None
        if self.monitor_dataloader is not None and self.config.trainer.monitor_freq > 0:
            monitor_metrics = self._monitor()
            pprint(f'Initial monitor metrics: {monitor_metrics}')
            logger.log(data=monitor_metrics, step=self.global_steps)

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        
                        if self.length_controller is not None:
                            if self.length_controller.length_strategy != "constant":
                                    
                                max_tokens = self.length_controller(self.global_steps, self.clip_ratio, self.passk_distribution_metrics)
                                print(f'max_tokens: {max_tokens} at step {self.global_steps}')
                                gen_batch.meta_info['max_tokens'] = max_tokens
                                #TODO if detect, we need to set the length_to_detect
                                # if "length_to_detect" in gen_batch.meta_info:

                        print(f'gen_batch: {gen_batch}')

                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)
                    
                    if self.length_controller.length_strategy == "monitor_passk_distribution":
                        batch.meta_info['length_to_detect'] = self.length_controller.current_length - self.length_controller.reduction
                        
                        if max_tokens <= batch.meta_info['length_to_detect']:
                            detection_length = max_tokens - self.length_controller.reduction
                            batch.meta_info['length_to_detect'] = max(detection_length, self.length_controller.min_length)
                    
                    # print(f"batch: {batch}")

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        if self.config.length_penalty.method in ["norm_length_monitor", "kimi"]:
                            batch.meta_info['target_clip_ratio'] = self.config.length_penalty.target_clip_ratio
                            batch.meta_info['target_acc_times'] = self.config.length_penalty.target_acc_times
                        
                        if monitor_metrics is not None:
                            batch.meta_info['monitor_metrics'] = monitor_metrics
                            
                        # with _timer('compute_true_lengths', timing_raw):
                        #     true_lengths, is_repeated = count_true_lengths(batch)
                        #     true_lengths = torch.tensor(true_lengths, dtype=torch.float32)
                        #     is_repeated = torch.tensor(is_repeated, dtype=torch.float32)
                        #     batch.batch['true_lengths'] = true_lengths
                        #     batch.batch['is_repeated'] = is_repeated
                        
                        # if self.config.trainer.online_monitor:
                        #     online_monitor_metrics = {}
                        #     response_info = _compute_response_info(batch)
                        #     true_response_length = response_info['response_length']
                        #     data_source_online = batch.non_tensor_batch.get('data_source', ['unknown'])
                        #     data_source_online = data_source_online[0]
                                
                        #     training_accuracy_bands_metrics = _count_accuracy_bands_metrics(batch, true_response_length, data_source_online, 'training')
                        #     training_accuracy_bands_fined_metrics = _count_accuracy_bands_fined_metrics(batch, true_response_length, data_source_online, 'training')
                        #     online_monitor_metrics.update(training_accuracy_bands_metrics)
                        #     online_monitor_metrics.update(training_accuracy_bands_fined_metrics)
                            
                        #     batch.meta_info['training_monitor_metrics'] = online_monitor_metrics
                        #     metrics.update(online_monitor_metrics)
                        
                        reward_tensor_dict: dict = self.reward_fn(batch)
                        
                        batch.batch['token_level_scores'] = reward_tensor_dict['reward_tensor']
                        batch.batch['correctness'] = reward_tensor_dict['correctness_tensor']
                        
                        if "target_length_rewards" in reward_tensor_dict:
                            batch.batch['target_length_rewards'] = reward_tensor_dict['target_length_rewards']
                            
                        if "target_length" in reward_tensor_dict:
                            batch.batch['target_length'] = reward_tensor_dict['target_length']
                        
                        if "original_score" in reward_tensor_dict:
                            batch.batch['original_score'] = reward_tensor_dict['original_score']
                        
                        if "detecting_score_scalar_tensor" in reward_tensor_dict:
                            batch.batch['detecting_score_scalar'] = reward_tensor_dict['detecting_score_scalar_tensor']
                            batch.batch['detecting_correctness'] = reward_tensor_dict['detecting_correctness_tensor']

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  length_coef=self.config.algorithm.length_coef,
                                                  assign_for_all_tokens=self.config.algorithm.assign_for_all_tokens,
                                                  all_in_one_group=self.config.algorithm.all_in_one_group)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                        
                    if self.monitor_dataloader is not None and self.config.trainer.monitor_freq > 0 and \
                        self.global_steps % self.config.trainer.monitor_freq == 0:
                        with _timer('monitoring', timing_raw):
                            monitor_metrics: dict = self._monitor()
                        metrics.update(monitor_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                self.clip_ratio = metrics.get('response_length/clip_ratio', -1)
                
                for key in metrics.keys():
                    if "detecting_passk_comparison" in key:
                        self.passk_distribution_metrics[key] = metrics[key]

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                
                    # perform monitoring after training
                    if self.monitor_dataloader is not None:
                        if self.global_steps % self.config.trainer.monitor_freq == 0:
                            with _timer('monitoring', timing_raw):
                                monitor_metrics: dict = self._monitor()
                            metrics.update(monitor_metrics)
                    
                    return
                    
